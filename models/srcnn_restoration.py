"""
SRCNN étendu pour Restauration d'Images
Super-Resolution Convolutional Neural Network (Dong et al. 2014)
adapté pour la restauration générale (débruitage, défloutage, JPEG)

Architecture originale SRCNN : 3 couches seulement.
Cette version étend à SRCNN-Extended avec :
  - Résidual blocks (SRCNN-R) pour aller plus profond sans dégradation
  - Connexion résiduelle globale image → sortie (residual learning)
  - Multi-scale feature extraction en entrée
  - Compatible train.py : retourne (recon, None, None)

Avantages vs U-Net / VAE :
  - Très léger (< 1 M params en config standard)
  - Rapide à l'inférence (pas d'upsampling/downsampling)
  - Idéal quand l'image dégradée et propre ont la même résolution
  - Excellent point de départ / baseline très rapide

Limitations :
  - Pas de champ réceptif global (contrairement aux modèles avec pooling)
  - Moins efficace sur dégradations structurelles larges

Conseil d'usage :
  - Débruitage (bruit blanc, AWGN) : SRCNN > U-Net souvent
  - Super-résolution ×2/×4 : utiliser la version avec upsampling
  - Artefacts JPEG légers : SRCNN compétitif, rapide
  - Flou fort / pluie / brouillard : préférer U-Net ou VAE-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Bloc résiduel standard : Conv-BN-ReLU-Conv-BN + skip."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class SRCNN_Restoration(nn.Module):
    """
    SRCNN étendu pour restauration d'images (même résolution entrée/sortie).

    Architecture :
        1. Extraction multi-échelle  : convolutions 3×3, 5×5, 7×7 en parallèle
        2. Feature fusion            : concaténation + projection
        3. Residual blocks profonds  : N blocs pour mapping non-linéaire
        4. Reconstruction            : Conv 3×3 → image + résidu global

    Flux (feature_channels=64, num_residual_blocks=8) :
        Input (3, H, W)
          ↓  extraction multi-échelle : 3 branches → 3×64 = 192 canaux
          ↓  fusion : 192 → 64
          ↓  8 × ResidualBlock(64)
          ↓  Conv 3×3 → 3
          ↓  + résidu global (image dégradée)
        Output (3, H, W)
    """

    def __init__(
        self,
        input_channels=3,
        feature_channels=64,
        num_residual_blocks=8,
        use_residual_learning=True,    # image_out = net(x) + x
    ):
        super().__init__()

        self.use_residual_learning = use_residual_learning
        fc = feature_channels

        # ── 1. EXTRACTION MULTI-ÉCHELLE ─────────────────────────────
        # Inspiré de l'observation de Dong et al. : filtres larges capturent
        # les structures basse fréquence, filtres fins les hautes fréquences.
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, fc, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(input_channels, fc, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(input_channels, fc, 7, padding=3),
            nn.ReLU(inplace=True),
        )

        # ── 2. FUSION ───────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Conv2d(fc * 3, fc, 1),   # 1×1 conv pour réduire la dimension
            nn.BatchNorm2d(fc),
            nn.ReLU(inplace=True),
        )

        # ── 3. RESIDUAL BLOCKS ──────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(fc) for _ in range(num_residual_blocks)]
        )

        # ── 4. RECONSTRUCTION ───────────────────────────────────────
        self.reconstruct = nn.Sequential(
            nn.Conv2d(fc, fc // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fc // 2, input_channels, 3, padding=1),
        )

        # Tanh final uniquement si on n'utilise pas le residual learning
        # (avec residual, la sortie = résidu + image ∈ [-1,1] déjà normalisé)
        if not use_residual_learning:
            self.output_act = nn.Tanh()
        else:
            self.output_act = None

        self._init_weights()

    # ── init ─────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        # Dernière couche de reconstruction initialisée à zéro
        # → le réseau démarre comme une identité (bonne pratique residual learning)
        nn.init.zeros_(self.reconstruct[-1].weight)
        nn.init.zeros_(self.reconstruct[-1].bias)

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, x):
        # Multi-échelle
        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7(x)

        # Fusion
        h = self.fusion(torch.cat([f3, f5, f7], dim=1))

        # Residual blocks
        h = self.res_blocks(h)

        # Reconstruction
        residual = self.reconstruct(h)

        if self.use_residual_learning:
            # Apprend le résidu : net(x) + x → l'image propre
            out = torch.clamp(x + residual, -1.0, 1.0)
        else:
            out = self.output_act(residual)

        return out, None, None   # (recon, None, None) → KL ignoré dans vae_loss

    def sample(self, x, num_samples=1):
        """Déterministe."""
        recon, _, _ = self.forward(x)
        return recon.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)


class SRCNN_Lite(nn.Module):
    """
    SRCNN original fidèle à Dong et al. (2014) : 3 couches uniquement.
    Très léger (~57k paramètres). Utile comme baseline ultra-rapide.

    Couche 1 : extraction de patches (9×9)
    Couche 2 : mapping non-linéaire (1×1)
    Couche 3 : reconstruction (5×5)
    """

    def __init__(self, input_channels=3, n1=64, n2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n1, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(n1, n2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n2, input_channels, 5, padding=2),
        )

    def forward(self, x):
        out = torch.clamp(x + self.net(x), -1.0, 1.0)
        return out, None, None


def srcnn_loss(recon, target, mu=None, logvar=None,
               beta=0.0, perceptual_weight=0.1, lpips_fn=None):
    """
    Loss SRCNN = MSE + Perceptual.
    Signature identique à vae_loss pour interchangeabilité dans train.py.
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    perceptual_loss = torch.tensor(0.0, device=recon.device)
    if perceptual_weight > 0 and lpips_fn is not None:
        perceptual_loss = lpips_fn(recon, target).mean()

    total = recon_loss + perceptual_weight * perceptual_loss
    return total, {
        'total':          total.item(),
        'reconstruction': recon_loss.item(),
        'kl':             0.0,
        'perceptual':     perceptual_loss.item(),
    }


if __name__ == '__main__':
    print("=== SRCNN Extended ===")
    model = SRCNN_Restoration(feature_channels=64, num_residual_blocks=8)
    x = torch.randn(2, 3, 128, 128)
    recon, _, _ = model(x)
    print(f'Input : {x.shape}')
    print(f'Recon : {recon.shape}')
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n:,}')

    print("\n=== SRCNN Lite (original) ===")
    model_lite = SRCNN_Lite()
    recon_lite, _, _ = model_lite(x)
    print(f'Recon : {recon_lite.shape}')
    n_lite = sum(p.numel() for p in model_lite.parameters() if p.requires_grad)
    print(f'Params: {n_lite:,}')

    print('\nSRCNN OK')