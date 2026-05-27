"""
GAN pour Restauration d'Images
Generator : U-Net (reprend unet_restoration.py)
Discriminator : PatchGAN (70×70) — standard pour les tâches image-to-image

Architecture inspirée de pix2pix (Isola et al. 2017) adaptée au débruitage.

Loss :
    Generator     : L1 + Perceptual + Adversarial (BCE ou LSGAN)
    Discriminator : BCE ou LSGAN sur patchs réels/faux

Compatible avec train_gan.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  GENERATOR  (U-Net identique à unet_restoration.py)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Generator(nn.Module):
    """
    U-Net Generator pour restauration d'images.

    Flux des canaux (base_channels=64) :
        enc0 :  3 →  64,  H/2
        enc1 :  64 → 128, H/4
        enc2 : 128 → 256, H/8
        enc3 : 256 → 512, H/16
        bottleneck : 512+512 → 256  (cat + dec, même résolution)
        up → cat → dec × 3
    """

    def __init__(self, input_channels=3, output_channels=3, base_channels=64):
        super().__init__()
        bc = base_channels

        # ── Encodeur ────────────────────────────────────────────────
        self.enc0 = self._enc_block(input_channels, bc)
        self.enc1 = self._enc_block(bc,      bc * 2)
        self.enc2 = self._enc_block(bc * 2,  bc * 4)
        self.enc3 = self._enc_block(bc * 4,  bc * 8)

        # ── Bottleneck ──────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bc * 8,  bc * 16, 3, padding=1),
            nn.BatchNorm2d(bc * 16), nn.ReLU(inplace=True),
            nn.Conv2d(bc * 16, bc * 8,  3, padding=1),
            nn.BatchNorm2d(bc * 8),  nn.ReLU(inplace=True),
        )

        # ── Décodeur ────────────────────────────────────────────────
        # Ordre correct : cat(même résolution) → dec → up
        self.dec0 = self._dec_block(bc * 8 * 2, bc * 4)
        self.up0  = nn.ConvTranspose2d(bc * 4, bc * 4, 4, stride=2, padding=1)

        self.dec1 = self._dec_block(bc * 4 * 2, bc * 2)
        self.up1  = nn.ConvTranspose2d(bc * 2, bc * 2, 4, stride=2, padding=1)

        self.dec2 = self._dec_block(bc * 2 * 2, bc)
        self.up2  = nn.ConvTranspose2d(bc, bc, 4, stride=2, padding=1)

        self.dec3 = self._dec_block(bc * 2, bc)
        self.up3  = nn.ConvTranspose2d(bc, bc, 4, stride=2, padding=1)

        self.final_conv = nn.Sequential(
            nn.Conv2d(bc, output_channels, 1),
            nn.Tanh()
        )
        self._init_weights()

    def _enc_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def _dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.final_conv[0].weight)
        nn.init.zeros_(self.final_conv[0].bias)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bottleneck(s3)

        h = self.up0(self.dec0(torch.cat([b,  s3], dim=1)))
        h = self.up1(self.dec1(torch.cat([h,  s2], dim=1)))
        h = self.up2(self.dec2(torch.cat([h,  s1], dim=1)))
        h = self.up3(self.dec3(torch.cat([h,  s0], dim=1)))

        return self.final_conv(h), None, None   # interface commune (recon, None, None)

    def sample(self, x, num_samples=1):
        recon, _, _ = self.forward(x)
        return recon.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)


# ─────────────────────────────────────────────────────────────────────────────
#  DISCRIMINATOR  (PatchGAN 70×70)
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator — classifie des patchs 70×70 comme réels/faux.
    Entrée : concat(image_dégradée, image_propre_ou_générée) → 6 canaux.

    Avantage vs discriminateur global :
    - Plus de gradients (un score par patch, pas un seul scalaire)
    - Meilleure capture des textures haute fréquence
    - Moins de paramètres, plus stable à l'entraînement

    Architecture standard pix2pix :
        C64 → C128 → C256 → C512 → Conv 1 canal (score)
    """

    def __init__(self, input_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        layers = []
        in_ch  = input_channels * 2   # on conditionne : concat(dégradée, propre/fausse)

        # Première couche : pas de BN
        layers += [
            nn.Conv2d(in_ch, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            stride  = 2 if i < n_layers - 1 else 1
            layers += [
                nn.Conv2d(ch, ch_next, 4, stride=stride, padding=1),
                nn.BatchNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next

        # Couche finale : 1 canal, score par patch
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, degraded, target):
        """
        Args:
            degraded : (B, 3, H, W) — image dégradée (condition)
            target   : (B, 3, H, W) — image propre réelle ou générée
        Returns:
            score : (B, 1, H', W') — carte de scores par patch
        """
        x = torch.cat([degraded, target], dim=1)
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class GANLoss(nn.Module):
    """
    Loss adversariale avec deux modes :
        'bce'   : Binary Cross-Entropy classique (pix2pix original)
        'lsgan' : Least-Squares GAN — plus stable, gradients non nuls partout
    """

    def __init__(self, mode='lsgan'):
        super().__init__()
        assert mode in ('bce', 'lsgan'), f"mode doit être 'bce' ou 'lsgan', reçu : {mode}"
        self.mode = mode

        if mode == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.MSELoss()

    def _label(self, pred, is_real):
        val = 1.0 if is_real else 0.0
        return torch.full_like(pred, val)

    def forward(self, pred, is_real):
        return self.loss(pred, self._label(pred, is_real))


def generator_loss(
    fake_pred,          # sortie discriminateur sur image générée
    recon,              # image générée
    target,             # image propre cible
    lambda_l1=100.0,    # poids L1 (pix2pix utilise 100)
    lambda_perceptual=0.1,
    lpips_fn=None,
    gan_loss_fn=None,
):
    """
    Loss Generator = Adversarial + λ_L1 * L1 + λ_perceptual * LPIPS

    Retourne (total_loss, dict) — dict compatible avec train.py.
    """
    # Adversarial : le générateur veut tromper le discriminateur
    adv_loss = gan_loss_fn(fake_pred, is_real=True) if gan_loss_fn else torch.tensor(0.)

    # L1 : fidélité pixel (meilleure que MSE pour préserver les contours)
    l1_loss = F.l1_loss(recon, target)

    # Perceptual
    perc_loss = torch.tensor(0.0, device=recon.device)
    if lambda_perceptual > 0 and lpips_fn is not None:
        perc_loss = lpips_fn(recon, target).mean()

    total = adv_loss + lambda_l1 * l1_loss + lambda_perceptual * perc_loss

    return total, {
        'total':         total.item(),
        'adversarial':   adv_loss.item(),
        'l1':            l1_loss.item(),
        'perceptual':    perc_loss.item(),
        # aliases pour compatibilité avec train.py / TensorBoard
        'reconstruction': l1_loss.item(),
        'kl':             0.0,
    }


def discriminator_loss(real_pred, fake_pred, gan_loss_fn):
    """
    Loss Discriminateur = 0.5 * (loss_réel + loss_faux)
    Le facteur 0.5 ralentit la mise à jour du discriminateur (stabilité).
    """
    loss_real = gan_loss_fn(real_pred, is_real=True)
    loss_fake = gan_loss_fn(fake_pred, is_real=False)
    total = 0.5 * (loss_real + loss_fake)
    return total, {
        'total':      total.item(),
        'loss_real':  loss_real.item(),
        'loss_fake':  loss_fake.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TEST RAPIDE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, C, H, W = 2, 3, 128, 128

    G = Generator(base_channels=64)
    D = PatchDiscriminator(base_channels=64)

    x        = torch.randn(B, C, H, W)
    target   = torch.randn(B, C, H, W)
    recon, _, _ = G(x)

    real_pred = D(x, target)
    fake_pred = D(x, recon.detach())

    print(f"Input    : {x.shape}")
    print(f"Recon    : {recon.shape}")
    print(f"D(real)  : {real_pred.shape}")
    print(f"D(fake)  : {fake_pred.shape}")

    n_G = sum(p.numel() for p in G.parameters() if p.requires_grad)
    n_D = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Params G : {n_G:,}")
    print(f"Params D : {n_D:,}")
    print("GAN OK")