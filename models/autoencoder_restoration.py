"""
Autoencoder pour Restauration d'Images
Architecture déterministe (pas d'espace latent probabiliste)

Différences vs VAE_Restoration :
- Pas de reparameterization trick (mu/logvar → z direct)
- Pas de KL divergence dans la loss
- Espace latent compact et régulier via L2 regularization optionnelle
- Plus rapide à entraîner, plus stable sur petits datasets
- Moins de diversité dans les reconstructions (déterministe)

Cas d'usage recommandé :
- Débruitage simple (bruit gaussien, JPEG, compression)
- Baseline rapide avant d'essayer le VAE ou U-Net
- Datasets < 10 000 images

Compatible avec train.py : retourne (recon, z, None)
→ mu=z, logvar=None → vae_loss avec beta=0 ignore le KL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder_Restoration(nn.Module):
    """
    Autoencoder convolutionnel pour restauration d'images.

    Architecture :
        Encodeur : 4 blocs Conv+BN+LeakyReLU avec stride 2
        Bottleneck : projection FC vers espace latent compact
        Décodeur : 4 blocs ConvTranspose+BN+ReLU
    """

    def __init__(
        self,
        input_channels=3,
        latent_dim=256,
        image_size=128,
        base_channels=64
    ):
        super().__init__()

        self.input_channels = input_channels
        self.latent_dim     = latent_dim
        self.image_size     = image_size
        self.encoded_size   = image_size // 16   # 128→8, 256→16
        self.encoded_ch     = base_channels * 8  # 512
        bc = base_channels

        # ── ENCODEUR ────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            self._enc_block(input_channels, bc),        # →  64, H/2
            self._enc_block(bc,      bc * 2),           # → 128, H/4
            self._enc_block(bc * 2,  bc * 4),           # → 256, H/8
            self._enc_block(bc * 4,  bc * 8),           # → 512, H/16
        )

        # ── BOTTLENECK ──────────────────────────────────────────────
        flat = self.encoded_ch * self.encoded_size * self.encoded_size
        self.fc_encode = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)

        # ── DÉCODEUR ────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            self._dec_block(bc * 8, bc * 4),   # → 256, H/8
            self._dec_block(bc * 4, bc * 2),   # → 128, H/4
            self._dec_block(bc * 2, bc),        # →  64, H/2
            self._dec_block(bc,     bc),        # →  64, H
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(bc, input_channels, 3, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    # ── blocs de base ────────────────────────────────────────────────

    def _enc_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),  # downsample
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
        )

    def _dec_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),  # upsample
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ── forward ──────────────────────────────────────────────────────

    def encode(self, x):
        """Retourne le vecteur latent z (déterministe)."""
        h = self.encoder(x)
        z = self.fc_encode(h.view(h.size(0), -1))
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.encoded_ch, self.encoded_size, self.encoded_size)
        return self.final_conv(self.decoder(h))

    def forward(self, x):
        """
        Retourne (recon, z, None) pour compatibilité avec vae_loss(beta=0).
        mu=z, logvar=None → le terme KL est ignoré si beta=0.
        """
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z, None

    def sample(self, x, num_samples=1):
        """Déterministe : toutes les reconstructions sont identiques."""
        recon, _, _ = self.forward(x)
        return recon.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)


def autoencoder_loss(recon, target, z=None, logvar=None,
                     beta=0.0, perceptual_weight=0.1, lpips_fn=None,
                     latent_reg=1e-4):
    """
    Loss Autoencoder = MSE + Perceptual + régularisation L2 latent (optionnel).

    Args:
        latent_reg : poids de la régularisation L2 sur z.
                     Encourage un espace latent compact, similaire à une KL douce.
                     Mettre à 0.0 pour désactiver.

    Signature compatible avec vae_loss pour train.py.
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # Régularisation L2 sur le vecteur latent (remplace KL)
    latent_loss = torch.tensor(0.0, device=recon.device)
    if latent_reg > 0 and z is not None:
        latent_loss = latent_reg * z.pow(2).mean()

    perceptual_loss = torch.tensor(0.0, device=recon.device)
    if perceptual_weight > 0 and lpips_fn is not None:
        perceptual_loss = lpips_fn(recon, target).mean()

    total = recon_loss + latent_loss + perceptual_weight * perceptual_loss
    return total, {
        'total':          total.item(),
        'reconstruction': recon_loss.item(),
        'kl':             latent_loss.item(),   # affiché comme 'kl' pour compatibilité
        'perceptual':     perceptual_loss.item(),
    }


if __name__ == '__main__':
    model = Autoencoder_Restoration(image_size=128, latent_dim=256)
    x = torch.randn(2, 3, 128, 128)
    recon, z, _ = model(x)
    print(f'Input  : {x.shape}')
    print(f'Recon  : {recon.shape}')
    print(f'Latent : {z.shape}')
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params : {n:,}')
    print('Autoencoder OK')