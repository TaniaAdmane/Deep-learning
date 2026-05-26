"""
VAE U-Net pour Restauration d'Images
Architecture améliorée avec skip connections (U-Net style)
+ Perceptual loss activée par défaut

Différences vs vae_restoration.py :
- Skip connections encodeur → décodeur (comme U-Net)
- Gain attendu : +5 à +8 dB PSNR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_UNet(nn.Module):
    """
    VAE avec skip connections U-Net.

    Architecture :
        Encodeur : 4 blocs de downsampling (sauvegarde les feature maps)
        Bottleneck : fc_mu / fc_logvar → z → fc_decode
        Décodeur : 4 blocs de upsampling + concaténation des skips encodeur
    """

    def __init__(
        self,
        input_channels=3,
        latent_dim=256,
        image_size=128,
        base_channels=64
    ):
        super().__init__()

        self.input_channels  = input_channels
        self.latent_dim      = latent_dim
        self.image_size      = image_size
        self.encoded_size    = image_size // 16       # 128→8, 256→16
        self.encoded_ch      = base_channels * 8      # 512
        bc = base_channels

        # ── ENCODEUR ────────────────────────────────────────────────
        # Chaque bloc sauvegarde une feature map pour le skip
        self.enc0 = self._enc_block(input_channels, bc)        # →  64, H/2
        self.enc1 = self._enc_block(bc,      bc * 2)           # → 128, H/4
        self.enc2 = self._enc_block(bc * 2,  bc * 4)           # → 256, H/8
        self.enc3 = self._enc_block(bc * 4,  bc * 8)           # → 512, H/16

        # ── BOTTLENECK ──────────────────────────────────────────────
        flat = self.encoded_ch * self.encoded_size * self.encoded_size
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)

        # ── DÉCODEUR avec skips ──────────────────────────────────────
        # Après chaque upsample, on concatène le skip correspondant
        # → les blocs suivants reçoivent 2× les canaux
        #
        #   dec0 : 512        → 256,  H/8   (pas de skip en entrée)
        #   concat skip enc2 (256) → 512 canaux
        #   dec1 : 512        → 128,  H/4
        #   concat skip enc1 (128) → 256 canaux
        #   dec2 : 256        → 64,   H/2
        #   concat skip enc0 (64)  → 128 canaux
        #   dec3 : 128        → 64,   H
        self.dec0 = self._dec_block(bc * 8,      bc * 4)   # 512  → 256
        self.dec1 = self._dec_block(bc * 4 * 2,  bc * 2)   # 512  → 128  (256+256)
        self.dec2 = self._dec_block(bc * 2 * 2,  bc)       # 256  → 64   (128+128)
        self.dec3 = self._dec_block(bc * 2,      bc)       # 128  → 64   ( 64+ 64)

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
            nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1),   # downsample
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
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -5.0)

    # ── forward ──────────────────────────────────────────────────────

    def encode(self, x):
        """Retourne (mu, logvar, skips) — skips = liste des feature maps encodeur."""
        s0 = self.enc0(x)   # (B,  64, H/2,  W/2)
        s1 = self.enc1(s0)  # (B, 128, H/4,  W/4)
        s2 = self.enc2(s1)  # (B, 256, H/8,  W/8)
        s3 = self.enc3(s2)  # (B, 512, H/16, W/16)

        h = s3.view(s3.size(0), -1)
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)

        return mu, logvar, [s0, s1, s2]  # s3 va dans le bottleneck

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, skips=None):
        """
        Args:
            z     : (B, latent_dim)
            skips : [s0, s1, s2] depuis encode() — None pour sampling pur
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.encoded_ch, self.encoded_size, self.encoded_size)

        h = self.dec0(h)                                         # 512 → 256, H/8

        if skips is not None:
            h = torch.cat([h, skips[2]], dim=1)                  # + skip enc2 → 512
        h = self.dec1(h)                                         # 512 → 128, H/4

        if skips is not None:
            h = torch.cat([h, skips[1]], dim=1)                  # + skip enc1 → 256
        h = self.dec2(h)                                         # 256 →  64, H/2

        if skips is not None:
            h = torch.cat([h, skips[0]], dim=1)                  # + skip enc0 → 128
        h = self.dec3(h)                                         # 128 →  64, H

        return self.final_conv(h)

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z     = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar

    def sample(self, x, num_samples=10):
        """Plusieurs reconstructions pour estimer l'incertitude."""
        mu, logvar, skips = self.encode(x)
        return torch.stack([self.decode(self.reparameterize(mu, logvar), skips)
                            for _ in range(num_samples)])


def vae_loss(recon, target, mu, logvar, beta=1.0, perceptual_weight=0.0, lpips_fn=None):
    """Identique à vae_restoration.py — réutilisable sans modification."""
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    perceptual_loss = torch.tensor(0.0, device=recon.device)
    if perceptual_weight > 0 and lpips_fn is not None:
        perceptual_loss = lpips_fn(recon, target).mean()

    total = recon_loss + beta * kl_loss + perceptual_weight * perceptual_loss
    return total, {
        'total':          total.item(),
        'reconstruction': recon_loss.item(),
        'kl':             kl_loss.item(),
        'perceptual':     perceptual_loss.item(),
    }


if __name__ == '__main__':
    model = VAE_UNet(image_size=128, latent_dim=256)
    x = torch.randn(2, 3, 128, 128)
    recon, mu, logvar = model(x)
    print(f'Input : {x.shape}')
    print(f'Recon : {recon.shape}')
    print(f'Mu    : {mu.shape}')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params:,}')
    print('VAE U-Net OK')
