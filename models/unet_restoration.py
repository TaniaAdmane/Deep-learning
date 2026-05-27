"""
U-Net pour Restauration d'Images
Architecture U-Net classique (sans bottleneck VAE)

Différences vs VAE_UNet :
- Pas d'espace latent probabiliste (pas de mu/logvar/KL)
- Skip connections directes encodeur → décodeur
- Loss = MSE + Perceptual uniquement (pas de KL)
- Plus déterministe → meilleure reconstruction pixel-perfect
- Gain attendu sur tâches simples : +2 à +4 dB PSNR vs VAE

Compatible avec train.py : retourne (recon, None, None)
pour que vae_loss ignore automatiquement le terme KL (beta=0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_Restoration(nn.Module):
    """
    U-Net pour restauration d'images.

    Architecture :
        Encodeur : 4 blocs de downsampling + sauvegarde des feature maps
        Bottleneck : 2 convolutions profondes (sans latent)
        Décodeur  : 4 blocs de upsampling + concaténation des skips

    Flux des canaux (base_channels=64) :
        enc0 :  3  →  64,  H/2
        enc1 :  64 → 128,  H/4
        enc2 : 128 → 256,  H/8
        enc3 : 256 → 512,  H/16
        bottleneck : 512 → 1024 → 512
        dec0 : 512+512 → 256,  H/8
        dec1 : 256+256 → 128,  H/4
        dec2 : 128+128 →  64,  H/2
        dec3 :  64+ 64 →  64,  H
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        base_channels=64
    ):
        super().__init__()

        bc = base_channels

        # ── ENCODEUR ────────────────────────────────────────────────
        self.enc0 = self._enc_block(input_channels, bc)      # →  64, H/2
        self.enc1 = self._enc_block(bc,      bc * 2)         # → 128, H/4
        self.enc2 = self._enc_block(bc * 2,  bc * 4)         # → 256, H/8
        self.enc3 = self._enc_block(bc * 4,  bc * 8)         # → 512, H/16

        # ── BOTTLENECK ──────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bc * 8,  bc * 16, 3, padding=1),
            nn.BatchNorm2d(bc * 16), nn.ReLU(inplace=True),
            nn.Conv2d(bc * 16, bc * 8,  3, padding=1),
            nn.BatchNorm2d(bc * 8),  nn.ReLU(inplace=True),
        )

        # ── DÉCODEUR ────────────────────────────────────────────────
        # Flux : cat(bottleneck, skip) → dec → up → cat(h, skip) → dec → up ...
        #
        #   dec0 : cat(b=512, s3=512) → 1024 → 256   (même résolution H/16)
        #   up0  : 256 → 256,  H/8
        #   dec1 : cat(256, s2=256)   →  512 → 128
        #   up1  : 128 → 128,  H/4
        #   dec2 : cat(128, s1=128)   →  256 →  64
        #   up2  :  64 →  64,  H/2
        #   dec3 : cat( 64, s0= 64)   →  128 →  64
        #   up3  :  64 →  64,  H
        self.dec0  = self._dec_block(bc * 8 * 2, bc * 4)   # 1024 → 256
        self.up0   = nn.ConvTranspose2d(bc * 4,  bc * 4,  4, stride=2, padding=1)  # H/16→H/8

        self.dec1  = self._dec_block(bc * 4 * 2, bc * 2)   #  512 → 128
        self.up1   = nn.ConvTranspose2d(bc * 2,  bc * 2,  4, stride=2, padding=1)  # H/8→H/4

        self.dec2  = self._dec_block(bc * 2 * 2, bc)       #  256 →  64
        self.up2   = nn.ConvTranspose2d(bc,      bc,      4, stride=2, padding=1)   # H/4→H/2

        self.dec3  = self._dec_block(bc * 2,     bc)       #  128 →  64
        self.up3   = nn.ConvTranspose2d(bc,      bc,      4, stride=2, padding=1)   # H/2→H

        # ── SORTIE ──────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(bc, output_channels, 1),
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

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, x):
        # Encodeur + sauvegarde des skips
        s0 = self.enc0(x)   # ( 64, H/2)
        s1 = self.enc1(s0)  # (128, H/4)
        s2 = self.enc2(s1)  # (256, H/8)
        s3 = self.enc3(s2)  # (512, H/16)

        # Bottleneck
        b = self.bottleneck(s3)  # (512, H/16)

        # Décodeur avec skips
        # Ordre : cat(même résolution) → dec → up → cat → dec → up ...
        h = self.up0(self.dec0(torch.cat([b,  s3], dim=1)))  # cat H/16 → dec → up → H/8
        h = self.up1(self.dec1(torch.cat([h,  s2], dim=1)))  # cat H/8  → dec → up → H/4
        h = self.up2(self.dec2(torch.cat([h,  s1], dim=1)))  # cat H/4  → dec → up → H/2
        h = self.up3(self.dec3(torch.cat([h,  s0], dim=1)))  # cat H/2  → dec → up → H

        return self.final_conv(h), None, None   # (recon, None, None) → KL ignoré

    def sample(self, x, num_samples=1):
        """Compatibilité avec l'interface VAE (déterministe ici)."""
        recon, _, _ = self.forward(x)
        return recon.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)


def unet_loss(recon, target, mu=None, logvar=None,
              beta=0.0, perceptual_weight=0.1, lpips_fn=None):
    """
    Loss U-Net = MSE + Perceptual Loss (pas de KL).
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
    model = UNet_Restoration(base_channels=64)
    x = torch.randn(2, 3, 128, 128)
    recon, _, _ = model(x)
    print(f'Input : {x.shape}')
    print(f'Recon : {recon.shape}')
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n:,}')
    print('U-Net OK')