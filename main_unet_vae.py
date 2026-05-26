"""
Entraînement VAE U-Net (avec skip connections + perceptual loss)
Architecture améliorée pour meilleure reconstruction
"""

import torch
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/project')

from models.vae_unet import VAE_UNet
from dataset import create_dataloaders
from train_vae import VAETrainer


def train():
    config = {
        'patch_size':        128,
        'batch_size':        16,
        'num_epochs':        30,
        'learning_rate':     1e-4,
        'beta':              0.0,    # Phase 1 : reconstruction pure
        'perceptual_weight': 0.05,   # Activé (améliore SSIM/LPIPS)
        'latent_dim':        256,
        'noise_sigma':       (5, 75),
        'generate_noise':    True
    }

    data_dir = Path.home() / 'work/data'
    paths = {
        'train_clean':    str(data_dir / 'train/clean'),
        'train_degraded': None,
        'val_clean':      str(data_dir / 'val/clean'),
        'val_degraded':   None,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    print(f'Mode   : blind denoising σ∈{config["noise_sigma"]} + skip connections + perceptual')

    train_loader, val_loader = create_dataloaders(
        train_clean_dir=paths['train_clean'],
        train_degraded_dir=paths['train_degraded'],
        val_clean_dir=paths['val_clean'],
        val_degraded_dir=paths['val_degraded'],
        patch_size=config['patch_size'],
        batch_size=config['batch_size'],
        num_workers=4,
        num_patches_per_image=4,
        noise_sigma=config['noise_sigma'],
        generate_noise=config['generate_noise']
    )

    model = VAE_UNet(
        input_channels=3,
        latent_dim=config['latent_dim'],
        image_size=config['patch_size']
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres : {n_params:,}')

    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        beta=config['beta'],
        perceptual_weight=config['perceptual_weight'],
        checkpoint_dir='/tmp/checkpoints_unet',
        log_dir='/tmp/logs_unet'
    )

    trainer.train(num_epochs=config['num_epochs'])
    print('Entraînement terminé !')


if __name__ == '__main__':
    train()
