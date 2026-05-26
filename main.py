"""
DENOISING avec génération de bruit dynamique
Zéro stockage supplémentaire!
"""

import torch
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/project')

from models.vae_restoration import VAE_Restoration
from dataset import create_dataloaders
from train_vae import VAETrainer


def train_example():
    """Entraînement DENOISING avec bruit généré à la volée"""
    
    config = {
        'patch_size': 128,
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'beta': 0.0,               
        'perceptual_weight': 0.0,
        'latent_dim': 256,
        'noise_sigma': (5, 75),     # Plage aléatoire : blind denoising (valeur fixe ex: 25 pour mode ciblé)
        'generate_noise': True      # Génération dynamique !
    }
    
    # Chemins (seulement clean, pas besoin de noisy!)
    data_dir = Path.home() / 'work/data'
    
    paths = {
        'train_clean': str(data_dir / 'train/clean'),
        'train_degraded': None,  # Pas utilisé en mode denoising dynamique
        'val_clean': str(data_dir / 'val/clean'),
        'val_degraded': None     # Pas utilisé
    }
    
    # Vérifier que clean existe
    if not Path(paths['train_clean']).exists():
        raise FileNotFoundError(f"❌ {paths['train_clean']} not found!")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using {device}")
    sigma_info = f"σ∈{config['noise_sigma']}" if isinstance(config['noise_sigma'], tuple) else f"σ={config['noise_sigma']}"
    print(f"🎯 Mode: DENOISING dynamique ({sigma_info})")
    
    # Data loaders avec génération de bruit dynamique
    print("\nCreating data loaders...")
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
    
    # Model
    print("\n🏗️  Building model...")
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=config['latent_dim'],
        image_size=config['patch_size']
    )
    
    # Trainer
    print("\n⚙️  Initializing trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        beta=config['beta'],
        perceptual_weight=config['perceptual_weight'],
        checkpoint_dir='/tmp/checkpoints',
        log_dir='/tmp/logs'
    )
    
    # Train!
    print("\n🎯 Starting training...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_denoising.py train")
        sys.exit(1)
    
    if sys.argv[1] == 'train':
        train_example()