"""
Exemple d'utilisation - Version Simple (données locales)
"""

import torch
from pathlib import Path
from models.vae_restoration import VAE_Restoration
from dataset import create_dataloaders
from train_vae import VAETrainer


def train_example():
    """Entraînement avec données locales"""
    
    config = {
        'patch_size': 128,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'beta': 0.001,
        'perceptual_weight': 0.1,
        'latent_dim': 256
    }
    
    # Chemins locaux
    data_dir = Path.home() / 'work/data'
    
    paths = {
        'train_clean': str(data_dir / 'train/clean'),
        'train_degraded': str(data_dir / 'train/degraded'),
        'val_clean': str(data_dir / 'val/clean'),
        'val_degraded': str(data_dir / 'val/degraded')
    }
    
    # Vérifier
    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"❌ {name} not found at {path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using {device}")
    
    # Data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_clean_dir=paths['train_clean'],
        train_degraded_dir=paths['train_degraded'],
        val_clean_dir=paths['val_clean'],
        val_degraded_dir=paths['val_degraded'],
        patch_size=config['patch_size'],
        batch_size=config['batch_size'],
        num_workers=4  # OK pour local
    )
    
    # Model
    print("\n🏗️ Building model...")
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=config['latent_dim'],
        image_size=config['patch_size']
    )
    
    # Trainer
    print("\n🎯 Initializing trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        beta=config['beta'],
        perceptual_weight=config['perceptual_weight'],
        checkpoint_dir=str(Path.home() / 'work/checkpoints'),
        log_dir=str(Path.home() / 'work/logs')
    )
    
    # Train!
    print("\n🚀 Starting training...")
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py train")
        sys.exit(1)
    
    if sys.argv[1] == 'train':
        train_example()