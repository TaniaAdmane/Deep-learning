"""
Exemple d'utilisation rapide du VAE
"""

import torch
from models.vae_restoration import VAE_Restoration
from dataset import create_dataloaders
from train_vae import VAETrainer

# ============================================
# EXEMPLE 1 : Entraînement depuis zéro
# ============================================

def train_example():
    """Exemple d'entraînement complet"""
    
    # 1. Configuration
    config = {
        'patch_size': 128,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'beta': 1.0,  # β-VAE parameter
        'perceptual_weight': 0.1,
        'latent_dim': 256
    }
    
    # 2. Paths vers vos données
    paths = {
        'train_clean': 'data/train/clean',
        'train_degraded': 'data/train/degraded',
        'val_clean': 'data/val/clean',
        'val_degraded': 'data/val/degraded'
    }
    
    # 3. Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    
    # 4. Data loaders
    train_loader, val_loader = create_dataloaders(
        train_clean_dir=paths['train_clean'],
        train_degraded_dir=paths['train_degraded'],
        val_clean_dir=paths['val_clean'],
        val_degraded_dir=paths['val_degraded'],
        patch_size=config['patch_size'],
        batch_size=config['batch_size']
    )
    
    # 5. Model
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=config['latent_dim'],
        image_size=config['patch_size']
    )
    
    # 6. Trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        beta=config['beta'],
        perceptual_weight=config['perceptual_weight'],
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    # 7. Train !
    trainer.train(num_epochs=config['num_epochs'])
    
    print("Training completed!")


# ============================================
# EXEMPLE 2 : Inférence avec modèle entraîné
# ============================================

def inference_example():
    """Exemple d'inférence sur une image"""
    
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    
    # 1. Charger le modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=256,
        image_size=128
    )
    
    # Charger weights
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded!")
    
    # 2. Charger une image dégradée
    degraded_img = Image.open('path/to/degraded_image.png').convert('RGB')
    
    # 3. Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Ou 256x256
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    degraded_tensor = transform(degraded_img).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # 4. Reconstruction
    with torch.no_grad():
        recon, mu, logvar = model(degraded_tensor)
    
    # 5. Post-processing
    recon_img = ((recon.squeeze(0).cpu() + 1) / 2).numpy()  # [0, 1]
    recon_img = np.transpose(recon_img, (1, 2, 0))  # (H, W, 3)
    recon_img = (recon_img * 255).astype(np.uint8)
    
    # 6. Sauvegarder
    Image.fromarray(recon_img).save('reconstructed.png')
    
    print("Reconstruction saved to reconstructed.png")


# ============================================
# EXEMPLE 3 : Échantillonnage multiple (incertitude)
# ============================================

def uncertainty_example():
    """Exemple d'analyse d'incertitude via échantillonnage"""
    
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from PIL import Image
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VAE_Restoration(image_size=128, latent_dim=256)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Charger image
    degraded_img = Image.open('path/to/degraded_image.png').convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    degraded_tensor = transform(degraded_img).unsqueeze(0).to(device)
    
    # Générer 10 reconstructions différentes
    with torch.no_grad():
        samples = model.sample(degraded_tensor, num_samples=10)  # (10, 1, 3, H, W)
    
    # Visualiser
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(10):
        sample = ((samples[i, 0] + 1) / 2).cpu().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        
        axes[i].imshow(sample)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('uncertainty_samples.png', dpi=150)
    
    print("10 different reconstructions saved!")
    
    # Calculer variance (incertitude)
    samples_np = ((samples.squeeze(1) + 1) / 2).cpu().numpy()
    variance = np.var(samples_np, axis=0)  # (3, H, W)
    
    # Plot carte d'incertitude
    plt.figure(figsize=(6, 6))
    plt.imshow(np.mean(variance, axis=0), cmap='hot')
    plt.title('Uncertainty Map (High = More Uncertain)')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('uncertainty_map.png', dpi=150, bbox_inches='tight')
    
    print("Uncertainty map saved!")


# ============================================
# EXEMPLE 4 : Comparaison Beta-VAE
# ============================================

def beta_vae_comparison():
    """Compare différentes valeurs de β"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Entraîner 3 modèles avec β différents
    betas = [0.1, 1.0, 10.0]
    
    for beta in betas:
        print(f"\n{'='*60}")
        print(f"Training with β = {beta}")
        print(f"{'='*60}")
        
        # Créer dataloaders
        train_loader, val_loader = create_dataloaders(
            train_clean_dir='data/train/clean',
            train_degraded_dir='data/train/degraded',
            val_clean_dir='data/val/clean',
            val_degraded_dir='data/val/degraded',
            patch_size=128,
            batch_size=16
        )
        
        # Créer modèle
        model = VAE_Restoration(image_size=128, latent_dim=256)
        
        # Trainer avec β spécifique
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=1e-4,
            beta=beta,  # <-- Différent pour chaque modèle
            checkpoint_dir=f'checkpoints_beta_{beta}',
            log_dir=f'logs_beta_{beta}'
        )
        
        trainer.train(num_epochs=30)
        
        print(f"β = {beta} completed!")
    
    print("\nβ-VAE comparison completed!")
    print("Compare results in TensorBoard:")
    print("  tensorboard --logdir logs_beta_0.1")
    print("  tensorboard --logdir logs_beta_1.0")
    print("  tensorboard --logdir logs_beta_10.0")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_usage.py train       - Train VAE from scratch")
        print("  python example_usage.py inference   - Run inference on image") #Prend une seule image dégradée et la restaure avec un modèle déjà entraîné.
        print("  python example_usage.py uncertainty - Analyze uncertainty") #Génère 10 reconstructions différentes de la même image pour analyser l'incertitude du VAE.
        print("  python example_usage.py beta        - Compare β-VAE variants") #Entraîne 3 modèles différents avec des valeurs de β différentes pour comparer.
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'train':
        train_example()
    elif mode == 'inference':
        inference_example()
    elif mode == 'uncertainty':
        uncertainty_example()
    elif mode == 'beta':
        beta_vae_comparison()
    else:
        print(f"Unknown mode: {mode}")