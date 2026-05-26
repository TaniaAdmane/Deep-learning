"""
Création d'un dataset avec bruit gaussien contrôlé
Pour denoising avec VAE
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse


def add_gaussian_noise(img_array, sigma):
    """
    Ajoute du bruit gaussien à une image
    
    Args:
        img_array: numpy array (H, W, 3) dans [0, 255]
        sigma: écart-type du bruit gaussien (typiquement 15-50)
    
    Returns:
        noisy_array: image bruitée dans [0, 255]
    """
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = img_array + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def create_noisy_dataset(clean_dir, noisy_dir, sigma=25, num_samples=None):
    """
    Crée un dataset bruité à partir d'images propres
    
    Args:
        clean_dir: dossier source avec images propres
        noisy_dir: dossier destination pour images bruitées
        sigma: niveau de bruit (15=léger, 25=moyen, 50=fort)
        num_samples: nombre d'images à traiter (None = toutes)
    """
    clean_dir = Path(clean_dir)
    noisy_dir = Path(noisy_dir)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    
    # Trouver toutes les images
    extensions = ['*.png', '*.jpg', '*.jpeg']
    clean_images = []
    for ext in extensions:
        clean_images.extend(list(clean_dir.rglob(ext)))
    
    if num_samples:
        clean_images = clean_images[:num_samples]
    
    print(f"\n{'='*60}")
    print(f"CRÉATION DATASET BRUITÉ")
    print(f"{'='*60}")
    print(f"Source: {clean_dir}")
    print(f"Destination: {noisy_dir}")
    print(f"Niveau de bruit (σ): {sigma}")
    print(f"Nombre d'images: {len(clean_images)}")
    print(f"{'='*60}\n")
    
    for clean_path in tqdm(clean_images, desc="Adding noise"):
        # Chemin relatif pour préserver structure
        rel_path = clean_path.relative_to(clean_dir)
        noisy_path = noisy_dir / rel_path
        
        # Créer sous-dossiers si nécessaire
        noisy_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Charger image
            img = Image.open(clean_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32)
            
            # Ajouter bruit
            noisy_array = add_gaussian_noise(img_array, sigma)
            
            # Sauvegarder
            noisy_img = Image.fromarray(noisy_array)
            noisy_img.save(noisy_path)
            
        except Exception as e:
            print(f"  ⚠️  Erreur avec {clean_path}: {e}")
    
    print(f"\n✅ Dataset bruité créé!")
    print(f"   {len(clean_images)} images traitées")
    print(f"   Sauvegardées dans: {noisy_dir}\n")


def create_multi_level_dataset(clean_dir, base_output_dir, sigmas=[15, 25, 50]):
    """
    Crée plusieurs datasets avec différents niveaux de bruit
    
    Args:
        clean_dir: dossier source
        base_output_dir: dossier de base pour outputs
        sigmas: liste de niveaux de bruit à générer
    """
    base_output_dir = Path(base_output_dir)
    
    for sigma in sigmas:
        noisy_dir = base_output_dir / f'noisy_sigma{sigma}'
        create_noisy_dataset(clean_dir, noisy_dir, sigma=sigma)


def visualize_noise_levels(clean_path, output_path='noise_comparison.png', sigmas=[15, 25, 50]):
    """
    Visualise différents niveaux de bruit sur une image
    """
    import matplotlib.pyplot as plt
    
    img = Image.open(clean_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(5 * (len(sigmas) + 1), 5))
    
    # Image propre
    axes[0].imshow(img_array.astype(np.uint8))
    axes[0].set_title('Clean (σ=0)')
    axes[0].axis('off')
    
    # Images bruitées
    for i, sigma in enumerate(sigmas):
        noisy = add_gaussian_noise(img_array, sigma)
        axes[i + 1].imshow(noisy)
        axes[i + 1].set_title(f'Noisy (σ={sigma})')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualisation sauvegardée: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create noisy dataset for denoising')
    
    parser.add_argument('--clean_dir', type=str, required=True,
                        help='Directory with clean images')
    parser.add_argument('--noisy_dir', type=str, required=True,
                        help='Output directory for noisy images')
    parser.add_argument('--sigma', type=int, default=25,
                        help='Noise level (15=light, 25=medium, 50=heavy)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of images to process (None=all)')
    parser.add_argument('--visualize', type=str, default=None,
                        help='Path to an image to visualize noise levels')
    
    args = parser.parse_args()
    
    if args.visualize:
        # Mode visualisation
        visualize_noise_levels(args.visualize)
    else:
        # Mode création dataset
        create_noisy_dataset(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            sigma=args.sigma,
            num_samples=args.num_samples
        )