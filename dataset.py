"""
Dataset pour chargement d'images propres et dégradées
Supporte extraction de patches, augmentation, et génération de bruit À LA VOLÉE
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random


class ImageRestorationDataset(Dataset):
    """
    Dataset pour restauration d'images
    Charge paires (image_dégradée, image_propre)
    Supporte génération de bruit DYNAMIQUE pour denoising (zéro stockage!)
    """
    
    def __init__(
        self,
        clean_dir,
        degraded_dir,
        patch_size=128,
        num_patches_per_image=8,
        augment=True,
        is_train=True,
        noise_sigma=25,  # Niveau de bruit pour denoising dynamique
        generate_noise=True  # Si True, génère le bruit à la volée
    ):
        """
        Args:
            clean_dir: dossier contenant images propres
            degraded_dir: dossier images dégradées (ignoré si generate_noise=True)
            patch_size: taille des patches (128 ou 256)
            num_patches_per_image: nombre de patches par image (training only)
            augment: appliquer augmentation de données
            is_train: mode training ou validation
            noise_sigma: niveau de bruit gaussien (15=léger, 25=moyen, 50=fort)
            generate_noise: Si True, génère bruit à la volée (DENOISING MODE)
        """
        self.clean_dir = Path(clean_dir)
        self.degraded_dir = Path(degraded_dir) if degraded_dir else None
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.augment = augment
        self.is_train = is_train
        self.noise_sigma = noise_sigma
        self.generate_noise = generate_noise
        
        # Lister les images propres
        clean_relpaths = []
        for ext in ['png', 'jpg', 'jpeg']:
            clean_relpaths.extend(sorted([p.relative_to(self.clean_dir) for p in self.clean_dir.rglob(f'*.{ext}')]))
        self.clean_images = sorted(set(clean_relpaths))
        
        print(f"Found {len(self.clean_images)} clean images in {clean_dir}")
        
        if generate_noise:
            print(f"🎯 Mode: DENOISING DYNAMIQUE (σ={noise_sigma}) - Zéro stockage supplémentaire!")
        else:
            print(f"🎯 Mode: Utilisation d'images dégradées existantes")
        
        # Transformations
        self.to_tensor = transforms.ToTensor()  # [0, 1]
        self.normalize = transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
        
        # Augmentation
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
    
    def __len__(self):
        if self.is_train:
            return len(self.clean_images) * self.num_patches_per_image
        else:
            return len(self.clean_images)
    
    def _load_image(self, path):
        """Charge une image et convertit en RGB"""
        img = Image.open(path).convert('RGB')
        return img
    
    def _extract_patch(self, img, top, left):
        """Extrait un patch de l'image"""
        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
        return patch
    
    def _add_gaussian_noise(self, clean_tensor, sigma):
        """
        Ajoute du bruit gaussien à un tensor [0, 1]
        
        Args:
            clean_tensor: torch.Tensor (C, H, W) dans [0, 1]
            sigma: écart-type du bruit (échelle 0-255)
        
        Returns:
            noisy_tensor: avec bruit ajouté, clampé dans [0, 1]
        """
        # Normaliser sigma de [0, 255] vers [0, 1]
        sigma_normalized = sigma / 255.0
        
        # Générer bruit gaussien
        noise = torch.randn_like(clean_tensor) * sigma_normalized
        
        # Ajouter bruit et clamper
        noisy = clean_tensor + noise
        noisy = torch.clamp(noisy, 0, 1)
        
        return noisy
    
    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return self._load_item(idx)
            except Exception:
                idx = random.randint(0, len(self) - 1)
        return self._load_item(idx)

    def _load_item(self, idx):
        if self.is_train:
            # Mode training: extraire patches aléatoires
            img_idx = idx // self.num_patches_per_image
            clean_relpath = self.clean_images[img_idx]

            # Charger image propre
            clean_img = self._load_image(self.clean_dir / clean_relpath)
            
            # Vérifier taille minimale
            if clean_img.width < self.patch_size or clean_img.height < self.patch_size:
                # Image trop petite, resize
                target_size = (self.patch_size, self.patch_size)
                clean_img = clean_img.resize(target_size, Image.BICUBIC)
                top, left = 0, 0
            else:
                # Coordonnées aléatoires
                left = random.randint(0, clean_img.width - self.patch_size)
                top = random.randint(0, clean_img.height - self.patch_size)

            # Extraire patch
            clean_patch = self._extract_patch(clean_img, top, left)
            
            # Augmentation AVANT conversion en tenseur
            if self.augment:
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed)
                torch.manual_seed(seed)
                clean_patch = self.augment_transforms(clean_patch)

            # Convertir en tenseur
            clean_patch = self.to_tensor(clean_patch)
            
            # GÉNÉRER BRUIT DYNAMIQUEMENT
            if self.generate_noise:
                degraded_patch = self._add_gaussian_noise(clean_patch, self.noise_sigma)
            else:
                # Charger degraded existant (mode SR ou autre)
                degraded_img = self._load_image(self.degraded_dir / clean_relpath)
                if degraded_img.size != (clean_img.width, clean_img.height):
                    degraded_img = degraded_img.resize((clean_img.width, clean_img.height), Image.BICUBIC)
                degraded_patch_pil = self._extract_patch(degraded_img, top, left)
                if self.augment:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    degraded_patch_pil = self.augment_transforms(degraded_patch_pil)
                degraded_patch = self.to_tensor(degraded_patch_pil)

        else:
            # Mode validation: patch central
            clean_relpath = self.clean_images[idx]
            clean_img = self._load_image(self.clean_dir / clean_relpath)

            # Crop central si image > patch_size
            if clean_img.width >= self.patch_size and clean_img.height >= self.patch_size:
                left = (clean_img.width - self.patch_size) // 2
                top = (clean_img.height - self.patch_size) // 2
                clean_patch = self._extract_patch(clean_img, top, left)
            else:
                # Resize si trop petite
                clean_patch = clean_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            
            # Convertir en tenseur
            clean_patch = self.to_tensor(clean_patch)
            
            # GÉNÉRER BRUIT
            if self.generate_noise:
                degraded_patch = self._add_gaussian_noise(clean_patch, self.noise_sigma)
            else:
                degraded_img = self._load_image(self.degraded_dir / clean_relpath)
                if degraded_img.size != (clean_img.width, clean_img.height):
                    degraded_img = degraded_img.resize((clean_img.width, clean_img.height), Image.BICUBIC)
                if degraded_img.width >= self.patch_size and degraded_img.height >= self.patch_size:
                    degraded_patch_pil = self._extract_patch(degraded_img, top, left)
                else:
                    degraded_patch_pil = degraded_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
                degraded_patch = self.to_tensor(degraded_patch_pil)
        
        # Normaliser [-1, 1]
        degraded_patch = self.normalize(degraded_patch)
        clean_patch = self.normalize(clean_patch)
        
        # DEBUG : Vérifier normalisation (premier échantillon seulement)
        if idx == 0 and self.is_train:
            print(f"\n{'='*60}")
            print(f"DEBUG DATASET (Training):")
            print(f"  Degraded - min: {degraded_patch.min():.4f}, max: {degraded_patch.max():.4f}")
            print(f"  Clean - min: {clean_patch.min():.4f}, max: {clean_patch.max():.4f}")
            print(f"  Mean degraded: {degraded_patch.mean():.4f}")
            print(f"  Mean clean: {clean_patch.mean():.4f}")
            if self.generate_noise:
                print(f"  Noise sigma: {self.noise_sigma}")
            print(f"{'='*60}\n")
        
        return degraded_patch, clean_patch, str(clean_relpath)


def create_dataloaders(
    train_clean_dir,
    train_degraded_dir,
    val_clean_dir,
    val_degraded_dir,
    patch_size=128,
    batch_size=16,
    num_workers=4,
    num_patches_per_image=8,
    noise_sigma=25,
    generate_noise=True
):
    """
    Crée les dataloaders pour training et validation
    
    Args:
        train_clean_dir: dossier images propres train
        train_degraded_dir: dossier images dégradées train (ignoré si generate_noise=True)
        val_clean_dir: dossier images propres val
        val_degraded_dir: dossier images dégradées val (ignoré si generate_noise=True)
        patch_size: taille patches (128 ou 256)
        batch_size: taille des batchs
        num_workers: nombre de workers pour chargement
        num_patches_per_image: nombre de patches par image (train)
        noise_sigma: niveau de bruit pour denoising (15, 25, 50)
        generate_noise: Si True, génère bruit dynamiquement (DENOISING)
    
    Returns:
        train_loader, val_loader
    """
    
    # Training dataset
    train_dataset = ImageRestorationDataset(
        clean_dir=train_clean_dir,
        degraded_dir=train_degraded_dir,
        patch_size=patch_size,
        num_patches_per_image=num_patches_per_image,
        augment=True,
        is_train=True,
        noise_sigma=noise_sigma,
        generate_noise=generate_noise
    )
    
    # Validation dataset
    val_dataset = ImageRestorationDataset(
        clean_dir=val_clean_dir,
        degraded_dir=val_degraded_dir,
        patch_size=patch_size,
        num_patches_per_image=1,
        augment=False,
        is_train=False,
        noise_sigma=noise_sigma,
        generate_noise=generate_noise
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Test
if __name__ == "__main__":
    print("Testing dataset with dynamic noise generation...")
    
    # Simuler création de quelques images de test
    os.makedirs("test_data/train/clean", exist_ok=True)
    
    # Créer images dummy
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img.save(f"test_data/train/clean/img_{i:03d}.png")
    
    # Test dataset avec génération de bruit dynamique
    dataset = ImageRestorationDataset(
        clean_dir="test_data/train/clean",
        degraded_dir=None,  # Pas besoin!
        patch_size=128,
        num_patches_per_image=4,
        is_train=True,
        noise_sigma=25,
        generate_noise=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test un échantillon
    degraded, clean, name = dataset[0]
    print(f"Degraded shape: {degraded.shape}")
    print(f"Clean shape: {clean.shape}")
    print(f"Value range: [{degraded.min():.2f}, {degraded.max():.2f}]")
    print(f"Image name: {name}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_degraded, batch_clean, names in loader:
        print(f"\nBatch degraded: {batch_degraded.shape}")
        print(f"Batch clean: {batch_clean.shape}")
        break
    
    print("\n✅ Dataset avec bruit dynamique OK!")