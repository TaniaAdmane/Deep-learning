"""
Dataset pour chargement d'images propres et dégradées
Supporte extraction de patches et augmentation de données
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
    """
    
    def __init__(
        self,
        clean_dir,
        degraded_dir,
        patch_size=128,
        num_patches_per_image=8,
        augment=True,
        is_train=True
    ):
        """
        Args:
            clean_dir: dossier contenant images HD propres
            degraded_dir: dossier contenant images bruitées/dégradées
            patch_size: taille des patches (128 ou 256)
            num_patches_per_image: nombre de patches à extraire par image (training only)
            augment: appliquer augmentation de données
            is_train: mode training (extraction patches) ou validation (images complètes)
        """
        self.clean_dir = Path(clean_dir)
        self.degraded_dir = Path(degraded_dir)
        self.patch_size = patch_size
        self.num_patches_per_image = num_patches_per_image
        self.augment = augment
        self.is_train = is_train
        
        # Lister les images (supposer même structure dans clean et degraded)
        self.image_names = sorted([f.name for f in self.clean_dir.glob("*.png")])
        
        if len(self.image_names) == 0:
            self.image_names = sorted([f.name for f in self.clean_dir.glob("*.jpg")])
        
        print(f"Found {len(self.image_names)} image pairs in {clean_dir}")
        
        # Transformations
        self.to_tensor = transforms.ToTensor()  # [0, 1]
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        
        # Augmentation
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
    
    def __len__(self):
        if self.is_train:
            return len(self.image_names) * self.num_patches_per_image
        else:
            return len(self.image_names)
    
    def _load_image(self, path):
        """Charge une image et convertit en RGB"""
        img = Image.open(path).convert('RGB')
        return img
    
    def _extract_patch(self, img, top, left):
        """Extrait un patch de l'image"""
        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
        return patch
    
    def _get_random_patch_coords(self, img_width, img_height):
        """Génère des coordonnées aléatoires pour extraction de patch"""
        left = random.randint(0, img_width - self.patch_size)
        top = random.randint(0, img_height - self.patch_size)
        return top, left
    
    def __getitem__(self, idx):
        if self.is_train:
            # Mode training: extraire patches aléatoires
            img_idx = idx // self.num_patches_per_image
            img_name = self.image_names[img_idx]
            
            # Charger images complètes
            clean_img = self._load_image(self.clean_dir / img_name)
            degraded_img = self._load_image(self.degraded_dir / img_name)
            
            # Coordonnées aléatoires (mêmes pour clean et degraded)
            top, left = self._get_random_patch_coords(clean_img.width, clean_img.height)
            
            # Extraire patches
            clean_patch = self._extract_patch(clean_img, top, left)
            degraded_patch = self._extract_patch(degraded_img, top, left)
            
        else:
            # Mode validation: utiliser images complètes ou patches centraux
            img_name = self.image_names[idx]
            
            clean_img = self._load_image(self.clean_dir / img_name)
            degraded_img = self._load_image(self.degraded_dir / img_name)
            
            # Crop central si image > patch_size
            if clean_img.width > self.patch_size or clean_img.height > self.patch_size:
                # Patch central
                left = (clean_img.width - self.patch_size) // 2
                top = (clean_img.height - self.patch_size) // 2
                
                clean_patch = self._extract_patch(clean_img, top, left)
                degraded_patch = self._extract_patch(degraded_img, top, left)
            else:
                clean_patch = clean_img
                degraded_patch = degraded_img
        
        # Convertir en tenseurs
        clean_patch = self.to_tensor(clean_patch)
        degraded_patch = self.to_tensor(degraded_patch)
        
        # Augmentation (même transformation pour les deux)
        if self.augment and self.is_train:
            # Stack pour appliquer même transformation
            stacked = torch.stack([degraded_patch, clean_patch], dim=0)
            stacked = self.augment_transforms(stacked)
            degraded_patch, clean_patch = stacked[0], stacked[1]
        
        # Normaliser [-1, 1]
        degraded_patch = self.normalize(degraded_patch)
        clean_patch = self.normalize(clean_patch)
        
        return degraded_patch, clean_patch, img_name


class PatchDataset(Dataset):
    """
    Dataset alternative: charge des patches pré-extraits
    Utile si vous avez déjà extrait les patches sur disque
    """
    
    def __init__(
        self,
        clean_patches_dir,
        degraded_patches_dir,
        augment=True
    ):
        self.clean_dir = Path(clean_patches_dir)
        self.degraded_dir = Path(degraded_patches_dir)
        self.augment = augment
        
        # Lister patches
        self.patch_names = sorted([f.name for f in self.clean_dir.glob("*.png")])
        print(f"Found {len(self.patch_names)} pre-extracted patches")
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90, expand=False),
            ])
    
    def __len__(self):
        return len(self.patch_names)
    
    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        
        # Charger patches
        clean = Image.open(self.clean_dir / patch_name).convert('RGB')
        degraded = Image.open(self.degraded_dir / patch_name).convert('RGB')
        
        # Convertir en tenseurs
        clean = self.to_tensor(clean)
        degraded = self.to_tensor(degraded)
        
        # Augmentation
        if self.augment:
            stacked = torch.stack([degraded, clean], dim=0)
            stacked = self.augment_transforms(stacked)
            degraded, clean = stacked[0], stacked[1]
        
        # Normaliser
        degraded = self.normalize(degraded)
        clean = self.normalize(clean)
        
        return degraded, clean, patch_name


def create_dataloaders(
    train_clean_dir,
    train_degraded_dir,
    val_clean_dir,
    val_degraded_dir,
    patch_size=128,
    batch_size=16,
    num_workers=4,
    num_patches_per_image=8
):
    """
    Crée les dataloaders pour training et validation
    
    Args:
        train_clean_dir: dossier images propres train
        train_degraded_dir: dossier images dégradées train
        val_clean_dir: dossier images propres val
        val_degraded_dir: dossier images dégradées val
        patch_size: taille patches (128 ou 256)
        batch_size: taille des batchs
        num_workers: nombre de workers pour chargement
        num_patches_per_image: nombre de patches par image (train)
    
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
        is_train=True
    )
    
    # Validation dataset
    val_dataset = ImageRestorationDataset(
        clean_dir=val_clean_dir,
        degraded_dir=val_degraded_dir,
        patch_size=patch_size,
        num_patches_per_image=1,  # Pas de multi-patches en validation
        augment=False,
        is_train=False
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
    # Exemple d'utilisation
    print("Testing dataset...")
    
    # Simuler création de quelques images de test
    import os
    os.makedirs("test_data/train/clean", exist_ok=True)
    os.makedirs("test_data/train/degraded", exist_ok=True)
    
    # Créer images dummy
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img.save(f"test_data/train/clean/img_{i:03d}.png")
        img.save(f"test_data/train/degraded/img_{i:03d}.png")
    
    # Test dataset
    dataset = ImageRestorationDataset(
        clean_dir="test_data/train/clean",
        degraded_dir="test_data/train/degraded",
        patch_size=128,
        num_patches_per_image=4,
        is_train=True
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
    
    print("\n✅ Dataset OK!")