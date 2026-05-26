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
        
        # Lister les images en recherchant récursivement dans les dossiers
        self.image_pairs = self._find_image_pairs(self.clean_dir, self.degraded_dir)
        print(f"Found {len(self.image_pairs)} image pairs in {clean_dir}")
        
        # Transformations
        self.to_tensor = transforms.ToTensor()  # [0, 1]
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        
        # Augmentation - CORRIGÉ
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
    
    def __len__(self):
        if self.is_train:
            return len(self.image_pairs) * self.num_patches_per_image
        else:
            return len(self.image_pairs)
    
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

    def _find_image_pairs(self, clean_dir, degraded_dir):
        """Recherche récursive des images propres et dégradées et crée une liste de paires."""
        clean_dir = Path(clean_dir)
        degraded_dir = Path(degraded_dir)

        clean_relpaths = []
        for ext in ['png', 'jpg', 'jpeg']:
            clean_relpaths.extend(sorted([p.relative_to(clean_dir) for p in clean_dir.rglob(f'*.{ext}')]))

        clean_relpaths = sorted(set(clean_relpaths))
        degraded_relpaths = []
        for ext in ['png', 'jpg', 'jpeg']:
            degraded_relpaths.extend(sorted([p.relative_to(degraded_dir) for p in degraded_dir.rglob(f'*.{ext}')]))

        degraded_set = set(degraded_relpaths)
        degraded_parent_map = {}
        for p in degraded_relpaths:
            degraded_parent_map.setdefault(p.parent, []).append(p)

        pairs = []
        for clean_rel in clean_relpaths:
            if clean_rel in degraded_set:
                pairs.append((clean_rel, clean_rel))
                continue

            candidate = clean_rel.with_name(clean_rel.stem + 'x2' + clean_rel.suffix)
            if candidate in degraded_set:
                pairs.append((clean_rel, candidate))
                continue

            siblings = degraded_parent_map.get(clean_rel.parent, [])
            matches = [p for p in siblings if p.stem == clean_rel.stem or p.stem.startswith(clean_rel.stem + 'x2')]
            if len(matches) == 1:
                pairs.append((clean_rel, matches[0]))
                continue
            elif len(matches) > 1:
                pairs.append((clean_rel, sorted(matches)[0]))
                continue

            print(f"Warning: No degraded pair found for {clean_rel}")
        return pairs
    
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
            clean_relpath, degraded_relpath = self.image_pairs[img_idx]

            # Charger images complètes
            clean_img = self._load_image(self.clean_dir / clean_relpath)
            degraded_img = self._load_image(self.degraded_dir / degraded_relpath)
            
            # CRITICAL: Si l'image dégradée est plus petite (LR), l'upsampler à la taille de clean
            if degraded_img.size != clean_img.size:
                degraded_img = degraded_img.resize(clean_img.size, Image.BICUBIC)

            # Coordonnées aléatoires (mêmes pour clean et degraded)
            # S'assurer qu'elles sont valides après upsampling
            max_left = min(clean_img.width, degraded_img.width) - self.patch_size
            max_top = min(clean_img.height, degraded_img.height) - self.patch_size
            
            if max_left <= 0 or max_top <= 0:
                # Image trop petite, resize à la taille minimale requise
                target_size = (self.patch_size, self.patch_size)
                clean_img = clean_img.resize(target_size, Image.BICUBIC)
                degraded_img = degraded_img.resize(target_size, Image.BICUBIC)
                top, left = 0, 0
            else:
                left = random.randint(0, max_left)
                top = random.randint(0, max_top)

            # Extraire patches
            clean_patch = self._extract_patch(clean_img, top, left)
            degraded_patch = self._extract_patch(degraded_img, top, left)
            
            # Augmentation AVANT conversion en tenseur
            if self.augment:
                # Appliquer les mêmes transformations aux deux patches
                seed = random.randint(0, 2**32 - 1)
                
                random.seed(seed)
                torch.manual_seed(seed)
                clean_patch = self.augment_transforms(clean_patch)
                
                random.seed(seed)
                torch.manual_seed(seed)
                degraded_patch = self.augment_transforms(degraded_patch)

            # Convertir en tenseurs APRÈS augmentation
            clean_patch = self.to_tensor(clean_patch)
            degraded_patch = self.to_tensor(degraded_patch)
            
            # SAFETY CHECK: Vérifier que le patch n'est pas complètement noir/blanc
            # (peut arriver avec des coordonnées hors limites ou images corrompues)
            if degraded_patch.max() - degraded_patch.min() < 0.01:
                # Patch invalide, forcer re-sampling
                raise ValueError("Invalid patch: no variation in degraded image")

        else:
            # Mode validation: utiliser images complètes ou patches centraux
            clean_relpath, degraded_relpath = self.image_pairs[idx]
            clean_img = self._load_image(self.clean_dir / clean_relpath)
            degraded_img = self._load_image(self.degraded_dir / degraded_relpath)
            
            # CRITICAL: Si l'image dégradée est plus petite (LR), l'upsampler à la taille de clean
            if degraded_img.size != clean_img.size:
                degraded_img = degraded_img.resize(clean_img.size, Image.BICUBIC)

            # Crop central si image > patch_size
            if clean_img.width > self.patch_size or clean_img.height > self.patch_size:
                # Patch central - vérifier les dimensions
                if clean_img.width >= self.patch_size and clean_img.height >= self.patch_size:
                    left = (clean_img.width - self.patch_size) // 2
                    top = (clean_img.height - self.patch_size) // 2

                    clean_patch = self._extract_patch(clean_img, top, left)
                    degraded_patch = self._extract_patch(degraded_img, top, left)
                else:
                    # Image trop petite, resize
                    target_size = (self.patch_size, self.patch_size)
                    clean_patch = clean_img.resize(target_size, Image.BICUBIC)
                    degraded_patch = degraded_img.resize(target_size, Image.BICUBIC)
            else:
                # Image déjà assez petite, resize direct
                target_size = (self.patch_size, self.patch_size)
                clean_patch = clean_img.resize(target_size, Image.BICUBIC)
                degraded_patch = degraded_img.resize(target_size, Image.BICUBIC)
            
            # Convertir en tenseurs (pas d'augmentation en validation)
            clean_patch = self.to_tensor(clean_patch)
            degraded_patch = self.to_tensor(degraded_patch)
        
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
            print(f"{'='*60}\n")
        
        return degraded_patch, clean_patch, str(clean_relpath)

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
        
        # Augmentation
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
    print("Testing dataset...")
    
    # Simuler création de quelques images de test
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