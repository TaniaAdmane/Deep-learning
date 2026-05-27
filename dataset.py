"""
Dataset pour chargement d'images propres et dégradées
Supporte S3 (s3://bucket/prefix) ET chemins locaux transparement
Génération de bruit à la volée pour denoising
"""

import os
import io
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random


# ── helpers S3 ───────────────────────────────────────────────────────────────

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _parse_s3(path: str):
    """Retourne (bucket, prefix) depuis 's3://bucket/prefix'."""
    m = re.match(r"s3://([^/]+)/?(.*)", str(path))
    if not m:
        raise ValueError(f"Chemin S3 invalide : {path}")
    bucket = m.group(1)
    prefix = m.group(2).rstrip("/")
    return bucket, prefix


def _s3_list_images(bucket: str, prefix: str):
    """
    Liste récursivement toutes les images sous s3://bucket/prefix.
    Retourne une liste de clés S3 (strings).
    """
    import boto3
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg")):
                keys.append(key)
    return sorted(keys)


def _s3_load_image(bucket: str, key: str) -> Image.Image:
    import boto3
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _local_list_images(root: str):
    """Liste récursivement les images sous un dossier local."""
    root = Path(root)
    keys = []
    for ext in ("png", "jpg", "jpeg"):
        keys.extend(sorted(str(p) for p in root.rglob(f"*.{ext}")))
    return sorted(set(keys))


def _local_load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── Dataset ──────────────────────────────────────────────────────────────────

class ImageRestorationDataset(Dataset):
    """
    Dataset pour restauration d'images.
    Supporte S3 (s3://bucket/prefix) et chemins locaux de façon transparente.

    Mode DENOISING (generate_noise=True) :
        - Seul clean_dir est utilisé
        - Le bruit gaussien est généré à la volée (σ réglable)

    Mode PAIRES (generate_noise=False) :
        - clean_dir + degraded_dir doivent avoir les mêmes noms de fichiers
    """

    def __init__(
        self,
        clean_dir,
        degraded_dir=None,
        patch_size=128,
        num_patches_per_image=8,
        augment=True,
        is_train=True,
        noise_sigma=25,
        generate_noise=True,
    ):
        self.patch_size             = patch_size
        self.num_patches_per_image  = num_patches_per_image
        self.augment                = augment
        self.is_train               = is_train
        self.noise_sigma            = noise_sigma
        self.generate_noise         = generate_noise

        # ── indexation des images propres ────────────────────────────
        self._clean_s3  = _is_s3(clean_dir)
        self._clean_dir = str(clean_dir)

        if self._clean_s3:
            self._clean_bucket, self._clean_prefix = _parse_s3(clean_dir)
            all_keys = _s3_list_images(self._clean_bucket, self._clean_prefix)
            # Stocker les clés S3 complètes
            self.clean_images = all_keys
        else:
            self.clean_images = _local_list_images(clean_dir)

        print(f"Found {len(self.clean_images)} clean images in {clean_dir}")

        # ── indexation des images dégradées (mode paires) ────────────
        self._degraded_s3  = False
        self._degraded_dir = None
        if not generate_noise and degraded_dir:
            self._degraded_s3  = _is_s3(degraded_dir)
            self._degraded_dir = str(degraded_dir)
            if self._degraded_s3:
                self._degraded_bucket, self._degraded_prefix = _parse_s3(degraded_dir)

        if generate_noise:
            print(f"🎯 Mode: DENOISING DYNAMIQUE (σ={noise_sigma}) - Zéro stockage supplémentaire!")
        else:
            print(f"🎯 Mode: Utilisation d'images dégradées existantes")

        # ── transformations ──────────────────────────────────────────
        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize([0.5] * 3, [0.5] * 3)
        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])

    # ── longueur ─────────────────────────────────────────────────────

    def __len__(self):
        if self.is_train:
            return len(self.clean_images) * self.num_patches_per_image
        return len(self.clean_images)

    # ── chargement ───────────────────────────────────────────────────

    def _load_clean(self, idx) -> Image.Image:
        key = self.clean_images[idx]
        if self._clean_s3:
            return _s3_load_image(self._clean_bucket, key)
        return _local_load_image(key)

    def _load_degraded(self, clean_idx) -> Image.Image:
        """Charge l'image dégradée correspondante (mode paires)."""
        if self._clean_s3:
            clean_key = self.clean_images[clean_idx]
            # Remplace le préfixe clean par le préfixe degraded
            rel = clean_key[len(self._clean_prefix):]   # /0001000/img.png
            degraded_key = self._degraded_prefix + rel
            return _s3_load_image(self._degraded_bucket, degraded_key)
        else:
            clean_path = Path(self.clean_images[clean_idx])
            rel        = clean_path.relative_to(self._clean_dir)
            degraded_path = Path(self._degraded_dir) / rel
            return _local_load_image(str(degraded_path))

    # ── bruit ────────────────────────────────────────────────────────

    def _add_gaussian_noise(self, tensor, sigma):
        sigma_n = sigma / 255.0
        noisy   = tensor + torch.randn_like(tensor) * sigma_n
        return torch.clamp(noisy, 0.0, 1.0)

    # ── patch ────────────────────────────────────────────────────────

    def _get_patch(self, img: Image.Image, top, left) -> Image.Image:
        return img.crop((left, top, left + self.patch_size, top + self.patch_size))

    def _random_coords(self, img: Image.Image):
        """Renvoie (top, left) pour un patch aléatoire."""
        left = random.randint(0, max(img.width  - self.patch_size, 0))
        top  = random.randint(0, max(img.height - self.patch_size, 0))
        return top, left

    def _center_coords(self, img: Image.Image):
        left = (img.width  - self.patch_size) // 2
        top  = (img.height - self.patch_size) // 2
        return max(top, 0), max(left, 0)

    def _ensure_min_size(self, img: Image.Image) -> Image.Image:
        if img.width < self.patch_size or img.height < self.patch_size:
            return img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        return img

    # ── __getitem__ ──────────────────────────────────────────────────

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return self._load_item(idx)
            except Exception:
                idx = random.randint(0, len(self) - 1)
        return self._load_item(idx)

    def _load_item(self, idx):
        if self.is_train:
            img_idx   = idx // self.num_patches_per_image
            clean_img = self._ensure_min_size(self._load_clean(img_idx))
            top, left = self._random_coords(clean_img)
            clean_patch = self._get_patch(clean_img, top, left)

            # Augmentation synchronisée
            if self.augment:
                seed = random.randint(0, 2 ** 32 - 1)
                random.seed(seed); torch.manual_seed(seed)
                clean_patch = self.augment_transforms(clean_patch)

            clean_t = self.to_tensor(clean_patch)

            if self.generate_noise:
                sigma       = (random.uniform(*self.noise_sigma)
                               if isinstance(self.noise_sigma, (list, tuple))
                               else self.noise_sigma)
                degraded_t  = self._add_gaussian_noise(clean_t, sigma)
            else:
                deg_img   = self._ensure_min_size(self._load_degraded(img_idx))
                deg_patch = self._get_patch(deg_img, top, left)
                if self.augment:
                    random.seed(seed); torch.manual_seed(seed)
                    deg_patch = self.augment_transforms(deg_patch)
                degraded_t = self.to_tensor(deg_patch)

        else:
            clean_img   = self._ensure_min_size(self._load_clean(idx))
            top, left   = self._center_coords(clean_img)
            clean_patch = self._get_patch(clean_img, top, left)
            clean_t     = self.to_tensor(clean_patch)

            if self.generate_noise:
                sigma      = (random.uniform(*self.noise_sigma)
                              if isinstance(self.noise_sigma, (list, tuple))
                              else self.noise_sigma)
                degraded_t = self._add_gaussian_noise(clean_t, sigma)
            else:
                deg_img   = self._ensure_min_size(self._load_degraded(idx))
                deg_patch = self._get_patch(deg_img, top, left)
                degraded_t = self.to_tensor(deg_patch)

        # Normalisation [-1, 1]
        degraded_t = self.normalize(degraded_t)
        clean_t    = self.normalize(clean_t)

        # Clé lisible pour le logging
        key = self.clean_images[idx if not self.is_train else idx // self.num_patches_per_image]
        name = key.split("/")[-1]   # fonctionne S3 et local

        return degraded_t, clean_t, name


# ── create_dataloaders ───────────────────────────────────────────────────────

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
    generate_noise=True,
):
    train_dataset = ImageRestorationDataset(
        clean_dir=train_clean_dir,
        degraded_dir=train_degraded_dir,
        patch_size=patch_size,
        num_patches_per_image=num_patches_per_image,
        augment=True,
        is_train=True,
        noise_sigma=noise_sigma,
        generate_noise=generate_noise,
    )

    val_dataset = ImageRestorationDataset(
        clean_dir=val_clean_dir,
        degraded_dir=val_degraded_dir,
        patch_size=patch_size,
        num_patches_per_image=1,
        augment=False,
        is_train=False,
        noise_sigma=noise_sigma,
        generate_noise=generate_noise,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader