"""
Noise2Void — entraînement self-supervised
Même architecture U-Net que unet.py (UNet_Restoration)
Seule différence : on masque des pixels en entrée et on
prédit uniquement sur ces pixels masqués (pas d'image propre nécessaire)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob, os, numpy as np
from pathlib import Path

from unet_restoration import UNet_Restoration

import torch.nn.functional as F

class N2VUNet(nn.Module):
    """Wrapper N2V autour du U-Net de ta copine — corrige les tailles sans modifier son code."""
    def __init__(self):
        super().__init__()
        from unet_restoration import UNet_Restoration
        self.unet = UNet_Restoration(input_channels=3, output_channels=3, base_channels=64)

    def forward(self, x):
        # Pad l'input à un multiple de 16 si nécessaire
        B, C, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        recon, _, _ = self.unet(x)
        
        # Crop pour revenir à la taille originale
        if pad_h > 0 or pad_w > 0:
            recon = recon[:, :, :H, :W]
        
        return recon, None, None

# ── Config ──────────────────────────────────────────────────────────
CLEAN_DIR = Path.home() / 'work/train'
VAL_DIR     = Path.home() / 'work/val/clean'
SAVE_PATH   = Path.home() / 'work/checkpoints/best_n2v.pth'
IMG_SIZE    = 128
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-3
NOISE_STD   = 25 / 255.0
MASK_RATIO  = 0.02   # 2% des pixels masqués, comme dans le papier
# ────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')


# ── Dataset ─────────────────────────────────────────────────────────

class N2VDataset(Dataset):
    """
    Charge des images propres, applique du bruit gaussien à la volée.
    Ne retourne QUE l'image bruitée — pas l'image propre.
    C'est le principe N2V : apprentissage self-supervised.
    """
    def __init__(self, img_dir, img_size=128, noise_std=0.1):
        self.paths = (glob.glob(str(img_dir / '**/*.png'), recursive=True) +
                      glob.glob(str(img_dir / '**/*.jpg'), recursive=True))
        self.noise_std = noise_std
        self.transform = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1] comme ta copine
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        clean = self.transform(img)
        noisy = clean + torch.randn_like(clean) * self.noise_std
        noisy = noisy.clamp(-1., 1.)
        return noisy  # pas de clean !


# ── Masquage N2V ────────────────────────────────────────────────────

def n2v_mask(batch, mask_ratio=0.02, neighborhood=5):
    """
    Pour chaque pixel masqué :
      - INPUT  : remplacé par un voisin aléatoire (le réseau ne voit pas le pixel)
      - TARGET : valeur originale bruitée (ce qu'on veut prédire)

    Retourne :
      masked_input : ce qu'on donne au réseau
      mask         : booléen (True = pixel masqué)
    """
    B, C, H, W = batch.shape
    masked = batch.clone()
    mask   = torch.zeros(B, 1, H, W, dtype=torch.bool, device=batch.device)
    half   = neighborhood // 2
    n_px   = int(H * W * mask_ratio)

    for b in range(B):
        coords = torch.randperm(H * W, device=batch.device)[:n_px]
        rows, cols = coords // W, coords % W

        for r, c in zip(rows.tolist(), cols.tolist()):
            # Voisin aléatoire ≠ pixel lui-même
            while True:
                dr = np.random.randint(-half, half + 1)
                dc = np.random.randint(-half, half + 1)
                if dr != 0 or dc != 0:
                    break
            nr = min(max(r + dr, 0), H - 1)
            nc = min(max(c + dc, 0), W - 1)
            masked[b, :, r, c] = batch[b, :, nr, nc]
            mask[b, 0, r, c]   = True

    return masked, mask


# ── Métriques ───────────────────────────────────────────────────────

def psnr(pred, target):
    # Dénormalise [-1,1] → [0,1] avant calcul
    p = (pred.clamp(-1, 1)   + 1) / 2
    t = (target.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(p, t)
    return 20 * torch.log10(1.0 / (mse.sqrt() + 1e-8))


# ── Entraînement ────────────────────────────────────────────────────

def train():
    train_set = N2VDataset(CLEAN_DIR, IMG_SIZE, NOISE_STD)
    val_set   = N2VDataset(VAL_DIR,   IMG_SIZE, NOISE_STD)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f'Train : {len(train_set)} images | Val : {len(val_set)} images')

    # Même U-Net que ta copine, sans modification
    model     = model = UNet_Restoration(input_channels=3, output_channels=3, base_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres : {n_params:,}')

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.
        for noisy in train_loader:
            noisy = noisy.to(device)
            masked_input, mask = n2v_mask(noisy, MASK_RATIO)

            pred, _, _ = model(masked_input)  # (recon, None, None)

            # Loss uniquement sur les pixels masqués
            mask_exp = mask.expand_as(pred)
            loss = F.mse_loss(pred[mask_exp], noisy[mask_exp])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ──
        model.eval()
        val_loss, val_psnr = 0., 0.
        with torch.no_grad():
            for noisy in val_loader:
                noisy = noisy.to(device)
                masked_input, mask = n2v_mask(noisy, MASK_RATIO)
                pred, _, _ = model(masked_input)
                mask_exp = mask.expand_as(pred)
                val_loss += F.mse_loss(pred[mask_exp], noisy[mask_exp]).item()
                val_psnr += psnr(pred, noisy).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        avg_psnr  = val_psnr   / len(val_loader)
        scheduler.step()

        print(f'Epoch {epoch:03d}/{EPOCHS} | '
              f'Train loss: {avg_train:.6f} | '
              f'Val loss: {avg_val:.6f} | '
              f'PSNR: {avg_psnr:.2f} dB')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_psnr': avg_psnr,
            }, SAVE_PATH)
            print(f'  → Checkpoint sauvegardé')


    print(f'\nTerminé. Meilleur checkpoint : {SAVE_PATH}')


if __name__ == '__main__':
    train()
