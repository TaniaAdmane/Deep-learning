"""
Baseline : Simple upsampling bicubique
"""
from PIL import Image
import torch
from pathlib import Path
from metrics import MetricsCalculator
from tqdm import tqdm

data_dir = Path.home() / 'work/data'
val_clean = data_dir / 'val/clean'
val_degraded = data_dir / 'val/degraded'

device = 'cuda'
metrics_calc = MetricsCalculator(device=device)

from torchvision import transforms
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.5]*3, [0.5]*3)

psnrs, ssims, lpips = [], [], []

for clean_path in tqdm(list(val_clean.glob('**/*.png'))[:50]):
    rel_path = clean_path.relative_to(val_clean)
    degraded_path = val_degraded / rel_path.with_name(rel_path.stem + 'x2' + rel_path.suffix)
    
    if not degraded_path.exists():
        continue
    
    clean_img = Image.open(clean_path).convert('RGB')
    degraded_img = Image.open(degraded_path).convert('RGB')
    
    # Upsampling bicubique
    bicubic = degraded_img.resize(clean_img.size, Image.BICUBIC)
    
    # Convertir en tensors
    clean_t = normalize(to_tensor(clean_img)).unsqueeze(0).to(device)
    bicubic_t = normalize(to_tensor(bicubic)).unsqueeze(0).to(device)
    
    # Métriques
    m = metrics_calc.calculate_all_metrics(bicubic_t, clean_t)
    psnrs.append(m['psnr'])
    ssims.append(m['ssim'])
    lpips.append(m['lpips'])

import numpy as np
print(f"\n📊 BASELINE BICUBIC:")
print(f"  PSNR: {np.mean(psnrs):.2f} dB")
print(f"  SSIM: {np.mean(ssims):.4f}")
print(f"  LPIPS: {np.mean(lpips):.4f}")