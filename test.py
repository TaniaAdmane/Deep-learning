"""
Test rapide du dataset avec upsampling
"""

import sys
sys.path.insert(0, '/mnt/project')

from dataset import ImageRestorationDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Créer dataset
data_dir = Path.home() / 'work/data'

dataset = ImageRestorationDataset(
    clean_dir=str(data_dir / 'train/clean'),
    degraded_dir=str(data_dir / 'train/degraded'),
    patch_size=128,
    num_patches_per_image=1,
    augment=False,
    is_train=True
)

print("="*60)
print("TEST DU DATASET AVEC UPSAMPLING")
print("="*60)

# Charger 3 échantillons
for i in range(3):
    degraded, clean, name = dataset[i]
    
    print(f"\nÉchantillon {i}:")
    print(f"  Name: {name}")
    print(f"  Degraded shape: {degraded.shape}")
    print(f"  Clean shape: {clean.shape}")
    print(f"  Degraded range: [{degraded.min():.4f}, {degraded.max():.4f}]")
    print(f"  Clean range: [{clean.min():.4f}, {clean.max():.4f}]")
    print(f"  Degraded mean: {degraded.mean():.4f}")
    print(f"  Clean mean: {clean.mean():.4f}")
    
    # Vérification
    if degraded.min() == degraded.max() == -1.0:
        print("  ❌ STILL BLACK!")
    else:
        print("  ✅ Looks good!")

# Visualiser
print("\nCréation de visualisation...")
degraded, clean, name = dataset[0]

# Dénormaliser pour visualisation
degraded_vis = (degraded.numpy() + 1) / 2
clean_vis = (clean.numpy() + 1) / 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(np.transpose(degraded_vis, (1, 2, 0)))
axes[0].set_title(f'Degraded (LR upsampled)\nRange: [{degraded.min():.2f}, {degraded.max():.2f}]')
axes[0].axis('off')

axes[1].imshow(np.transpose(clean_vis, (1, 2, 0)))
axes[1].set_title(f'Clean (HR)\nRange: [{clean.min():.2f}, {clean.max():.2f}]')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('test_upsampled_patches.png', dpi=150, bbox_inches='tight')
print("✅ Visualisation sauvegardée : test_upsampled_patches.png")

print("\n" + "="*60)
print("TEST TERMINÉ")
print("="*60)