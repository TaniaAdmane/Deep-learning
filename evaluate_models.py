"""
Script d'évaluation unifié pour la restauration d'images
Supporte : VAE, VAE-UNet, U-Net, Autoencoder, SRCNN, SRCNN-Lite

Usage :
    python evaluate.py --model unet      --checkpoint checkpoints/unet/best_model.pth ...
    python evaluate.py --model vae       --checkpoint checkpoints/vae/best_model.pth  ...
    python evaluate.py --model srcnn     --checkpoint checkpoints/srcnn/best_model.pth ...

    # Comparer plusieurs modèles d'un coup :
    python evaluate.py --model unet vae srcnn \
        --checkpoint checkpoints/unet/best_model.pth \
                       checkpoints/vae/best_model.pth \
                       checkpoints/srcnn/best_model.pth \
        --test_clean s3://taniaadmane/dossier/val/clean \
        --test_degraded s3://taniaadmane/dossier/val/clean
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from dataset import ImageRestorationDataset
from metrics import MetricsCalculator
from torch.utils.data import DataLoader

# ── imports modèles ──────────────────────────────────────────────────────────
from models.vae_restoration          import VAE_Restoration
from models.vae_unet                 import VAE_UNet
from models.unet_restoration         import UNet_Restoration
from models.autoencoder_restoration  import Autoencoder_Restoration
from models.srcnn_restoration        import SRCNN_Restoration, SRCNN_Lite


# ── chargement du modèle depuis checkpoint ───────────────────────────────────

def load_model(model_name: str, checkpoint_path: str, patch_size: int, device: str):
    """
    Charge un modèle depuis son checkpoint.
    Retourne le modèle en mode eval, prêt à l'inférence.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    name = model_name.lower()

    if name == 'vae':
        model = VAE_Restoration(
            input_channels=3,
            latent_dim=checkpoint.get('latent_dim', 256),
            image_size=patch_size,
        )
    elif name == 'vae_unet':
        model = VAE_UNet(
            input_channels=3,
            latent_dim=checkpoint.get('latent_dim', 256),
            image_size=patch_size,
        )
    elif name == 'unet':
        model = UNet_Restoration(base_channels=64)

    elif name == 'autoencoder':
        model = Autoencoder_Restoration(
            input_channels=3,
            latent_dim=checkpoint.get('latent_dim', 256),
            image_size=patch_size,
        )
    elif name == 'srcnn':
        model = SRCNN_Restoration(feature_channels=64, num_residual_blocks=8)

    elif name == 'srcnn_lite':
        model = SRCNN_Lite()

    else:
        raise ValueError(f"Modèle inconnu : '{model_name}'")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    epoch = checkpoint.get('epoch', '?')
    psnr  = checkpoint.get('best_val_psnr', '?')
    print(f"  ✅ {model_name} chargé — epoch {epoch}  |  best val PSNR : {psnr}")
    return model


# ── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Évaluateur générique compatible avec tous les modèles de restauration.
    Interface commune : model(x) → (recon, *, *)
    """

    def __init__(self, model, model_name: str, device='cuda'):
        self.model      = model
        self.model_name = model_name
        self.device     = device
        self.metrics_calculator = MetricsCalculator(device=device)

    # ── évaluation complète ───────────────────────────────────────────

    @torch.no_grad()
    def evaluate_dataset(self, dataloader):
        """
        Calcule PSNR / SSIM / LPIPS sur tout le dataset.
        Retourne un dict avec moyennes, écarts-types et listes complètes.
        """
        all_psnr, all_ssim, all_lpips = [], [], []

        for degraded, clean, _ in tqdm(dataloader, desc=f"Eval {self.model_name}"):
            degraded = degraded.to(self.device)
            clean    = clean.to(self.device)

            recon, _, _ = self.model(degraded)

            metrics = self.metrics_calculator.calculate_all_metrics(recon, clean)
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            all_lpips.append(metrics['lpips'])

        return {
            'psnr_mean':  np.mean(all_psnr),
            'psnr_std':   np.std(all_psnr),
            'psnr_all':   all_psnr,
            'ssim_mean':  np.mean(all_ssim),
            'ssim_std':   np.std(all_ssim),
            'ssim_all':   all_ssim,
            'lpips_mean': np.mean(all_lpips),
            'lpips_std':  np.std(all_lpips),
            'lpips_all':  all_lpips,
        }

    # ── visualisations ────────────────────────────────────────────────

    @torch.no_grad()
    def visualize_reconstructions(self, dataloader, num_samples=8, save_path='results'):
        """Sauvegarde des triplets (dégradé | reconstruction | référence)."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        degraded, clean, names = next(iter(dataloader))
        degraded = degraded[:num_samples].to(self.device)
        clean    = clean[:num_samples].to(self.device)

        recon, _, _ = self.model(degraded)

        to_np = lambda t: ((t.clamp(-1, 1) + 1) / 2).cpu().numpy()
        deg_np   = to_np(degraded)
        clean_np = to_np(clean)
        rec_np   = to_np(recon)

        for i in range(num_samples):
            m = self.metrics_calculator.calculate_all_metrics(
                recon[i:i+1], clean[i:i+1]
            )
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(np.transpose(deg_np[i],   (1, 2, 0)))
            axes[0].set_title('Dégradée (entrée)')
            axes[0].axis('off')

            axes[1].imshow(np.transpose(rec_np[i],   (1, 2, 0)))
            axes[1].set_title(
                f'{self.model_name} — Reconstruction\n'
                f'PSNR: {m["psnr"]:.2f} dB  |  '
                f'SSIM: {m["ssim"]:.4f}  |  '
                f'LPIPS: {m["lpips"]:.4f}'
            )
            axes[1].axis('off')

            axes[2].imshow(np.transpose(clean_np[i], (1, 2, 0)))
            axes[2].set_title('Référence (propre)')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_path / f'reconstruction_{i:02d}.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Reconstructions sauvegardées → {save_path}")

    @torch.no_grad()
    def analyze_uncertainty(self, dataloader, num_samples=10, num_images=4, save_path='results'):
        """
        Analyse d'incertitude par échantillonnage multiple.
        Pertinent pour VAE / VAE-UNet (stochastiques).
        Pour U-Net / SRCNN / Autoencoder : les N reconstructions sont identiques
        → la carte de variance sera nulle, ce qui est une info utile aussi.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        degraded, clean, _ = next(iter(dataloader))
        degraded = degraded[:num_images].to(self.device)
        clean    = clean[:num_images].to(self.device)

        for img_idx in range(num_images):
            x       = degraded[img_idx:img_idx + 1]
            samples = self.model.sample(x, num_samples=num_samples)
            # samples : (num_samples, 1, 3, H, W)  ou  (num_samples, B, 3, H, W)
            samples_np = ((samples.squeeze(1).clamp(-1, 1) + 1) / 2).cpu().numpy()

            mean_recon    = np.mean(samples_np, axis=0)
            variance_gray = np.mean(np.var(samples_np, axis=0), axis=0)

            ncols = num_samples + 4   # input + GT + N samples + mean + variance
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 4))

            axes[0].imshow(np.transpose(
                ((degraded[img_idx].clamp(-1,1)+1)/2).cpu().numpy(), (1,2,0)))
            axes[0].set_title('Entrée')
            axes[0].axis('off')

            axes[1].imshow(np.transpose(
                ((clean[img_idx].clamp(-1,1)+1)/2).cpu().numpy(), (1,2,0)))
            axes[1].set_title('Référence')
            axes[1].axis('off')

            for i in range(num_samples):
                axes[i + 2].imshow(np.transpose(samples_np[i], (1, 2, 0)))
                axes[i + 2].set_title(f'#{i+1}')
                axes[i + 2].axis('off')

            axes[num_samples + 2].imshow(np.transpose(mean_recon, (1, 2, 0)))
            axes[num_samples + 2].set_title('Moyenne')
            axes[num_samples + 2].axis('off')

            im = axes[num_samples + 3].imshow(variance_gray, cmap='hot')
            axes[num_samples + 3].set_title('Incertitude')
            axes[num_samples + 3].axis('off')
            plt.colorbar(im, ax=axes[num_samples + 3], fraction=0.046)

            plt.suptitle(f'{self.model_name} — analyse incertitude (image {img_idx})',
                         fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(save_path / f'uncertainty_{img_idx:02d}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Incertitude sauvegardée → {save_path}")

    def plot_metrics_distribution(self, results, save_path='results'):
        """Histogrammes PSNR / SSIM / LPIPS."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Distribution des métriques — {self.model_name}', fontsize=13)

        specs = [
            ('psnr',  'PSNR (dB)',           'blue',   '{:.2f}',  '{:.2f}'),
            ('ssim',  'SSIM',                'green',  '{:.4f}',  '{:.4f}'),
            ('lpips', 'LPIPS (↓ meilleur)',  'orange', '{:.4f}',  '{:.4f}'),
        ]

        for ax, (key, xlabel, color, fmt_mean, fmt_std) in zip(axes, specs):
            ax.hist(results[f'{key}_all'], bins=30, alpha=0.7,
                    color=color, edgecolor='black')
            ax.axvline(results[f'{key}_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Moy. : {fmt_mean.format(results[f"{key}_mean"])}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Fréquence')
            ax.set_title(f'Std : {fmt_std.format(results[f"{key}_std"])}')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Distribution métriques → {save_path}")


# ── comparaison multi-modèles ────────────────────────────────────────────────

def plot_comparison(all_results: dict, save_path: str):
    """
    Graphe comparatif barres pour plusieurs modèles.
    all_results : {model_name: results_dict}
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model_names = list(all_results.keys())
    metrics     = ['psnr', 'ssim', 'lpips']
    titles      = ['PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓']
    colors      = plt.cm.tab10(np.linspace(0, 0.5, len(model_names)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparaison des modèles', fontsize=14, fontweight='bold')

    for ax, metric, title in zip(axes, metrics, titles):
        means = [all_results[m][f'{metric}_mean'] for m in model_names]
        stds  = [all_results[m][f'{metric}_std']  for m in model_names]

        bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(title.split()[0])
        ax.grid(axis='y', alpha=0.3)

        # Valeur au-dessus de chaque barre
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = save_path / 'model_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparaison sauvegardée → {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Évaluation unifiée — restauration d\'images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model', nargs='+', required=True,
                        choices=['vae', 'vae_unet', 'unet',
                                 'autoencoder', 'srcnn', 'srcnn_lite'],
                        help='Modèle(s) à évaluer (un ou plusieurs)')
    parser.add_argument('--checkpoint', nargs='+', required=True,
                        help='Chemin(s) vers le(s) checkpoint(s) — même ordre que --model')

    parser.add_argument('--test_clean',    type=str, required=True)
    parser.add_argument('--test_degraded', type=str, required=True)

    parser.add_argument('--patch_size',          type=int, default=128, choices=[128, 256])
    parser.add_argument('--batch_size',          type=int, default=8)
    parser.add_argument('--num_workers',         type=int, default=4)
    parser.add_argument('--noise_sigma',         type=int, default=25)
    parser.add_argument('--generate_noise',      action='store_true', default=True)

    parser.add_argument('--output_dir',          type=str, default='eval_results')
    parser.add_argument('--num_vis_samples',     type=int, default=8)
    parser.add_argument('--uncertainty_samples', type=int, default=10)
    parser.add_argument('--skip_uncertainty',    action='store_true',
                        help='Passer l\'analyse d\'incertitude (plus rapide)')

    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    if len(args.model) != len(args.checkpoint):
        raise ValueError(
            f"--model ({len(args.model)}) et --checkpoint ({len(args.checkpoint)}) "
            "doivent avoir le même nombre d'arguments."
        )

    # ── Dataset de test ──────────────────────────────────────────────
    print("\nCréation du dataset de test...")
    test_dataset = ImageRestorationDataset(
        clean_dir=args.test_clean,
        degraded_dir=args.test_degraded,
        patch_size=args.patch_size,
        num_patches_per_image=1,
        augment=False,
        is_train=False,
        noise_sigma=args.noise_sigma,
        generate_noise=args.generate_noise,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"  {len(test_dataset)} images de test")

    # ── Évaluation de chaque modèle ──────────────────────────────────
    all_results = {}

    for model_name, ckpt_path in zip(args.model, args.checkpoint):
        print(f"\n{'='*60}")
        print(f"  Modèle : {model_name}  |  checkpoint : {ckpt_path}")
        print(f"{'='*60}")

        out_dir = Path(args.output_dir) / model_name

        # Chargement
        model     = load_model(model_name, ckpt_path, args.patch_size, device)
        evaluator = Evaluator(model, model_name, device)

        # 1. Métriques globales
        print("\n→ Calcul des métriques...")
        results = evaluator.evaluate_dataset(test_loader)
        all_results[model_name] = results

        print(f"\n  PSNR  : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"  SSIM  : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"  LPIPS : {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")

        # Sauvegarde JSON (sans les listes complètes)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(
                {k: v for k, v in results.items() if not k.endswith('_all')},
                f, indent=2
            )

        # 2. Visualisations
        print("\n→ Visualisations reconstructions...")
        evaluator.visualize_reconstructions(
            test_loader,
            num_samples=args.num_vis_samples,
            save_path=out_dir / 'reconstructions',
        )

        # 3. Incertitude
        if not args.skip_uncertainty:
            print("\n→ Analyse d'incertitude...")
            evaluator.analyze_uncertainty(
                test_loader,
                num_samples=args.uncertainty_samples,
                num_images=4,
                save_path=out_dir / 'uncertainty',
            )

        # 4. Distribution des métriques
        print("\n→ Distribution des métriques...")
        evaluator.plot_metrics_distribution(results, save_path=out_dir)

    # ── Comparaison si plusieurs modèles ────────────────────────────
    if len(args.model) > 1:
        print(f"\n{'='*60}")
        print("  Comparaison multi-modèles")
        print(f"{'='*60}")
        plot_comparison(all_results, save_path=args.output_dir)

        # Tableau récapitulatif
        print(f"\n{'Modèle':<15} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
        print("-" * 47)
        for name, res in all_results.items():
            print(f"{name:<15} "
                  f"{res['psnr_mean']:>8.2f} dB  "
                  f"{res['ssim_mean']:>8.4f}  "
                  f"{res['lpips_mean']:>8.4f}")

    print(f"\n✅ Évaluation terminée — résultats dans : {args.output_dir}/")


if __name__ == '__main__':
    main()