"""
Script d'évaluation du VAE entraîné
Calcul métriques, visualisations, analyse d'incertitude
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from models.vae_restoration import VAE_Restoration
from dataset import ImageRestorationDataset
from metrics import MetricsCalculator
from torch.utils.data import DataLoader


class VAEEvaluator:
    """
    Évaluateur pour le VAE
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.metrics_calculator = MetricsCalculator(device=device)
    
    @torch.no_grad()
    def evaluate_dataset(self, dataloader):
        """
        Évalue le modèle sur un dataset complet
        
        Returns:
            dict avec métriques moyennes et par image
        """
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        for degraded, clean, names in tqdm(dataloader, desc="Evaluating"):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Reconstruction
            recon, _, _ = self.model(degraded)
            
            # Métriques
            metrics = self.metrics_calculator.calculate_all_metrics(recon, clean)
            
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            all_lpips.append(metrics['lpips'])
        
        results = {
            'psnr_mean': np.mean(all_psnr),
            'psnr_std': np.std(all_psnr),
            'psnr_all': all_psnr,
            'ssim_mean': np.mean(all_ssim),
            'ssim_std': np.std(all_ssim),
            'ssim_all': all_ssim,
            'lpips_mean': np.mean(all_lpips),
            'lpips_std': np.std(all_lpips),
            'lpips_all': all_lpips,
        }
        
        return results
    
    @torch.no_grad()
    def visualize_reconstructions(self, dataloader, num_samples=8, save_path='results'):
        """
        Visualise des exemples de reconstructions
        
        Args:
            dataloader: dataloader de validation
            num_samples: nombre d'exemples à visualiser
            save_path: dossier de sauvegarde
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Prendre un batch
        degraded, clean, names = next(iter(dataloader))
        degraded = degraded[:num_samples].to(self.device)
        clean = clean[:num_samples].to(self.device)
        
        # Reconstruction
        recon, mu, logvar = self.model(degraded)
        
        # Dénormaliser pour visualisation
        degraded_vis = ((degraded + 1) / 2).cpu().numpy()
        clean_vis = ((clean + 1) / 2).cpu().numpy()
        recon_vis = ((recon + 1) / 2).cpu().numpy()
        
        # Calculer métriques pour chaque image
        for i in range(num_samples):
            metrics = self.metrics_calculator.calculate_all_metrics(
                recon[i:i+1], clean[i:i+1]
            )
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Degraded
            axes[0].imshow(np.transpose(degraded_vis[i], (1, 2, 0)))
            axes[0].set_title('Degraded Input')
            axes[0].axis('off')
            
            # Reconstructed
            axes[1].imshow(np.transpose(recon_vis[i], (1, 2, 0)))
            axes[1].set_title(f'VAE Reconstruction\n'
                             f'PSNR: {metrics["psnr"]:.2f} dB | '
                             f'SSIM: {metrics["ssim"]:.4f} | '
                             f'LPIPS: {metrics["lpips"]:.4f}')
            axes[1].axis('off')
            
            # Clean
            axes[2].imshow(np.transpose(clean_vis[i], (1, 2, 0)))
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path / f'reconstruction_{i}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {save_path}")
    
    @torch.no_grad()
    def analyze_uncertainty(self, dataloader, num_samples=10, num_images=4, save_path='results'):
        """
        Analyse l'incertitude du VAE via échantillonnage multiple
        
        Args:
            dataloader: dataloader de validation
            num_samples: nombre d'échantillons à générer par image
            num_images: nombre d'images à analyser
            save_path: dossier de sauvegarde
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Prendre un batch
        degraded, clean, names = next(iter(dataloader))
        degraded = degraded[:num_images].to(self.device)
        clean = clean[:num_images].to(self.device)
        
        for img_idx in range(num_images):
            # Générer multiples reconstructions
            x = degraded[img_idx:img_idx+1]
            samples = self.model.sample(x, num_samples=num_samples)  # (num_samples, 1, 3, H, W)
            
            # Calculer variance pixel-wise
            samples_np = ((samples.squeeze(1) + 1) / 2).cpu().numpy()  # (num_samples, 3, H, W)
            variance = np.var(samples_np, axis=0)  # (3, H, W)
            
            # Moyenne des échantillons
            mean_recon = np.mean(samples_np, axis=0)
            
            # Visualisation
            fig, axes = plt.subplots(2, num_samples // 2 + 2, figsize=(20, 8))
            axes = axes.flatten()
            
            # Input
            axes[0].imshow(np.transpose(((degraded[img_idx] + 1) / 2).cpu().numpy(), (1, 2, 0)))
            axes[0].set_title('Input (Degraded)')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(np.transpose(((clean[img_idx] + 1) / 2).cpu().numpy(), (1, 2, 0)))
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Échantillons
            for i in range(num_samples):
                axes[i + 2].imshow(np.transpose(samples_np[i], (1, 2, 0)))
                axes[i + 2].set_title(f'Sample {i+1}')
                axes[i + 2].axis('off')
            
            # Moyenne
            axes[num_samples + 2].imshow(np.transpose(mean_recon, (1, 2, 0)))
            axes[num_samples + 2].set_title('Mean Reconstruction')
            axes[num_samples + 2].axis('off')
            
            # Carte de variance (en grayscale)
            variance_gray = np.mean(variance, axis=0)  # Moyenne sur RGB
            axes[num_samples + 3].imshow(variance_gray, cmap='hot')
            axes[num_samples + 3].set_title('Uncertainty Map\n(High = uncertain)')
            axes[num_samples + 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path / f'uncertainty_analysis_{img_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Uncertainty analysis saved to {save_path}")
    
    def plot_metrics_distribution(self, results, save_path='results'):
        """
        Plot la distribution des métriques
        
        Args:
            results: dict de evaluate_dataset()
            save_path: dossier de sauvegarde
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # PSNR
        axes[0].hist(results['psnr_all'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(results['psnr_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {results["psnr_mean"]:.2f} dB')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'PSNR Distribution\nStd: {results["psnr_std"]:.2f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # SSIM
        axes[1].hist(results['ssim_all'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(results['ssim_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {results["ssim_mean"]:.4f}')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'SSIM Distribution\nStd: {results["ssim_std"]:.4f}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # LPIPS
        axes[2].hist(results['lpips_all'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(results['lpips_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {results["lpips_mean"]:.4f}')
        axes[2].set_xlabel('LPIPS (lower is better)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'LPIPS Distribution\nStd: {results["lpips_std"]:.4f}')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics distribution plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE for image restoration')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_clean', type=str, required=True,
                        help='Path to test clean images')
    parser.add_argument('--test_degraded', type=str, required=True,
                        help='Path to test degraded images')
    parser.add_argument('--patch_size', type=int, default=128,
                        choices=[128, 256], help='Patch size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Output directory')
    parser.add_argument('--num_vis_samples', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--uncertainty_samples', type=int, default=10,
                        help='Number of samples for uncertainty analysis')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=checkpoint.get('latent_dim', 256),
        image_size=args.patch_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded (epoch {checkpoint['epoch']})")
    
    # Create dataset
    print("\nCreating test dataset...")
    test_dataset = ImageRestorationDataset(
        clean_dir=args.test_clean,
        degraded_dir=args.test_degraded,
        patch_size=args.patch_size,
        num_patches_per_image=1,
        augment=False,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test set: {len(test_dataset)} images")
    
    # Evaluator
    evaluator = VAEEvaluator(model, device=device)
    
    # 1. Evaluate metrics
    print("\n" + "="*60)
    print("Evaluating metrics on test set...")
    print("="*60)
    results = evaluator.evaluate_dataset(test_loader)
    
    print(f"\nResults:")
    print(f"  PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"  SSIM:  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"  LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    
    # Save results
    with open(output_dir / 'metrics.json', 'w') as f:
        # Remove full lists for JSON (too large)
        save_results = {k: v for k, v in results.items() if not k.endswith('_all')}
        json.dump(save_results, f, indent=2)
    
    # 2. Visualize reconstructions
    print("\n" + "="*60)
    print("Creating reconstruction visualizations...")
    print("="*60)
    evaluator.visualize_reconstructions(
        test_loader,
        num_samples=args.num_vis_samples,
        save_path=output_dir / 'reconstructions'
    )
    
    # 3. Uncertainty analysis
    print("\n" + "="*60)
    print("Analyzing uncertainty via sampling...")
    print("="*60)
    evaluator.analyze_uncertainty(
        test_loader,
        num_samples=args.uncertainty_samples,
        num_images=4,
        save_path=output_dir / 'uncertainty'
    )
    
    # 4. Plot distributions
    print("\n" + "="*60)
    print("Plotting metrics distributions...")
    print("="*60)
    evaluator.plot_metrics_distribution(results, save_path=output_dir)
    
    print("\n" + "="*60)
    print(f"Evaluation completed! Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()