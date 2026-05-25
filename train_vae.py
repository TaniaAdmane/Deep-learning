"""
Script d'entraînement du VAE pour restauration d'images
Avec logging TensorBoard, checkpointing, et validation
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from models.vae_restoration import VAE_Restoration, vae_loss
from dataset import create_dataloaders
from metrics import MetricsCalculator, MetricsTracker
import lpips


class VAETrainer:
    """
    Classe pour gérer l'entraînement du VAE
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-4,
        beta=1.0,
        perceptual_weight=0.1,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Hyperparamètres
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Scheduler (optionnel)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # LPIPS pour loss perceptuelle
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # Métriques
        self.metrics_calculator = MetricsCalculator(device=device)
        
        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Beta (KL weight): {beta}")
        print(f"  Perceptual weight: {perceptual_weight}")
    
    def train_epoch(self):
        """Entraîne le modèle pour une epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_perceptual_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(degraded)
            
            # Loss
            loss, loss_dict = vae_loss(
                recon, clean, mu, logvar,
                beta=self.beta,
                perceptual_weight=self.perceptual_weight,
                lpips_fn=self.lpips_fn
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (évite explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Tracking
            epoch_loss += loss_dict['total']
            epoch_recon_loss += loss_dict['reconstruction']
            epoch_kl_loss += loss_dict['kl']
            if batch_idx == 0:  # Premier batch seulement
                print(f"DEBUG: KL from loss_dict = {loss_dict['kl']:.4f}")
            epoch_perceptual_loss += loss_dict['perceptual']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'kl': f"{loss_dict['kl']:.4f}"
            })
            
            # Log batch
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss_batch', loss_dict['total'], global_step)
                self.writer.add_scalar('Train/Recon_batch', loss_dict['reconstruction'], global_step)
                self.writer.add_scalar('Train/KL_batch', loss_dict['kl'], global_step)
        
        # Moyennes epoch
        num_batches = len(self.train_loader)
        epoch_stats = {
            'loss': epoch_loss / num_batches,
            'recon': epoch_recon_loss / num_batches,
            'kl': epoch_kl_loss / num_batches,
            'perceptual': epoch_perceptual_loss / num_batches
        }
        
        return epoch_stats
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        val_loss = 0.0
        val_kl_loss = 0.0  # ⬅️ AJOUT
        metrics_tracker = MetricsTracker()
        
        for degraded, clean, _ in tqdm(self.val_loader, desc="Validation"):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward
            recon, mu, logvar = self.model(degraded)
            
            # Loss
            loss, loss_dict = vae_loss(
                recon, clean, mu, logvar,
                beta=self.beta,
                perceptual_weight=self.perceptual_weight,
                lpips_fn=self.lpips_fn
            )
            
            val_loss += loss_dict['total']
            val_kl_loss += loss_dict['kl']  
            
            # Métriques
            metrics = self.metrics_calculator.calculate_all_metrics(recon, clean)
            metrics_tracker.update(metrics)
        
        # Moyennes
        val_loss /= len(self.val_loader)
        val_kl_loss /= len(self.val_loader)  
        avg_metrics = metrics_tracker.get_average()
        avg_metrics['kl'] = val_kl_loss  
        
        return val_loss, avg_metrics
        
    def save_checkpoint(self, is_best=False, filename='checkpoint.pth'):
        """Sauvegarde checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr,
            'beta': self.beta,
            'perceptual_weight': self.perceptual_weight
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            print(f"✅ Best model saved at epoch {self.current_epoch}")
    
    def load_checkpoint(self, filepath):
        """Charge checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_psnr = checkpoint['best_val_psnr']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def log_images(self, num_images=4):
        """Log des exemples d'images dans TensorBoard"""
        self.model.eval()
        
        with torch.no_grad():
            # Prendre un batch de validation
            degraded, clean, _ = next(iter(self.val_loader))
            degraded = degraded[:num_images].to(self.device)
            clean = clean[:num_images].to(self.device)
            
            # Reconstruction
            recon, _, _ = self.model(degraded)
            
            # Dénormaliser pour visualisation [0, 1]
            degraded_vis = (degraded + 1) / 2
            clean_vis = (clean + 1) / 2
            recon_vis = (recon + 1) / 2
            
            # Log dans TensorBoard
            self.writer.add_images('Val/Degraded', degraded_vis, self.current_epoch)
            self.writer.add_images('Val/Reconstructed', recon_vis, self.current_epoch)
            self.writer.add_images('Val/Clean', clean_vis, self.current_epoch)
    
    def train(self, num_epochs, log_images_every=5):
        """
        Boucle d'entraînement complète
        
        Args:
            num_epochs: nombre d'epochs
            log_images_every: fréquence de logging d'images
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_stats = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Logging
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_stats['loss']:.4f} "
                  f"(Recon: {train_stats['recon']:.4f}, KL: {train_stats['kl']:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Val LPIPS: {val_metrics['lpips']:.4f}")
            
            # TensorBoard
            self.writer.add_scalar('Train/Loss_epoch', train_stats['loss'], epoch)
            self.writer.add_scalar('Train/Recon_epoch', train_stats['recon'], epoch)
            self.writer.add_scalar('Train/KL_epoch', train_stats['kl'], epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/PSNR', val_metrics['psnr'], epoch)
            self.writer.add_scalar('Val/SSIM', val_metrics['ssim'], epoch)
            self.writer.add_scalar('Val/LPIPS', val_metrics['lpips'], epoch)
            self.writer.add_scalar('Train/Learning_Rate', 
                                   self.optimizer.param_groups[0]['lr'], epoch)
            
            # Log images
            if epoch % log_images_every == 0:
                self.log_images()
            
            # Checkpoint
            is_best = val_metrics['psnr'] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics['psnr']
                self.best_val_loss = val_loss
            
            if is_best:
                self.save_checkpoint(is_best=True, filename='best_model.pth')
            
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best val PSNR: {self.best_val_psnr:.2f} dB")
        print(f"{'='*60}\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train VAE for image restoration')
    
    # Données
    parser.add_argument('--train_clean', type=str, required=True,
                        help='Path to training clean images')
    parser.add_argument('--train_degraded', type=str, required=True,
                        help='Path to training degraded images')
    parser.add_argument('--val_clean', type=str, required=True,
                        help='Path to validation clean images')
    parser.add_argument('--val_degraded', type=str, required=True,
                        help='Path to validation degraded images')
    
    # Hyperparamètres
    parser.add_argument('--patch_size', type=int, default=128,
                        choices=[128, 256], help='Patch size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta for KL divergence (β-VAE)')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss (LPIPS)')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    
    # Autres
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_clean_dir=args.train_clean,
        train_degraded_dir=args.train_degraded,
        val_clean_dir=args.val_clean,
        val_degraded_dir=args.val_degraded,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating model...")
    model = VAE_Restoration(
        input_channels=3,
        latent_dim=args.latent_dim,
        image_size=args.patch_size
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        beta=args.beta,
        perceptual_weight=args.perceptual_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Resume si demandé
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()