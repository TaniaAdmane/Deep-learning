"""
Script d'entraînement du GAN pour restauration d'images

Différences vs train.py (VAE/UNet/SRCNN) :
    - Deux optimiseurs : opt_G et opt_D
    - Deux backward séparés par batch
    - Détachement du graphe pour la mise à jour du discriminateur
    - Warm-up optionnel : pre-train G en L1 pur avant d'activer D

Usage :
    python train_gan.py \
        --train_clean    s3://taniaadmane/dossier/train/clean \
        --train_degraded s3://taniaadmane/dossier/train/clean \
        --val_clean      s3://taniaadmane/dossier/val/clean \
        --val_degraded   s3://taniaadmane/dossier/val/clean
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm
import lpips

from dataset import create_dataloaders
from metrics import MetricsCalculator, MetricsTracker
from models.gan_restoration import (
    Generator, PatchDiscriminator,
    GANLoss, generator_loss, discriminator_loss,
)


# ── Trainer GAN ──────────────────────────────────────────────────────────────

class GANTrainer:
    """
    Entraîneur GAN conditionnel (pix2pix style) pour restauration d'images.

    Stratégie d'entraînement par batch :
        1. Forward G : recon = G(dégradée)
        2. Update D  : max log D(x,y) + log(1 - D(x, G(x)))
        3. Update G  : min log(1 - D(x, G(x))) + λ_L1 * ||y - G(x)||
    """

    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        val_loader,
        device='cuda',
        lr_G=2e-4,
        lr_D=2e-4,
        lambda_l1=100.0,
        lambda_perceptual=0.1,
        gan_mode='lsgan',
        warmup_epochs=5,
        checkpoint_dir='checkpoints/gan',
        log_dir='logs/gan',
    ):
        self.G   = generator.to(device)
        self.D   = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.lambda_l1          = lambda_l1
        self.lambda_perceptual  = lambda_perceptual
        self.warmup_epochs      = warmup_epochs   # epochs en L1 pur avant d'activer D

        # Optimiseurs — Adam avec β1=0.5 (recommandé GAN)
        self.opt_G = optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

        # Schedulers
        self.sched_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_G, mode='min', factor=0.5, patience=5)
        self.sched_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_D, mode='min', factor=0.5, patience=5)

        # Mixed precision
        self.scaler_G = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
        self.scaler_D = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

        # Losses
        self.gan_loss_fn = GANLoss(mode=gan_mode).to(device)
        self.lpips_fn    = lpips.LPIPS(net='alex').to(device)

        # Métriques
        self.metrics_calculator = MetricsCalculator(device=device)

        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # État
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.best_val_loss = float('inf')

        n_G = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
        n_D = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"\nGAN Trainer initialisé")
        print(f"  Device        : {device}")
        print(f"  Params G      : {n_G:,}")
        print(f"  Params D      : {n_D:,}")
        print(f"  lr_G / lr_D   : {lr_G} / {lr_D}")
        print(f"  lambda_L1     : {lambda_l1}")
        print(f"  Perceptual    : {lambda_perceptual}")
        print(f"  GAN mode      : {gan_mode}")
        print(f"  Warmup epochs : {warmup_epochs}")

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def _adv_active(self):
        """True si on est passé la phase de warm-up."""
        return self.current_epoch >= self.warmup_epochs

    def _set_requires_grad(self, net, flag):
        for p in net.parameters():
            p.requires_grad = flag

    # ── train epoch ──────────────────────────────────────────────────

    def train_epoch(self):
        self.G.train()
        self.D.train()

        stats = {k: 0. for k in
                 ['G_total', 'G_adv', 'G_l1', 'G_perc', 'D_total', 'D_real', 'D_fake']}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]")

        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = degraded.to(self.device)
            clean    = clean.to(self.device)

            # ── 1. Forward G ────────────────────────────────────────
            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                recon, _, _ = self.G(degraded)

            # ── 2. Update D ─────────────────────────────────────────
            if self._adv_active:
                self._set_requires_grad(self.D, True)
                self.opt_D.zero_grad()

                with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                    real_pred = self.D(degraded, clean)
                    fake_pred = self.D(degraded, recon.detach())   # detach !
                    loss_D, d_dict = discriminator_loss(
                        real_pred, fake_pred, self.gan_loss_fn)

                self.scaler_D.scale(loss_D).backward()
                self.scaler_D.unscale_(self.opt_D)
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                self.scaler_D.step(self.opt_D)
                self.scaler_D.update()

                stats['D_total'] += d_dict['total']
                stats['D_real']  += d_dict['loss_real']
                stats['D_fake']  += d_dict['loss_fake']

            # ── 3. Update G ─────────────────────────────────────────
            self._set_requires_grad(self.D, False)   # économie mémoire
            self.opt_G.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                # Recalculer fake_pred avec graphe complet (pas de detach)
                fake_pred_for_G = (self.D(degraded, recon)
                                   if self._adv_active else None)

                loss_G, g_dict = generator_loss(
                    fake_pred      = fake_pred_for_G,
                    recon          = recon,
                    target         = clean,
                    lambda_l1      = self.lambda_l1,
                    lambda_perceptual = self.lambda_perceptual,
                    lpips_fn       = self.lpips_fn,
                    gan_loss_fn    = self.gan_loss_fn if self._adv_active else None,
                )

            self.scaler_G.scale(loss_G).backward()
            self.scaler_G.unscale_(self.opt_G)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()

            stats['G_total'] += g_dict['total']
            stats['G_adv']   += g_dict['adversarial']
            stats['G_l1']    += g_dict['l1']
            stats['G_perc']  += g_dict['perceptual']

            # Progress bar
            pbar.set_postfix({
                'G': f"{g_dict['total']:.3f}",
                'D': f"{d_dict['total']:.3f}" if self._adv_active else 'warmup',
                'L1': f"{g_dict['l1']:.4f}",
            })

            # TensorBoard batch
            step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/G_loss_batch', g_dict['total'],       step)
                self.writer.add_scalar('Train/G_l1_batch',   g_dict['l1'],          step)
                self.writer.add_scalar('Train/G_adv_batch',  g_dict['adversarial'], step)
                if self._adv_active:
                    self.writer.add_scalar('Train/D_loss_batch', d_dict['total'],   step)

        n = len(self.train_loader)
        return {k: v / n for k, v in stats.items()}

    # ── validate ─────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self):
        self.G.eval()

        val_loss = 0.
        metrics_tracker = MetricsTracker()

        for degraded, clean, _ in tqdm(self.val_loader, desc="Validation"):
            degraded = degraded.to(self.device)
            clean    = clean.to(self.device)

            recon, _, _ = self.G(degraded)

            # Loss G uniquement (pas D en val)
            loss_G, g_dict = generator_loss(
                fake_pred         = None,
                recon             = recon,
                target            = clean,
                lambda_l1         = self.lambda_l1,
                lambda_perceptual = self.lambda_perceptual,
                lpips_fn          = self.lpips_fn,
                gan_loss_fn       = None,
            )
            val_loss += g_dict['total']

            metrics = self.metrics_calculator.calculate_all_metrics(recon, clean)
            metrics_tracker.update(metrics)

        return val_loss / len(self.val_loader), metrics_tracker.get_average()

    # ── checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, is_best=False, filename='checkpoint.pth'):
        ckpt = {
            'epoch':            self.current_epoch,
            'G_state_dict':     self.G.state_dict(),
            'D_state_dict':     self.D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'best_val_psnr':    self.best_val_psnr,
            'best_val_loss':    self.best_val_loss,
            'lambda_l1':        self.lambda_l1,
            'lambda_perceptual':self.lambda_perceptual,
        }
        torch.save(ckpt, self.checkpoint_dir / filename)
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
            print(f"  ✅ Best GAN saved (epoch {self.current_epoch})")

    def load_checkpoint(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        self.G.load_state_dict(ckpt['G_state_dict'])
        self.D.load_state_dict(ckpt['D_state_dict'])
        self.opt_G.load_state_dict(ckpt['opt_G_state_dict'])
        self.opt_D.load_state_dict(ckpt['opt_D_state_dict'])
        self.current_epoch = ckpt['epoch']
        self.best_val_psnr = ckpt['best_val_psnr']
        self.best_val_loss = ckpt['best_val_loss']
        print(f"Checkpoint GAN chargé — epoch {self.current_epoch}")

    # ── log images ───────────────────────────────────────────────────

    def log_images(self, num_images=4):
        self.G.eval()
        with torch.no_grad():
            degraded, clean, _ = next(iter(self.val_loader))
            degraded = degraded[:num_images].to(self.device)
            clean    = clean[:num_images].to(self.device)
            recon, _, _ = self.G(degraded)

            to_vis = lambda t: (t.clamp(-1, 1) + 1) / 2
            self.writer.add_images('Val/Degraded',      to_vis(degraded), self.current_epoch)
            self.writer.add_images('Val/Reconstructed', to_vis(recon),    self.current_epoch)
            self.writer.add_images('Val/Clean',         to_vis(clean),    self.current_epoch)

    # ── boucle principale ────────────────────────────────────────────

    def train(self, num_epochs, log_images_every=5):
        print(f"\n{'='*60}")
        print(f"  GAN — {num_epochs} epochs  "
              f"(warmup L1 pur : {self.warmup_epochs} epochs)")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            if epoch == self.warmup_epochs:
                print("\n🔥 Warm-up terminé — activation du discriminateur !\n")

            train_stats           = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.sched_G.step(val_loss)
            if self._adv_active:
                self.sched_D.step(train_stats['D_total'])

            elapsed = time.time() - t0
            mode    = "GAN" if self._adv_active else "WARMUP (L1)"

            print(f"\nEpoch {epoch}/{num_epochs-1}  [{mode}]  ({elapsed:.1f}s)")
            print(f"  G  — total: {train_stats['G_total']:.4f}  "
                  f"l1: {train_stats['G_l1']:.4f}  "
                  f"adv: {train_stats['G_adv']:.4f}  "
                  f"perc: {train_stats['G_perc']:.4f}")
            if self._adv_active:
                print(f"  D  — total: {train_stats['D_total']:.4f}  "
                      f"real: {train_stats['D_real']:.4f}  "
                      f"fake: {train_stats['D_fake']:.4f}")
            print(f"  Val — loss: {val_loss:.4f}  "
                  f"PSNR: {val_metrics['psnr']:.2f} dB  "
                  f"SSIM: {val_metrics['ssim']:.4f}  "
                  f"LPIPS: {val_metrics['lpips']:.4f}")

            # TensorBoard epoch
            self.writer.add_scalar('Train/G_total',   train_stats['G_total'],   epoch)
            self.writer.add_scalar('Train/G_l1',      train_stats['G_l1'],      epoch)
            self.writer.add_scalar('Train/G_adv',     train_stats['G_adv'],     epoch)
            self.writer.add_scalar('Train/G_perc',    train_stats['G_perc'],    epoch)
            self.writer.add_scalar('Train/D_total',   train_stats['D_total'],   epoch)
            self.writer.add_scalar('Val/Loss',        val_loss,                 epoch)
            self.writer.add_scalar('Val/PSNR',        val_metrics['psnr'],      epoch)
            self.writer.add_scalar('Val/SSIM',        val_metrics['ssim'],      epoch)
            self.writer.add_scalar('Val/LPIPS',       val_metrics['lpips'],     epoch)
            self.writer.add_scalar('LR/G', self.opt_G.param_groups[0]['lr'],    epoch)
            self.writer.add_scalar('LR/D', self.opt_D.param_groups[0]['lr'],    epoch)

            if epoch % log_images_every == 0:
                self.log_images()

            is_best = val_metrics['psnr'] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics['psnr']
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

            if epoch % 10 == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch{epoch:04d}.pth')

        print(f"\n{'='*60}")
        print(f"  Entraînement GAN terminé !")
        print(f"  Best PSNR : {self.best_val_psnr:.2f} dB")
        print(f"{'='*60}\n")
        self.writer.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entraînement GAN — restauration d\'images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group('Données')
    data.add_argument('--train_clean',    type=str, required=True)
    data.add_argument('--train_degraded', type=str, required=True)
    data.add_argument('--val_clean',      type=str, required=True)
    data.add_argument('--val_degraded',   type=str, required=True)

    hp = parser.add_argument_group('Hyperparamètres')
    hp.add_argument('--patch_size',        type=int,   default=128, choices=[128, 256])
    hp.add_argument('--batch_size',        type=int,   default=16)
    hp.add_argument('--num_epochs',        type=int,   default=100)
    hp.add_argument('--lr_G',              type=float, default=2e-4)
    hp.add_argument('--lr_D',              type=float, default=2e-4)
    hp.add_argument('--lambda_l1',         type=float, default=100.0)
    hp.add_argument('--lambda_perceptual', type=float, default=0.1)
    hp.add_argument('--gan_mode',          type=str,   default='lsgan',
                    choices=['lsgan', 'bce'])
    hp.add_argument('--warmup_epochs',     type=int,   default=5,
                    help='Epochs de pré-entraînement G en L1 pur avant activation D')
    hp.add_argument('--noise_sigma',       type=int,   default=25)

    infra = parser.add_argument_group('Infrastructure')
    infra.add_argument('--num_workers',      type=int, default=4)
    infra.add_argument('--checkpoint_dir',   type=str, default='checkpoints/gan')
    infra.add_argument('--log_dir',          type=str, default='logs/gan')
    infra.add_argument('--log_images_every', type=int, default=5)
    infra.add_argument('--resume',           type=str, default=None)

    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # Data
    print("\nCréation des data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_clean_dir=args.train_clean,
        train_degraded_dir=args.train_degraded,
        val_clean_dir=args.val_clean,
        val_degraded_dir=args.val_degraded,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        noise_sigma=args.noise_sigma,
        generate_noise=True,
    )

    # Modèles
    G = Generator(base_channels=64)
    D = PatchDiscriminator(base_channels=64)

    # Trainer
    trainer = GANTrainer(
        generator=G,
        discriminator=D,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_G=args.lr_G,
        lr_D=args.lr_D,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
        gan_mode=args.gan_mode,
        warmup_epochs=args.warmup_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_epochs=args.num_epochs,
        log_images_every=args.log_images_every,
    )


if __name__ == '__main__':
    main()
