"""
Script d'entraînement unifié pour la restauration d'images
Supporte : VAE, VAE-UNet, U-Net, Autoencoder, SRCNN

Usage :
    python train.py --model vae       --train_clean ... --train_degraded ...
    python train.py --model vae_unet  --train_clean ... --train_degraded ...
    python train.py --model unet      --train_clean ... --train_degraded ...
    python train.py --model autoencoder --train_clean ... --train_degraded ...
    python train.py --model srcnn     --train_clean ... --train_degraded ...
    python train.py --model srcnn_lite --train_clean ... --train_degraded ...
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from pathlib import Path
import argparse
from tqdm import tqdm

from dataset import create_dataloaders
from metrics import MetricsCalculator, MetricsTracker
import lpips

# ── imports modèles ──────────────────────────────────────────────────────────

from models.vae_restoration  import VAE_Restoration,       vae_loss
from models.vae_unet         import VAE_UNet,               vae_loss         as vae_unet_loss
from models.unet_restoration import UNet_Restoration,       unet_loss
from models.autoencoder_restoration import Autoencoder_Restoration, autoencoder_loss
from models.srcnn_restoration import SRCNN_Restoration, SRCNN_Lite, srcnn_loss


# ── registre des modèles ─────────────────────────────────────────────────────

def build_model(args):
    """
    Instancie le modèle et la fonction de loss selon args.model.

    Retourne :
        model    : nn.Module
        loss_fn  : callable(recon, target, mu, logvar, **kwargs) → (loss, dict)
        model_kw : dict des kwargs supplémentaires pour loss_fn
    """
    name = args.model.lower()

    if name == 'vae':
        model = VAE_Restoration(
            input_channels=3,
            latent_dim=args.latent_dim,
            image_size=args.patch_size,
        )
        loss_fn = vae_loss
        model_kw = {'beta': args.beta}

    elif name == 'vae_unet':
        model = VAE_UNet(
            input_channels=3,
            latent_dim=args.latent_dim,
            image_size=args.patch_size,
        )
        loss_fn = vae_unet_loss
        model_kw = {'beta': args.beta}

    elif name == 'unet':
        model = UNet_Restoration(base_channels=64)
        loss_fn = unet_loss
        model_kw = {'beta': 0.0}

    elif name == 'autoencoder':
        model = Autoencoder_Restoration(
            input_channels=3,
            latent_dim=args.latent_dim,
            image_size=args.patch_size,
        )
        loss_fn = autoencoder_loss
        model_kw = {'beta': 0.0, 'latent_reg': args.latent_reg}

    elif name == 'srcnn':
        model = SRCNN_Restoration(
            feature_channels=args.srcnn_channels,
            num_residual_blocks=args.srcnn_blocks,
        )
        loss_fn = srcnn_loss
        model_kw = {'beta': 0.0}

    elif name == 'srcnn_lite':
        model = SRCNN_Lite()
        loss_fn = srcnn_loss
        model_kw = {'beta': 0.0}

    else:
        raise ValueError(
            f"Modèle inconnu : '{name}'. "
            "Choix : vae | vae_unet | unet | autoencoder | srcnn | srcnn_lite"
        )

    return model, loss_fn, model_kw


# ── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Entraîneur générique compatible avec tous les modèles de restauration.

    Tous les modèles exposent la même interface :
        forward(x) → (recon, mu_or_z_or_None, logvar_or_None)
    La loss_fn reçoit ces valeurs + les kwargs spécifiques au modèle.
    """

    def __init__(
        self,
        model,
        loss_fn,
        loss_kw,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=1e-4,
        perceptual_weight=0.1,
        checkpoint_dir='checkpoints',
        log_dir='logs',
        model_name='model',
    ):
        self.model        = model.to(device)
        self.loss_fn      = loss_fn
        self.loss_kw      = loss_kw          # ex: {'beta': 1.0} ou {'latent_reg': 1e-4}
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.perceptual_weight = perceptual_weight
        self.model_name   = model_name

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

        # Métriques
        self.metrics_calculator = MetricsCalculator(device=device)

        # Logging / checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # État
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTrainer initialisé — modèle : {model_name}")
        print(f"  Device          : {device}")
        print(f"  Paramètres      : {n_params:,}")
        print(f"  Learning rate   : {learning_rate}")
        print(f"  Perceptual      : {perceptual_weight}")
        for k, v in loss_kw.items():
            print(f"  {k:16s}: {v}")

    # ── helpers ──────────────────────────────────────────────────────

    def _compute_loss(self, recon, clean, mu, logvar):
        """Appelle la bonne loss_fn avec les kwargs du modèle courant."""
        return self.loss_fn(
            recon, clean, mu, logvar,
            perceptual_weight=self.perceptual_weight,
            lpips_fn=self.lpips_fn,
            **self.loss_kw,
        )

    # ── train epoch ──────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()

        totals = {'loss': 0., 'recon': 0., 'kl': 0., 'perceptual': 0.}
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]")

        for batch_idx, (degraded, clean, _) in enumerate(pbar):
            degraded = degraded.to(self.device)
            clean    = clean.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device == 'cuda')):
                recon, mu, logvar = self.model(degraded)
                loss, loss_dict   = self._compute_loss(recon, clean, mu, logvar)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += loss_dict.get(
                    k if k != 'recon' else 'reconstruction', 0.
                )

            pbar.set_postfix({
                'loss':  f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'kl':    f"{loss_dict['kl']:.4f}",
            })

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Train/Loss_batch',  loss_dict['total'],          global_step)
                self.writer.add_scalar('Train/Recon_batch', loss_dict['reconstruction'], global_step)
                self.writer.add_scalar('Train/KL_batch',    loss_dict['kl'],             global_step)

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    # ── validate ─────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        val_loss    = 0.
        val_kl_loss = 0.
        metrics_tracker = MetricsTracker()

        for degraded, clean, _ in tqdm(self.val_loader, desc="Validation"):
            degraded = degraded.to(self.device)
            clean    = clean.to(self.device)

            recon, mu, logvar  = self.model(degraded)
            loss, loss_dict    = self._compute_loss(recon, clean, mu, logvar)

            val_loss    += loss_dict['total']
            val_kl_loss += loss_dict['kl']

            metrics = self.metrics_calculator.calculate_all_metrics(recon, clean)
            metrics_tracker.update(metrics)

        n = len(self.val_loader)
        avg_metrics       = metrics_tracker.get_average()
        avg_metrics['kl'] = val_kl_loss / n

        return val_loss / n, avg_metrics

    # ── checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, is_best=False, filename='checkpoint.pth'):
        checkpoint = {
            'epoch':                self.current_epoch,
            'model_name':           self.model_name,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'best_val_psnr':        self.best_val_psnr,
            'loss_kw':              self.loss_kw,
            'perceptual_weight':    self.perceptual_weight,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"  ✅ Best model saved (epoch {self.current_epoch})")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_psnr = checkpoint['best_val_psnr']
        print(f"Checkpoint chargé — epoch {self.current_epoch}")

    # ── log images ───────────────────────────────────────────────────

    def log_images(self, num_images=4):
        self.model.eval()
        with torch.no_grad():
            degraded, clean, _ = next(iter(self.val_loader))
            degraded = degraded[:num_images].to(self.device)
            clean    = clean[:num_images].to(self.device)
            recon, _, _ = self.model(degraded)

            to_vis = lambda t: (t.clamp(-1, 1) + 1) / 2
            self.writer.add_images('Val/Degraded',      to_vis(degraded), self.current_epoch)
            self.writer.add_images('Val/Reconstructed', to_vis(recon),    self.current_epoch)
            self.writer.add_images('Val/Clean',         to_vis(clean),    self.current_epoch)

    # ── boucle principale ────────────────────────────────────────────

    def train(self, num_epochs, log_images_every=5):
        print(f"\n{'='*60}")
        print(f"  Démarrage : {num_epochs} epochs  |  modèle : {self.model_name}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_stats          = self.train_epoch()
            val_loss, val_metrics = self.validate()
            self.scheduler.step(val_loss)

            elapsed = time.time() - t0

            print(f"\nEpoch {epoch}/{num_epochs - 1}  ({elapsed:.1f}s)")
            print(f"  Train  — loss: {train_stats['loss']:.4f}  "
                  f"recon: {train_stats['recon']:.4f}  "
                  f"kl: {train_stats['kl']:.4f}")
            print(f"  Val    — loss: {val_loss:.4f}  "
                  f"PSNR: {val_metrics['psnr']:.2f} dB  "
                  f"SSIM: {val_metrics['ssim']:.4f}  "
                  f"LPIPS: {val_metrics['lpips']:.4f}")

            # TensorBoard
            self.writer.add_scalar('Train/Loss_epoch',  train_stats['loss'],  epoch)
            self.writer.add_scalar('Train/Recon_epoch', train_stats['recon'], epoch)
            self.writer.add_scalar('Train/KL_epoch',    train_stats['kl'],    epoch)
            self.writer.add_scalar('Val/Loss',          val_loss,             epoch)
            self.writer.add_scalar('Val/PSNR',          val_metrics['psnr'],  epoch)
            self.writer.add_scalar('Val/SSIM',          val_metrics['ssim'],  epoch)
            self.writer.add_scalar('Val/LPIPS',         val_metrics['lpips'], epoch)
            self.writer.add_scalar('Train/LR',
                                   self.optimizer.param_groups[0]['lr'], epoch)

            if epoch % log_images_every == 0:
                self.log_images()

            # Checkpoint
            is_best = val_metrics['psnr'] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics['psnr']
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True, filename='best_model.pth')

            # Checkpoint périodique toutes les 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(filename=f'checkpoint_epoch{epoch:04d}.pth')

        print(f"\n{'='*60}")
        print(f"  Entraînement terminé !")
        print(f"  Best PSNR : {self.best_val_psnr:.2f} dB")
        print(f"{'='*60}\n")
        self.writer.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entraînement unifié — restauration d\'images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Données
    data = parser.add_argument_group('Données')
    data.add_argument('--train_clean',    type=str, required=True)
    data.add_argument('--train_degraded', type=str, required=True)
    data.add_argument('--val_clean',      type=str, required=True)
    data.add_argument('--val_degraded',   type=str, required=True)

    # ── Modèle
    mdl = parser.add_argument_group('Modèle')
    mdl.add_argument('--model', type=str, default='vae',
                     choices=['vae', 'vae_unet', 'unet',
                              'autoencoder', 'srcnn', 'srcnn_lite'],
                     help='Architecture à entraîner')
    mdl.add_argument('--latent_dim',      type=int,   default=256,
                     help='Dimension latente (VAE / Autoencoder)')
    mdl.add_argument('--srcnn_channels',  type=int,   default=64,
                     help='Canaux internes SRCNN')
    mdl.add_argument('--srcnn_blocks',    type=int,   default=8,
                     help='Nombre de residual blocks SRCNN')

    # ── Hyperparamètres
    hp = parser.add_argument_group('Hyperparamètres')
    hp.add_argument('--patch_size',        type=int,   default=128, choices=[128, 256])
    hp.add_argument('--batch_size',        type=int,   default=16)
    hp.add_argument('--num_epochs',        type=int,   default=50)
    hp.add_argument('--lr',                type=float, default=1e-4)
    hp.add_argument('--beta',              type=float, default=1.0,
                    help='Poids KL (VAE uniquement)')
    hp.add_argument('--perceptual_weight', type=float, default=0.1,
                    help='Poids loss perceptuelle LPIPS')
    hp.add_argument('--latent_reg',        type=float, default=1e-4,
                    help='Régularisation L2 latent (Autoencoder uniquement)')

    # ── Infra
    infra = parser.add_argument_group('Infrastructure')
    infra.add_argument('--num_workers',     type=int, default=4)
    infra.add_argument('--checkpoint_dir',  type=str, default='checkpoints')
    infra.add_argument('--log_dir',         type=str, default='logs')
    infra.add_argument('--log_images_every',type=int, default=5)
    infra.add_argument('--resume',          type=str, default=None,
                       help='Chemin vers un checkpoint pour reprendre')

    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # ── Data loaders
    print("\nCréation des data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_clean_dir=args.train_clean,
        train_degraded_dir=args.train_degraded,
        val_clean_dir=args.val_clean,
        val_degraded_dir=args.val_degraded,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Modèle + loss
    print(f"\nConstruction du modèle : {args.model}")
    model, loss_fn, loss_kw = build_model(args)

    # ── Sous-dossier de log par modèle
    log_dir  = os.path.join(args.log_dir,         args.model)
    ckpt_dir = os.path.join(args.checkpoint_dir,  args.model)

    # ── Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        loss_kw=loss_kw,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        perceptual_weight=args.perceptual_weight,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        model_name=args.model,
    )

    # ── Reprise éventuelle
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Entraînement
    trainer.train(
        num_epochs=args.num_epochs,
        log_images_every=args.log_images_every,
    )


if __name__ == '__main__':
    main()