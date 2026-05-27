import torch
import torch.nn.functional as F
import numpy as np
from math import log10
import lpips


class MetricsCalculator:
    """
    Calculateur de métriques pour évaluation de restauration d'images
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialiser LPIPS (Learned Perceptual Image Patch Similarity)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        
        print(f"Metrics calculator initialized on {device}")
    
    def denormalize(self, tensor):
        return (tensor + 1.0) / 2.0
    
    def calculate_psnr(self, img1, img2):
        # Dénormaliser
        img1 = self.denormalize(img1)
        img2 = self.denormalize(img2)
        
        # MSE
        mse = F.mse_loss(img1, img2, reduction='none')
        mse = mse.mean(dim=[1, 2, 3])  # Moyenne par image
        
        # PSNR = 10 * log10(1 / MSE)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        
        return psnr.mean().item()
    
    def calculate_ssim(self, img1, img2, window_size=11):
        # Dénormaliser
        img1 = self.denormalize(img1)
        img2 = self.denormalize(img2)
        
        # Convertir en grayscale pour SSIM
        # Formule standard: 0.299*R + 0.587*G + 0.114*B
        def rgb_to_gray(img):
            weights = torch.tensor([0.299, 0.587, 0.114]).to(img.device)
            weights = weights.view(1, 3, 1, 1)
            gray = (img * weights).sum(dim=1, keepdim=True)
            return gray
        
        img1_gray = rgb_to_gray(img1)
        img2_gray = rgb_to_gray(img2)
        
        # Créer fenêtre gaussienne
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.outer(g).unsqueeze(0).unsqueeze(0)
        
        window = gaussian_window(window_size).to(img1.device)
        
        # Constantes pour stabilité
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Moyennes
        mu1 = F.conv2d(img1_gray, window, padding=window_size // 2)
        mu2 = F.conv2d(img2_gray, window, padding=window_size // 2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Variances et covariance
        sigma1_sq = F.conv2d(img1_gray ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2_gray ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1_gray * img2_gray, window, padding=window_size // 2) - mu1_mu2
        
        # SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def calculate_lpips(self, img1, img2):
        with torch.no_grad():
            # LPIPS attend des images dans [-1, 1]
            lpips_value = self.lpips_fn(img1, img2)
        
        return lpips_value.mean().item()
    
    def calculate_all_metrics(self, predictions, targets):
        with torch.no_grad():
            metrics = {
                'psnr': self.calculate_psnr(predictions, targets),
                'ssim': self.calculate_ssim(predictions, targets),
                'lpips': self.calculate_lpips(predictions, targets)
            }
        
        return metrics
    
    def batch_metrics(self, predictions, targets):
        batch_size = predictions.size(0)
        
        psnr_list = []
        ssim_list = []
        lpips_list = []
        
        for i in range(batch_size):
            pred = predictions[i:i+1]
            target = targets[i:i+1]
            
            psnr_list.append(self.calculate_psnr(pred, target))
            ssim_list.append(self.calculate_ssim(pred, target))
            lpips_list.append(self.calculate_lpips(pred, target))
        
        return {
            'psnr_per_image': psnr_list,
            'ssim_per_image': ssim_list,
            'lpips_per_image': lpips_list,
            'psnr_mean': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim_mean': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list),
            'lpips_mean': np.mean(lpips_list),
            'lpips_std': np.std(lpips_list),
        }


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise les compteurs"""
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.lpips_sum = 0.0
        self.count = 0
    
    def update(self, metrics):
        self.psnr_sum += metrics['psnr']
        self.ssim_sum += metrics['ssim']
        self.lpips_sum += metrics['lpips']
        self.count += 1
    
    def get_average(self):
        """Retourne les moyennes"""
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count,
            'lpips': self.lpips_sum / self.count
        }
    
    def get_summary_string(self):
        """Retourne un string formaté des métriques"""
        avg = self.get_average()
        return (f"PSNR: {avg['psnr']:.2f} dB | "
                f"SSIM: {avg['ssim']:.4f} | "
                f"LPIPS: {avg['lpips']:.4f}")


# Test
if __name__ == "__main__":
    print("Testing metrics...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = MetricsCalculator(device=device)
    
    # Images de test
    img1 = torch.randn(4, 3, 128, 128).to(device)  # [-1, 1]
    img2 = img1 + torch.randn_like(img1) * 0.1  # Ajouter un peu de bruit
    
    # Test métriques individuelles
    print("\nTesting individual metrics:")
    psnr = calculator.calculate_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    ssim = calculator.calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    lpips_score = calculator.calculate_lpips(img1, img2)
    print(f"LPIPS: {lpips_score:.4f}")
    
    # Test toutes les métriques
    print("\nTesting all metrics:")
    metrics = calculator.calculate_all_metrics(img1, img2)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Test batch metrics
    print("\nTesting batch metrics:")
    batch_metrics = calculator.batch_metrics(img1, img2)
    print(f"PSNR mean: {batch_metrics['psnr_mean']:.2f} ± {batch_metrics['psnr_std']:.2f}")
    print(f"SSIM mean: {batch_metrics['ssim_mean']:.4f} ± {batch_metrics['ssim_std']:.4f}")
    print(f"LPIPS mean: {batch_metrics['lpips_mean']:.4f} ± {batch_metrics['lpips_std']:.4f}")
    
    # Test tracker
    print("\nTesting metrics tracker:")
    tracker = MetricsTracker()
    for _ in range(5):
        metrics = calculator.calculate_all_metrics(img1, img2)
        tracker.update(metrics)
    
    print(tracker.get_summary_string())
    
    print("\n✅ Metrics OK!")
