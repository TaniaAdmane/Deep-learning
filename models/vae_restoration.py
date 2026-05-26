"""
VAE pour Restauration d'Images
Architecture complète avec encoder probabiliste et decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_Restoration(nn.Module):
    """
    Variational Autoencoder pour restauration d'images
    Prend une image dégradée et reconstruit l'image propre
    """
    
    def __init__(
        self, 
        input_channels=3,
        latent_dim=256,
        image_size=128,  # 128 ou 256
        base_channels=64
    ):
        super(VAE_Restoration, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Calculer la taille du feature map après encodage
        # Pour 128x128: 128 -> 64 -> 32 -> 16 -> 8
        # Pour 256x256: 256 -> 128 -> 64 -> 32 -> 16
        self.encoded_size = image_size // 16
        self.encoded_features = base_channels * 8  # 512
        
        # ============== ENCODER ==============
        # Input: (batch, 3, H, W)
        # Output: (batch, 512, H/16, W/16)
        
        self.encoder = nn.ModuleList([
            # Block 1: H -> H/2
            self._make_encoder_block(input_channels, base_channels),
            
            # Block 2: H/2 -> H/4
            self._make_encoder_block(base_channels, base_channels * 2),
            
            # Block 3: H/4 -> H/8
            self._make_encoder_block(base_channels * 2, base_channels * 4),
            
            # Block 4: H/8 -> H/16
            self._make_encoder_block(base_channels * 4, base_channels * 8),
        ])
        
        # Latent space projection
        flatten_size = self.encoded_features * self.encoded_size * self.encoded_size
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(flatten_size, latent_dim)
        
        # ============== DECODER ==============
        # Projection from latent to spatial features
        self.fc_decode = nn.Linear(latent_dim, flatten_size)
        
        # Decoder blocks (mirror of encoder)
        self.decoder = nn.ModuleList([
            # Block 1: H/16 -> H/8
            self._make_decoder_block(base_channels * 8, base_channels * 4),
            
            # Block 2: H/8 -> H/4
            self._make_decoder_block(base_channels * 4, base_channels * 2),
            
            # Block 3: H/4 -> H/2
            self._make_decoder_block(base_channels * 2, base_channels),
            
            # Block 4: H/2 -> H
            self._make_decoder_block(base_channels, base_channels),
        ])
        
        # Final convolution to output image
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, input_channels, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Initialize weights for numerical stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Weight initialization for numerical stability
        Critical for preventing KL divergence explosion
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Special initialization for logvar: start with small variance
        # logvar = -5 → exp(-5) = 0.0067 → very small initial variance
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_logvar.bias, -5.0)
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Encoder block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Downsample"""
        return nn.Sequential(
            # First conv
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsample
            nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Decoder block: Upsample -> Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            # Upsample
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # First conv
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        Args:
            x: (batch, 3, H, W) - degraded images
        Returns:
            mu: (batch, latent_dim) - mean of latent distribution
            logvar: (batch, latent_dim) - log variance of latent distribution
        """
        h = x
        
        # Pass through encoder blocks
        for block in self.encoder:
            h = block(h)
        
        # Flatten
        h = h.view(h.size(0), -1)
        
        # Project to latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # CRITICAL: Clip logvar to prevent numerical overflow
        # exp(10) = 22026, exp(-10) = 0.000045
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        Args:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        Returns:
            z: (batch, latent_dim) - sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to image
        Args:
            z: (batch, latent_dim)
        Returns:
            recon: (batch, 3, H, W) - reconstructed clean image
        """
        # Project to spatial features
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.encoded_features, self.encoded_size, self.encoded_size)
        
        # Pass through decoder blocks
        for block in self.decoder:
            h = block(h)
        
        # Final convolution
        recon = self.final_conv(h)
        
        return recon
    
    def forward(self, x):
        """
        Forward pass through VAE
        Args:
            x: (batch, 3, H, W) - degraded images
        Returns:
            recon: (batch, 3, H, W) - reconstructed clean images
            mu: (batch, latent_dim) - latent mean
            logvar: (batch, latent_dim) - latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return recon, mu, logvar
    
    def sample(self, x, num_samples=10):
        """
        Generate multiple reconstructions by sampling from latent distribution
        Useful for analyzing uncertainty
        
        Args:
            x: (batch, 3, H, W) - degraded images
            num_samples: number of samples to generate
        Returns:
            samples: (num_samples, batch, 3, H, W)
        """
        mu, logvar = self.encode(x)
        
        samples = []
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            samples.append(recon)
        
        return torch.stack(samples)


def vae_loss(recon, target, mu, logvar, beta=1.0, perceptual_weight=0.0, lpips_fn=None):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence + Perceptual Loss
    
    Args:
        recon: (batch, 3, H, W) - reconstructed images
        target: (batch, 3, H, W) - target clean images
        mu: (batch, latent_dim) - latent mean
        logvar: (batch, latent_dim) - latent log variance
        beta: weight for KL divergence (β-VAE)
        perceptual_weight: weight for perceptual loss
        lpips_fn: LPIPS loss function (optional)
    
    Returns:
        total_loss: scalar
        loss_dict: dictionary with individual losses
    """
    # 1. Reconstruction Loss (MSE) - moyenne sur tout
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    
    # 2. KL Divergence - moyenne sur batch et latent dim
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Perceptual Loss (LPIPS) - optionnel
    perceptual_loss = torch.tensor(0.0, device=recon.device)
    if perceptual_weight > 0 and lpips_fn is not None:
        perceptual_loss = lpips_fn(recon, target).mean()
    
    # Total Loss
    total_loss = recon_loss + beta * kl_loss + perceptual_weight * perceptual_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'reconstruction': recon_loss.item(),
        'kl': kl_loss.item(),
        'perceptual': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0
    }
    
    return total_loss, loss_dict


# Test rapide
if __name__ == "__main__":
    # Test avec images 128x128
    model_128 = VAE_Restoration(image_size=128, latent_dim=256)
    x_128 = torch.randn(4, 3, 128, 128)
    recon_128, mu_128, logvar_128 = model_128(x_128)
    print(f"Input 128: {x_128.shape}")
    print(f"Recon 128: {recon_128.shape}")
    print(f"Latent mu: {mu_128.shape}")
    
    # Test avec images 256x256
    model_256 = VAE_Restoration(image_size=256, latent_dim=256)
    x_256 = torch.randn(4, 3, 256, 256)
    recon_256, mu_256, logvar_256 = model_256(x_256)
    print(f"\nInput 256: {x_256.shape}")
    print(f"Recon 256: {recon_256.shape}")
    
    # Test sampling
    samples = model_128.sample(x_128, num_samples=5)
    print(f"\nSamples: {samples.shape}")  # (5, 4, 3, 128, 128)
    
    print("\n✅ VAE Architecture OK!")