"""
Discriminator Model
Day 2: Updated for CLIP embeddings (512-dim)

Changes from Day 1:
- Flexible semantic_dim (works with both GloVe 300 and CLIP 512)
- Spectral normalization throughout
- Projection discriminator design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """
    Conditional Discriminator with Semantic Embeddings
    
    Architecture:
    - Feature extraction from images
    - Semantic projection network
    - Projection discriminator (combines image features + semantic)
    """
    
    def __init__(
        self,
        nc: int = 3,
        ndf: int = 64,
        semantic_dim: int = 512,  # Changed: Now 512 for CLIP (was 300 for GloVe)
        semantic_proj_dim: int = 256
    ):
        """
        Args:
            nc: Number of image channels
            ndf: Base channel multiplier
            semantic_dim: Semantic embedding dimension (512 for CLIP, 300 for GloVe)
            semantic_proj_dim: Projected semantic dimension
        """
        super(Discriminator, self).__init__()
        
        self.ndf = ndf
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim
        
        # Semantic projection network
        self.semantic_proj = nn.Sequential(
            spectral_norm(nn.Linear(semantic_dim, semantic_proj_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(semantic_proj_dim, semantic_proj_dim)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Image feature extraction
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 (keep spatial dimensions)
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Unconditional output (for WGAN)
        self.output = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0))
        
        # Conditional projection (for class conditioning)
        self.embed_output = spectral_norm(nn.Linear(semantic_proj_dim, ndf * 8))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.Linear)) and not isinstance(m, nn.BatchNorm2d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels, semantic_embeddings):
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, nc, 32, 32]
            labels: Class labels [batch_size] (indices)
            semantic_embeddings: All semantic embeddings [num_classes, semantic_dim]
            
        Returns:
            Discriminator score [batch_size] (real vs fake)
        """
        # Extract image features
        h = self.features(x)  # [batch_size, ndf*8, 4, 4]
        
        # Unconditional output
        output = self.output(h).squeeze()  # [batch_size]
        
        # Conditional output (projection)
        sem_features = self.semantic_proj(semantic_embeddings[labels])  # [batch_size, semantic_proj_dim]
        projection = self.embed_output(sem_features).view(-1, self.ndf * 8, 1, 1)  # [batch_size, ndf*8, 1, 1]
        
        # Pool features
        h_pooled = F.adaptive_avg_pool2d(h, 1).view(-1, self.ndf * 8)  # [batch_size, ndf*8]
        
        # Compute conditional score (inner product)
        cond_output = torch.sum(projection.squeeze() * h_pooled, dim=1)  # [batch_size]
        
        # Combine unconditional and conditional scores
        return output + cond_output


def test_discriminator():
    """Test discriminator with CLIP embeddings"""
    print("Testing Discriminator with CLIP embeddings...")
    
    # Configuration
    batch_size = 8
    num_classes = 100
    semantic_dim = 512  # CLIP dimension
    
    # Create discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator = Discriminator(
        nc=3,
        ndf=64,
        semantic_dim=semantic_dim,
        semantic_proj_dim=256
    ).to(device)
    
    # Create dummy inputs
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    semantic_embeddings = torch.randn(num_classes, semantic_dim, device=device)
    
    # Forward pass
    scores = discriminator(images, labels, semantic_embeddings)
    
    print(f"✓ Input images shape: {images.shape}")
    print(f"✓ Input labels shape: {labels.shape}")
    print(f"✓ Semantic embeddings shape: {semantic_embeddings.shape}")
    print(f"✓ Output scores shape: {scores.shape}")
    print(f"✓ Output scores range: [{scores.min().item():.2f}, {scores.max().item():.2f}]")
    print(f"✓ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Check output is valid
    assert scores.shape == (batch_size,), "Wrong output shape!"
    
    print("\n✓ Discriminator test passed!")
    return discriminator


def test_gan_pair():
    """Test Generator + Discriminator together"""
    print("\n" + "="*70)
    print("Testing Generator + Discriminator Pair")
    print("="*70)
    
    from models.generator import Generator
    
    # Configuration
    batch_size = 8
    nz = 128
    num_classes = 100
    semantic_dim = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    G = Generator(nz=nz, semantic_dim=semantic_dim).to(device)
    D = Discriminator(semantic_dim=semantic_dim).to(device)
    
    # Create inputs
    z = torch.randn(batch_size, nz, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    semantic_embeddings = torch.randn(num_classes, semantic_dim, device=device)
    
    # Generate fake images
    fake_images = G(z, labels, semantic_embeddings)
    
    # Discriminate
    fake_scores = D(fake_images, labels, semantic_embeddings)
    
    print(f"✓ Generated images: {fake_images.shape}")
    print(f"✓ Fake scores: {fake_scores.shape}")
    print(f"✓ Mean fake score: {fake_scores.mean().item():.4f}")
    
    # Test with real images
    real_images = torch.randn(batch_size, 3, 32, 32, device=device)
    real_scores = D(real_images, labels, semantic_embeddings)
    
    print(f"✓ Real scores: {real_scores.shape}")
    print(f"✓ Mean real score: {real_scores.mean().item():.4f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    fake_loss = -fake_scores.mean()
    fake_loss.backward()
    
    print("✓ Backward pass successful!")
    
    print("\n✓ GAN pair test passed!")
    print(f"\nTotal parameters:")
    print(f"  Generator: {sum(p.numel() for p in G.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in D.parameters()):,}")
    print(f"  Total: {sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in D.parameters()):,}")


if __name__ == "__main__":
    test_discriminator()
    test_gan_pair()