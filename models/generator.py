"""
Generator Model
Day 2: Updated for CLIP embeddings (512-dim)

Changes from Day 1:
- Flexible semantic_dim (works with both GloVe 300 and CLIP 512)
- Spectral normalization added
- Better initialization
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """
    Conditional Generator with Semantic Embeddings
    
    Architecture:
    - Noise (nz) + Semantic embedding (semantic_dim) -> Combined input
    - Semantic projection network
    - Main generation network (Conv layers with upsampling)
    """
    
    def __init__(
        self,
        nz: int = 128,
        ngf: int = 64,
        nc: int = 3,
        semantic_dim: int = 512,  # Changed: Now 512 for CLIP (was 300 for GloVe)
        semantic_proj_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Args:
            nz: Latent noise dimension
            ngf: Base channel multiplier
            nc: Number of image channels (3 for RGB)
            semantic_dim: Input semantic embedding dimension (512 for CLIP, 300 for GloVe)
            semantic_proj_dim: Projected semantic dimension
            dropout: Dropout rate for semantic projection
        """
        super(Generator, self).__init__()
        
        self.nz = nz
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim
        
        # Semantic projection network
        # Maps semantic embeddings to a learned space
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(semantic_proj_dim, semantic_proj_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Combined input dimension
        combined_dim = nz + semantic_proj_dim
        
        # Initial projection from latent to feature map
        self.project = nn.Sequential(
            spectral_norm(nn.Linear(combined_dim, ngf * 8 * 4 * 4)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main generation network
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.main = nn.Sequential(
            # 4x4 -> 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            spectral_norm(nn.Conv2d(ngf * 2, nc, 3, 1, 1)),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels, semantic_embeddings):
        """
        Forward pass
        
        Args:
            z: Noise tensor [batch_size, nz]
            labels: Class labels [batch_size] (indices)
            semantic_embeddings: All semantic embeddings [num_classes, semantic_dim]
            
        Returns:
            Generated images [batch_size, nc, 32, 32]
        """
        # Get semantic features for the given labels
        # semantic_embeddings[labels] -> [batch_size, semantic_dim]
        sem_features = self.semantic_proj(semantic_embeddings[labels])
        
        # Concatenate noise and semantic features
        x = torch.cat([z, sem_features], dim=1)  # [batch_size, nz + semantic_proj_dim]
        
        # Project to initial feature map
        x = self.project(x)
        x = x.view(-1, self.ngf * 8, 4, 4)  # Reshape to [batch_size, ngf*8, 4, 4]
        
        # Generate image
        return self.main(x)
    
    @property
    def ngf(self):
        """Get base channel multiplier"""
        return 64  # Default, can be made configurable


def test_generator():
    """Test generator with CLIP embeddings"""
    print("Testing Generator with CLIP embeddings...")
    
    # Configuration
    batch_size = 8
    nz = 128
    num_classes = 100
    semantic_dim = 512  # CLIP dimension
    
    # Create generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(
        nz=nz,
        ngf=64,
        nc=3,
        semantic_dim=semantic_dim,
        semantic_proj_dim=256
    ).to(device)
    
    # Create dummy inputs
    z = torch.randn(batch_size, nz, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    semantic_embeddings = torch.randn(num_classes, semantic_dim, device=device)
    
    # Forward pass
    fake_images = generator(z, labels, semantic_embeddings)
    
    print(f"✓ Input noise shape: {z.shape}")
    print(f"✓ Input labels shape: {labels.shape}")
    print(f"✓ Semantic embeddings shape: {semantic_embeddings.shape}")
    print(f"✓ Output images shape: {fake_images.shape}")
    print(f"✓ Output range: [{fake_images.min().item():.2f}, {fake_images.max().item():.2f}]")
    print(f"✓ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Check output is valid
    assert fake_images.shape == (batch_size, 3, 32, 32), "Wrong output shape!"
    assert fake_images.min() >= -1 and fake_images.max() <= 1, "Output not in [-1, 1]!"
    
    print("\n✓ Generator test passed!")
    return generator


if __name__ == "__main__":
    test_generator()