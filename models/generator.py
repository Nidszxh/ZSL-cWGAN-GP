"""
Conditional Generator with Semantic Projection and Spectral Normalization.
Includes SAGAN-style self-attention at 8x8 resolution.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """SAGAN-style self-attention block."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, C, H, W = x.shape
        q = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, H * W)
        v = self.value(x).view(batch, -1, H * W)

        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        return self.gamma * out + x


class Generator(nn.Module):
    """
    Conditional Generator with Semantic Embeddings and Self-Attention.

    Architecture:
    - Noise (nz) + Semantic embedding (semantic_dim) -> Combined input
    - Semantic projection network
    - Main generation network (Conv layers with upsampling)
    - Self-attention at 8x8 resolution
    """

    def __init__(self, nz: int = 128, ngf: int = 64, nc: int = 3, semantic_dim: int = 512, semantic_proj_dim: int = 256, dropout: float = 0.2):
        super(Generator, self).__init__()

        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim

        # Semantic projection network
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, semantic_proj_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout), nn.Linear(semantic_proj_dim, semantic_proj_dim), nn.LeakyReLU(0.2, inplace=True)
        )

        combined_dim = nz + semantic_proj_dim

        # Initial projection from latent to feature map
        self.project = nn.Sequential(spectral_norm(nn.Linear(combined_dim, ngf * 8 * 4 * 4)), nn.LeakyReLU(0.2, inplace=True))

        # 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2, mode="nearest"),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = SelfAttention(ngf * 4)

        # 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 16x16 -> 32x32
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), spectral_norm(nn.Conv2d(ngf * 2, nc, 3, 1, 1)), nn.Tanh())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, labels, semantic_embeddings):
        sem_features = self.semantic_proj(semantic_embeddings[labels])
        x = torch.cat([z, sem_features], dim=1)
        x = self.project(x)
        x = x.view(-1, self.ngf * 8, 4, 4)
        x = self.up1(x)
        x = self.attn(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


def test_generator():
    """Test generator with CLIP embeddings"""
    print("Testing Generator with CLIP embeddings...")

    # Configuration
    batch_size = 8
    nz = 128
    num_classes = 100
    semantic_dim = 512  # CLIP dimension

    # Create generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(nz=nz, ngf=64, nc=3, semantic_dim=semantic_dim, semantic_proj_dim=256).to(device)

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
