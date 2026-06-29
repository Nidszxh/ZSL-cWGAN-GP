"""
Conditional Generator with Semantic Projection and Spectral Normalization.
Includes SAGAN-style self-attention at configurable resolutions.
Improvements: spectral norm on semantic projection, optional multi-scale attention.
"""

from typing import Optional

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

    Improvements over baseline:
    - Spectral normalization on semantic projection MLP (stabilizes conditioning)
    - Configurable self-attention at multiple resolutions
    """

    def __init__(
        self,
        nz: int = 128,
        ngf: int = 64,
        nc: int = 3,
        semantic_dim: int = 512,
        semantic_proj_dim: int = 256,
        dropout: float = 0.2,
        use_spectral_norm_semantic: bool = True,
        attention_resolutions: Optional[list] = None,
    ):
        super(Generator, self).__init__()

        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim

        if attention_resolutions is None:
            attention_resolutions = [8]

        self.attention_resolutions = attention_resolutions

        # Semantic projection network with optional spectral norm
        sn = spectral_norm if use_spectral_norm_semantic else lambda x: x
        self.semantic_proj = nn.Sequential(
            sn(nn.Linear(semantic_dim, semantic_proj_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            sn(nn.Linear(semantic_proj_dim, semantic_proj_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        combined_dim = nz + semantic_proj_dim

        # Initial projection from latent to feature map
        self.project = nn.Sequential(
            spectral_norm(nn.Linear(combined_dim, ngf * 8 * 4 * 4)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.Upsample(scale_factor=2, mode="nearest"),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn8 = SelfAttention(ngf * 4) if 8 in attention_resolutions else nn.Identity()

        # 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn16 = SelfAttention(ngf * 2) if 16 in attention_resolutions else nn.Identity()

        # 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            spectral_norm(nn.Conv2d(ngf * 2, nc, 3, 1, 1)),
            nn.Tanh(),
        )

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
        x = self.attn8(x)
        x = self.up2(x)
        x = self.attn16(x)
        x = self.up3(x)
        return x
