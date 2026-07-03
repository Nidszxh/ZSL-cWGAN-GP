"""
Conditional Discriminator with Spectral Normalization and Semantic Projection.
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

    def __init__(self, nc: int = 3, ndf: int = 64, semantic_dim: int = 512, semantic_proj_dim: int = 256):
        super(Discriminator, self).__init__()

        self.ndf = ndf
        self.semantic_dim = semantic_dim
        self.semantic_proj_dim = semantic_proj_dim

        # Semantic projection network
        self.semantic_proj = nn.Sequential(
            spectral_norm(nn.Linear(semantic_dim, semantic_proj_dim, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(semantic_proj_dim, semantic_proj_dim, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Image feature extraction
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        # L1-F18: bias-free critic — learnable biases let D inflate an unbounded
        # offset (observed +1.7M) that WGAN-GP cannot constrain; this pins D(0)=0.
        self.features = nn.Sequential(
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 (keep spatial dimensions)
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Unconditional output
        self.output = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))

        # Conditional projection
        self.embed_output = spectral_norm(nn.Linear(semantic_proj_dim, ndf * 8, bias=False))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels, semantic_embeddings, return_features=False):
        h = self.features(x)  # [B, ndf*8, 4, 4]

        # Unconditional output
        output = self.output(h).view(-1)

        # Conditional output (projection)
        sem_features = self.semantic_proj(semantic_embeddings[labels])
        projection = self.embed_output(sem_features).view(-1, self.ndf * 8, 1, 1)
        h_pooled = F.adaptive_avg_pool2d(h, 1).view(-1, self.ndf * 8)
        cond_output = torch.sum(projection.view(-1, self.ndf * 8) * h_pooled, dim=1)

        if return_features:
            return output + cond_output, h
        return output + cond_output
