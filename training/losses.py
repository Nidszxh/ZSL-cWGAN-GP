"""
Loss Functions
Day 2: WGAN-GP losses
"""

import torch
import torch.nn as nn


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor,
    semantic_embeddings: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Compute WGAN-GP gradient penalty
    
    Args:
        discriminator: Discriminator model
        real_images: Real images [batch_size, nc, h, w]
        fake_images: Fake images [batch_size, nc, h, w]
        labels: Class labels [batch_size]
        semantic_embeddings: Semantic embeddings [num_classes, semantic_dim]
        device: Device to compute on
        lambda_gp: Gradient penalty coefficient
        
    Returns:
        gradient_penalty: Scalar gradient penalty
    """
    batch_size = real_images.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Get discriminator output on interpolated images
    d_interpolates = discriminator(interpolates, labels, semantic_embeddings)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty * lambda_gp


def wasserstein_loss_discriminator(
    real_output: torch.Tensor,
    fake_output: torch.Tensor
) -> tuple:
    """
    Compute Wasserstein loss for discriminator
    
    Args:
        real_output: Discriminator output on real images
        fake_output: Discriminator output on fake images
        
    Returns:
        d_loss: Discriminator loss
        wasserstein_distance: Estimated Wasserstein distance
    """
    wasserstein_distance = fake_output.mean() - real_output.mean()
    d_loss = wasserstein_distance
    
    return d_loss, wasserstein_distance


def wasserstein_loss_generator(
    fake_output: torch.Tensor
) -> torch.Tensor:
    """
    Compute Wasserstein loss for generator
    
    Args:
        fake_output: Discriminator output on fake images
        
    Returns:
        g_loss: Generator loss
    """
    g_loss = -fake_output.mean()
    return g_loss


class WGANGPLoss:
    """Unified WGAN-GP Loss Calculator"""
    
    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp
    
    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        device: torch.device
    ) -> dict:
        """
        Compute full discriminator loss with gradient penalty
        
        Returns:
            Dictionary with:
                - d_loss: Total discriminator loss
                - wasserstein_distance: W-distance estimate
                - gradient_penalty: GP value
        """
        # Get discriminator outputs
        real_output = discriminator(real_images, labels, semantic_embeddings)
        fake_output = discriminator(fake_images.detach(), labels, semantic_embeddings)
        
        # Wasserstein loss
        d_loss_wgan, w_dist = wasserstein_loss_discriminator(real_output, fake_output)
        
        # Gradient penalty
        gp = compute_gradient_penalty(
            discriminator, real_images, fake_images, labels,
            semantic_embeddings, device, self.lambda_gp
        )
        
        # Total loss
        d_loss = d_loss_wgan + gp
        
        return {
            'd_loss': d_loss,
            'wasserstein_distance': w_dist.item(),
            'gradient_penalty': gp.item() / self.lambda_gp  # Normalized GP
        }
    
    def generator_loss(
        self,
        discriminator: nn.Module,
        fake_images: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor
    ) -> dict:
        """
        Compute generator loss
        
        Returns:
            Dictionary with:
                - g_loss: Generator loss
        """
        # Get discriminator output on fake images
        fake_output = discriminator(fake_images, labels, semantic_embeddings)
        
        # Generator loss
        g_loss = wasserstein_loss_generator(fake_output)
        
        return {
            'g_loss': g_loss
        }


if __name__ == "__main__":
    # Test loss functions
    print("Testing WGAN-GP losses...")
    
    from models.generator import Generator
    from models.discriminator import Discriminator
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    G = Generator(semantic_dim=512).to(device)
    D = Discriminator(semantic_dim=512).to(device)
    
    # Create dummy data
    batch_size = 8
    z = torch.randn(batch_size, 128, device=device)
    labels = torch.randint(0, 80, (batch_size,), device=device)
    semantic_embeddings = torch.randn(80, 512, device=device)
    real_images = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Generate fake images
    fake_images = G(z, labels, semantic_embeddings)
    
    # Initialize loss calculator
    loss_fn = WGANGPLoss(lambda_gp=10.0)
    
    # Test discriminator loss
    d_losses = loss_fn.discriminator_loss(
        D, real_images, fake_images, labels, semantic_embeddings, device
    )
    
    print(f"✓ Discriminator loss: {d_losses['d_loss'].item():.4f}")
    print(f"✓ Wasserstein distance: {d_losses['wasserstein_distance']:.4f}")
    print(f"✓ Gradient penalty: {d_losses['gradient_penalty']:.4f}")
    
    # Test generator loss
    g_losses = loss_fn.generator_loss(
        D, fake_images, labels, semantic_embeddings
    )
    
    print(f"✓ Generator loss: {g_losses['g_loss'].item():.4f}")
    
    # Test backward
    d_losses['d_loss'].backward()
    print("✓ Discriminator backward pass successful")
    
    G.zero_grad()
    D.zero_grad()
    
    fake_images = G(z, labels, semantic_embeddings)
    g_losses = loss_fn.generator_loss(D, fake_images, labels, semantic_embeddings)
    g_losses['g_loss'].backward()
    print("✓ Generator backward pass successful")
    
    print("\n✓ All loss tests passed!")