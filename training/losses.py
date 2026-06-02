"""
Loss Functions — WGAN-GP losses
"""

import torch
import torch.nn as nn


def compute_gradient_penalty(
    discriminator: nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor, labels: torch.Tensor, semantic_embeddings: torch.Tensor, device: torch.device, lambda_gp: float = 10.0
) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels, semantic_embeddings)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=torch.ones_like(d_interpolates), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty * lambda_gp


def wasserstein_loss_discriminator(real_output: torch.Tensor, fake_output: torch.Tensor) -> tuple:
    wasserstein_distance = fake_output.mean() - real_output.mean()
    d_loss = wasserstein_distance
    return d_loss, wasserstein_distance


def wasserstein_loss_generator(fake_output: torch.Tensor) -> torch.Tensor:
    g_loss = -fake_output.mean()
    return g_loss


class WGANGPLoss:
    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp

    def discriminator_loss(self, discriminator: nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor, labels: torch.Tensor, semantic_embeddings: torch.Tensor, device: torch.device) -> dict:
        real_output = discriminator(real_images, labels, semantic_embeddings)
        fake_output = discriminator(fake_images.detach(), labels, semantic_embeddings)
        d_loss_wgan, w_dist = wasserstein_loss_discriminator(real_output, fake_output)
        gp = compute_gradient_penalty(discriminator, real_images, fake_images, labels, semantic_embeddings, device, self.lambda_gp)
        d_loss = d_loss_wgan + gp
        return {"d_loss": d_loss, "wasserstein_distance": w_dist.item(), "gradient_penalty": gp.item() / self.lambda_gp}

    def generator_loss(self, discriminator: nn.Module, fake_images: torch.Tensor, labels: torch.Tensor, semantic_embeddings: torch.Tensor) -> dict:
        fake_output = discriminator(fake_images, labels, semantic_embeddings)
        g_loss = wasserstein_loss_generator(fake_output)
        return {"g_loss": g_loss}


if __name__ == "__main__":
    print("Testing WGAN-GP losses...")
    from models.generator import Generator
    from models.discriminator import Discriminator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(semantic_dim=512).to(device)
    D = Discriminator(semantic_dim=512).to(device)
    batch_size = 8
    z = torch.randn(batch_size, 128, device=device)
    labels = torch.randint(0, 80, (batch_size,), device=device)
    semantic_embeddings = torch.randn(80, 512, device=device)
    real_images = torch.randn(batch_size, 3, 32, 32, device=device)
    fake_images = G(z, labels, semantic_embeddings)
    loss_fn = WGANGPLoss(lambda_gp=10.0)
    d_losses = loss_fn.discriminator_loss(D, real_images, fake_images, labels, semantic_embeddings, device)
    print(f"D loss: {d_losses['d_loss'].item():.4f}, W-dist: {d_losses['wasserstein_distance']:.4f}, GP: {d_losses['gradient_penalty']:.4f}")
    g_losses = loss_fn.generator_loss(D, fake_images, labels, semantic_embeddings)
    print(f"G loss: {g_losses['g_loss'].item():.4f}")
    d_losses["d_loss"].backward()
    G.zero_grad()
    D.zero_grad()
    fake_images = G(z, labels, semantic_embeddings)
    g_losses = loss_fn.generator_loss(D, fake_images, labels, semantic_embeddings)
    g_losses["g_loss"].backward()
    print("All loss tests passed!")
