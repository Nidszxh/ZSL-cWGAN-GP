"""
Loss Functions — WGAN-GP losses with feature matching.
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
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    # L1-F1: GP must not backprop into the generator.
    fake_images = fake_images.detach()
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    # L1-F10: GP in explicit fp32 (autocast would zero it under AMP).
    with torch.autocast(device_type="cuda", enabled=False):
        d_interpolates = discriminator(interpolates, labels, semantic_embeddings)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
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

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        device: torch.device,
    ) -> dict:
        real_output = discriminator(real_images, labels, semantic_embeddings)
        fake_output = discriminator(fake_images.detach(), labels, semantic_embeddings)
        d_loss_wgan, w_dist = wasserstein_loss_discriminator(real_output, fake_output)
        gp = compute_gradient_penalty(
            discriminator, real_images, fake_images, labels, semantic_embeddings, device, self.lambda_gp
        )
        d_loss = d_loss_wgan + gp
        return {
            "d_loss": d_loss,
            "wasserstein_distance": w_dist.item(),
            "gradient_penalty": gp.item() / self.lambda_gp,
        }

    def generator_loss(
        self,
        discriminator: nn.Module,
        fake_images: torch.Tensor,
        labels: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        real_images: torch.Tensor = None,
        feature_matching_weight: float = 0.0,
    ) -> dict:
        # L1-F2: one forward supplies both score and features for feature matching.
        fake_output, fake_features = discriminator(fake_images, labels, semantic_embeddings, return_features=True)
        g_loss = wasserstein_loss_generator(fake_output)
        result = {"g_loss": g_loss}

        if feature_matching_weight > 0 and real_images is not None:
            _, real_features = discriminator(real_images, labels, semantic_embeddings, return_features=True)
            fm_loss = nn.functional.mse_loss(fake_features, real_features)
            weighted_fm = feature_matching_weight * fm_loss
            result["feature_matching_loss"] = weighted_fm.item()
            g_loss = g_loss + weighted_fm
            result["g_loss"] = g_loss

        return result
