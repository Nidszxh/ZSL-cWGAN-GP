"""
Quick Training Test - 5-epoch sanity check for the full pipeline.
"""

import warnings

import torch
import torch.optim as optim

warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.utils.embeddings import EmbeddingManager
from src.utils.data_loader import get_class_names, get_class_split, get_data_loaders
from src.training.losses import WGANGPLoss


def quick_train_test(num_epochs=5):
    """Quick training test with CLIP embeddings"""

    print("=" * 70)
    print("QUICK TRAINING TEST")
    print("5-epoch sanity check for the full pipeline")
    print("=" * 70 + "\n")

    config_path = Path("src/configs/config.yaml")
    if not config_path.exists():
        print("Error: config.yaml not found!")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["num_epochs"] = num_epochs
    config["training"]["batch_size"] = 64
    config["training"]["mixed_precision"] = False
    config["training"]["use_ema"] = False

    # Determinism: WGAN at aggressive TTUR is seed-sensitive.
    torch.manual_seed(config["experiment"]["seed"])
    torch.cuda.manual_seed_all(config["experiment"]["seed"])

    device = torch.device(config["experiment"]["device"])
    print(f"Device: {device}\n")

    print("Step 1: Loading data...")
    seen_classes, unseen_classes = get_class_split(
        num_classes=config["dataset"]["num_classes"],
        seen_count=config["dataset"]["seen_classes"],
        cache_dir=config["paths"]["cache_dir"],
        seed=config["experiment"]["seed"],
    )

    train_loader, val_loader, class_info = get_data_loaders(config, seen_classes)
    num_seen_classes = class_info["num_seen_classes"]

    class_names = get_class_names(config["paths"]["data_root"])

    print(f"Data loaded: {len(train_loader)} batches\n")

    print("Step 2: Loading semantic embeddings...")
    embedding_manager = EmbeddingManager(config)

    seen_embeddings, embedding_dim = embedding_manager.get_embeddings(class_names, seen_classes)
    del embedding_manager
    torch.cuda.empty_cache()

    print(f"CLIP embeddings loaded, dim={embedding_dim}\n")

    print("Step 3: Initializing models...")

    gen_cfg = config["model"]["generator"]
    disc_cfg = config["model"]["discriminator"]

    netG = Generator(
        nz=gen_cfg["nz"],
        ngf=gen_cfg["ngf"],
        nc=gen_cfg["nc"],
        semantic_dim=embedding_dim,
        semantic_proj_dim=gen_cfg["semantic_proj_dim"],
        dropout=gen_cfg["dropout"],
        use_spectral_norm_semantic=gen_cfg.get("use_spectral_norm_semantic", True),
        attention_resolutions=gen_cfg.get("attention_resolutions", [8]),
    ).to(device)

    netD = Discriminator(
        nc=disc_cfg["nc"],
        ndf=disc_cfg["ndf"],
        semantic_dim=embedding_dim,
        semantic_proj_dim=disc_cfg["semantic_proj_dim"],
    ).to(device)

    print(f"Generator params: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}\n")

    print("Step 4: Setting up optimization...")

    optimizerD = optim.Adam(
        netD.parameters(),
        lr=config["training"]["lr_d"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=config["training"]["lr_g"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )

    loss_fn = WGANGPLoss(lambda_gp=config["training"]["lambda_gp"])

    print(f"LR_G: {config['training']['lr_g']}, LR_D: {config['training']['lr_d']}\n")

    nz = gen_cfg["nz"]
    fixed_noise = torch.randn(16, nz, device=device)
    fixed_labels = torch.randint(0, num_seen_classes, (16,), device=device)

    print("Step 5: Starting training...")
    print("=" * 70 + "\n")

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        netG.train()
        netD.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batches = 0

        for real_images, labels in pbar:
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            batches += 1

            for _ in range(config["training"]["n_critic"]):
                netD.zero_grad(set_to_none=True)

                z = torch.randn(batch_size, nz, device=device)
                fake_images = netG(z, labels, seen_embeddings)

                d_loss_dict = loss_fn.discriminator_loss(
                    netD, real_images, fake_images, labels, seen_embeddings, device
                )

                d_loss_dict["d_loss"].backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), config["training"]["grad_clip"])
                optimizerD.step()

            netG.zero_grad(set_to_none=True)

            z = torch.randn(batch_size, nz, device=device)
            fake_images = netG(z, labels, seen_embeddings)

            g_loss_dict = loss_fn.generator_loss(netD, fake_images, labels, seen_embeddings)

            g_loss_dict["g_loss"].backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), config["training"]["grad_clip"])
            optimizerG.step()

            epoch_g_loss += g_loss_dict["g_loss"].item()
            epoch_d_loss += d_loss_dict["d_loss"].item()

            g_losses.append(g_loss_dict["g_loss"].item())
            d_losses.append(d_loss_dict["d_loss"].item())

            pbar.set_postfix(
                {
                    "G": f"{epoch_g_loss/batches:.4f}",
                    "D": f"{epoch_d_loss/batches:.4f}",
                    "W": f"{d_loss_dict['wasserstein_distance']:.4f}",
                }
            )

        if (epoch + 1) % 2 == 0 or epoch == 0:
            netG.eval()
            with torch.no_grad():
                fake_samples = netG(fixed_noise, fixed_labels, seen_embeddings).cpu()

                Path("results/test").mkdir(parents=True, exist_ok=True)
                grid = vutils.make_grid(fake_samples, padding=2, normalize=True, nrow=4)

                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title(f"Epoch {epoch+1}")
                plt.imshow(grid.permute(1, 2, 0))
                plt.savefig(f"results/test/samples_epoch_{epoch+1}.png", dpi=150, bbox_inches="tight")
                plt.close()

                print(f"Saved samples for epoch {epoch+1}")
            netG.train()

    print("\nStep 6: Plotting training curves...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(g_losses, label="Generator", alpha=0.7)
    ax1.plot(d_losses, label="Discriminator", alpha=0.7)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Losses")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    window = 50
    if len(g_losses) > window:
        g_smooth = [sum(g_losses[max(0, i - window) : i + 1]) / min(i + 1, window) for i in range(len(g_losses))]
        d_smooth = [sum(d_losses[max(0, i - window) : i + 1]) / min(i + 1, window) for i in range(len(d_losses))]

        ax2.plot(g_smooth, label="Generator (smoothed)", alpha=0.7)
        ax2.plot(d_smooth, label="Discriminator (smoothed)", alpha=0.7)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Loss")
        ax2.set_title(f"Smoothed Losses (window={window})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/test/training_curves.png", dpi=150)
    plt.close()

    print("Saved training curves\n")

    print("Step 7: Generating final samples...")

    netG.eval()
    with torch.no_grad():
        n_samples = 25
        z = torch.randn(n_samples, nz, device=device)
        labels = torch.randint(0, num_seen_classes, (n_samples,), device=device)

        fake_images = netG(z, labels, seen_embeddings).cpu()

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.flatten()

        for i, (img, label) in enumerate(zip(fake_images, labels)):
            img_np = ((img + 1) / 2).numpy().transpose(1, 2, 0)
            img_np = img_np.clip(0, 1)

            axes[i].imshow(img_np)
            axes[i].axis("off")

            orig_class = class_info["new_to_org"][label.item()]
            class_name = class_names[orig_class]
            axes[i].set_title(class_name, fontsize=8)

        plt.tight_layout()
        plt.savefig("results/test/final_samples.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("Saved final samples\n")

    print("=" * 70)
    print("TRAINING TEST COMPLETE")
    print("=" * 70)
    print(f"Successfully trained for {num_epochs} epochs")
    print(f"Final G loss: {g_losses[-1]:.4f}")
    print(f"Final D loss: {d_losses[-1]:.4f}")
    print("\nResults saved in: results/test/")
    print("  - samples_epoch_*.png : Generated samples per epoch")
    print("  - training_curves.png : Loss curves")
    print("  - final_samples.png : Final generated samples with labels")
    print("\nPipeline integration successful!")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    import sys

    try:
        success = quick_train_test(num_epochs=5)
        sys.exit(0 if success else 1)
    except Exception as e:
        print("\nTraining test failed with error:")
        print(f"  {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
