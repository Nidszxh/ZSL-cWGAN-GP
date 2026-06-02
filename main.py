import sys
import warnings
from pathlib import Path

# Suppress NumPy 2.4+ deprecation warning from torchvision's CIFAR-100 loader
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from torchvision import datasets

from evaluation.gan_eval import compute_fid, save_fake_images, save_real_images
from evaluation.zsl_eval import evaluate_zsl, train_zsl_classifier
from models.discriminator import Discriminator
from models.generator import Generator
from training.trainer import train_gan
from utils.data_loader import get_class_split, get_data_loaders, get_test_loader
from utils.embeddings import EmbeddingManager
from utils.visualization import (
    create_experiment_summary,
    generate_final_sample_grid,
    plot_training_curves,
    plot_metrics_progress,
    plot_zsl_class_accuracy,
    plot_zsl_confusion_matrix,
)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("Error: configs/config.yaml not found")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["seed"])
    device = torch.device(config["experiment"]["device"])
    print(f"Using device: {device}")

    for dir_path in [
        config["paths"]["results_dir"],
        config["paths"]["checkpoints_dir"],
        config["paths"]["cache_dir"],
        f"{config['paths']['results_dir']}/fake",
        f"{config['paths']['results_dir']}/real",
        f"{config['paths']['results_dir']}/unseen_synthetic",
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Data
    seen_classes, unseen_classes = get_class_split(
        num_classes=config["dataset"]["num_classes"],
        seen_count=config["dataset"]["seen_classes"],
        cache_dir=config["paths"]["cache_dir"],
        seed=config["experiment"]["seed"],
    )

    train_loader, val_loader, class_info = get_data_loaders(config, seen_classes)
    num_seen_classes = class_info["num_seen_classes"]

    cifar100 = datasets.CIFAR100(root=config["paths"]["data_root"], download=True)
    class_names = cifar100.classes

    # Embeddings
    print("\nLoading embeddings...")
    embedding_manager = EmbeddingManager(config)
    all_embeddings, embedding_dim = embedding_manager.get_embeddings(class_names)
    seen_embeddings, _ = embedding_manager.get_embeddings(class_names, seen_classes)
    unseen_embeddings, _ = embedding_manager.get_embeddings(class_names, unseen_classes)
    seen_embeddings = seen_embeddings.to(device)
    unseen_embeddings = unseen_embeddings.to(device)

    # Models
    gen_cfg = config["model"]["generator"]
    disc_cfg = config["model"]["discriminator"]

    netG = Generator(
        nz=gen_cfg["nz"],
        ngf=gen_cfg["ngf"],
        nc=gen_cfg["nc"],
        semantic_dim=embedding_dim,
        semantic_proj_dim=gen_cfg["semantic_proj_dim"],
        dropout=gen_cfg["dropout"],
    ).to(device)

    netD = Discriminator(
        nc=disc_cfg["nc"],
        ndf=disc_cfg["ndf"],
        semantic_dim=embedding_dim,
        semantic_proj_dim=disc_cfg["semantic_proj_dim"],
    ).to(device)

    print(f"\nGenerator params: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in netD.parameters()):,}")

    # Save real images once
    real_images_dir = f"{config['paths']['results_dir']}/real"
    if not Path(real_images_dir, "real_00000.png").exists():
        save_real_images(val_loader.dataset, save_dir=real_images_dir)

    # Train
    tracker = train_gan(netG, netD, train_loader, seen_embeddings, device, config, real_images_dir)

    # Post-training plots
    print("\nGenerating training visualizations...")
    plot_training_curves(tracker, save_dir=config["paths"]["results_dir"])
    plot_metrics_progress(tracker, save_dir=config["paths"]["results_dir"])

    # Load best model
    print(f"\nLoading best model with FID: {tracker.best_fid:.2f}")
    checkpoint = torch.load(Path(config["paths"]["checkpoints_dir"]) / "best_model.pth")
    netG.load_state_dict(checkpoint["generator"])
    netD.load_state_dict(checkpoint["discriminator"])

    # Final evaluation
    print("\nGenerating final evaluation images...")
    final_fake_dir = save_fake_images(
        netG,
        "final",
        device,
        num_samples=config["evaluation"]["fid_samples"],
        batch_size=config["training"]["batch_size"],
        nz=gen_cfg["nz"],
        num_seen_classes=num_seen_classes,
        seen_embeddings=seen_embeddings,
        save_dir=f"{config['paths']['results_dir']}/fake_final",
    )
    final_metrics = compute_fid(real_images_dir, final_fake_dir)
    print(f"Final FID: {final_metrics['fid']:.2f} | IS: {final_metrics['is_mean']:.2f} +- {final_metrics['is_std']:.2f}")

    # Sample grid
    generate_final_sample_grid(
        netG,
        device,
        num_seen_classes,
        seen_embeddings,
        class_names,
        class_info,
        nz=gen_cfg["nz"],
        save_dir=config["paths"]["results_dir"],
    )

    # Unseen class images
    print("\nGenerating synthetic images for unseen classes...")
    samples_per_class = config["evaluation"]["synthetic_samples_per_class"]
    save_dir = Path(config["paths"]["results_dir"]) / "unseen_synthetic"
    save_dir.mkdir(parents=True, exist_ok=True)

    netG.eval()
    with torch.no_grad():
        for i, unseen_class_idx in enumerate(unseen_classes):
            z = torch.randn(samples_per_class, gen_cfg["nz"], device=device)
            z = torch.clamp(z, -2, 2)
            labels = torch.full((samples_per_class,), i, device=device)
            fake_imgs = netG(z, labels, unseen_embeddings)
            fake_imgs = (fake_imgs + 1) / 2
            class_dir = save_dir / f"{i:02d}_{class_names[unseen_class_idx]}"
            class_dir.mkdir(parents=True, exist_ok=True)

            for j, img in enumerate(fake_imgs):
                vutils.save_image(img, class_dir / f"{j:04d}.png")
            if samples_per_class >= 16:
                grid = vutils.make_grid(fake_imgs[:16], nrow=4, padding=2, normalize=False)
                vutils.save_image(grid, class_dir / "samples_grid.png")

    # ZSL evaluation
    print("\n" + "=" * 70)
    print("ZERO-SHOT LEARNING EVALUATION")
    print("=" * 70)

    test_loader, test_class_info = get_test_loader(config, unseen_classes)
    classifier = train_zsl_classifier(netG, unseen_embeddings, unseen_classes, test_loader, device, config)
    zsl_metrics = evaluate_zsl(
        classifier,
        test_loader,
        unseen_classes,
        class_names,
        device,
        results_dir=config["paths"]["results_dir"],
    )

    # ZSL plots
    plot_zsl_confusion_matrix(zsl_metrics, save_dir=config["paths"]["results_dir"])
    plot_zsl_class_accuracy(zsl_metrics, save_dir=config["paths"]["results_dir"])

    # Summary
    create_experiment_summary(
        tracker,
        zsl_metrics,
        final_metrics["fid"],
        config,
        save_dir=config["paths"]["results_dir"],
    )

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print("\nGAN Metrics:")
    print(f"  Best FID: {tracker.best_fid:.2f}")
    print(f"  Final FID: {final_metrics['fid']:.2f}")
    print("\nZero-Shot Learning:")
    print(f"  Top-1 Accuracy: {zsl_metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {zsl_metrics['top5_accuracy']:.2f}%")
    print(f"  Mean Class Accuracy: {zsl_metrics['mean_class_accuracy']:.2f}%")
    print(f"\nResults: {config['paths']['results_dir']}/")
    print(f"Checkpoints: {config['paths']['checkpoints_dir']}/")
    print(f"TensorBoard: {config['paths']['tensorboard_dir']}/")
    print(f"\n  tensorboard --logdir={config['paths']['tensorboard_dir']}")


if __name__ == "__main__":
    main()
