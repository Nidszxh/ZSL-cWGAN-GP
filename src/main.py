import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

import numpy as np
import torch
import torchvision.utils as vutils
import yaml

from src.evaluation.gan_eval import compute_fid, save_fake_images, save_real_images, _save_images_concurrent
from src.evaluation.gzsl_eval import evaluate_gzsl, plot_gzsl_results, train_gzsl_classifier
from src.evaluation.zsl_eval import evaluate_zsl, train_zsl_classifier
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.training.trainer import train_gan
from src.utils.data_loader import get_class_names, get_class_split, get_data_loaders, get_test_loader
from src.utils.embeddings import EmbeddingManager
from src.utils.visualization import (
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


def validate_config(config: dict) -> None:
    """Validate required config keys exist with sensible values."""
    required_paths = {
        "paths": ["data_root", "results_dir", "checkpoints_dir", "cache_dir"],
        "dataset": ["num_classes", "seen_classes"],
        "embeddings": ["type"],
        "model.generator": ["nz", "ngf", "nc", "semantic_proj_dim"],
        "model.discriminator": ["ndf", "nc", "semantic_proj_dim"],
        "model.classifier": ["backbone"],
        "training": ["num_epochs", "batch_size", "lr_g", "lr_d", "n_critic", "lambda_gp"],
    }
    errors = []
    for section, keys in required_paths.items():
        parent = config
        for part in section.split("."):
            if isinstance(parent, dict):
                parent = parent.get(part, {})
        if not isinstance(parent, dict):
            errors.append(f"Missing config section: {section}")
            continue
        for key in keys:
            if key not in parent:
                errors.append(f"Missing config key: {section}.{key}")

    embed_type = config.get("embeddings", {}).get("type", "")
    if embed_type != "clip":
        errors.append(f"embeddings.type must be 'clip', got {embed_type!r}")
    elif not config.get("embeddings", {}).get("clip_model"):
        errors.append("embeddings.clip_model required")

    if errors:
        for e in errors:
            print(f"Config error: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="ZSL-cWGAN-GP Training")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (e.g. checkpoints/checkpoint_epoch_050.pth)",
    )
    parser.add_argument("--config", type=str, default="src/configs/config.yaml", help="Path to config file")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: {args.config} not found")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["resume_path"] = args.resume

    validate_config(config)
    set_seed(config["experiment"]["seed"])
    device = torch.device(config["experiment"]["device"])
    print(f"Using device: {device}")

    for dir_path in [
        config["paths"]["results_dir"],
        config["paths"]["checkpoints_dir"],
        config["paths"]["cache_dir"],
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
    # L1-F13: class arrays must be sorted to match FilteredCIFAR100's sorted-rank labels.
    assert np.array_equal(np.sort(seen_classes), seen_classes)
    assert np.array_equal(np.sort(unseen_classes), unseen_classes)

    train_loader, val_loader, class_info = get_data_loaders(config, seen_classes)
    num_seen_classes = class_info["num_seen_classes"]

    class_names = get_class_names(config["paths"]["data_root"])

    # Embeddings
    print("\nLoading embeddings...")
    embedding_manager = EmbeddingManager(config)
    seen_embeddings, embedding_dim = embedding_manager.get_embeddings(class_names, seen_classes)
    unseen_embeddings, _ = embedding_manager.get_embeddings(class_names, unseen_classes)
    # L2-F1b: free the CLIP encoder (~1.5GB) before GAN training.
    del embedding_manager
    torch.cuda.empty_cache()
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
        use_spectral_norm_semantic=gen_cfg.get("use_spectral_norm_semantic", True),
        attention_resolutions=gen_cfg.get("attention_resolutions", [8]),
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
        save_real_images(
            val_loader.dataset,
            num_samples=config["evaluation"]["fid_samples"],
            save_dir=real_images_dir,
        )

    # Train (with optional resume)
    tracker = train_gan(
        netG,
        netD,
        train_loader,
        seen_embeddings,
        device,
        config,
        real_images_dir,
        resume_path=config.get("resume_path"),
    )
    torch.cuda.empty_cache()

    # Post-training plots
    print("\nGenerating training visualizations...")
    plot_training_curves(tracker, save_dir=config["paths"]["results_dir"])
    plot_metrics_progress(tracker, save_dir=config["paths"]["results_dir"])

    # Load best model
    print(f"\nLoading best model with FID: {tracker.best_fid:.2f}")
    checkpoint = torch.load(Path(config["paths"]["checkpoints_dir"]) / "best_model.pth", weights_only=True)
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
    # L2-F1: free Inception caches before the ZSL/GZSL phases (8GB VRAM).
    torch.cuda.empty_cache()
    print(
        f"Final FID: {final_metrics['fid']:.2f} | IS: {final_metrics['is_mean']:.2f} +- {final_metrics['is_std']:.2f}"
    )

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
            labels = torch.full((samples_per_class,), i, device=device)
            fake_imgs = netG(z, labels, unseen_embeddings)
            fake_imgs = ((fake_imgs + 1) / 2).clamp(0, 1)
            class_dir = save_dir / f"{i:02d}_{class_names[unseen_class_idx]}"
            class_dir.mkdir(parents=True, exist_ok=True)

            _save_images_concurrent(fake_imgs, class_dir, lambda j: f"{j:04d}.png")
            if samples_per_class >= 16:
                grid = vutils.make_grid(fake_imgs[:16], nrow=4, padding=2, normalize=False)
                vutils.save_image(grid, class_dir / "samples_grid.png")

    # =====================================================
    # ZSL Evaluation (unseen classes only)
    # =====================================================
    print("\n" + "=" * 70)
    print("ZERO-SHOT LEARNING EVALUATION")
    print("=" * 70)

    test_loader, _ = get_test_loader(config, unseen_classes)
    classifier = train_zsl_classifier(netG, unseen_embeddings, unseen_classes, device, config)
    zsl_metrics = evaluate_zsl(
        classifier,
        test_loader,
        unseen_classes,
        class_names,
        device,
    )

    plot_zsl_confusion_matrix(zsl_metrics, save_dir=config["paths"]["results_dir"])
    plot_zsl_class_accuracy(zsl_metrics, save_dir=config["paths"]["results_dir"])

    # =====================================================
    # GZSL Evaluation (seen + unseen classes)
    # =====================================================
    gzsl_cfg = config["evaluation"].get("gzsl", {})
    gzsl_classifier = None
    gzsl_metrics = None
    if gzsl_cfg.get("enabled", False):
        print("\n" + "=" * 70)
        print("GENERALIZED ZERO-SHOT LEARNING (GZSL) EVALUATION")
        print("=" * 70)

        gzsl_classifier = train_gzsl_classifier(
            generator=netG,
            unseen_semantic_embeddings=unseen_embeddings,
            seen_classes=seen_classes,
            unseen_classes=unseen_classes,
            train_loader=train_loader,
            device=device,
            config=config,
        )

        if gzsl_classifier is not None:
            gzsl_metrics = evaluate_gzsl(
                classifier=gzsl_classifier,
                seen_classes=seen_classes,
                unseen_classes=unseen_classes,
                test_loader=test_loader,
                seen_eval_loader=val_loader,
                class_names=class_names,
                device=device,
            )
            plot_gzsl_results(gzsl_metrics, save_dir=config["paths"]["results_dir"])

    # =====================================================
    # Summary
    # =====================================================
    create_experiment_summary(
        tracker,
        zsl_metrics,
        final_metrics["fid"],
        config,
        gzsl_metrics=gzsl_metrics,
        save_dir=config["paths"]["results_dir"],
    )

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print("\nGAN Metrics:")
    print(f"  Best FID: {tracker.best_fid:.2f}")
    print(f"  Final FID: {final_metrics['fid']:.2f}")
    print("\nZero-Shot Learning (unseen only):")
    print(f"  Top-1 Accuracy: {zsl_metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {zsl_metrics['top5_accuracy']:.2f}%")
    print(f"  Mean Class Accuracy: {zsl_metrics['mean_class_accuracy']:.2f}%")

    if gzsl_cfg.get("enabled", False) and gzsl_classifier is not None:
        print("\nGeneralized Zero-Shot Learning (seen + unseen):")
        print(f"  Seen Accuracy: {gzsl_metrics['seen_accuracy']:.2f}%")
        print(f"  Unseen Accuracy: {gzsl_metrics['unseen_accuracy']:.2f}%")
        print(f"  Harmonic Mean (H): {gzsl_metrics['harmonic_mean']:.2f}%")

    print(f"\nResults: {config['paths']['results_dir']}/")
    print(f"Checkpoints: {config['paths']['checkpoints_dir']}/")
    print(f"TensorBoard: {config['paths']['tensorboard_dir']}/")
    print(f"\n  tensorboard --logdir={config['paths']['tensorboard_dir']}")


if __name__ == "__main__":
    main()
