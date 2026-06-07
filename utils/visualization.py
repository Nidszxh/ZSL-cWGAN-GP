from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


def plot_training_curves(tracker, save_dir: str = "results"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(tracker.g_losses, label="G Loss", alpha=0.7)
    ax1.plot(tracker.d_losses, label="D Loss", alpha=0.7)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Generator and Discriminator Losses")
    ax1.grid(True, alpha=0.3)

    ax2.plot(tracker.w_distances, label="Wasserstein Distance", alpha=0.7)
    ax2.plot(tracker.gp_values, label="Gradient Penalty", alpha=0.7)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.set_title("Wasserstein Distance and Gradient Penalty")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(save_dir) / "training_curves.png", dpi=150)
    plt.close()


def plot_metrics_progress(tracker, save_dir: str = "results"):
    if not tracker.metrics_history:
        return

    epochs = [m[0] for m in tracker.metrics_history]
    fids = [m[1]["fid"] for m in tracker.metrics_history]
    is_means = [m[1].get("is_mean", 0) for m in tracker.metrics_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, fids, "o-", label="FID Score", markersize=6)
    ax1.axhline(y=tracker.best_fid, color="r", linestyle="--", label=f"Best: {tracker.best_fid:.2f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("FID Score")
    ax1.set_title("FID Score During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, is_means, "o-", color="green", label="Inception Score", markersize=6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Inception Score")
    ax2.set_title("Inception Score During Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "metrics_progress.png", dpi=150)
    plt.close()


def save_sample_grid(generator, fixed_noise, fixed_labels, seen_embeddings, epoch, device, save_dir: str = "results"):
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(fixed_noise, fixed_labels, seen_embeddings).cpu()
        grid = vutils.make_grid(fake_samples, padding=2, normalize=True)

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Epoch {epoch}")
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / f"samples_epoch_{epoch:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()
    generator.train()
    return grid


def plot_zsl_confusion_matrix(zsl_metrics, save_dir: str = "results"):
    cm = zsl_metrics["confusion_matrix"]
    num_classes = len(cm)
    class_labels = zsl_metrics["class_names"]

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Zero-Shot Learning Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_labels, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_labels, fontsize=9)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "zsl_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_zsl_class_accuracy(zsl_metrics, save_dir: str = "results"):
    per_class_acc = zsl_metrics["per_class_accuracy"]
    class_labels = zsl_metrics["class_names"]
    mean_acc = zsl_metrics["mean_class_accuracy"]
    top1_acc = zsl_metrics["top1_accuracy"]
    num_classes = len(per_class_acc)

    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_indices]
    sorted_names = [class_labels[i] for i in sorted_indices]

    colors = ["#d73027" if acc < mean_acc else "#4575b4" for acc in sorted_acc]

    plt.figure(figsize=(16, 6))
    plt.bar(range(num_classes), sorted_acc, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    plt.xticks(range(num_classes), sorted_names, rotation=90, fontsize=9)
    plt.ylim(0, 100)
    plt.title("Zero-Shot Learning Accuracy by Class", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.axhline(y=mean_acc, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_acc:.2f}%")
    plt.axhline(y=top1_acc, color="green", linestyle="--", linewidth=2, label=f"Overall: {top1_acc:.2f}%")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "zsl_class_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()


def create_experiment_summary(tracker, zsl_metrics, final_fid, config, save_dir: str = "results", gzsl_metrics=None):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    sample_grid_path = Path(save_dir) / "generated_samples_grid.png"
    if sample_grid_path.exists():
        img = plt.imread(sample_grid_path)
        ax1.imshow(img)
        ax1.set_title("Generated Samples (Seen Classes)", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    if tracker.g_losses and tracker.d_losses:
        window = min(500, len(tracker.g_losses))
        ax2.plot(tracker.g_losses[-window:], label="G Loss", alpha=0.7, linewidth=1.5)
        ax2.plot(tracker.d_losses[-window:], label="D Loss", alpha=0.7, linewidth=1.5)
        ax2.set_title("Training Losses (Last 500)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Iterations", fontsize=9)
        ax2.set_ylabel("Loss", fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    if tracker.metrics_history:
        epochs = [m[0] for m in tracker.metrics_history]
        fids = [m[1]["fid"] for m in tracker.metrics_history]
        ax3.plot(epochs, fids, "o-", color="steelblue", linewidth=2, markersize=6)
        ax3.axhline(y=tracker.best_fid, color="r", linestyle="--", linewidth=2, label=f"Best: {tracker.best_fid:.2f}")
        ax3.set_title("FID Progression", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Epoch", fontsize=9)
        ax3.set_ylabel("FID Score", fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if tracker.metrics_history:
        epochs = [m[0] for m in tracker.metrics_history]
        is_means = [m[1].get("is_mean", 0) for m in tracker.metrics_history]
        ax4.plot(epochs, is_means, "o-", color="green", linewidth=2, markersize=6)
        ax4.set_title("Inception Score", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Epoch", fontsize=9)
        ax4.set_ylabel("IS", fontsize=9)
        ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    if tracker.w_distances:
        window = min(500, len(tracker.w_distances))
        ax5.plot(tracker.w_distances[-window:], color="orange", alpha=0.7, linewidth=1.5)
        ax5.set_title("Wasserstein Distance", fontsize=11, fontweight="bold")
        ax5.set_xlabel("Iterations", fontsize=9)
        ax5.set_ylabel("W-Distance", fontsize=9)
        ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, :2])
    if gzsl_metrics is not None:
        gzsl_path = Path(save_dir) / "gzsl_results.png"
        if gzsl_path.exists():
            img = plt.imread(gzsl_path)
            ax6.imshow(img)
            ax6.set_title("GZSL Results (Seen + Unseen)", fontsize=14, fontweight="bold")
        else:
            zsl_acc_path = Path(save_dir) / "zsl_class_accuracy.png"
            if zsl_acc_path.exists():
                img = plt.imread(zsl_acc_path)
                ax6.imshow(img)
                ax6.set_title("Zero-Shot Learning Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax6.axis("off")
    else:
        zsl_acc_path = Path(save_dir) / "zsl_class_accuracy.png"
        if zsl_acc_path.exists():
            img = plt.imread(zsl_acc_path)
            ax6.imshow(img)
            ax6.set_title("Zero-Shot Learning Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax6.axis("off")

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    num_epochs = tracker.metrics_history[-1][0] if tracker.metrics_history else "?"

    gzsl_section = ""
    if gzsl_metrics is not None:
        gzsl_section = f"""
GZSL:
  Seen Acc: {gzsl_metrics['seen_accuracy']:.2f}%
  Unseen Acc: {gzsl_metrics['unseen_accuracy']:.2f}%
  Harmonic Mean: {gzsl_metrics['harmonic_mean']:.2f}%
"""

    summary_text = f"""
EXPERIMENT SUMMARY
{'=' * 30}

GAN Training:
  Best FID: {tracker.best_fid:.2f}
  Final FID: {final_fid:.2f}

Zero-Shot Learning:
  Top-1 Acc: {zsl_metrics['top1_accuracy']:.2f}%
  Top-5 Acc: {zsl_metrics['top5_accuracy']:.2f}%
  Mean Class: {zsl_metrics['mean_class_accuracy']:.2f}%
{gzsl_section}
Training:
  Epochs: {num_epochs}
  Batch Size: {config['training']['batch_size']}
  LR G: {config['training']['lr_g']}
  LR D: {config['training']['lr_d']}
  Classifier: {config.get('model', {}).get('classifier', {}).get('backbone', 'resnet18')}
  Embeddings: {config['embeddings']['type']}
"""
    ax7.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle("ZSL-cWGAN-GP Training Summary", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(Path(save_dir) / "experiment_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_final_sample_grid(generator, device, num_seen_classes, seen_embeddings, class_names, class_info, nz: int = 128, n_rows: int = 4, n_cols: int = 5, save_dir: str = "results"):
    generator.eval()
    noise = torch.randn(n_rows * n_cols, nz, device=device)
    labels = torch.tensor([i % num_seen_classes for i in range(n_rows * n_cols)], device=device)

    with torch.no_grad():
        fake_images = generator(noise, labels, seen_embeddings).cpu()
        fake_images = (fake_images + 1) / 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for i, (img, label) in enumerate(zip(fake_images, labels)):
            img_np = img.numpy().transpose(1, 2, 0)
            axes[i].imshow(img_np)
            orig_class_idx = class_info["new_to_org"][label.item()]
            class_name = class_names[orig_class_idx]
            axes[i].set_title(class_name, fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "generated_samples_grid.png", dpi=200, bbox_inches="tight")
        plt.close()
    generator.train()
