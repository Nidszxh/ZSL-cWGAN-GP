"""
Generalized Zero-Shot Learning (GZSL) Evaluation Module.

Evaluates on BOTH seen + unseen classes simultaneously, reporting:
- Seen accuracy, Unseen accuracy, Harmonic mean (H)
- Calibrated stacking to reduce seen-class bias
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm

from src.utils.data_loader import FilteredCIFAR100


class _LabelShiftedDataset(Dataset):
    """Shifts labels by a fixed offset for ConcatDataset compatibility."""

    def __init__(self, dataset, offset):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label + self.offset


zsl_augment = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
)


class CalibratedClassifier(nn.Module):
    """Wrapper that applies learned calibration to reduce seen/unseen bias."""

    def __init__(self, base_classifier, num_seen, num_unseen, calibration="learned_temperature"):
        super().__init__()
        self.base_classifier = base_classifier
        self.calibration = calibration
        self.num_seen = num_seen
        self.num_unseen = num_unseen

        if calibration == "learned_temperature":
            self.temperature_s = nn.Parameter(torch.ones(1))
            self.temperature_u = nn.Parameter(torch.ones(1))
            self.bias_shift = nn.Parameter(torch.zeros(1))
        elif calibration == "fixed":
            self.register_buffer("temperature_s", torch.tensor(1.0))
            self.register_buffer("temperature_u", torch.tensor(1.0))
            self.register_buffer("bias_shift", torch.tensor(0.0))

    def forward(self, x):
        logits = self.base_classifier(x)

        if self.calibration == "none":
            return logits

        seen_logits = logits[:, : self.num_seen] / self.temperature_s
        unseen_logits = logits[:, self.num_seen :] / self.temperature_u + self.bias_shift
        return torch.cat([seen_logits, unseen_logits], dim=1)


def _mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple:
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def _generate_synthetic_data(
    generator: torch.nn.Module,
    num_samples: int,
    num_classes: int,
    semantic_embeddings: torch.Tensor,
    nz: int,
    batch_size: int,
    device: torch.device,
) -> TensorDataset:
    generator.eval()
    all_images, all_labels = [], []
    samples_per_class = num_samples // num_classes

    with torch.no_grad():
        for cls in range(num_classes):
            n_remaining = samples_per_class
            while n_remaining > 0:
                curr_bs = min(batch_size, n_remaining)
                z = torch.randn(curr_bs, nz, device=device)
                labels = torch.full((curr_bs,), cls, device=device, dtype=torch.long)
                fake_imgs = generator(z, labels, semantic_embeddings)
                all_images.append(fake_imgs.cpu())
                all_labels.append(labels.cpu())
                n_remaining -= curr_bs

    generator.train()
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return TensorDataset(images, labels)


def train_gzsl_classifier(
    generator: torch.nn.Module,
    unseen_semantic_embeddings: torch.Tensor,
    seen_classes: np.ndarray,
    unseen_classes: np.ndarray,
    train_loader: DataLoader,
    device: torch.device,
    config: dict,
) -> Optional[torch.nn.Module]:
    """
    Train a GZSL classifier on:
    - Real seen-class images (from train_loader)
    - Synthetic unseen-class images (from generator)

    The classifier sees num_seen + num_unseen output classes.
    """
    gzsl_cfg = config["evaluation"].get("gzsl", {})
    if not gzsl_cfg.get("enabled", False):
        return None

    num_seen = len(seen_classes)
    num_unseen = len(unseen_classes)
    num_total = num_seen + num_unseen

    nz = config["model"]["generator"]["nz"]
    samples_per_class = config["evaluation"]["synthetic_samples_per_class"]
    zsl_epochs = config["evaluation"]["zsl_epochs"]
    zsl_lr = config["evaluation"]["zsl_lr"]
    zsl_batch_size = config["evaluation"]["zsl_batch_size"]
    checkpoints_dir = config["paths"]["checkpoints_dir"]

    mixup_alpha = config["evaluation"].get("zsl_mixup_alpha", 0.2)
    label_smoothing = config["evaluation"].get("zsl_label_smoothing", 0.1)

    regenerate_every = config["evaluation"].get("zsl_regenerate_every", 1)
    calibration_mode = gzsl_cfg.get("calibration", "learned_temperature")

    from src.models.zsl_classifier import build_classifier_from_config

    base_classifier = build_classifier_from_config(num_total, config).to(device)
    classifier = CalibratedClassifier(base_classifier, num_seen, num_unseen, calibration=calibration_mode).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(classifier.parameters(), lr=zsl_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    total_synth_samples = samples_per_class * num_unseen
    synth_train_size = int(0.8 * total_synth_samples)
    synth_val_size = total_synth_samples - synth_train_size

    best_val_acc = 0
    patience = 10
    epochs_without_improv = 0

    plain_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    plain_seen = FilteredCIFAR100(
        root=config["paths"]["data_root"],
        train=True,
        download=True,
        transform=plain_transform,
        allowed_classes=seen_classes,
    )
    split_gen = torch.Generator().manual_seed(config["experiment"]["seed"])
    plain_size = int(0.9 * len(plain_seen))
    seen_dataset, _ = random_split(plain_seen, [plain_size, len(plain_seen) - plain_size], generator=split_gen)

    print("\nTraining GZSL classifier (seen real + synthetic unseen)...")
    for epoch in range(zsl_epochs):
        if epoch % regenerate_every == 0:
            print(f"  Regenerating synthetic unseen data (epoch {epoch + 1})...")
            synth_dataset = _generate_synthetic_data(
                generator,
                total_synth_samples,
                num_unseen,
                unseen_semantic_embeddings,
                nz,
                batch_size=zsl_batch_size,
                device=device,
            )
            num_workers = min(config.get("dataset", {}).get("num_workers", 4), 4)
            split_gen = torch.Generator().manual_seed(42)
            synth_train, synth_val = random_split(
                synth_dataset, [synth_train_size, synth_val_size], generator=split_gen
            )

            shifted_synth_train = _LabelShiftedDataset(synth_train, num_seen)
            combined_dataset = ConcatDataset([seen_dataset, shifted_synth_train])
            combined_train_loader = DataLoader(
                combined_dataset,
                batch_size=zsl_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=True,
            )

            shifted_synth_val = _LabelShiftedDataset(synth_val, num_seen)
            # L1-F14: val labels need the same num_seen shift as train.
            synth_val_loader = DataLoader(
                shifted_synth_val,
                batch_size=100,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(combined_train_loader, desc=f"GZSL Epoch {epoch + 1}/{zsl_epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = zsl_augment(images)

            optimizer.zero_grad(set_to_none=True)

            use_mixup = mixup_alpha > 0 and np.random.random() < 0.5
            if use_mixup:
                mixed_images, labels_a, labels_b, lam = _mixup_data(images, labels, mixup_alpha)
                outputs = classifier(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = classifier(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if not use_mixup:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(
                {
                    "Loss": f"{train_loss / max(1, pbar.n + 1):.4f}",
                    "Acc": f"{100.0 * correct / max(1, total):.2f}%",
                }
            )

        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in synth_val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = classifier(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        print(f"  GZSL Validation Acc: {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
            torch.save(classifier.state_dict(), Path(checkpoints_dir) / "best_gzsl_classifier.pth")
            epochs_without_improv = 0
        else:
            epochs_without_improv += 1
            if epochs_without_improv >= patience:
                print(f"  Early stopping GZSL at epoch {epoch + 1}")
                break

    classifier.load_state_dict(torch.load(Path(checkpoints_dir) / "best_gzsl_classifier.pth", weights_only=True))
    return classifier


def evaluate_gzsl(
    classifier: torch.nn.Module,
    seen_classes: np.ndarray,
    unseen_classes: np.ndarray,
    test_loader: DataLoader,
    seen_eval_loader: DataLoader,
    class_names: list,
    device: torch.device,
) -> dict:
    """
    Evaluate GZSL: classifier must predict correctly for both seen and unseen classes.

    Uses held-out seen validation set (not training set) for realistic seen accuracy.

    Returns:
        dict with seen_acc, unseen_acc, harmonic_mean, confusion_matrix, etc.
    """
    num_seen = len(seen_classes)
    num_unseen = len(unseen_classes)
    num_total = num_seen + num_unseen
    classifier.eval()

    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0
    all_correct = 0
    all_total = 0
    confusion_matrix = torch.zeros(num_total, num_total)
    per_class_correct = [0] * num_total
    per_class_total = [0] * num_total

    with torch.no_grad():
        for images, labels in tqdm(seen_eval_loader, desc="Evaluating seen classes"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            batch_size = labels.size(0)
            seen_total += batch_size
            all_total += batch_size
            correct_mask = predicted.eq(labels)
            seen_correct += correct_mask.sum().item()
            all_correct += correct_mask.sum().item()
            cnt = torch.bincount(labels, minlength=num_total)
            corr = torch.bincount(labels[correct_mask], minlength=num_total)
            per_class_total = [t + c.item() for t, c in zip(per_class_total, cnt)]
            per_class_correct = [t + c.item() for t, c in zip(per_class_correct, corr)]
            flat = (labels * num_total + predicted).flatten()
            confusion_matrix += torch.bincount(flat, minlength=num_total * num_total).view(num_total, num_total).cpu()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating unseen classes"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels = labels + num_seen
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            batch_size = labels.size(0)
            unseen_total += batch_size
            all_total += batch_size
            correct_mask = predicted.eq(labels)
            unseen_correct += correct_mask.sum().item()
            all_correct += correct_mask.sum().item()
            cnt = torch.bincount(labels, minlength=num_total)
            corr = torch.bincount(labels[correct_mask], minlength=num_total)
            per_class_total = [t + c.item() for t, c in zip(per_class_total, cnt)]
            per_class_correct = [t + c.item() for t, c in zip(per_class_correct, corr)]
            flat = (labels * num_total + predicted).flatten()
            confusion_matrix += torch.bincount(flat, minlength=num_total * num_total).view(num_total, num_total).cpu()

    seen_acc = 100.0 * seen_correct / seen_total if seen_total > 0 else 0
    unseen_acc = 100.0 * unseen_correct / unseen_total if unseen_total > 0 else 0
    # L1-F9: H is sample-level accuracy (balanced CIFAR-100, so ~per-class-mean).
    harmonic_mean = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) > 0 else 0
    overall_acc = 100.0 * all_correct / all_total if all_total > 0 else 0

    per_class_acc = [
        100.0 * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in range(num_total)
    ]
    mean_class_acc = float(np.mean(per_class_acc))

    for i in range(num_total):
        if per_class_total[i] > 0:
            confusion_matrix[i] = confusion_matrix[i] / per_class_total[i]

    all_class_names = [class_names[c] for c in seen_classes] + [class_names[c] for c in unseen_classes]
    seen_class_names = [class_names[c] for c in seen_classes]
    unseen_class_names = [class_names[c] for c in unseen_classes]

    print(f"\n{'=' * 70}")
    print("GENERALIZED ZERO-SHOT LEARNING (GZSL) RESULTS")
    print(f"{'=' * 70}")
    print(f"Seen Accuracy:      {seen_acc:.2f}%")
    print(f"Unseen Accuracy:    {unseen_acc:.2f}%")
    print(f"Harmonic Mean (H):  {harmonic_mean:.2f}%")
    print(f"Overall Accuracy:   {overall_acc:.2f}%")

    return {
        "seen_accuracy": seen_acc,
        "unseen_accuracy": unseen_acc,
        "harmonic_mean": harmonic_mean,
        "overall_accuracy": overall_acc,
        "mean_class_accuracy": mean_class_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.cpu().numpy(),
        "all_class_names": all_class_names,
        "seen_class_names": seen_class_names,
        "unseen_class_names": unseen_class_names,
        "num_seen": num_seen,
        "num_unseen": num_unseen,
    }


def plot_gzsl_results(gzsl_metrics: dict, save_dir: str = "results"):
    """Generate GZSL-specific visualizations."""
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    seen_acc = gzsl_metrics["seen_accuracy"]
    unseen_acc = gzsl_metrics["unseen_accuracy"]
    h_mean = gzsl_metrics["harmonic_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    categories = ["Seen\nAccuracy", "Unseen\nAccuracy", "Harmonic\nMean (H)"]
    values = [seen_acc, unseen_acc, h_mean]
    colors = ["#4575b4", "#d73027", "#fdae61"]
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("GZSL: Seen vs Unseen vs Harmonic Mean", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax2 = axes[1]
    num_total = gzsl_metrics["num_seen"] + gzsl_metrics["num_unseen"]
    per_class_acc = gzsl_metrics["per_class_accuracy"]
    seen_names = gzsl_metrics["seen_class_names"]
    unseen_names = gzsl_metrics["unseen_class_names"]
    all_names = seen_names + unseen_names
    is_unseen = [False] * len(seen_names) + [True] * len(unseen_names)
    sorted_indices = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_indices]
    sorted_names = [all_names[i] for i in sorted_indices]
    sorted_unseen = [is_unseen[i] for i in sorted_indices]
    bar_colors = ["#d73027" if u else "#4575b4" for u in sorted_unseen]
    ax2.bar(range(len(sorted_acc)), sorted_acc, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(sorted_names)))
    ax2.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("GZSL Per-Class Accuracy (blue=seen, red=unseen)", fontsize=13, fontweight="bold")
    ax2.axhline(y=h_mean, color="orange", linestyle="--", linewidth=2, label=f"H = {h_mean:.1f}%")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path / "gzsl_results.png", dpi=200, bbox_inches="tight")
    plt.close()

    cm = gzsl_metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("GZSL Confusion Matrix (Seen + Unseen)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(num_total)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(all_names, rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(all_names, fontsize=7)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    ax.axhline(y=len(seen_names) - 0.5, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=len(seen_names) - 0.5, color="red", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(save_path / "gzsl_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"GZSL plots saved to {save_dir}/")
