"""
Generalized Zero-Shot Learning (GZSL) Evaluation Module.

Evaluates on BOTH seen + unseen classes simultaneously, reporting:
- Seen accuracy, Unseen accuracy, Harmonic mean (H)
- Calibrated stacking to reduce seen-class bias
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm


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


def _mixup_data(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def _generate_synthetic_data(generator, num_samples, num_classes, semantic_embeddings, nz, batch_size, device):
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
    generator,
    seen_semantic_embeddings,
    unseen_semantic_embeddings,
    seen_classes,
    unseen_classes,
    train_loader,
    test_loader,
    device,
    config,
):
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

    from models.zsl_classifier import ZSLClassifier

    base_classifier = ZSLClassifier(num_total).to(device)
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

    seen_dataset = train_loader.dataset
    seen_indices = list(range(len(seen_dataset)))

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
            split_gen = torch.Generator().manual_seed(42)
            synth_train, synth_val = random_split(synth_dataset, [synth_train_size, synth_val_size], generator=split_gen)
            synth_train_loader = DataLoader(synth_train, batch_size=zsl_batch_size, shuffle=True, num_workers=0)

            combined_train_dataset = _CombinedDataset(seen_dataset, synth_train.dataset, num_seen)
            combined_train_loader = DataLoader(combined_train_dataset, batch_size=zsl_batch_size, shuffle=True, num_workers=0, drop_last=True)

            synth_val_loader = DataLoader(synth_val, batch_size=100, shuffle=False, num_workers=0)

        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(combined_train_loader, desc=f"GZSL Epoch {epoch + 1}/{zsl_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            images = zsl_augment(images)

            optimizer.zero_grad(set_to_none=True)

            if mixup_alpha > 0 and np.random.random() < 0.5:
                mixed_images, labels_a, labels_b, lam = _mixup_data(images, labels, mixup_alpha)
                outputs = classifier(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = classifier(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({"Loss": f"{train_loss / max(1, pbar.n + 1):.4f}", "Acc": f"{100.0 * correct / total:.2f}%"})

        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in synth_val_loader:
                images, labels = images.to(device), labels.to(device)
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


class _CombinedDataset(Dataset):
    """Combines real seen-class dataset and synthetic unseen-class dataset into one."""

    def __init__(self, seen_dataset, synth_dataset, seen_class_offset):
        self.seen_images = []
        self.seen_labels = []
        for i in range(len(seen_dataset)):
            img, label = seen_dataset[i]
            self.seen_images.append(img)
            self.seen_labels.append(label)

        self.synth_images = []
        self.synth_labels = []
        for i in range(len(synth_dataset)):
            img, label = synth_dataset[i]
            self.synth_images.append(img)
            self.synth_labels.append(label + seen_class_offset)

        self.all_images = self.seen_images + self.synth_images
        self.all_labels = self.seen_labels + self.synth_labels
        self.length = len(self.all_images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.all_images[idx], self.all_labels[idx]


def evaluate_gzsl(
    classifier,
    seen_classes,
    unseen_classes,
    test_loader,
    seen_train_loader,
    class_names,
    device,
    results_dir="results",
):
    """
    Evaluate GZSL: classifier must predict correctly for both seen and unseen classes.

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
        for images, labels in tqdm(seen_train_loader, desc="Evaluating seen classes"):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            batch_size = labels.size(0)
            seen_total += batch_size
            all_total += batch_size
            correct_mask = predicted.eq(labels)
            seen_correct += correct_mask.sum().item()
            all_correct += correct_mask.sum().item()
            for i in range(batch_size):
                lbl = labels[i].item()
                per_class_correct[lbl] += (predicted[i] == lbl).item()
                per_class_total[lbl] += 1
                confusion_matrix[lbl][predicted[i]] += 1

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating unseen classes"):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = outputs.max(1)
            batch_size = labels.size(0)
            unseen_total += batch_size
            all_total += batch_size
            correct_mask = predicted.eq(labels)
            unseen_correct += correct_mask.sum().item()
            all_correct += correct_mask.sum().item()
            for i in range(batch_size):
                true_lbl = labels[i].item()
                pred_lbl = predicted[i].item()
                per_class_correct[true_lbl] += (pred_lbl == true_lbl).item()
                per_class_total[true_lbl] += 1
                confusion_matrix[true_lbl][pred_lbl] += 1

    seen_acc = 100.0 * seen_correct / seen_total if seen_total > 0 else 0
    unseen_acc = 100.0 * unseen_correct / unseen_total if unseen_total > 0 else 0
    harmonic_mean = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) > 0 else 0
    overall_acc = 100.0 * all_correct / all_total if all_total > 0 else 0

    per_class_acc = [100.0 * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in range(num_total)]
    mean_class_acc = float(np.mean([a for a in per_class_acc if a > 0]))

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


def plot_gzsl_results(gzsl_metrics, save_dir="results"):
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
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

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
