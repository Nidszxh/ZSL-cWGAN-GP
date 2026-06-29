"""
ZSL Classifier Training and Evaluation.

Uses a pretrained-backbone classifier trained on synthetic unseen-class images.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm

from src.models.zsl_classifier import ZSLClassifier, build_classifier_from_config

zsl_augment = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
)


def _generate_synthetic_data(
    generator: torch.nn.Module,
    num_samples: int,
    num_classes: int,
    semantic_embeddings: torch.Tensor,
    nz: int,
    batch_size: int,
    device: torch.device,
) -> TensorDataset:
    """Per-class balanced synthetic data generation."""
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


def _mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple:
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def train_zsl_classifier(
    generator: torch.nn.Module,
    unseen_semantic_embeddings: torch.Tensor,
    unseen_classes: np.ndarray,
    device: torch.device,
    config: dict,
) -> torch.nn.Module:
    num_unseen = len(unseen_classes)
    nz = config["model"]["generator"]["nz"]
    samples_per_class = config["evaluation"]["synthetic_samples_per_class"]
    zsl_epochs = config["evaluation"]["zsl_epochs"]
    zsl_lr = config["evaluation"]["zsl_lr"]
    zsl_batch_size = config["evaluation"]["zsl_batch_size"]
    checkpoints_dir = config["paths"]["checkpoints_dir"]

    regenerate_every = config["evaluation"].get("zsl_regenerate_every", 1)
    mixup_alpha = config["evaluation"].get("zsl_mixup_alpha", 0.2)
    label_smoothing = config["evaluation"].get("zsl_label_smoothing", 0.1)

    total_samples = samples_per_class * num_unseen
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    classifier = build_classifier_from_config(num_unseen, config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(classifier.parameters(), lr=zsl_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0
    patience = 10
    epochs_without_improv = 0
    synth_train_loader = None
    synth_val_loader = None

    print("\nTraining ZSL classifier (per-epoch synthetic data regeneration)...")
    for epoch in range(zsl_epochs):
        if epoch % regenerate_every == 0 or synth_train_loader is None:
            print(f"  Regenerating synthetic data (epoch {epoch + 1})...")
            full_dataset = _generate_synthetic_data(
                generator,
                total_samples,
                num_unseen,
                unseen_semantic_embeddings,
                nz,
                batch_size=zsl_batch_size,
                device=device,
            )
            split_gen = torch.Generator().manual_seed(42)
            synth_train, synth_val = random_split(full_dataset, [train_size, val_size], generator=split_gen)
            synth_train_loader = DataLoader(synth_train, batch_size=zsl_batch_size, shuffle=True)
            synth_val_loader = DataLoader(synth_val, batch_size=100, shuffle=False)

        classifier.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(synth_train_loader, desc=f"ZSL Epoch {epoch + 1}/{zsl_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
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
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
            torch.save(classifier.state_dict(), Path(checkpoints_dir) / "best_zsl_classifier.pth")
            epochs_without_improv = 0
        else:
            epochs_without_improv += 1
            if epochs_without_improv >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    classifier.load_state_dict(
        torch.load(Path(checkpoints_dir) / "best_zsl_classifier.pth", weights_only=True)
    )
    return classifier


def evaluate_zsl(
    classifier: torch.nn.Module,
    test_loader: DataLoader,
    unseen_classes: np.ndarray,
    class_names: list,
    device: torch.device,
    results_dir: str = "results",
) -> dict:
    num_unseen = len(unseen_classes)
    classifier.eval()

    correct = 0
    total = 0
    class_correct = [0] * num_unseen
    class_total = [0] * num_unseen
    top5_correct = 0
    confusion_matrix = torch.zeros(num_unseen, num_unseen)

    print("\nEvaluating on real unseen class data...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
                confusion_matrix[label][predicted[i]] += 1

    top1_accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * top5_correct / total
    per_class_acc = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_unseen)
    ]
    mean_class_acc = float(np.mean(per_class_acc))

    for i in range(num_unseen):
        if class_total[i] > 0:
            confusion_matrix[i] = confusion_matrix[i] / class_total[i]

    print(f"\n{'=' * 70}")
    print("ZSL RESULTS")
    print(f"{'=' * 70}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"Mean Class Accuracy: {mean_class_acc:.2f}%")

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_class_accuracy": mean_class_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.cpu().numpy(),
        "class_names": [class_names[cls] for cls in unseen_classes],
    }
