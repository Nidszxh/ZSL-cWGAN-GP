"""
Data Loader Module
Day 2: CIFAR-100 with seen/unseen split
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class FilteredCIFAR100(datasets.CIFAR100):
    """CIFAR-100 filtered to specific classes"""
    
    def __init__(self, *args, allowed_classes=None, transform=None, **kwargs):
        super().__init__(*args, transform=transform, **kwargs)
        
        if allowed_classes is not None:
            # Filter data
            mask = np.isin(self.targets, allowed_classes)
            indices = np.where(mask)[0]
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]
            
            # Remap class indices
            self.org_to_new = {
                old_cls: i for i, old_cls in enumerate(sorted(set(self.targets)))
            }
            self.targets = [self.org_to_new[target] for target in self.targets]


def get_class_split(
    num_classes: int = 100,
    seen_count: int = 80,
    cache_dir: str = "./cache",
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get or create seen/unseen class split
    
    Returns:
        seen_classes: Array of seen class indices
        unseen_classes: Array of unseen class indices
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_file = cache_dir / 'class_split.json'
    
    if split_file.exists():
        print(f"Loading class split from {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
            seen_classes = np.array(split_data['seen'])
            unseen_classes = np.array(split_data['unseen'])
    else:
        print(f"Creating new class split (seed={seed})")
        rng = np.random.RandomState(seed)
        all_classes = np.arange(num_classes)
        rng.shuffle(all_classes)
        
        seen_classes = all_classes[:seen_count]
        unseen_classes = all_classes[seen_count:]
        
        # Save split
        with open(split_file, 'w') as f:
            json.dump({
                'seen': seen_classes.tolist(),
                'unseen': unseen_classes.tolist(),
                'seed': seed
            }, f, indent=2)
        
        print(f"Saved class split to {split_file}")
    
    print(f"Seen classes: {len(seen_classes)}, Unseen classes: {len(unseen_classes)}")
    
    return seen_classes, unseen_classes


def get_data_loaders(
    config: dict,
    seen_classes: np.ndarray
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation data loaders
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        class_info: Dictionary with class mapping info
    """
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    train_dataset = FilteredCIFAR100(
        root=config['paths']['data_root'],
        train=True,
        download=True,
        transform=transform_train,
        allowed_classes=seen_classes
    )
    
    # Split into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['experiment']['seed'])
    )
    
    # Update val subset transform
    val_dataset = FilteredCIFAR100(
        root=config['paths']['data_root'],
        train=True,
        download=True,
        transform=transform_val,
        allowed_classes=seen_classes
    )
    val_subset.dataset = val_dataset
    
    # Create loaders
    num_workers = min(config['dataset']['num_workers'], 4)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Class info
    class_info = {
        'org_to_new': train_dataset.org_to_new,
        'new_to_org': {v: k for k, v in train_dataset.org_to_new.items()},
        'num_seen_classes': len(train_dataset.org_to_new)
    }
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")
    print(f"Seen classes (remapped): {class_info['num_seen_classes']}")
    
    return train_loader, val_loader, class_info


def get_test_loader(
    config: dict,
    unseen_classes: np.ndarray
) -> Tuple[DataLoader, dict]:
    """
    Create test data loader for unseen classes
    
    Returns:
        test_loader: Test data loader
        class_info: Dictionary with class mapping info
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = FilteredCIFAR100(
        root=config['paths']['data_root'],
        train=False,
        download=True,
        transform=transform_test,
        allowed_classes=unseen_classes
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=min(config['dataset']['num_workers'], 4),
        pin_memory=True
    )
    
    class_info = {
        'org_to_new': test_dataset.org_to_new,
        'new_to_org': {v: k for k, v in test_dataset.org_to_new.items()},
        'num_unseen_classes': len(test_dataset.org_to_new)
    }
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Unseen classes: {class_info['num_unseen_classes']}")
    
    return test_loader, class_info


if __name__ == "__main__":
    # Test data loading
    import yaml
    
    print("Testing data loader...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get class split
    seen_classes, unseen_classes = get_class_split(
        num_classes=config['dataset']['num_classes'],
        seen_count=config['dataset']['seen_classes'],
        cache_dir=config['paths']['cache_dir'],
        seed=config['experiment']['seed']
    )
    
    # Get data loaders
    train_loader, val_loader, class_info = get_data_loaders(config, seen_classes)
    
    # Test batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"✓ Batch images shape: {images.shape}")
    print(f"✓ Batch labels shape: {labels.shape}")
    print(f"✓ Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"✓ Unique labels in batch: {labels.unique().tolist()}")
    
    print("\n✓ Data loader test passed!")