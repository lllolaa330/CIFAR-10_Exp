"""CIFAR-10 datasets and reproducible train/validation split."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(augmentation: Mapping | None = None):
    """Build stochastic train and deterministic evaluation transforms."""
    augmentation = dict(augmentation or {})
    train_steps = []

    crop_padding = int(augmentation.get("random_crop_padding", 0))
    if crop_padding > 0:
        train_steps.append(transforms.RandomCrop(32, padding=crop_padding))
    if augmentation.get("horizontal_flip", False):
        train_steps.append(transforms.RandomHorizontalFlip())

    common_steps = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_transform = transforms.Compose(train_steps + common_steps)
    eval_transform = transforms.Compose(common_steps)
    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    seed: int = 266978,
    num_workers: int = 0,
    augmentation: Mapping | None = None,
    download: bool = True,
    pin_memory: bool = False,
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_transform, eval_transform = build_transforms(augmentation)
    data_dir = str(data_dir)

    # Separate sources prevent validation from inheriting random augmentation.
    train_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    val_source = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )

    num_samples = len(train_source)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_samples, generator=split_generator).tolist()

    train_dataset = Subset(train_source, indices[:num_train])
    val_dataset = Subset(val_source, indices[num_train:])

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=shuffle_generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader, test_loader
