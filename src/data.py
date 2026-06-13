from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _limit_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def get_mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Ambil MNIST dan buat DataLoader untuk train/test."""
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = _limit_dataset(train_dataset, max_train_samples)
    test_dataset = _limit_dataset(test_dataset, max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
