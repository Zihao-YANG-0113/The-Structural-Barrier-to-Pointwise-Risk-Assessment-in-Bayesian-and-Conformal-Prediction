"""
splits.py
---------
DataLoader factory for the various data splits.
All scripts go through this module so that the splits stay consistent
and never leak across train / hold-out / calibration / test_id.

CIFAR-10 normalisation (computed on the full training set):
  mean = (0.4914, 0.4822, 0.4465)
  std  = (0.2023, 0.1994, 0.2010)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# ---------- CIFAR-10 standard normalisation ----------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# SVHN normalisation (CIFAR-10 stats are a fine approximation here).
_SVHN_TRANSFORM = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _load_splits(splits_dir: str) -> dict:
    path = os.path.join(splits_dir, "splits.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find split file {path}. Run data/prepare_data.py first."
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_loaders(cfg: dict, root: str, batch_size: int = None, num_workers: int = 4):
    """Return a dict with one DataLoader per split:
      - train       : with augmentation (40,000)
      - holdout     : no augmentation (5,000) — feeds B̂_local
      - calibration : no augmentation (5,000) — quantile calibration
      - test_id     : no augmentation (5,000) — ID test set
    No validation split: the backbone is trained with cosine annealing
    and the final-epoch checkpoint is the natural convergence point."""
    data_dir   = os.path.join(root, cfg.get("data_dir", "./data/raw"))
    splits_dir = os.path.join(root, "data")
    bs = batch_size or cfg.get("batch_size", 128)

    splits = _load_splits(splits_dir)

    cifar_train_full = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=_TRAIN_TRANSFORM
    )
    cifar_train_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=_EVAL_TRANSFORM
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=_EVAL_TRANSFORM
    )

    loaders = {
        "train": DataLoader(
            Subset(cifar_train_full, splits["train"]),
            batch_size=bs, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        ),
        "holdout": DataLoader(
            Subset(cifar_train_eval, splits["holdout"]),
            batch_size=bs, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "calibration": DataLoader(
            Subset(cifar_train_eval, splits["calibration"]),
            batch_size=bs, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test_id": DataLoader(
            Subset(cifar_test, splits["test_id"]),
            batch_size=bs, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders


def get_ood_loader(cfg: dict, root: str, batch_size: int = None, num_workers: int = 4):
    """Return the SVHN test loader for OOD evaluation."""
    data_dir = os.path.join(root, cfg.get("data_dir", "./data/raw"))
    bs = batch_size or cfg.get("batch_size", 128)
    dataset = torchvision.datasets.SVHN(
        root=data_dir, split='test', download=False, transform=_SVHN_TRANSFORM
    )
    # Take the first 5000 to match the ID test set size.
    subset = Subset(dataset, list(range(5000)))
    return DataLoader(subset, batch_size=bs, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_cifar10c_loader(cfg: dict, root: str, corruption: str, severity: int,
                         batch_size: int = None, num_workers: int = 4):
    """Return a DataLoader for one CIFAR-10-C corruption + severity.
    CIFAR-10-C layout: one .npy per corruption containing 10,000 × 5 = 50,000
    images, with labels.npy holding 50,000 labels (severities are concatenated
    in order)."""
    from torch.utils.data import TensorDataset

    cifar10c_dir = os.path.join(root, cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C"))
    bs = batch_size or cfg.get("batch_size", 128)

    images_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
    labels_path = os.path.join(cifar10c_dir, "labels.npy")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Could not find {images_path}. "
                                f"Download CIFAR-10-C manually.")

    images = np.load(images_path)   # (50000, 32, 32, 3), uint8
    labels = np.load(labels_path)   # (50000,), int64

    # Severities 1..5, 10,000 images each.
    start = (severity - 1) * 10000
    end   = severity * 10000
    imgs  = images[start:end]       # (10000, 32, 32, 3)
    lbls  = labels[start:end]

    # Normalise
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    imgs_tensor = (imgs_tensor - mean) / std

    dataset = TensorDataset(imgs_tensor, torch.from_numpy(lbls).long())
    return DataLoader(dataset, batch_size=bs, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def collect_features_and_labels(loader, device="cuda"):
    """Helper: iterate over a loader and return (images_tensor, labels_tensor).
    Used to batch-collect inputs prior to feature extraction."""
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
