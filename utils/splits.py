"""
splits.py
---------
提供加载各数据分割的 DataLoader 工厂函数。
所有脚本通过此模块获取一致的数据集，避免分割泄漏。

CIFAR-10 归一化参数（在完整训练集上计算）：
  mean = (0.4914, 0.4822, 0.4465)
  std  = (0.2023, 0.1994, 0.2010)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# ---------- CIFAR-10 标准归一化 ----------
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

# SVHN 归一化（ImageNet 统计近似即可）
_SVHN_TRANSFORM = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _load_splits(splits_dir: str) -> dict:
    path = os.path.join(splits_dir, "splits.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到分割文件 {path}。请先运行 data/prepare_data.py。"
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_loaders(cfg: dict, root: str, batch_size: int = None, num_workers: int = 4):
    """
    返回 dict，包含各数据分割的 DataLoader：
      - train       : 带 augmentation（40,000）
      - holdout     : 无 augmentation（5,000，用于 B̂_local）
      - calibration : 无 augmentation（5,000，分位数校准）
      - test_id     : 无 augmentation（5,000，ID 测试）
    注：无 Val 集，backbone 用余弦退火收敛，保存最终 epoch。
    """
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
    """返回 SVHN 测试集 DataLoader（OOD 评估）。"""
    data_dir = os.path.join(root, cfg.get("data_dir", "./data/raw"))
    bs = batch_size or cfg.get("batch_size", 128)
    dataset = torchvision.datasets.SVHN(
        root=data_dir, split='test', download=False, transform=_SVHN_TRANSFORM
    )
    # 取前 5000 以匹配 ID test 集大小
    subset = Subset(dataset, list(range(5000)))
    return DataLoader(subset, batch_size=bs, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_cifar10c_loader(cfg: dict, root: str, corruption: str, severity: int,
                         batch_size: int = None, num_workers: int = 4):
    """
    返回单个 CIFAR-10-C 腐蚀类型+严重程度的 DataLoader。
    CIFAR-10-C 文件结构：每个腐蚀类型一个 .npy 文件，共 10,000*5 张图。
    labels.npy 共 50,000 个标签（每个严重程度 10,000 张顺序叠加）。
    """
    from torch.utils.data import TensorDataset

    cifar10c_dir = os.path.join(root, cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C"))
    bs = batch_size or cfg.get("batch_size", 128)

    images_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
    labels_path = os.path.join(cifar10c_dir, "labels.npy")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"未找到 {images_path}，请手动下载 CIFAR-10-C。")

    images = np.load(images_path)   # (50000, 32, 32, 3), uint8
    labels = np.load(labels_path)   # (50000,), int64

    # severity 1~5，每段 10000 张
    start = (severity - 1) * 10000
    end   = severity * 10000
    imgs  = images[start:end]       # (10000, 32, 32, 3)
    lbls  = labels[start:end]

    # 归一化
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    imgs_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    imgs_tensor = (imgs_tensor - mean) / std

    dataset = TensorDataset(imgs_tensor, torch.from_numpy(lbls).long())
    return DataLoader(dataset, batch_size=bs, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def collect_features_and_labels(loader, device="cuda"):
    """
    辅助函数：遍历 loader，返回 (images_tensor, labels_tensor)。
    用于特征提取前的批量收集。
    """
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
