"""
backbone.py
-----------
CIFAR-10 backbone networks (ResNet-20 default, plus a WideResNet-28-10
option for experiments outside the dissertation's main pipeline).

The dissertation (Chapter 4, §4.2.1) uses ResNet-20 — the standard
3-stage / 3-blocks-per-stage CIFAR-10 ResNet from He et al. 2016 — as
the vision backbone. `build_resnet20` below implements that architecture
exactly (depth = 6n + 2 with n = 3 → 20 layers; channel widths
[16, 32, 64]; final feature dim = 64).

A second builder, `build_resnet18`, is kept for reproducibility of
auxiliary ablations that used the torchvision ResNet-18 with 32×32 input
adaptation. Set `backbone: resnet20` (default) in `configs/default.yaml`
for the main paper pipeline.

Usage:
    # Single model
    python utils/backbone.py --config configs/default.yaml

    # Deep Ensemble (5 seeds)
    python utils/backbone.py --config configs/default.yaml --mode ensemble

    # MC-Dropout
    python utils/backbone.py --config configs/default.yaml --mode mc_dropout

Note: B̂_local uses the audited model's own penultimate-layer features
for the k-NN, so no separate reference encoder is needed. ImageNet
pretrained features are used for an ablation only.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.models as tv_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.splits import get_loaders


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  模型构建                                                            #
# ------------------------------------------------------------------ #

def build_resnet18(num_classes: int = 10, dropout_rate: float = 0.0) -> nn.Module:
    """CIFAR-10-adapted torchvision ResNet-18. Used only by auxiliary ablations.
    First conv is changed to 3×3 stride=1 and the maxpool is dropped to suit
    32×32 inputs. dropout_rate > 0 inserts Dropout before the FC head."""
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if dropout_rate > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
    else:
        model.fc = nn.Linear(512, num_classes)

    return model


# ------------------------------------------------------------------ #
#  ResNet-20 — CIFAR-10 main backbone used in Chapter 4              #
# ------------------------------------------------------------------ #

class _CifarBasicBlock(nn.Module):
    """Pre-activation basic block for the CIFAR ResNet family."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.dropout_rate = dropout_rate

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    """
    CIFAR-style ResNet (He et al. 2016, §4.2): three stages with
    [16, 32, 64] channels, n basic blocks per stage, total depth = 6n + 2.
    n = 3 → ResNet-20 (default in this codebase). The penultimate-layer
    feature dim is 64 — this is the space in which B̂_local's k-NN is
    computed for the vision experiments.
    """
    def __init__(self, depth: int = 20, num_classes: int = 10,
                 dropout_rate: float = 0.0):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (e.g. 20, 32, 44)."
        n = (depth - 2) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32, n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64, n, stride=2, dropout_rate=dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.feat_dim = 64

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, n_blocks, stride, dropout_rate):
        layers = [_CifarBasicBlock(self.in_planes, planes, stride, dropout_rate)]
        self.in_planes = planes
        for _ in range(1, n_blocks):
            layers.append(_CifarBasicBlock(planes, planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_resnet20(num_classes: int = 10, dropout_rate: float = 0.0) -> nn.Module:
    """ResNet-20 for CIFAR-10 — the dissertation's main vision backbone."""
    return CifarResNet(depth=20, num_classes=num_classes, dropout_rate=dropout_rate)


# ------------------------------------------------------------------ #
#  WideResNet-28-10                                                   #
# ------------------------------------------------------------------ #

class _WideBasicBlock(nn.Module):
    """WideResNet 基本残差块。"""
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.dropout_rate = dropout_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    """
    WideResNet-d-w for CIFAR-10 (32×32).
    默认 d=28, w=10 → WideResNet-28-10.
    特征维度 = 64 * widen_factor (= 640 for w=10).
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = nStages[0]

        self.conv1 = nn.Conv2d(3, nStages[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(nStages[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nStages[3], num_classes)

        # 记录特征维度（供 feature_extractor 使用）
        self.feat_dim = nStages[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, n_blocks, stride, dropout_rate):
        layers = [_WideBasicBlock(self.in_planes, planes, stride, dropout_rate)]
        self.in_planes = planes
        for _ in range(1, n_blocks):
            layers.append(_WideBasicBlock(planes, planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_model(cfg: dict, dropout_rate: float = 0.0) -> nn.Module:
    """Build the backbone selected by `backbone:` in the YAML config.
    Defaults to ResNet-20 (dissertation main pipeline)."""
    backbone = cfg.get("backbone", "resnet20")
    num_classes = cfg.get("num_classes", 10)

    if backbone == "resnet20":
        return build_resnet20(num_classes, dropout_rate)
    elif backbone == "resnet18":
        return build_resnet18(num_classes, dropout_rate)
    elif backbone == "wrn28_10":
        return WideResNet(depth=28, widen_factor=10, num_classes=num_classes,
                          dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


# ------------------------------------------------------------------ #
#  训练循环                                                            #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, 100.0 * correct / total


# ------------------------------------------------------------------ #
#  单模型训练                                                          #
# ------------------------------------------------------------------ #

def train_model(cfg: dict, root: str, seed: int, save_name: str,
                dropout_rate: float = 0.0, use_amp: bool = True):
    """
    训练单个 ResNet-18，保存最终 epoch checkpoint。
    使用余弦退火调度（200 epoch），自然收敛，无需 Val 集早停。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}  (seed={seed})")

    loaders = get_loaders(cfg, root)
    model = build_model(cfg, dropout_rate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["lr"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"], nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    ckpt_dir = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{save_name}.pt")

    epochs = cfg["epochs"]
    train_loss, train_acc = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, scaler
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}%"
            )

    # 保存最终 epoch（余弦退火末尾为自然收敛点）
    torch.save({
        "epoch":        epochs,
        "state_dict":   model.state_dict(),
        "train_acc":    train_acc,
        "cfg":          cfg,
        "seed":         seed,
        "dropout_rate": dropout_rate,
    }, save_path)
    print(f"训练完成。最终训练准确率: {train_acc:.2f}%  已保存到 {save_path}")
    return save_path


# ------------------------------------------------------------------ #
#  加载已训练模型（供其他模块使用）                                    #
# ------------------------------------------------------------------ #

def load_model(ckpt_path: str, device="cuda") -> nn.Module:
    """加载已保存的 checkpoint，根据 cfg 中的 backbone 字段自动选择架构。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    dr   = ckpt.get("dropout_rate", 0.0)
    model = build_model(cfg, dr).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_ensemble(root: str, cfg: dict, device="cuda") -> list:
    """加载 5 个 ensemble 模型，返回模型列表。"""
    ckpt_dir = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))
    models = []
    for i, seed in enumerate(cfg["ensemble_seeds"]):
        path = os.path.join(ckpt_dir, f"ensemble_{i}_seed{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {path}，请先运行 ensemble 训练。")
        models.append(load_model(path, device))
    return models


# ------------------------------------------------------------------ #
#  主入口                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["single", "ensemble", "mc_dropout"],
        default="single",
        help=(
            "single: 训练一个模型（backbone_single）; "
            "ensemble: 训练 5 个 ensemble 模型; "
            "mc_dropout: 训练带 dropout 的模型（backbone_mcdropout）"
        ),
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.mode == "single":
        train_model(cfg, root, seed=cfg["seed"], save_name="backbone_single")

    elif args.mode == "ensemble":
        for i, seed in enumerate(cfg["ensemble_seeds"]):
            print(f"\n===== 训练 Ensemble 模型 {i+1}/5  seed={seed} =====")
            train_model(cfg, root, seed=seed, save_name=f"ensemble_{i}_seed{seed}")

    elif args.mode == "mc_dropout":
        train_model(cfg, root, seed=cfg["seed"], save_name="backbone_mcdropout",
                    dropout_rate=cfg.get("dropout_rate", 0.1))


if __name__ == "__main__":
    main()
