"""
feature_extractor.py
--------------------
从训练好的 ResNet-18 中提取倒数第二层（avgpool 后）的 512 维特征向量。
特征会被缓存到磁盘，避免重复计算。

用法：
    python models/feature_extractor.py --config configs/default.yaml --model backbone_single
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.splits import get_loaders, get_ood_loader
from models.backbone import load_model, load_ensemble, build_resnet18


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Hook-based 特征提取                                                 #
# ------------------------------------------------------------------ #

class FeatureExtractorHook:
    """
    通过 forward hook 提取 ResNet-18 avgpool 输出（512 维）。
    支持批量推理。
    """
    def __init__(self, model: nn.Module, device="cuda"):
        self.model  = model.to(device)
        self.device = device
        self._features = []
        # 挂载 hook 到 avgpool
        self._handle = model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # output: (B, 512, 1, 1) → 压平为 (B, 512)
        self._features.append(output.squeeze(-1).squeeze(-1).detach().cpu())

    @torch.no_grad()
    def extract(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        遍历 loader，返回 (features, labels)。
        features: (N, 512)
        labels:   (N,)
        """
        self.model.eval()
        self._features = []
        all_labels = []
        for x, y in tqdm(loader, desc="提取特征", leave=False):
            x = x.to(self.device)
            self.model(x)          # forward 触发 hook
            all_labels.append(y)
        features = torch.cat(self._features, dim=0)
        labels   = torch.cat(all_labels, dim=0)
        return features, labels

    def close(self):
        self._handle.remove()


# ------------------------------------------------------------------ #
#  缓存管理                                                            #
# ------------------------------------------------------------------ #

def _cache_path(root: str, model_name: str, split: str) -> str:
    cache_dir = os.path.join(root, "data", "features")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{model_name}_{split}.pt")


def extract_and_cache(
    model: nn.Module,
    loaders: dict,
    root: str,
    model_name: str,
    device="cuda",
    splits=("train", "holdout", "calibration", "test_id"),
    ood_loader=None,
) -> dict:
    """
    对指定 splits 提取特征并缓存到磁盘。
    返回 {split: {"features": Tensor, "labels": Tensor}}。
    """
    extractor = FeatureExtractorHook(model, device)
    results = {}

    for split in splits:
        cache = _cache_path(root, model_name, split)
        if os.path.exists(cache):
            print(f"  [{split}] 已有缓存，加载中...")
            results[split] = torch.load(cache, weights_only=True)
            continue
        print(f"  [{split}] 提取特征...")
        feats, labels = extractor.extract(loaders[split])
        data = {"features": feats, "labels": labels}
        torch.save(data, cache)
        results[split] = data
        print(f"  [{split}] 特征形状: {feats.shape}  已保存到 {cache}")

    if ood_loader is not None:
        split = "test_ood"
        cache = _cache_path(root, model_name, split)
        if os.path.exists(cache):
            print(f"  [{split}] 已有缓存，加载中...")
            results[split] = torch.load(cache, weights_only=True)
        else:
            print(f"  [{split}] 提取特征...")
            feats, labels = extractor.extract(ood_loader)
            data = {"features": feats, "labels": labels}
            torch.save(data, cache)
            results[split] = data
            print(f"  [{split}] 特征形状: {feats.shape}")

    extractor.close()
    return results


def load_cached_features(root: str, model_name: str, split: str) -> dict:
    """加载已缓存的特征，返回 {"features": Tensor, "labels": Tensor}。"""
    cache = _cache_path(root, model_name, split)
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"未找到特征缓存 {cache}。请先运行 feature_extractor.py。"
        )
    return torch.load(cache, weights_only=True)


# ------------------------------------------------------------------ #
#  Ensemble 特征：取 5 个模型特征的均值                                #
# ------------------------------------------------------------------ #

def extract_imagenet_pretrained_features(
    root: str,
    loaders: dict,
    device="cuda",
    ood_loader=None,
) -> dict:
    """
    用 ImageNet 预训练 ResNet-18（冻结，不需训练）提取特征。
    缓存以 "imagenet_pretrained" 为 model_name。
    用于 B̂_local 消融：比较主模型特征 vs ImageNet 预训练特征。
    """
    import torchvision.models as tv_models

    model_name = "imagenet_pretrained"
    # 检查是否已完整缓存
    splits = ("holdout", "test_id", "calibration")
    all_cached = all(
        os.path.exists(_cache_path(root, model_name, s)) for s in splits
    )
    if all_cached:
        print("imagenet_pretrained 特征已全部缓存，直接加载。")
        results = {s: load_cached_features(root, model_name, s) for s in splits}
        if ood_loader is not None and os.path.exists(_cache_path(root, model_name, "test_ood")):
            results["test_ood"] = load_cached_features(root, model_name, "test_ood")
        return results

    # 加载 ImageNet 预训练权重，移除分类头（只要 512 维特征）
    pretrained = tv_models.resnet18(weights="IMAGENET1K_V1")
    pretrained.fc = nn.Identity()
    pretrained = pretrained.to(device)
    pretrained.eval()
    print("使用 ImageNet 预训练 ResNet-18 提取特征（冻结，无微调）")

    return extract_and_cache(
        pretrained, loaders, root, model_name, device,
        splits=splits, ood_loader=ood_loader,
    )


def extract_ensemble_features(
    models: list,
    loaders: dict,
    root: str,
    cfg: dict,
    device="cuda",
    splits=("train", "holdout", "calibration", "test_id"),
    ood_loader=None,
) -> dict:
    """
    逐个提取 5 个 ensemble 模型的特征，然后取均值。
    缓存以 "ensemble_mean" 为 model_name。
    """
    model_name = "ensemble_mean"
    # 先检查是否已有完整缓存
    all_cached = all(
        os.path.exists(_cache_path(root, model_name, s)) for s in splits
    )
    if all_cached:
        print("ensemble_mean 特征已全部缓存，直接加载。")
        return {s: load_cached_features(root, model_name, s) for s in splits}

    # 逐模型提取，累积后取均值
    feature_accum = {s: None for s in splits}
    labels_store  = {s: None for s in splits}

    for idx, m in enumerate(models):
        print(f"  Ensemble 模型 {idx+1}/{len(models)} 提取特征...")
        extractor = FeatureExtractorHook(m, device)
        for split in splits:
            feats, labels = extractor.extract(loaders[split])
            if feature_accum[split] is None:
                feature_accum[split] = feats.float()
                labels_store[split]  = labels
            else:
                feature_accum[split] += feats.float()
        extractor.close()

    results = {}
    for split in splits:
        mean_feats = feature_accum[split] / len(models)
        data = {"features": mean_feats, "labels": labels_store[split]}
        cache = _cache_path(root, model_name, split)
        torch.save(data, cache)
        results[split] = data
        print(f"  [{split}] ensemble_mean 特征形状: {mean_feats.shape}")

    return results


# ------------------------------------------------------------------ #
#  主入口                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--model",   default="backbone_single",
                        help="checkpoint 名称（不含 .pt 后缀）")
    parser.add_argument("--all",     action="store_true",
                        help="提取所有模型（single + ensemble + reference）的特征")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders     = get_loaders(cfg, root)
    ood_loader  = get_ood_loader(cfg, root)
    ckpt_dir    = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))

    if args.all:
        model_names = (
            ["backbone_single", "backbone_mcdropout"]
            + [f"ensemble_{i}_seed{s}" for i, s in enumerate(cfg["ensemble_seeds"])]
        )
    else:
        model_names = [args.model]

    for name in model_names:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 未找到 checkpoint: {ckpt_path}")
            continue
        print(f"\n=== 提取特征: {name} ===")
        model = load_model(ckpt_path, device)
        extract_and_cache(model, loaders, root, name, device, ood_loader=ood_loader)

    print("\n特征提取完成！")


if __name__ == "__main__":
    main()
