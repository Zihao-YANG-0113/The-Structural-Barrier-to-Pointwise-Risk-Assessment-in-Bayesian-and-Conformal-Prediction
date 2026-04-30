"""
feature_extractor.py
--------------------
Extract penultimate-layer (post-avgpool) features from a trained backbone
and cache them on disk so that subsequent k-NN queries are cheap.

Usage:
    python utils/feature_extractor.py --config configs/default.yaml --model backbone_single
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
from utils.splits import get_loaders, get_ood_loader
from utils.backbone import load_model, load_ensemble, build_resnet18


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Hook-based feature extraction                                      #
# ------------------------------------------------------------------ #

class FeatureExtractorHook:
    """Extract avgpool outputs of a CIFAR backbone via a forward hook
    (e.g. 64-d for ResNet-20, 512-d for ResNet-18). Batch-friendly."""
    def __init__(self, model: nn.Module, device="cuda"):
        self.model  = model.to(device)
        self.device = device
        self._features = []
        # Hook on avgpool
        self._handle = model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # output: (B, D, 1, 1) → flatten to (B, D)
        self._features.append(output.squeeze(-1).squeeze(-1).detach().cpu())

    @torch.no_grad()
    def extract(self, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Iterate over the loader and return (features, labels).
        features: (N, D)
        labels:   (N,)"""
        self.model.eval()
        self._features = []
        all_labels = []
        for x, y in tqdm(loader, desc="Extracting features", leave=False):
            x = x.to(self.device)
            self.model(x)          # forward triggers the hook
            all_labels.append(y)
        features = torch.cat(self._features, dim=0)
        labels   = torch.cat(all_labels, dim=0)
        return features, labels

    def close(self):
        self._handle.remove()


# ------------------------------------------------------------------ #
#  Cache management                                                   #
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
    """Extract features for the given splits and cache them on disk.
    Returns {split: {"features": Tensor, "labels": Tensor}}."""
    extractor = FeatureExtractorHook(model, device)
    results = {}

    for split in splits:
        cache = _cache_path(root, model_name, split)
        if os.path.exists(cache):
            print(f"  [{split}] cache hit, loading...")
            results[split] = torch.load(cache, weights_only=True)
            continue
        print(f"  [{split}] extracting features...")
        feats, labels = extractor.extract(loaders[split])
        data = {"features": feats, "labels": labels}
        torch.save(data, cache)
        results[split] = data
        print(f"  [{split}] feature shape: {feats.shape}  saved to {cache}")

    if ood_loader is not None:
        split = "test_ood"
        cache = _cache_path(root, model_name, split)
        if os.path.exists(cache):
            print(f"  [{split}] cache hit, loading...")
            results[split] = torch.load(cache, weights_only=True)
        else:
            print(f"  [{split}] extracting features...")
            feats, labels = extractor.extract(ood_loader)
            data = {"features": feats, "labels": labels}
            torch.save(data, cache)
            results[split] = data
            print(f"  [{split}] feature shape: {feats.shape}")

    extractor.close()
    return results


def load_cached_features(root: str, model_name: str, split: str) -> dict:
    """Load a cached feature file. Returns {"features": Tensor, "labels": Tensor}."""
    cache = _cache_path(root, model_name, split)
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"Could not find feature cache {cache}. Run feature_extractor.py first."
        )
    return torch.load(cache, weights_only=True)


# ------------------------------------------------------------------ #
#  Auxiliary: ImageNet pretrained features (for the B̂_local ablation) #
# ------------------------------------------------------------------ #

def extract_imagenet_pretrained_features(
    root: str,
    loaders: dict,
    device="cuda",
    ood_loader=None,
) -> dict:
    """Extract features from a frozen ImageNet-pretrained ResNet-18.
    Cached under model_name = "imagenet_pretrained". Used by the B̂_local
    ablation that compares the audited backbone's own features against
    ImageNet pretrained ones."""
    import torchvision.models as tv_models

    model_name = "imagenet_pretrained"
    splits = ("holdout", "test_id", "calibration")
    all_cached = all(
        os.path.exists(_cache_path(root, model_name, s)) for s in splits
    )
    if all_cached:
        print("imagenet_pretrained features fully cached, loading from disk.")
        results = {s: load_cached_features(root, model_name, s) for s in splits}
        if ood_loader is not None and os.path.exists(_cache_path(root, model_name, "test_ood")):
            results["test_ood"] = load_cached_features(root, model_name, "test_ood")
        return results

    pretrained = tv_models.resnet18(weights="IMAGENET1K_V1")
    pretrained.fc = nn.Identity()
    pretrained = pretrained.to(device)
    pretrained.eval()
    print("Using a frozen ImageNet-pretrained ResNet-18 to extract features.")

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
    """Extract features from each of the 5 ensemble members and average
    them. Cached under model_name = "ensemble_mean"."""
    model_name = "ensemble_mean"
    all_cached = all(
        os.path.exists(_cache_path(root, model_name, s)) for s in splits
    )
    if all_cached:
        print("ensemble_mean features fully cached, loading from disk.")
        return {s: load_cached_features(root, model_name, s) for s in splits}

    feature_accum = {s: None for s in splits}
    labels_store  = {s: None for s in splits}

    for idx, m in enumerate(models):
        print(f"  Ensemble model {idx+1}/{len(models)} extracting features...")
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
        print(f"  [{split}] ensemble_mean feature shape: {mean_feats.shape}")

    return results


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--model",   default="backbone_single",
                        help="Checkpoint name (no .pt suffix).")
    parser.add_argument("--all",     action="store_true",
                        help="Extract features for every model "
                             "(single + ensemble + reference).")
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
            print(f"[skip] checkpoint not found: {ckpt_path}")
            continue
        print(f"\n=== Extracting features: {name} ===")
        model = load_model(ckpt_path, device)
        extract_and_cache(model, loaders, root, name, device, ood_loader=ood_loader)

    print("\nFeature extraction done.")


if __name__ == "__main__":
    main()
