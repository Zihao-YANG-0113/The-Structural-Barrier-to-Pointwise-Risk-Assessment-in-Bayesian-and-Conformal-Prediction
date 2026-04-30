"""
swag.py
-------
Last-Layer SWAG (Stochastic Weight Averaging - Gaussian) inference.

SWAG collects last-layer weight snapshots during the SWA phase, fits a
low-rank + diagonal Gaussian to them, and at inference time samples from
that Gaussian to obtain a Bayesian predictive.

Two-step workflow:
  1. swag_train  : continue training on top of backbone_single with a
                   high constant LR while collecting snapshots.
  2. swag_predict: draw weight samples from the fitted Gaussian and
                   average several forward passes.

Usage:
    # Step 1: SWAG collection (assumes backbone_single is trained)
    python utils/swag.py --config configs/default.yaml --mode train

    # Step 2: SWAG inference
    python utils/swag.py --config configs/default.yaml --mode predict
"""

import argparse
import copy
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.splits import get_loaders, get_ood_loader
from utils.backbone import load_model, build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=-1)


# ------------------------------------------------------------------ #
#  Last-Layer SWAG collector                                          #
# ------------------------------------------------------------------ #

class LastLayerSWAG:
    """Collect SWAG statistics on the last (fc) layer only.

    Tracked quantities:
      - θ̄      : SWA running mean of the weights.
      - Σ_diag  : diagonal variance (running second moment - mean²).
      - D_cols  : columns of the low-rank deviation matrix (the most
                  recent K snapshots minus the running mean).

    At inference time, sampling is:
      θ ~ θ̄ + (1/√2) * Σ_diag^{1/2} * z₁ + (1/√(2(K-1))) * D * z₂
    """

    def __init__(self, base_model: nn.Module, max_rank: int = 20):
        self.base_model = base_model
        self.max_rank = max_rank

        # Reference to the last-layer parameters
        fc = self._get_fc(base_model)
        self.fc_weight_shape = fc.weight.shape
        self.fc_bias_shape = fc.bias.shape if fc.bias is not None else None

        # SWA statistics
        self.n_snapshots = 0
        self.mean_w = torch.zeros_like(fc.weight.data).cpu()
        self.sq_mean_w = torch.zeros_like(fc.weight.data).cpu()
        self.mean_b = torch.zeros_like(fc.bias.data).cpu() if fc.bias is not None else None
        self.sq_mean_b = torch.zeros_like(fc.bias.data).cpu() if fc.bias is not None else None

        # Low-rank deviation columns
        self.dev_w = []  # list of (flattened) weight deviations
        self.dev_b = []  # list of bias deviations

    @staticmethod
    def _get_fc(model: nn.Module) -> nn.Linear:
        """Return the model's final linear layer."""
        fc = model.fc
        if isinstance(fc, nn.Sequential):
            # MC-Dropout model: Sequential(Dropout, Linear)
            for m in fc:
                if isinstance(m, nn.Linear):
                    return m
        return fc

    def collect_snapshot(self):
        """Call after each SWA epoch to take a last-layer weight snapshot."""
        fc = self._get_fc(self.base_model)
        w = fc.weight.data.cpu().clone()
        b = fc.bias.data.cpu().clone() if fc.bias is not None else None

        self.n_snapshots += 1
        n = self.n_snapshots

        # Online update of the mean and the second moment
        self.mean_w = self.mean_w + (w - self.mean_w) / n
        self.sq_mean_w = self.sq_mean_w + (w ** 2 - self.sq_mean_w) / n
        if b is not None:
            self.mean_b = self.mean_b + (b - self.mean_b) / n
            self.sq_mean_b = self.sq_mean_b + (b ** 2 - self.sq_mean_b) / n

        # Low-rank deviation (relative to the current mean)
        self.dev_w.append((w - self.mean_w).flatten())
        if b is not None:
            self.dev_b.append(b - self.mean_b)

        # Cap the rank
        if len(self.dev_w) > self.max_rank:
            self.dev_w.pop(0)
            if self.dev_b:
                self.dev_b.pop(0)

    def _diag_var(self):
        """Diagonal variance = E[w²] - E[w]², clamped ≥ 0."""
        var_w = (self.sq_mean_w - self.mean_w ** 2).clamp(min=1e-10)
        var_b = None
        if self.mean_b is not None:
            var_b = (self.sq_mean_b - self.mean_b ** 2).clamp(min=1e-10)
        return var_w, var_b

    @torch.no_grad()
    def sample_and_set(self, scale: float = 1.0):
        """Sample one set of last-layer weights from the SWAG posterior
        and write them into base_model.
        scale: scaling factor on the sampling variance (default 1.0)."""
        var_w, var_b = self._diag_var()
        K = len(self.dev_w)

        # Diagonal part (computed on CPU, copied to device at the end)
        z1_w = torch.randn_like(self.mean_w)
        w_sample = self.mean_w + scale * (1.0 / np.sqrt(2.0)) * var_w.sqrt() * z1_w

        # Low-rank part
        if K > 1:
            z2 = torch.randn(K, device="cpu")
            D_w = torch.stack(self.dev_w, dim=0)  # (K, d) on CPU
            lr_w = (scale / np.sqrt(2.0 * (K - 1))) * (z2 @ D_w)
            w_sample = w_sample + lr_w.view_as(self.mean_w)

        # Write to the model
        fc = self._get_fc(self.base_model)
        fc.weight.data.copy_(w_sample.to(fc.weight.device))

        if self.mean_b is not None:
            z1_b = torch.randn_like(self.mean_b)
            b_sample = self.mean_b + scale * (1.0 / np.sqrt(2.0)) * var_b.sqrt() * z1_b
            if K > 1:
                D_b = torch.stack(self.dev_b, dim=0)  # (K, d_b) on CPU
                lr_b = (scale / np.sqrt(2.0 * (K - 1))) * (z2 @ D_b)
                b_sample = b_sample + lr_b
            fc.bias.data.copy_(b_sample.to(fc.bias.device))

    def set_swa_mean(self):
        """Set the last-layer weights to the SWA mean (e.g. for BN updates)."""
        fc = self._get_fc(self.base_model)
        fc.weight.data.copy_(self.mean_w.to(fc.weight.device))
        if self.mean_b is not None:
            fc.bias.data.copy_(self.mean_b.to(fc.bias.device))

    def state_dict(self):
        return {
            "n_snapshots": self.n_snapshots,
            "max_rank": self.max_rank,
            "mean_w": self.mean_w,
            "sq_mean_w": self.sq_mean_w,
            "mean_b": self.mean_b,
            "sq_mean_b": self.sq_mean_b,
            "dev_w": self.dev_w,
            "dev_b": self.dev_b,
        }

    def load_state_dict(self, d):
        self.n_snapshots = d["n_snapshots"]
        self.max_rank = d["max_rank"]
        self.mean_w = d["mean_w"].cpu()
        self.sq_mean_w = d["sq_mean_w"].cpu()
        self.mean_b = d["mean_b"].cpu() if d["mean_b"] is not None else None
        self.sq_mean_b = d["sq_mean_b"].cpu() if d["sq_mean_b"] is not None else None
        self.dev_w = [t.cpu() for t in d["dev_w"]]
        self.dev_b = [t.cpu() for t in d["dev_b"]]


# ------------------------------------------------------------------ #
#  SWAG training (continuation on top of an already trained backbone) #
# ------------------------------------------------------------------ #

def swag_train(cfg: dict, root: str, device="cuda"):
    """Load backbone_single, continue training the last layer with a high
    constant LR, and snapshot the weights at the end of each epoch."""
    ckpt_path = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                              "backbone_single.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find {ckpt_path}. "
                                f"Train backbone_single first.")

    model = load_model(ckpt_path, device)
    model.train()

    swag_cfg = cfg.get("swag", {})
    swa_lr = swag_cfg.get("swa_lr", 0.01)
    swa_epochs = swag_cfg.get("swa_epochs", 40)
    max_rank = swag_cfg.get("max_rank", 20)
    collect_freq = swag_cfg.get("collect_freq", 1)  # snapshot every N epochs

    print(f"SWAG training: swa_lr={swa_lr}, swa_epochs={swa_epochs}, "
          f"max_rank={max_rank}, collect_freq={collect_freq}")

    loaders = get_loaders(cfg, root)
    criterion = nn.CrossEntropyLoss()

    # Optimise the last layer only (last-layer SWAG)
    fc = LastLayerSWAG._get_fc(model)
    optimizer = torch.optim.SGD(
        fc.parameters(),
        lr=swa_lr, momentum=0.9, weight_decay=cfg.get("weight_decay", 5e-4),
    )

    swag = LastLayerSWAG(model, max_rank=max_rank)

    for epoch in range(1, swa_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(loaders["train"], desc=f"SWAG epoch {epoch}/{swa_epochs}",
                         leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        acc = 100.0 * correct / total
        if epoch % collect_freq == 0:
            swag.collect_snapshot()
            tag = " [collected]"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == swa_epochs:
            print(f"  Epoch {epoch:3d}/{swa_epochs} | "
                  f"Loss {total_loss/total:.4f} Acc {acc:.2f}%{tag}")

    print(f"Collected {swag.n_snapshots} snapshots in total.")

    # Save SWAG state + base model
    save_dir = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))
    save_path = os.path.join(save_dir, "swag_last_layer.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "swag_state": swag.state_dict(),
        "cfg": cfg,
    }, save_path)
    print(f"SWAG model saved: {save_path}")
    return save_path


# ------------------------------------------------------------------ #
#  SWAG inference                                                     #
# ------------------------------------------------------------------ #

@torch.no_grad()
def predict_swag(model, swag: LastLayerSWAG, loader, n_samples: int = 30,
                 scale: float = 1.0, device="cuda") -> dict:
    """Sample n_samples sets of last-layer weights from the SWAG posterior,
    do one forward pass per sample, and average the predictions.
    Output schema matches LLLA / MC-Dropout."""
    model.to(device)
    model.eval()

    all_sample_probs = []
    all_labels = []

    # Collect all inputs first
    inputs, labels_list = [], []
    for x, y in loader:
        inputs.append(x)
        labels_list.append(y)
    all_x = torch.cat(inputs, dim=0)
    all_y = torch.cat(labels_list, dim=0)

    # n_samples weight draws
    all_probs = []
    for s in tqdm(range(n_samples), desc=f"SWAG samples ({n_samples})"):
        swag.sample_and_set(scale=scale)
        model.eval()

        batch_probs = []
        for x in inputs:
            x = x.to(device)
            logits = model(x)
            batch_probs.append(F.softmax(logits, dim=-1).cpu())
        probs_s = torch.cat(batch_probs, dim=0)  # (N, C)
        all_probs.append(probs_s)

    # Restore the SWA mean
    swag.set_swa_mean()

    probs_stack = torch.stack(all_probs, dim=1)  # (N, n_samples, C)
    mean_probs = probs_stack.mean(dim=1)          # (N, C)

    pred_ent = entropy(mean_probs)
    h_per = entropy(probs_stack.reshape(-1, probs_stack.shape[-1]))
    h_per = h_per.view(-1, n_samples).mean(dim=1)
    mi = pred_ent - h_per
    max_prob = mean_probs.max(dim=-1).values
    correct = (mean_probs.argmax(dim=-1) == all_y)

    return {
        "probs": mean_probs,
        "pred_entropy": pred_ent,
        "mutual_info": mi,
        "max_prob": max_prob,
        "labels": all_y,
        "correct": correct,
    }


def save_results(results: dict, root: str, split: str):
    save_dir = os.path.join(root, "results", "swag")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split}.pt")
    torch.save(results, path)
    print(f"SWAG results saved: {path}")


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all",
                        help="train: SWAG collection; predict: inference; all: both.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    swag_ckpt = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                              "swag_last_layer.pt")

    if args.mode in ("train", "all"):
        print("=== SWAG training (last-layer weight snapshots) ===")
        swag_train(cfg, root, device)

    if args.mode in ("predict", "all"):
        print("\n=== SWAG inference ===")
        if not os.path.exists(swag_ckpt):
            raise FileNotFoundError(f"Could not find {swag_ckpt}. "
                                    f"Run --mode train first.")

        ckpt = torch.load(swag_ckpt, map_location=device, weights_only=False)
        model = build_model(ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        swag = LastLayerSWAG(model, max_rank=ckpt["swag_state"]["max_rank"])
        swag.load_state_dict(ckpt["swag_state"])

        loaders = get_loaders(cfg, root)
        n_samples = cfg.get("swag", {}).get("n_samples", 30)
        scale = cfg.get("swag", {}).get("scale", 1.0)

        for split in ["holdout", "calibration", "test_id"]:
            print(f"\n--- SWAG predict [{split}] ---")
            results = predict_swag(model, swag, loaders[split],
                                   n_samples=n_samples, scale=scale, device=device)
            save_results(results, root, split)
            acc = results["correct"].float().mean() * 100
            print(f"  accuracy: {acc:.2f}%")

        print("\n--- SWAG predict [test_ood] ---")
        ood_loader = get_ood_loader(cfg, root)
        results = predict_swag(model, swag, ood_loader,
                               n_samples=n_samples, scale=scale, device=device)
        save_results(results, root, "test_ood")

        print("\nSWAG inference done.")


if __name__ == "__main__":
    main()
