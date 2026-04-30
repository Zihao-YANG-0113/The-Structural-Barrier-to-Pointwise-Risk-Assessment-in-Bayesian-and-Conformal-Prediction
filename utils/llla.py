"""
llla.py
-------
Last-Layer Laplace Approximation (LLLA) inference.

Uses `laplace-torch` to fit a last-layer Laplace approximation on the
trained CIFAR-10 backbone (ResNet-20 in the dissertation main pipeline).
The prior precision is selected by marginal likelihood optimisation, as
in Dissertation §4.2.1.

Outputs the uncertainty measures used in Chapter 4:
  - predictive_entropy : H[p̄(y|x)]
  - mutual_information : MI = H[p̄] - E_π[H[p_θ]]
  - max_prob_mean      : max class probability under the predictive mean

Usage:
    python utils/llla.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.splits import get_loaders, get_ood_loader
from utils.backbone import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Uncertainty helpers                                                #
# ------------------------------------------------------------------ #

def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample predictive entropy. probs: (N, C) → (N,)."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def mutual_information(
    mean_probs: torch.Tensor,
    probs_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MI = H[p̄] - E[H[p_θ]]
    mean_probs:    (N, C)
    probs_samples: (N, S, C) — S posterior samples."""
    h_mean   = entropy(mean_probs, eps)
    h_per    = -(probs_samples * (probs_samples + eps).log()).sum(dim=-1)  # (N, S)
    e_h_per  = h_per.mean(dim=1)                                            # (N,)
    return h_mean - e_h_per


# ------------------------------------------------------------------ #
#  LLLA inference                                                     #
# ------------------------------------------------------------------ #

def fit_llla(model, train_loader, device="cuda"):
    """Fit last-layer Laplace via `laplace-torch` and optimise the prior
    precision. Returns the Laplace object."""
    try:
        from laplace import Laplace
    except ImportError:
        raise ImportError("Install laplace-torch first: pip install laplace-torch")

    model = model.to(device)
    model.eval()

    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    print("Fitting last-layer Laplace approximation (Kron Hessian)...")
    la.fit(train_loader)
    print("Optimising prior precision (marginal likelihood)...")
    la.optimize_prior_precision(method="marglik")
    print(f"Optimised prior precision: {la.prior_precision.item():.4f}")
    return la


@torch.no_grad()
def predict_llla(la, loader, n_samples: int = 100, device="cuda"):
    """Run LLLA prediction on every sample in the loader.
    Returns a dict with:
      probs          : (N, C) mean predictive probabilities
      pred_entropy   : (N,)   predictive entropy
      mutual_info    : (N,)   mutual information
      max_prob       : (N,)   max-class probability
      labels         : (N,)   true labels
      correct        : (N,)   bool, prediction correct or not."""
    all_probs, all_labels = [], []

    for x, y in tqdm(loader, desc="LLLA predict", leave=False):
        x = x.to(device)
        # la() returns the mean predictive probs (B, C); link_approx='probit'
        # is the standard probit link for classification.
        with torch.no_grad():
            p = la(x, link_approx="probit")
        all_probs.append(p.cpu())
        all_labels.append(y)

    probs  = torch.cat(all_probs,  dim=0)   # (N, C)
    labels = torch.cat(all_labels, dim=0)   # (N,)

    # Estimate MI by Monte Carlo sampling from the posterior weights.
    # `laplace-torch` exposes la.predictive_samples for this.
    print("Estimating mutual information via posterior MC samples...")
    sample_probs_list = []
    for x, _ in tqdm(loader, desc="LLLA MC samples", leave=False):
        x = x.to(device)
        # Draw n_samples per batch; each draw returns a (B, C) tensor.
        s = la.predictive_samples(x, pred_type="glm", n_samples=n_samples)
        # s: (n_samples, B, C) or (B, n_samples, C) depending on the
        # laplace-torch version — normalise to (B, n_samples, C).
        if s.shape[0] == n_samples:
            s = s.permute(1, 0, 2)
        sample_probs_list.append(s.cpu())

    probs_samples = torch.cat(sample_probs_list, dim=0)  # (N, n_samples, C)

    pred_ent = entropy(probs)
    mi       = mutual_information(probs, probs_samples)
    max_prob = probs.max(dim=-1).values
    correct  = (probs.argmax(dim=-1) == labels)

    return {
        "probs":        probs,
        "pred_entropy": pred_ent,
        "mutual_info":  mi,
        "max_prob":     max_prob,
        "labels":       labels,
        "correct":      correct,
    }


# ------------------------------------------------------------------ #
#  Save & load                                                        #
# ------------------------------------------------------------------ #

def save_results(results: dict, root: str, split: str):
    save_dir = os.path.join(root, "results", "llla")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split}.pt")
    torch.save(results, path)
    print(f"LLLA results saved: {path}")


def load_results(root: str, split: str) -> dict:
    path = os.path.join(root, "results", "llla", f"{split}.pt")
    return torch.load(path, weights_only=False)


# ------------------------------------------------------------------ #
#  Main entry point                                                   #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
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

    ckpt_path = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                              "backbone_single.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint: {ckpt_path}. "
                                f"Train the backbone first.")

    model   = load_model(ckpt_path, device)
    loaders = get_loaders(cfg, root)

    # Fit Laplace
    la = fit_llla(model, loaders["train"], device)

    # Save Laplace object
    la_save = os.path.join(root, "checkpoints", "llla.pt")
    torch.save(la, la_save)
    print(f"Laplace object saved: {la_save}")

    # Predict on each split
    for split in ["val", "calibration", "test_id"]:
        print(f"\n=== LLLA predict [{split}] ===")
        results = predict_llla(la, loaders[split], device=device)
        save_results(results, root, split)

    # OOD
    print("\n=== LLLA predict [test_ood] ===")
    ood_loader = get_ood_loader(cfg, root)
    results    = predict_llla(la, ood_loader, device=device)
    save_results(results, root, "test_ood")

    print("\nLLLA inference done.")


if __name__ == "__main__":
    main()
