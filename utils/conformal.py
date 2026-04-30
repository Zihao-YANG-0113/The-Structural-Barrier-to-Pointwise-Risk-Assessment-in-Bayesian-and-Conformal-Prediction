"""
conformal.py
------------
Split Conformal Prediction with the APS (Adaptive Prediction Sets) score.

APS score: for a sample (x, y),
  score = cumulative softmax probability up to and including the true class y
  (sorted descending and accumulated).

Calibration: compute every APS score on the calibration set and take the
⌈(n+1)(1-α)⌉/n quantile q̂.
Prediction: prediction set = {y : APS_score(x, y) ≤ q̂}.

Usage:
    python utils/conformal.py --config configs/default.yaml --method llla
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
#  APS score                                                          #
# ------------------------------------------------------------------ #

def aps_scores(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample APS score.
    probs:  (N, C) softmax probabilities
    labels: (N,)   true labels
    returns: (N,)  APS score in [0, 1]"""
    N, C = probs.shape
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)  # (N, C)
    cumsum = sorted_probs.cumsum(dim=-1)                              # (N, C)

    scores = torch.zeros(N)
    for i in range(N):
        # Position of the true label in the sort
        rank = (sorted_idx[i] == labels[i]).nonzero(as_tuple=True)[0].item()
        scores[i] = cumsum[i, rank].item()
    return scores


def aps_scores_batch(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Vectorised APS score (faster)."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    # Rank of the true label per sample
    label_expand = labels.unsqueeze(1).expand_as(sorted_idx)    # (N, C)
    match_mask   = (sorted_idx == label_expand)                  # (N, C) bool
    # First match
    ranks = match_mask.float().argmax(dim=-1)                    # (N,)
    scores = cumsum[torch.arange(len(labels)), ranks]            # (N,)
    return scores


def prediction_sets_from_scores(probs: torch.Tensor, quantile: float) -> list[list[int]]:
    """Generate prediction sets from a calibration quantile.
    For each sample, collect classes whose cumulative probability does not
    exceed the quantile (always including at least the top class)."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    pred_sets = []
    for i in range(probs.shape[0]):
        included = (cumsum[i] <= quantile).nonzero(as_tuple=True)[0]
        # First position whose cumulative prob exceeds the quantile (so the
        # true class is included with probability ≥ 1-α).
        exceed = (cumsum[i] > quantile).nonzero(as_tuple=True)[0]
        if len(exceed) == 0:
            cutoff = probs.shape[1] - 1
        else:
            cutoff = exceed[0].item()
        set_indices = sorted_idx[i, :cutoff + 1].tolist()
        pred_sets.append(set_indices)
    return pred_sets


# ------------------------------------------------------------------ #
#  Calibration & prediction                                          #
# ------------------------------------------------------------------ #

class SplitConformal:
    """Split conformal predictor with APS scores."""
    def __init__(self, alpha: float = 0.1):
        self.alpha    = alpha
        self.quantile = None

    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor):
        """Estimate the quantile q̂ on the calibration set.
        cal_probs:  (n, C)
        cal_labels: (n,)"""
        scores = aps_scores_batch(cal_probs, cal_labels)
        n      = len(scores)
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        level  = min(level, 1.0)
        self.quantile = float(torch.quantile(scores, level))
        print(f"Conformal quantile q̂ = {self.quantile:.4f}  (α={self.alpha}, n={n})")
        return self.quantile

    def predict(self, test_probs: torch.Tensor) -> list[list[int]]:
        """Return the prediction set (list of class indices) for each test sample."""
        if self.quantile is None:
            raise RuntimeError("Call calibrate() first.")
        return prediction_sets_from_scores(test_probs, self.quantile)

    def coverage(self, pred_sets: list[list[int]], labels: torch.Tensor) -> float:
        """Empirical coverage rate."""
        covered = sum(
            1 for ps, y in zip(pred_sets, labels.tolist()) if y in ps
        )
        return covered / len(labels)

    def avg_set_size(self, pred_sets: list[list[int]]) -> float:
        return sum(len(ps) for ps in pred_sets) / len(pred_sets)


# ------------------------------------------------------------------ #
#  Load saved predictive probabilities                                #
# ------------------------------------------------------------------ #

def load_probs(root: str, method: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the saved probs / labels produced by the chosen method."""
    path = os.path.join(root, "results", method, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Run the corresponding method first.")
    data = torch.load(path, weights_only=False)
    return data["probs"], data["labels"]


# ------------------------------------------------------------------ #
#  Run & save                                                         #
# ------------------------------------------------------------------ #

def run_conformal(root: str, cfg: dict, method: str = "llla"):
    """Run split conformal for the given method and save prediction sets
    + coverage."""
    alpha = cfg.get("alpha", 0.1)
    cp    = SplitConformal(alpha=alpha)

    # Calibrate
    cal_probs, cal_labels = load_probs(root, method, "calibration")
    cp.calibrate(cal_probs, cal_labels)

    results = {}
    for split in ["test_id"]:
        test_probs, test_labels = load_probs(root, method, split)
        pred_sets = cp.predict(test_probs)
        cov       = cp.coverage(pred_sets, test_labels)
        avg_size  = cp.avg_set_size(pred_sets)
        print(f"  [{method}|{split}] coverage={cov:.4f}  avg set size={avg_size:.2f}")

        # APS scores (used downstream by the B_local audit pipeline)
        aps = aps_scores_batch(test_probs, test_labels)
        results[split] = {
            "pred_sets":  pred_sets,
            "coverage":   cov,
            "avg_size":   avg_size,
            "aps_scores": aps,
            "quantile":   cp.quantile,
            "labels":     test_labels,
            "probs":      test_probs,
        }

    save_dir = os.path.join(root, "results", f"conformal_{method}")
    os.makedirs(save_dir, exist_ok=True)
    for split, res in results.items():
        path = os.path.join(save_dir, f"{split}.pt")
        torch.save(res, path)
        print(f"Conformal results saved: {path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla",
                        choices=["llla", "ensemble", "mc_dropout"],
                        help="Method whose predictive probabilities feed conformal.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"=== Split Conformal (APS) with {args.method} ===")
    run_conformal(root, cfg, args.method)
    print("\nConformal inference done.")


if __name__ == "__main__":
    main()
