"""
b_local.py
----------
B̂_local estimator: estimate the pointwise risk gap from a held-out sample.

Core algorithm (Theorem 3.3 in the dissertation):
  1. Find k-NN neighbours in the audited model's penultimate-layer feature
     space by default (an ablation can swap in ImageNet pretrained features
     via --feature_model imagenet_pretrained).
  2. actual_loss   = neighbourhood-weighted cross-entropy at the held-out
                     true labels.
  3. believed_loss = neighbourhood-weighted predictive entropy (the loss
                     the model expects under its own predictive law).
  4. B̂(x) = actual_loss - believed_loss.
  5. D(x) = mean distance to the k neighbours (density signal).

On the held-out set, B̂ is computed via Leave-One-Out (LOO):
  for each held-out point we exclude its own index and take the k nearest
  among the remaining N-1. The LOO outputs are consumed by flag.py to
  derive τ_B (Youden's J) and d_max (95th percentile).

Usage:
    python core/b_local.py --config configs/default.yaml --method llla
    # ablation: use ImageNet pretrained features
    python core/b_local.py --method llla --feature_model imagenet_pretrained
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import load_cached_features


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def method_to_feature_model(method: str) -> str:
    """Return the feature-cache name corresponding to the audited method
    (its own penultimate layer)."""
    return {
        "llla":       "backbone_single",
        "ensemble":   "ensemble_mean",
        "mc_dropout": "backbone_mcdropout",
        "swag":       "backbone_single",
    }.get(method, "backbone_single")


# ------------------------------------------------------------------ #
#  Core B̂_local computation                                         #
# ------------------------------------------------------------------ #

class BLocalEstimator:
    """
    Pointwise risk-gap estimator.
      fit()         : build a k-NN index over held-out features and store
                      predictive probabilities and labels.
      estimate()    : estimate B̂(x) and D(x) for new test points.
      estimate_loo(): leave-one-out estimate on the held-out set itself
                      (each point excludes its own index).
    """

    def __init__(self, k: int = 50, kernel_bandwidth: float = None):
        self.k = k
        self.h = kernel_bandwidth
        self.knn = None
        self._holdout_features_np = None
        self._model_probs         = None
        self._holdout_labels      = None

    def fit(
        self,
        holdout_features: np.ndarray,       # (n_ho, d) feature vectors
        model_holdout_probs: torch.Tensor,  # (n_ho, C) audited model probs
        holdout_labels: torch.Tensor,       # (n_ho,)
    ):
        """Build the k-NN index and store held-out predictions."""
        self._holdout_features_np = holdout_features
        self._model_probs         = model_holdout_probs.float()
        self._holdout_labels      = holdout_labels

        self.knn = NearestNeighbors(
            n_neighbors=self.k, algorithm="auto",
            metric="euclidean", n_jobs=-1,
        )
        self.knn.fit(holdout_features)

        if self.h is None:
            # Auto bandwidth: median k-NN distance
            sample_dists, _ = self.knn.kneighbors(holdout_features[:500])
            self.h = float(np.median(sample_dists[:, -1])) + 1e-6

        print(
            f"BLocalEstimator fitted."
            f"  k={self.k}  h={self.h:.4f}  "
            f"hold-out size={len(holdout_labels):,}"
        )

    def _compute_b(self, idx: np.ndarray, dist: np.ndarray) -> tuple:
        """Compute the weighted B̂ and the mean distance for a single
        point. Returns (b_value, mean_dist)."""
        w = np.exp(-0.5 * (dist / self.h) ** 2)
        w = w / (w.sum() + 1e-12)
        w_t = torch.from_numpy(w.astype(np.float32))

        k_actual = len(idx)
        p  = self._model_probs[idx]           # (k, C)
        y  = self._holdout_labels[idx]        # (k,)

        actual_loss   = -(p[torch.arange(k_actual), y].clamp(min=1e-8).log())   # (k,)
        believed_loss = -(p * p.clamp(min=1e-8).log()).sum(dim=-1)               # (k,)

        b = ((w_t * actual_loss).sum() - (w_t * believed_loss).sum()).item()
        return b, float(dist.mean())

    def estimate(
        self,
        test_features: np.ndarray,
        batch_size: int = 256,
    ) -> dict:
        """Estimate B̂(x) and D(x) for new test points.
        Returns {"b_hat": Tensor(N,), "knn_dist": Tensor(N,), "k": int}."""
        if self.knn is None:
            raise RuntimeError("Call fit() first.")

        b_hat_list, knn_dist_list = [], []
        for start in tqdm(range(0, len(test_features), batch_size),
                          desc="Estimating B̂_local", leave=False):
            batch = test_features[start:start + batch_size]
            dists, indices = self.knn.kneighbors(batch)
            for i in range(len(batch)):
                b, d = self._compute_b(indices[i], dists[i])
                b_hat_list.append(b)
                knn_dist_list.append(d)

        return {
            "b_hat":    torch.tensor(b_hat_list),
            "knn_dist": torch.tensor(knn_dist_list),
            "k":        self.k,
        }

    def estimate_loo(self, batch_size: int = 256) -> dict:
        """Leave-one-out B̂ estimate on the held-out set itself.

        Strategy: query k+1 neighbours (which includes the point itself),
        then drop the entry whose global index equals the query index, and
        keep the remaining k for a Nadaraya-Watson weighted estimate.

        Returns {"b_hat": Tensor(N,), "knn_dist": Tensor(N,), "k": int,
                 "loo": True}."""
        if self._holdout_features_np is None:
            raise RuntimeError("Call fit() first.")

        N = len(self._holdout_features_np)
        knn_loo = NearestNeighbors(
            n_neighbors=self.k + 1, algorithm="auto",
            metric="euclidean", n_jobs=-1,
        )
        knn_loo.fit(self._holdout_features_np)

        b_hat_list, knn_dist_list = [], []

        for start in tqdm(range(0, N, batch_size), desc="LOO B̂_local", leave=False):
            batch = self._holdout_features_np[start:start + batch_size]
            dists, indices = knn_loo.kneighbors(batch)  # (B, k+1)

            for i in range(len(batch)):
                global_i = start + i
                all_idx  = indices[i]   # (k+1,)
                all_dist = dists[i]     # (k+1,)

                # Exclude self (global index == global_i)
                mask = all_idx != global_i
                idx  = all_idx[mask][:self.k]
                dist = all_dist[mask][:self.k]

                b, d = self._compute_b(idx, dist)
                b_hat_list.append(b)
                knn_dist_list.append(d)

        return {
            "b_hat":    torch.tensor(b_hat_list),
            "knn_dist": torch.tensor(knn_dist_list),
            "k":        self.k,
            "loo":      True,
        }


# ------------------------------------------------------------------ #
#  Run B̂_local estimation                                           #
# ------------------------------------------------------------------ #

def run_b_local(
    root: str,
    cfg: dict,
    method: str = "llla",
    feature_model: str = None,
):
    """Estimate B̂_local for the given audited method and save the results.

    method:        Audited model name (llla / ensemble / mc_dropout / swag).
    feature_model: Name of the feature cache used for the k-NN.
                   Default (None) = the audited method's own backbone:
                     llla       → backbone_single
                     ensemble   → ensemble_mean
                     mc_dropout → backbone_mcdropout
                   Pass "imagenet_pretrained" for the ablation.
    """
    k = cfg.get("k_neighbors", 50)
    estimator = BLocalEstimator(k=k)

    if feature_model is None:
        feature_model = method_to_feature_model(method)

    print(f"\n[b_local | {method}] feature cache: {feature_model}")

    # Load held-out features
    ref_holdout  = load_cached_features(root, feature_model, "holdout")
    ref_feats_np = ref_holdout["features"].numpy()

    # Load the audited model's predictive probabilities on the held-out set
    method_holdout_path = os.path.join(root, "results", method, "holdout.pt")
    if not os.path.exists(method_holdout_path):
        raise FileNotFoundError(
            f"Could not find {method_holdout_path}. "
            f"Run predict() for {method} on the holdout split first."
        )
    holdout_result = torch.load(method_holdout_path, weights_only=False)
    model_probs    = holdout_result["probs"]    # (n_ho, C)
    holdout_labels = holdout_result["labels"]   # (n_ho,)

    estimator.fit(ref_feats_np, model_probs, holdout_labels)

    save_dir = os.path.join(root, "results", f"b_local_{method}")
    os.makedirs(save_dir, exist_ok=True)

    # holdout: Leave-One-Out estimate (used by flag.py for τ_B and d_max)
    print(f"\nEstimating B̂_local [{method}|holdout] (LOO) ...")
    result_loo = estimator.estimate_loo()
    path = os.path.join(save_dir, "holdout.pt")
    torch.save(result_loo, path)
    b = result_loo["b_hat"]
    print(
        f"  LOO B̂ mean={b.mean():.4f}  std={b.std():.4f}  "
        f"  positive fraction={( b > 0 ).float().mean():.4f}"
    )

    # Other splits: standard k-NN estimate
    for split in ["test_id", "calibration"]:
        ref_data   = load_cached_features(root, feature_model, split)
        test_feats = ref_data["features"].numpy()
        print(f"\nEstimating B̂_local [{method}|{split}] ...")
        result = estimator.estimate(test_feats)
        path   = os.path.join(save_dir, f"{split}.pt")
        torch.save(result, path)
        b = result["b_hat"]
        print(
            f"  B̂ mean={b.mean():.4f}  std={b.std():.4f}  "
            f"  positive fraction={( b > 0 ).float().mean():.4f}"
        )

    # OOD
    try:
        ref_ood   = load_cached_features(root, feature_model, "test_ood")
        ood_feats = ref_ood["features"].numpy()
        print(f"\nEstimating B̂_local [{method}|test_ood] ...")
        result = estimator.estimate(ood_feats)
        path   = os.path.join(save_dir, "test_ood.pt")
        torch.save(result, path)
        b = result["b_hat"]
        print(f"  B̂ mean={b.mean():.4f}  std={b.std():.4f}")
    except FileNotFoundError:
        print("  [skip] test_ood features have not been extracted yet.")

    return estimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--method",        default="llla",
                        choices=["llla", "ensemble", "mc_dropout", "swag"])
    parser.add_argument("--feature_model", default=None,
                        help="k-NN feature-cache name. None = audited model's own "
                             "backbone features. 'imagenet_pretrained' = ablation "
                             "with ImageNet pretrained features.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_b_local(root, cfg, method=args.method, feature_model=args.feature_model)
    print("\nB̂_local estimation done.")


if __name__ == "__main__":
    main()
