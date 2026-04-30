"""
flag.py
-------
Trustworthiness Flag System (Stage 2 of the audit pipeline).

Each test point is labelled with one of three flags based on B̂_local
and the k-NN distance:
  - "trustworthy"        : knn_dist ≤ d_max  AND  B̂ ≤ τ_B
  - "Bl_detected"        : knn_dist ≤ d_max  AND  B̂ > τ_B  (posterior blindspot)
  - "insufficient_data"  : knn_dist > d_max                  (sparse neighbourhood)

τ_B and d_max are derived from the held-out LOO results — no hand-tuned
constants:

  τ_B  ← Youden's J statistic on the LOO ROC
           fpr, tpr, thresholds = roc_curve(is_wrong, b_local_loo)
           τ_B = thresholds[(tpr - fpr).argmax()]

  d_max ← 95th percentile of held-out LOO k-NN distances
           d_max = np.percentile(knn_dist_loo, 95)

Usage:
    python core/flag.py --config configs/default.yaml --method llla
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


FLAG_TRUSTWORTHY        = "trustworthy"
FLAG_BL_DETECTED        = "Bl_detected"
FLAG_INSUFFICIENT_DATA  = "insufficient_data"


def determine_tau_and_dmax(
    b_hat_loo:  torch.Tensor,   # (N,) hold-out LOO B̂
    errors:     torch.Tensor,   # (N,) bool/0-1, 1 = misclassified
    knn_dists:  torch.Tensor,   # (N,) hold-out LOO k-NN distance
) -> tuple:
    """
    Derive τ_B and d_max from the held-out LOO results — no hand-tuned
    constants.

    τ_B  : Youden's J = thresholds[(tpr - fpr).argmax()]
    d_max: np.percentile(knn_dists, 95)

    Returns:
        (tau_b: float, d_max: float)
    """
    b_np = b_hat_loo.float().numpy() if isinstance(b_hat_loo, torch.Tensor) else np.asarray(b_hat_loo, dtype=float)
    e_np = errors.float().numpy()    if isinstance(errors,    torch.Tensor) else np.asarray(errors,    dtype=float)
    d_np = knn_dists.float().numpy() if isinstance(knn_dists, torch.Tensor) else np.asarray(knn_dists, dtype=float)

    # d_max: 95th percentile (data-driven)
    d_max = float(np.percentile(d_np, 95))

    # τ_B: Youden's J statistic (data-driven)
    if e_np.sum() == 0 or e_np.sum() == len(e_np):
        tau_b = float(np.median(b_np))
        print(f"  [warn] hold-out labels are all identical, falling back to median τ_B={tau_b:.4f}")
    else:
        fpr, tpr, thresholds = roc_curve(e_np, b_np)
        j_idx = int((tpr - fpr).argmax())
        tau_b = float(thresholds[j_idx])
        print(
            f"  Youden's J → τ_B = {tau_b:.4f}  "
            f"(TPR={tpr[j_idx]:.3f}, FPR={fpr[j_idx]:.3f})"
        )

    print(f"  d_max = {d_max:.4f}  (holdout LOO knn_dist 95th percentile)")
    return tau_b, d_max


def audit_flag(
    b_hat:    torch.Tensor,
    knn_dist: torch.Tensor,
    tau_b:    float,
    d_max:    float,
) -> list:
    """Generate the per-point trustworthiness flag.
    Priority: insufficient_data > Bl_detected > trustworthy."""
    flags = []
    for i in range(len(b_hat)):
        if knn_dist[i].item() > d_max:
            flags.append(FLAG_INSUFFICIENT_DATA)
        elif b_hat[i].item() > tau_b:
            flags.append(FLAG_BL_DETECTED)
        else:
            flags.append(FLAG_TRUSTWORTHY)
    return flags


def flag_statistics(flags: list) -> dict:
    """Count and proportion of each flag category."""
    n = len(flags)
    stats = {}
    for f in [FLAG_TRUSTWORTHY, FLAG_BL_DETECTED, FLAG_INSUFFICIENT_DATA]:
        cnt = sum(1 for x in flags if x == f)
        stats[f] = {"count": cnt, "ratio": cnt / n}
    return stats


def run_flagging(root: str, cfg: dict, method: str = "llla"):
    """Derive τ_B and d_max from the LOO outputs, then save flags for
    every test split. Both thresholds are fully data-driven."""
    b_local_dir = os.path.join(root, "results", f"b_local_{method}")
    flag_dir    = os.path.join(root, "results", f"flags_{method}")
    os.makedirs(flag_dir, exist_ok=True)

    # ---- 1. Determine τ_B and d_max from the LOO outputs ----
    holdout_b_path      = os.path.join(b_local_dir, "holdout.pt")
    method_holdout_path = os.path.join(root, "results", method, "holdout.pt")

    if not os.path.exists(holdout_b_path):
        raise FileNotFoundError(
            f"Could not find {holdout_b_path}. Run core/b_local.py first."
        )
    if not os.path.exists(method_holdout_path):
        raise FileNotFoundError(
            f"Could not find {method_holdout_path}. Run predict() for "
            f"{method} on the holdout split first."
        )

    ho_b_data   = torch.load(holdout_b_path, weights_only=False)
    ho_method   = torch.load(method_holdout_path, weights_only=False)
    b_hat_ho    = ho_b_data["b_hat"]       # LOO B̂ (N,)
    knn_dist_ho = ho_b_data["knn_dist"]    # LOO k-NN distance (N,)
    errors_ho   = (~ho_method["correct"]).float()

    print(f"\n[{method}] Determining τ_B and d_max from hold-out LOO ...")
    tau_b, d_max = determine_tau_and_dmax(b_hat_ho, errors_ho, knn_dist_ho)

    # ---- 2. Generate flags for each test split ----
    for split in ["calibration", "test_id", "test_ood"]:
        path = os.path.join(b_local_dir, f"{split}.pt")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue

        data     = torch.load(path, weights_only=False)
        b_hat    = data["b_hat"]
        knn_dist = data["knn_dist"]

        flags = audit_flag(b_hat, knn_dist, tau_b=tau_b, d_max=d_max)
        stats = flag_statistics(flags)

        print(f"\n[{method}|{split}] flag statistics (τ_B={tau_b:.4f}, d_max={d_max:.4f}):")
        for f, s in stats.items():
            print(f"  {f:25s}: {s['count']:5d}  ({s['ratio']*100:.1f}%)")

        result = {
            "flags":    flags,
            "b_hat":    b_hat,
            "knn_dist": knn_dist,
            "stats":    stats,
            "tau_b":    tau_b,
            "d_max":    d_max,
        }
        out_path = os.path.join(flag_dir, f"{split}.pt")
        torch.save(result, out_path)
        print(f"  flags saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla",
                        choices=["llla", "ensemble", "mc_dropout", "swag"])
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_flagging(root, cfg, args.method)
    print("\nFlag generation done.")


if __name__ == "__main__":
    main()
