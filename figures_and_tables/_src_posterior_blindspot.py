"""
posterior_blindspot.py
---------------------
Core experiment: Posterior blindspot analysis.

For each scenario (Mixed / CIFAR-10-C severity 3 / CIFAR-10-C severity 5):
  1. Sort samples by pred_entropy ascending (most confident first)
  2. Take the top 10%, 20%, 30%, 50% lowest-entropy samples
  3. Within each subset, split by B_local > tau_b:
     - "Double confirmed": low entropy AND B_local <= tau (model + audit agree: safe)
     - "Blindspot":        low entropy BUT B_local > tau (entropy says safe, audit disagrees)
  4. Report sample counts, error rates, risk reduction, and Fisher exact test p-value

Output:
  results/eval/posterior_blindspot.csv
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  CIFAR-10-C helpers                                                  #
# ------------------------------------------------------------------ #

CORRUPTION_TYPES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur",
]


def load_cifar10c_severity(cifar10c_dir: str, severity: int, max_corruptions=None):
    """Load all corruption types at a given severity, return (images, labels)."""
    all_imgs, all_labels = [], []
    types = CORRUPTION_TYPES[:max_corruptions] if max_corruptions else CORRUPTION_TYPES
    for corruption in types:
        img_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
        label_path = os.path.join(cifar10c_dir, "labels.npy")
        if not os.path.exists(img_path):
            continue
        imgs = np.load(img_path)
        labels = np.load(label_path)
        start = (severity - 1) * 10000
        all_imgs.append(imgs[start:start + 10000])
        all_labels.append(labels[start:start + 10000])
    if not all_imgs:
        return None, None
    return np.concatenate(all_imgs), np.concatenate(all_labels)


def run_inference_on_array(model, images_np, labels_np, device, batch_size=256):
    """
    Run model inference on raw uint8 images. Returns dict with
    probs, pred_entropy, correct, features (512-dim).
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    model.eval()
    all_probs, all_feats = [], []

    hook_feats = []
    def _hook(module, inp, out):
        hook_feats.append(out.squeeze(-1).squeeze(-1).detach().cpu())
    handle = model.avgpool.register_forward_hook(_hook)

    N = len(images_np)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch_np = images_np[start:start + batch_size]
            x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=-1).cpu()
            all_probs.append(probs)

    handle.remove()

    probs = torch.cat(all_probs, dim=0)
    features = torch.cat(hook_feats, dim=0).numpy()
    labels = torch.from_numpy(labels_np.astype(np.int64))

    eps = 1e-8
    pred_entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    correct = (probs.argmax(dim=-1) == labels)

    return {
        "probs": probs,
        "pred_entropy": pred_entropy,
        "correct": correct,
        "labels": labels,
        "features": features,
    }


# ------------------------------------------------------------------ #
#  Blindspot analysis core                                             #
# ------------------------------------------------------------------ #

def analyze_blindspot(pred_entropy, correct, b_hat, tau_b, knn_dist, d_max,
                      coverages=(0.10, 0.20, 0.30, 0.50, 0.80)):
    """
    For each coverage level, take the lowest-entropy samples of the FULL pool
    (no pre-filter), then partition into three groups by the deployer-honest
    rule:
      - insufficient : knn_dist > d_max  (B̂ estimate unreliable → reject)
      - Bl_detected  : knn_dist <= d_max AND b_hat > tau_b (blindspot → reject)
      - confirmed    : knn_dist <= d_max AND b_hat <= tau_b (keep)

    The 'blindspot' group for risk-reduction and Fisher test is the UNION
    (insufficient + Bl_detected) — both are rejected in deployment, and both
    must count against the rejection budget.
    """
    entropy_np = pred_entropy.numpy() if isinstance(pred_entropy, torch.Tensor) else np.asarray(pred_entropy)
    correct_np = correct.numpy() if isinstance(correct, torch.Tensor) else np.asarray(correct)
    b_hat_np   = b_hat.numpy() if isinstance(b_hat, torch.Tensor) else np.asarray(b_hat)
    knn_np     = knn_dist.numpy() if isinstance(knn_dist, torch.Tensor) else np.asarray(knn_dist)

    N = len(entropy_np)
    sorted_idx = entropy_np.argsort()  # ascending entropy = most confident first

    rows = []
    for cov in coverages:
        k = max(1, int(cov * N))
        sel = sorted_idx[:k]

        sel_correct = correct_np[sel]
        sel_b_hat   = b_hat_np[sel]
        sel_knn     = knn_np[sel]

        mask_insuf  = sel_knn > d_max
        mask_Bl     = (~mask_insuf) & (sel_b_hat > tau_b)
        mask_conf   = (~mask_insuf) & (sel_b_hat <= tau_b)
        mask_blind  = mask_insuf | mask_Bl        # deployer rejects both

        n_insuf = int(mask_insuf.sum())
        n_Bl    = int(mask_Bl.sum())
        n_conf  = int(mask_conf.sum())
        n_blind = int(mask_blind.sum())

        def _err(mask, n):
            return (1 - sel_correct[mask].mean()) if n > 0 else float("nan")

        err_all    = float(1 - sel_correct.mean())
        err_insuf  = float(_err(mask_insuf, n_insuf))
        err_Bl     = float(_err(mask_Bl,    n_Bl))
        err_conf   = float(_err(mask_conf,  n_conf))
        err_blind  = float(_err(mask_blind, n_blind))

        # Risk reduction: error drop if the deployer rejects ALL blindspot samples
        risk_reduction = (err_all - err_conf) / (err_all + 1e-12) if err_all > 0 else 0.0

        # Fisher: confirmed vs combined blindspot (one-sided, blindspot error > confirmed)
        if n_conf > 0 and n_blind > 0:
            err_b_cnt = int((~sel_correct[mask_blind].astype(bool)).sum())
            ok_b_cnt  = n_blind - err_b_cnt
            err_c_cnt = int((~sel_correct[mask_conf].astype(bool)).sum())
            ok_c_cnt  = n_conf  - err_c_cnt
            _, pval = fisher_exact([[err_b_cnt, ok_b_cnt],
                                    [err_c_cnt, ok_c_cnt]],
                                   alternative="greater")
        else:
            pval = float("nan")

        # Effective rejection rate the deployer actually sees
        eff_reject_rate = n_blind / k if k > 0 else 0.0

        rows.append({
            "coverage":        cov,
            "n_total":         k,
            "n_insufficient":  n_insuf,
            "n_Bl_detected":   n_Bl,
            "n_confirmed":     n_conf,
            "n_blindspot":     n_blind,   # = n_insufficient + n_Bl_detected
            "error_all":          err_all,
            "error_insufficient": err_insuf,
            "error_Bl_detected":  err_Bl,
            "error_confirmed":    err_conf,
            "error_blindspot":    err_blind,
            "risk_reduction":     float(risk_reduction),
            "effective_reject_rate": float(eff_reject_rate),
            "p_value":            float(pval),
        })
    return rows


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def _print_table(scenario_name, rows):
    """Pretty-print one scenario's blindspot table with separate columns
    for insufficient and Bl_detected, plus effective rejection rate."""
    print(f"\n=== {scenario_name} ===")
    header = (
        f"  {'Cov':>4} | {'N':>5} | {'All':>6} | "
        f"{'Conf err% (n)':>18} | "
        f"{'Insuf err% (n)':>18} | "
        f"{'Bl-det err% (n)':>18} | "
        f"{'Rej%':>6} | {'Reduc':>7} | {'p':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        pval_str = "N/A" if np.isnan(r['p_value']) else (
            f"{r['p_value']:.1e}" if r['p_value'] < 1e-3 else f"{r['p_value']:.2e}")

        def _fmt(err, n):
            if np.isnan(err):
                return f"{'--':>6} ({n:>5})"
            return f"{err*100:>5.2f}% ({n:>5})"

        print(
            f"  {int(r['coverage']*100):>3}% | "
            f"{r['n_total']:>5} | "
            f"{r['error_all']*100:>5.1f}% | "
            f"{_fmt(r['error_confirmed'],    r['n_confirmed']):>18} | "
            f"{_fmt(r['error_insufficient'], r['n_insufficient']):>18} | "
            f"{_fmt(r['error_Bl_detected'],  r['n_Bl_detected']):>18} | "
            f"{r['effective_reject_rate']*100:>5.1f}% | "
            f"{r['risk_reduction']*100:>6.1f}% | "
            f"{pval_str:>9}"
        )


def run_posterior_blindspot(root: str, cfg: dict, method: str = "llla",
                            ood_fraction: float = 0.5,
                            ood_dataset: str = "test_ood"):
    eval_dir = os.path.join(root, "results", "eval")
    fig_dir  = os.path.join(root, "results", "figures")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tau_b and d_max from flags
    flag_id_path = os.path.join(root, "results", f"flags_{method}", "test_id.pt")
    flag_id_data = torch.load(flag_id_path, weights_only=False)
    tau_b = flag_id_data["tau_b"]
    d_max = flag_id_data["d_max"]
    print(f"tau_b = {tau_b:.4f}  (Youden's J on holdout LOO)")
    print(f"d_max = {d_max:.4f}  (holdout LOO knn_dist 95th percentile)")

    all_results = {}
    all_csv_rows = []

    # Load ID and OOD data (needed for Mixed scenario)
    pred_id = torch.load(os.path.join(root, "results", method, "test_id.pt"), weights_only=False)
    b_local_id = torch.load(os.path.join(root, "results", f"b_local_{method}", "test_id.pt"), weights_only=False)
    ood_basename = ood_dataset if ood_dataset != "test_ood" else "test_ood"
    pred_ood = torch.load(os.path.join(root, "results", method, f"{ood_basename}.pt"), weights_only=False)
    b_local_ood = torch.load(os.path.join(root, "results", f"b_local_{method}", f"{ood_basename}.pt"), weights_only=False)
    print(f"  [OOD] using {ood_basename}.pt ({len(pred_ood['correct'])} samples)")

    # ============================================================
    # Scenario 1: Mixed (configurable ID:OOD ratio)
    # ============================================================
    n_id  = len(pred_id["correct"])
    # Sample n_ood so that OOD makes up `ood_fraction` of the total pool.
    n_ood_requested = int(round(n_id * ood_fraction / (1.0 - ood_fraction)))
    n_ood = min(n_ood_requested, len(pred_ood["correct"]))
    rng_ood = np.random.default_rng(123)
    ood_idx = rng_ood.choice(len(pred_ood["correct"]), size=n_ood, replace=False)
    print(f"  [Mixed] ID={n_id}, OOD={n_ood}  "
          f"(ood_fraction={n_ood/(n_id+n_ood):.2f})")

    correct_ood = torch.zeros(n_ood, dtype=torch.bool)

    rng = np.random.default_rng(42)
    N_mix = n_id + n_ood
    perm = rng.permutation(N_mix)

    mixed_entropy = torch.cat([pred_id["pred_entropy"],
                                pred_ood["pred_entropy"][ood_idx]])[perm]
    mixed_correct = torch.cat([pred_id["correct"], correct_ood])[perm]
    mixed_b_hat   = torch.cat([b_local_id["b_hat"],
                                b_local_ood["b_hat"][ood_idx]])[perm]
    mixed_knn     = torch.cat([b_local_id["knn_dist"],
                                b_local_ood["knn_dist"][ood_idx]])[perm]

    acc_mix = mixed_correct.float().mean() * 100
    rows_mixed = analyze_blindspot(mixed_entropy, mixed_correct, mixed_b_hat,
                                   tau_b, mixed_knn, d_max)
    all_results[f"Mixed (50% ID + 50% OOD, N={N_mix}, acc={acc_mix:.1f}%)"] = rows_mixed
    for r in rows_mixed:
        r["scenario"] = "mixed"
        all_csv_rows.append(r)
    _print_table(f"Mixed (50% ID + 50% OOD) (N={N_mix}, acc={acc_mix:.1f}%)", rows_mixed)

    # ============================================================
    # CIFAR-10-C scenarios (severity 3 and 5, full size)
    # ============================================================
    cifar10c_dir = cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C")
    if not os.path.isabs(cifar10c_dir):
        cifar10c_dir = os.path.join(root, cifar10c_dir)

    if os.path.isdir(cifar10c_dir):
        from models.backbone import load_model
        from audit.b_local import BLocalEstimator, method_to_feature_model
        from models.feature_extractor import load_cached_features

        # Load model
        ckpt_path = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                                  "backbone_single.pt")
        model = load_model(ckpt_path, device)

        # Pre-fit BLocalEstimator once (shared across severities)
        feature_model = method_to_feature_model(method)
        ref_holdout = load_cached_features(root, feature_model, "holdout")
        ref_feats_ho = ref_holdout["features"].numpy()
        ho_path = os.path.join(root, "results", method, "holdout.pt")
        ho_result = torch.load(ho_path, weights_only=False)

        k = cfg.get("k_neighbors", 50)
        est = BLocalEstimator(k=k)
        est.fit(ref_feats_ho, ho_result["probs"], ho_result["labels"])

        for severity in [3, 5]:
            sev_label = f"CIFAR-10-C severity {severity}"
            print(f"\n=== Loading {sev_label} (all corruptions, full size) ===")
            imgs_c, labels_c = load_cifar10c_severity(cifar10c_dir, severity=severity)

            if imgs_c is None:
                print(f"  [skip] No data loaded for {sev_label}")
                continue

            n_total_c = len(imgs_c)
            print(f"  Total samples: {n_total_c}")
            print(f"  Running inference...")

            result_c = run_inference_on_array(model, imgs_c, labels_c, device)
            acc_c = result_c["correct"].float().mean() * 100
            print(f"  Accuracy: {acc_c:.1f}%")

            print(f"  Computing B_local...")
            b_result_c = est.estimate(result_c["features"])

            rows_c = analyze_blindspot(
                result_c["pred_entropy"], result_c["correct"],
                b_result_c["b_hat"], tau_b,
                b_result_c["knn_dist"], d_max,
            )
            scenario_key = f"cifar10c_s{severity}"
            all_results[f"{sev_label} (N={n_total_c}, acc={acc_c:.1f}%)"] = rows_c
            for r in rows_c:
                r["scenario"] = scenario_key
                all_csv_rows.append(r)
            _print_table(f"{sev_label} (N={n_total_c}, acc={acc_c:.1f}%)", rows_c)
    else:
        print(f"  [skip] CIFAR-10-C dir not found: {cifar10c_dir}")

    # Save CSV
    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(eval_dir, "posterior_blindspot.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nCSV saved: {csv_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla")
    parser.add_argument("--ood_fraction", type=float, default=0.5,
                        help="Fraction of OOD in the Mixed scenario (0.5 = 50/50, 0.2 = 80/20 ID/OOD)")
    parser.add_argument("--ood_dataset", default="test_ood",
                        help="OOD basename under results/{method}/ and results/b_local_{method}/; "
                             "'test_ood' (default, SVHN) or 'cifar100'")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_posterior_blindspot(root, cfg, args.method,
                            ood_fraction=args.ood_fraction,
                            ood_dataset=args.ood_dataset)
