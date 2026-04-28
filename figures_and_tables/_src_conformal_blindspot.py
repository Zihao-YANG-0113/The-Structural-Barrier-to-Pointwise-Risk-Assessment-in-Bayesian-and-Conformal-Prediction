"""
conformal_blindspot.py
----------------------
Does B_local add value to conformal prediction?

Core experiment (mirrors the Bayesian blindspot analysis):
  Among samples where conformal gives SMALL prediction sets (high confidence),
  can B_local identify the ones that are actually wrong?

  Bayesian side:   low entropy → split by B_local → error gap
  Conformal side:  small set   → split by B_local → error gap

For each scenario (test_id, CIFAR-10-C sev3/5, Mixed):
  1. Take top-K% smallest prediction sets (conformal says "confident")
  2. Within those, split by B_local flag
  3. Compare: error rate of "double confirmed" vs "blindspot"
  4. Fisher exact test for significance

Uses raw softmax (point estimate) for conformal — NOT Bayesian posteriors.
B_local flags come from the Bayesian audit pipeline (independent).

Output:
  results/eval/conformal_blindspot.csv
  results/figures/conformal_blindspot.{png,pdf}
"""

import os, sys, yaml, torch, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.conformal import aps_scores_batch, prediction_sets_from_scores


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Core analysis: among small-set samples, B_local finds errors       #
# ------------------------------------------------------------------ #

def analyze_conformal_blindspot(set_sizes, correct, b_hat, flags,
                                 coverages=(0.10, 0.20, 0.30, 0.50, 0.80)):
    """
    Mirror of Bayesian blindspot analysis, conformal version.
    Deployer-honest accounting: NO pre-filter. Tier is taken over the
    full pool, and 'blindspot' (to be rejected) = Bl_detected ∪ insufficient_data.
    Both types count against the deployer's rejection budget.
    """
    correct_np = correct.numpy() if isinstance(correct, torch.Tensor) else np.array(correct)
    b_np = b_hat.numpy() if isinstance(b_hat, torch.Tensor) else np.array(b_hat)
    flags_np = np.array(flags)
    sizes_np = set_sizes.numpy() if isinstance(set_sizes, torch.Tensor) else np.array(set_sizes)

    N = len(sizes_np)
    # Sort by set size ascending (smallest = most confident)
    sorted_idx = np.argsort(sizes_np)

    rows = []
    for cov in coverages:
        k = max(1, int(cov * N))
        sel = sorted_idx[:k]

        sel_correct = correct_np[sel]
        sel_flags   = flags_np[sel]
        sel_sizes   = sizes_np[sel]

        mask_insuf = sel_flags == "insufficient_data"
        mask_Bl    = sel_flags == "Bl_detected"
        mask_conf  = sel_flags == "trustworthy"
        mask_blind = mask_insuf | mask_Bl     # deployer rejects both

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

        risk_reduction = (err_all - err_conf) / (err_all + 1e-12) if err_all > 0 else 0.0

        # Fisher: confirmed vs combined blindspot
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

        eff_reject_rate = n_blind / k if k > 0 else 0.0

        rows.append({
            "coverage":        cov,
            "n_total":         k,
            "max_set_size":    int(sel_sizes.max()),
            "n_insufficient":  n_insuf,
            "n_Bl_detected":   n_Bl,
            "n_confirmed":     n_conf,
            "n_blindspot":     n_blind,
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
#  Softmax inference helpers                                           #
# ------------------------------------------------------------------ #

def run_softmax_inference(model, loader, device):
    """Get raw softmax probabilities from the base model (no Bayesian)."""
    import torch.nn.functional as F
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_probs.append(F.softmax(logits, dim=-1).cpu())
            all_labels.append(y)
    return torch.cat(all_probs), torch.cat(all_labels)


def run_softmax_on_images(model, images_np, labels_np, device, batch_size=256):
    """Get raw softmax from uint8 images."""
    import torch.nn.functional as F
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(images_np), batch_size):
            batch = images_np[start:start + batch_size]
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            logits = model(x.to(device))
            all_probs.append(F.softmax(logits, dim=-1).cpu())
    probs = torch.cat(all_probs)
    labels = torch.from_numpy(labels_np.astype(np.int64))
    correct = (probs.argmax(dim=-1) == labels)
    return probs, labels, correct


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(root, cfg, method="llla", ood_fraction: float = 0.5,
         ood_dataset: str = "test_ood"):
    eval_dir = os.path.join(root, "results", "eval")
    fig_dir  = os.path.join(root, "results", "figures")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =================================================================
    # Use raw softmax for conformal (standard, independent of Bayesian)
    # B_local flags from the Bayesian audit pipeline (independent)
    # =================================================================

    from models.backbone import load_model as load_backbone
    from data.splits import get_loaders, get_ood_loader

    ckpt_dir = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))
    backbone = load_backbone(os.path.join(ckpt_dir, "backbone_single.pt"), device)

    print("Getting raw softmax probabilities (no Bayesian)...")
    loaders = get_loaders(cfg, root)

    if ood_dataset == "cifar100":
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.datasets import CIFAR100
        c100 = CIFAR100(root=os.path.join(root, "data/raw"), train=False, download=False)
        imgs_t = torch.from_numpy(c100.data).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        imgs_norm = (imgs_t - mean) / std
        c100_ds = TensorDataset(imgs_norm, torch.tensor(c100.targets, dtype=torch.long))
        ood_loader = DataLoader(c100_ds, batch_size=cfg.get("batch_size", 128),
                                shuffle=False, num_workers=0)
        print(f"  [OOD] using CIFAR-100 test set ({len(c100_ds)} images)")
    else:
        ood_loader = get_ood_loader(cfg, root)
        print(f"  [OOD] using SVHN test set (default)")

    cal_probs, cal_labels = run_softmax_inference(backbone, loaders["calibration"], device)
    id_probs,  id_labels  = run_softmax_inference(backbone, loaders["test_id"], device)
    ood_probs, ood_labels = run_softmax_inference(backbone, ood_loader, device)

    id_correct  = (id_probs.argmax(dim=-1) == id_labels)
    ood_correct = torch.zeros(len(ood_labels), dtype=torch.bool)

    # Calibrate conformal on raw softmax
    alpha = cfg.get("alpha", 0.1)
    cal_scores = aps_scores_batch(cal_probs, cal_labels)
    n_cal = len(cal_scores)
    level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    quantile = float(torch.quantile(cal_scores, level))
    print(f"Conformal quantile (softmax, α={alpha}): {quantile:.4f}")

    # Load B_local flags (from Bayesian audit — independent)
    flag_id  = torch.load(os.path.join(root, "results", f"flags_{method}", "test_id.pt"),
                          weights_only=False)
    ood_basename = ood_dataset if ood_dataset != "test_ood" else "test_ood"
    flag_ood = torch.load(os.path.join(root, "results", f"flags_{method}", f"{ood_basename}.pt"),
                          weights_only=False)
    tau_b = flag_id["tau_b"]
    d_max = flag_id["d_max"]

    all_results = {}
    all_csv_rows = []

    def _print_table(name, rows):
        print(f"\n=== {name} ===")
        header = (f"  {'Coverage':>10} | {'N':>6} | {'MaxSet':>6} | {'All err%':>9} | "
                  f"{'Confirmed err% (n)':>22} | {'Blindspot err% (n)':>22} | "
                  f"{'Reduction':>10} | {'p-value':>10}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            pval_str = f"{r['p_value']:.2e}" if not np.isnan(r['p_value']) else "N/A"
            print(f"  Top {int(r['coverage']*100):>3}% | {r['n_total']:>5} | "
                  f"  ≤{r['max_set_size']:<3} | {r['error_all']*100:>7.2f}% | "
                  f"{r['error_confirmed']*100:>7.2f}% ({r['n_confirmed']:>6}) | "
                  f"{r['error_blindspot']*100:>7.2f}% ({r['n_blindspot']:>6}) | "
                  f"{r['risk_reduction']*100:>8.1f}% | {pval_str:>10}")

    # =================================================================
    # Scenario 1: test_id
    # =================================================================
    ps_id = prediction_sets_from_scores(id_probs, quantile)
    sizes_id = np.array([len(ps) for ps in ps_id])

    rows_id = analyze_conformal_blindspot(
        sizes_id, id_correct, flag_id["b_hat"], flag_id["flags"])
    all_results["ID (test)"] = rows_id
    for r in rows_id:
        r["scenario"] = "test_id"
        all_csv_rows.append(r)
    _print_table("ID (test)", rows_id)

    # =================================================================
    # Scenario 2: Mixed (ID + OOD, configurable ratio)
    # =================================================================
    n_id = len(id_labels)
    n_ood_requested = int(round(n_id * ood_fraction / (1.0 - ood_fraction)))
    n_ood = min(n_ood_requested, len(ood_labels))
    rng_ood = np.random.default_rng(123)
    ood_sel = rng_ood.choice(len(ood_labels), size=n_ood, replace=False)
    print(f"  [Mixed] ID={n_id}, OOD={n_ood}  "
          f"(ood_fraction={n_ood/(n_id+n_ood):.2f})")

    rng = np.random.default_rng(42)
    N_mix = n_id + n_ood
    perm = rng.permutation(N_mix)

    probs_mix   = torch.cat([id_probs, ood_probs[ood_sel]])[perm]
    labels_mix  = torch.cat([id_labels, ood_labels[ood_sel]])[perm]
    correct_mix = torch.cat([id_correct, ood_correct[ood_sel]])[perm]
    b_mix       = torch.cat([flag_id["b_hat"], flag_ood["b_hat"][ood_sel]])[perm]
    flags_mix   = np.concatenate([np.array(flag_id["flags"]),
                                  np.array(flag_ood["flags"])[ood_sel]])[perm]

    ps_mix = prediction_sets_from_scores(probs_mix, quantile)
    sizes_mix = np.array([len(ps) for ps in ps_mix])

    rows_mix = analyze_conformal_blindspot(
        sizes_mix, correct_mix, b_mix, flags_mix)
    all_results["Mixed (ID+OOD)"] = rows_mix
    for r in rows_mix:
        r["scenario"] = "mixed"
        all_csv_rows.append(r)
    _print_table("Mixed (ID+OOD)", rows_mix)

    # =================================================================
    # Scenario 3 & 4: CIFAR-10-C
    # =================================================================
    cifar10c_dir = cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C")
    if not os.path.isabs(cifar10c_dir):
        cifar10c_dir = os.path.join(root, cifar10c_dir)

    if os.path.isdir(cifar10c_dir):
        from audit.b_local import BLocalEstimator, method_to_feature_model
        from models.feature_extractor import load_cached_features
        from eval.blindspot_map import load_cifar10c_subsample, extract_features_from_images

        feat_model = method_to_feature_model(method)
        bb_holdout = load_cached_features(root, feat_model, "holdout")
        ho_result = torch.load(os.path.join(root, "results", method, "holdout.pt"),
                               weights_only=False)
        k = cfg.get("k_neighbors", 50)
        estimator = BLocalEstimator(k=k)
        estimator.fit(bb_holdout["features"].numpy(), ho_result["probs"], ho_result["labels"])

        def compute_flags_arr(b_hat_np, knn_dist_np):
            flags = []
            for b, d in zip(b_hat_np, knn_dist_np):
                if d > d_max:
                    flags.append("insufficient_data")
                elif b > tau_b:
                    flags.append("Bl_detected")
                else:
                    flags.append("trustworthy")
            return flags

        for severity in [3, 5]:
            print(f"\n=== CIFAR-10-C severity {severity} ===")

            n_sub = 10000
            imgs_sub, labels_sub_np, sub_idx = load_cifar10c_subsample(
                cifar10c_dir, severity, n_subsample=n_sub)

            # Raw softmax
            print(f"  Running raw softmax inference...")
            probs_sub, labels_sub, correct_sub = run_softmax_on_images(
                backbone, imgs_sub, labels_sub_np, device)

            # Conformal prediction sets
            ps_sub = prediction_sets_from_scores(probs_sub, quantile)
            sizes_sub = np.array([len(ps) for ps in ps_sub])

            # B_local
            print(f"  Extracting features & computing B_local...")
            ref_feats = extract_features_from_images(backbone, imgs_sub, device)
            b_result = estimator.estimate(ref_feats)
            flags_sub = compute_flags_arr(
                b_result["b_hat"].numpy(), b_result["knn_dist"].numpy())

            scen_label = f"CIFAR-10-C sev{severity}"
            rows_c = analyze_conformal_blindspot(
                sizes_sub, correct_sub, b_result["b_hat"], flags_sub)
            all_results[scen_label] = rows_c
            for r in rows_c:
                r["scenario"] = f"cifar10c_s{severity}"
                all_csv_rows.append(r)
            _print_table(scen_label, rows_c)

    # Save
    df = pd.DataFrame(all_csv_rows)
    csv_path = os.path.join(eval_dir, "conformal_blindspot.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla")
    parser.add_argument("--ood_fraction", type=float, default=0.5)
    parser.add_argument("--ood_dataset", default="test_ood",
                        help="'test_ood' (SVHN, default) or 'cifar100'")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path)
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(root, cfg, args.method, ood_fraction=args.ood_fraction,
         ood_dataset=args.ood_dataset)
