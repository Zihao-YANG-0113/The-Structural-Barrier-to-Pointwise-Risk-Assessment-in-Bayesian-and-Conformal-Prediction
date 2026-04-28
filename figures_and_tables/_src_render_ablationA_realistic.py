"""Ablation A · Stage-2 rejector — third panel = Mixed (realistic, 80/20 ID/OOD).

Uses the SAME multilayer-feature B_local pipeline as the original sev3/sev5 and
mixed-stress panels in tier_risk_analysis.py, so panel (c) is directly comparable
to panels (a)/(b):

    1. Load backbone model
    2. Extract multilayer features (concat layer3 GAP + avgpool, 768-d)
       for holdout, test_id, CIFAR-100
    3. Fit BLocalEstimatorV2 on holdout, get d_max via LOO p95
    4. Estimate b_hat for test_id + CIFAR-100 with that estimator
    5. Build realistic 80/20 ID/OOD mix
    6. Compute rejector curves; render alongside sev3/sev5 (read from CSV)

Multilayer features and b_hat are cached to results/realistic_cache/. With cache
present, re-render is seconds (no model inference).

Outputs (originals untouched):
    ~/Desktop/Thesis-Ablations/ablation_A_rejectors_cov{TIER}_realistic_muted.{pdf,png}
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ABL_DIR    = os.path.join(os.path.expanduser("~"), "Desktop", "Thesis-Ablations")
CSV_PATH   = os.path.join(ABL_DIR, "ablation_A_rejectors_curves.csv")
LLLA_DIR   = os.path.join(ROOT, "results", "llla")
CACHE_DIR  = os.path.join(ROOT, "results", "realistic_cache")
CKPT_DIR   = os.path.join(ROOT, "checkpoints")
CFG_PATH   = os.path.join(ROOT, "configs", "default.yaml")

# ---------------- Knobs ----------------
ALPHA_APS    = 0.10
REJECT_PCTS  = [5, 10, 15, 20, 25, 30, 35, 40]
SEED_PERM    = 42
SEED_OOD_SEL = 123
K_NEIGHBORS  = 50  # matches cfg "k_neighbors" default; override via cfg if changed

# mode → (third-panel title, ood ratio, output filename suffix)
MODES = {
    "realistic": ("Mixed (realistic)", 0.20, "realistic_muted"),
    "id_only":   ("ID only (test_id)", 0.00, "id_only_muted"),
}


# ---------------- Helpers (copied verbatim from tier_risk_analysis) ----------------
def _aps_set_size(probs, alpha):
    srt = np.sort(probs, axis=-1)[:, ::-1]
    cs = np.cumsum(srt, axis=-1)
    return (cs < (1.0 - alpha)).sum(axis=-1) + 1


def _reject_curve(score_in_tier, correct_in_tier, reject_pcts,
                  pre_filter_in_tier=None):
    score_ = np.asarray(score_in_tier, dtype=float).copy()
    if pre_filter_in_tier is not None:
        score_[pre_filter_in_tier] = np.inf
    k = len(score_)
    err_base = (~correct_in_tier).mean() * 100
    errs = [err_base]
    for rp in reject_pcts:
        n_rej = int(rp / 100 * k)
        if n_rej == 0 or n_rej >= k:
            errs.append(err_base); continue
        keep = np.argsort(score_)[:k - n_rej]
        errs.append((~correct_in_tier[keep]).mean() * 100)
    return np.array(errs)


def compute_random_rejection_curve(uncertainty, correct, tier_frac, reject_pcts,
                                   n_trials=50, seed=0):
    N = len(uncertainty)
    k_tier = max(1, int(tier_frac * N))
    sel = uncertainty.argsort()[:k_tier]
    correct_in = correct[sel]
    err_base = (~correct_in).mean() * 100
    rng = np.random.default_rng(seed)
    means, stds = [err_base], [0.0]
    for rp in reject_pcts:
        n_rej = int(rp / 100 * k_tier)
        if n_rej >= k_tier or n_rej == 0:
            means.append(err_base); stds.append(0.0); continue
        trials = []
        for _ in range(n_trials):
            keep = rng.choice(k_tier, size=k_tier - n_rej, replace=False)
            trials.append((~correct_in[keep]).mean() * 100)
        means.append(np.mean(trials)); stds.append(np.std(trials))
    return np.array(means), np.array(stds)


# ---------------- Multilayer B_local pipeline (matches tier_risk_analysis) ----------------
def compute_multilayer_blocal(force=False):
    """Build the cache: multilayer features + b_hat/knn_dist for test_id and
    cifar100, plus d_max from holdout LOO p95. Same recipe as tier_risk_analysis.

    Cache contents:
        cache.npz: id_b_hat, id_knn_dist, ood_b_hat, ood_knn_dist, d_max
    """
    cache_path = os.path.join(CACHE_DIR, "id_ood_blocal_multilayer.npz")
    if (not force) and os.path.exists(cache_path):
        print(f"  [cache hit] {cache_path}")
        d = np.load(cache_path)
        return dict(d)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print("  Building multilayer B_local cache (this calls the model)...")

    from eval.b_local_ablation import (BLocalEstimatorV2,
                                        extract_multilayer_features,
                                        extract_multilayer_from_array)
    from models.backbone import load_model
    from data.splits import get_loaders

    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    k = cfg.get("k_neighbors", K_NEIGHBORS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(os.path.join(CKPT_DIR, "backbone_single.pt"), device)

    # --- holdout multilayer features (for fitting + LOO d_max) ---
    print("  Extracting multilayer features (holdout)...")
    loaders = get_loaders(cfg, ROOT)
    ho_feats, _ = extract_multilayer_features(model, loaders["holdout"], device)
    ho_result = torch.load(os.path.join(LLLA_DIR, "holdout.pt"), weights_only=False)

    est = BLocalEstimatorV2(k=k, use_pca=True, pca_dim=64, local_bandwidth=True)
    est.fit(ho_feats, ho_result["probs"], ho_result["labels"])

    print("  LOO on holdout for d_max (p95)...")
    loo = est.estimate_loo()
    d_max = float(np.percentile(loo["knn_dist"].numpy(), 95))
    print(f"  d_max = {d_max:.4f}")

    # --- ID multilayer features + b_hat ---
    print("  Extracting multilayer features (test_id)...")
    id_feats, _ = extract_multilayer_features(model, loaders["test_id"], device)
    id_b = est.estimate(id_feats)
    print(f"  test_id b_hat shape = {id_b['b_hat'].shape}")

    # --- OOD (CIFAR-100) multilayer features + b_hat ---
    print("  Extracting multilayer features (CIFAR-100)...")
    from torchvision.datasets import CIFAR100
    ds_c100 = CIFAR100(root=os.path.join(ROOT, "data", "raw"),
                       train=False, download=False)
    ood_feats = extract_multilayer_from_array(model, ds_c100.data, device)
    ood_b = est.estimate(ood_feats)
    print(f"  cifar100 b_hat shape = {ood_b['b_hat'].shape}")

    out = {
        "id_b_hat":    id_b["b_hat"].numpy(),
        "id_knn_dist": id_b["knn_dist"].numpy(),
        "ood_b_hat":   ood_b["b_hat"].numpy(),
        "ood_knn_dist": ood_b["knn_dist"].numpy(),
        "d_max":       np.array(d_max, dtype=float),
    }
    np.savez(cache_path, **out)
    print(f"  saved cache: {cache_path}")
    return out


# ---------------- Build third-panel dataset ----------------
def build_third_panel(blocal_cache, ood_ratio):
    """ood_ratio=0 → pure ID (test_id only); >0 → ID + OOD mix at given ratio."""
    id_data = torch.load(os.path.join(LLLA_DIR, "test_id.pt"), weights_only=False)
    n_id = len(id_data["pred_entropy"])

    if ood_ratio <= 0.0:
        out = {
            "pred_entropy": id_data["pred_entropy"].numpy(),
            "mutual_info":  id_data["mutual_info"].numpy(),
            "correct":      id_data["correct"].numpy().astype(bool),
            "b_hat":        blocal_cache["id_b_hat"],
            "knn_dist":     blocal_cache["id_knn_dist"],
            "aps_size":     _aps_set_size(id_data["probs"].numpy(), ALPHA_APS).astype(float),
        }
        print(f"  ID only: n_id={n_id}")
        print(f"  acc = {out['correct'].mean()*100:.2f}%")
        return out

    ood_data = torch.load(os.path.join(LLLA_DIR, "cifar100.pt"), weights_only=False)
    n_ood = int(round(n_id * ood_ratio / (1.0 - ood_ratio)))
    n_ood_avail = len(ood_data["pred_entropy"])
    if n_ood > n_ood_avail:
        raise RuntimeError(f"need {n_ood} OOD but only {n_ood_avail} available")

    rng_sel = np.random.default_rng(SEED_OOD_SEL)
    ood_sel = rng_sel.choice(n_ood_avail, size=n_ood, replace=False)
    rng = np.random.default_rng(SEED_PERM)
    perm = rng.permutation(n_id + n_ood)

    def _cat(id_arr, ood_arr):
        return np.concatenate([id_arr, ood_arr[ood_sel]])[perm]

    probs_mix = np.concatenate([id_data["probs"].numpy(),
                                 ood_data["probs"].numpy()[ood_sel]], axis=0)[perm]

    out = {
        "pred_entropy": _cat(id_data["pred_entropy"].numpy(),
                              ood_data["pred_entropy"].numpy()),
        "mutual_info":  _cat(id_data["mutual_info"].numpy(),
                              ood_data["mutual_info"].numpy()),
        "correct":      _cat(id_data["correct"].numpy().astype(bool),
                              np.zeros(n_ood_avail, dtype=bool)),
        "b_hat":        _cat(blocal_cache["id_b_hat"],
                              blocal_cache["ood_b_hat"]),
        "knn_dist":     _cat(blocal_cache["id_knn_dist"],
                              blocal_cache["ood_knn_dist"]),
        "aps_size":     _aps_set_size(probs_mix, ALPHA_APS).astype(float),
    }
    print(f"  realistic mix: n_id={n_id} + n_ood={n_ood} = {n_id+n_ood}")
    print(f"  acc = {out['correct'].mean()*100:.2f}%")
    return out


def compute_realistic_curves(ds, tier_frac, d_max):
    N = len(ds["correct"])
    k_tier = max(1, int(tier_frac * N))
    sel = ds["pred_entropy"].argsort()[:k_tier]
    correct_in = ds["correct"][sel]
    dist_flag_in = ds["knn_dist"][sel] > d_max

    rand_mean, _ = compute_random_rejection_curve(
        ds["pred_entropy"], ds["correct"], tier_frac, REJECT_PCTS)

    return {
        "random":  rand_mean,
        "entropy": _reject_curve(ds["pred_entropy"][sel], correct_in, REJECT_PCTS),
        "mi":      _reject_curve(ds["mutual_info"][sel],  correct_in, REJECT_PCTS),
        "aps":     _reject_curve(ds["aps_size"][sel],     correct_in, REJECT_PCTS),
        "blocal":  _reject_curve(ds["b_hat"][sel], correct_in, REJECT_PCTS,
                                 pre_filter_in_tier=dist_flag_in),
    }


# ---------------- Render ----------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
})

ablA_methods = [
    ("random",  "Random",                         "#9E9E9E", None, ":",  2, 0.75),
    ("aps",     r"APS size ($\alpha=0.1$)",       "#B087C5", "D",  "--", 3, 0.90),
    ("mi",      "Mutual Info (MI)",               "#7FB27F", "^",  "-",  4, 1.00),
    ("entropy", r"Entropy $H[\bar{p}]$",          "#6E9BC5", "o",  "-",  5, 1.00),
    ("blocal",  r"$\hat{B}_{local}$ (ours)",      "#FB8C00", "s",  "-",  6, 1.00),
]
DS_TITLES = {"sev3": "CIFAR-10-C Sev 3",
             "sev5": "CIFAR-10-C Sev 5"}
FIGSIZE = (15, 4.5)
X_VALS = np.array([0] + REJECT_PCTS)


def plot_one_panel(ax, panel_data, ds_key):
    for key, label, color, marker, ls, z, alpha in ablA_methods:
        if key not in panel_data:
            continue
        y = panel_data[key]
        if key == "random":
            ax.plot(X_VALS, y, color=color, linewidth=1.2, linestyle=ls,
                    alpha=alpha, label=label, zorder=z)
            std = float(np.std(y)) if len(y) > 1 else 0.0
            if std > 0:
                ax.fill_between(X_VALS, y - std, y + std,
                                color=color, alpha=0.15, zorder=1)
        else:
            lw = 2.4 if key == "blocal" else 1.7
            ax.plot(X_VALS, y, color=color, linewidth=lw, linestyle=ls,
                    marker=marker, markersize=5, markeredgewidth=0.6,
                    markeredgecolor="white", alpha=alpha,
                    label=label, zorder=z)

    if "blocal" in panel_data:
        y_b = panel_data["blocal"]
        baseline = float(y_b[0]); best_i = int(y_b.argmin()); best_e = float(y_b[best_i])
        if baseline > 0 and best_e < baseline:
            rd = (baseline - best_e) / baseline * 100
            ax.plot(X_VALS[best_i], best_e, marker="*", color="#FB8C00",
                    markersize=11, zorder=7, markeredgecolor="white",
                    markeredgewidth=0.4)
            ax.annotate(f"$-${rd:.0f}%",
                        xy=(X_VALS[best_i], best_e),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=8, fontweight="bold", color="#FB8C00",
                        ha="center", va="bottom", zorder=8)

    ax.set_title(DS_TITLES[ds_key], fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(r"Within-tier Rejection (%)", fontsize=10)
    ax.set_ylabel("Error Rate (%)", fontsize=9.5)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_xlim(-1, 42)
    ax.tick_params(labelsize=8.5)

    legend_order = ["random", "entropy", "mi", "aps", "blocal"]
    label_lookup = {m[0]: m[1] for m in ablA_methods}
    h_raw, l_raw = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(l_raw, h_raw))
    ordered_labels = [label_lookup[k] for k in legend_order
                       if label_lookup[k] in handle_by_label]
    ordered_handles = [handle_by_label[lbl] for lbl in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, fontsize=7,
              framealpha=0.85, edgecolor="0.8",
              loc="upper right", handlelength=2.2)


def main(mode="realistic", force_recompute=False):
    third_title, ood_ratio, suffix = MODES[mode]
    print(f"Mode: {mode}  (third panel = '{third_title}', OOD ratio = {ood_ratio})")

    blocal_cache = compute_multilayer_blocal(force=force_recompute)
    d_max = float(blocal_cache["d_max"])
    print(f"  using d_max = {d_max:.4f}")

    third_ds = build_third_panel(blocal_cache, ood_ratio)
    DS_TITLES["third"] = third_title

    df = pd.read_csv(CSV_PATH)
    tier_pcts = sorted(df["tier_pct"].unique().tolist(), reverse=True)
    for tier_pct in tier_pcts:
        tier_frac = tier_pct / 100.0
        sub_tier = df[df["tier_pct"] == tier_pct]
        third_curves = compute_realistic_curves(third_ds, tier_frac, d_max)

        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, squeeze=False)

        for ds_idx, ds_key in enumerate(["sev3", "sev5"]):
            sub_ds = sub_tier[sub_tier["dataset"] == ds_key]
            pivot = (sub_ds.pivot_table(index="reject_pct", columns="method",
                                         values="err").sort_index())
            panel_data = {col: pivot[col].values for col in pivot.columns}
            plot_one_panel(axes[0, ds_idx], panel_data, ds_key)

        plot_one_panel(axes[0, 2], third_curves, "third")

        fig.suptitle(
            f"Ablation A · Stage-2 rejector  (Coverage {tier_pct}% tier)",
            fontsize=13, fontweight="bold", y=1.005,
        )
        fig.tight_layout(h_pad=1.8, w_pad=1.2)

        for ext in ["pdf", "png"]:
            out = os.path.join(
                ABL_DIR, f"ablation_A_rejectors_cov{tier_pct}_{suffix}.{ext}")
            fig.savefig(out, bbox_inches="tight", dpi=200)
            print(f"saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()), default="realistic",
                        help="Third-panel mode: realistic (80/20) or id_only (100% ID).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute multilayer features + b_hat.")
    args = parser.parse_args()
    main(mode=args.mode, force_recompute=args.no_cache)
