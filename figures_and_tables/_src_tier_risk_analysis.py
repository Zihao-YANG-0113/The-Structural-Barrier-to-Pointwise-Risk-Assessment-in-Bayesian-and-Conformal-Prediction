"""
tier_risk_analysis.py
---------------------
Publication-quality analysis: within fixed uncertainty tiers, B_local
identifies risks invisible to Bayesian uncertainty measures.

Uses LLLA Bayesian outputs (pred_entropy, mutual_info) for tier selection.
Three tier-selection criteria (columns):
  - Predictive Entropy H[p̄] (total uncertainty)
  - Mutual Information MI (epistemic uncertainty)
  - Max Probability (confidence)

Datasets (rows):
  - CIFAR-10-C severity 3
  - CIFAR-10-C severity 5
  - Mixed (50% CIFAR-10 ID + 50% CIFAR-100 OOD, balanced)

Outputs:
  Figure: results/figures/tier_risk_curves.pdf
  Table:  results/eval/tier_risk_table.csv
  Curves: results/eval/tier_risk_curves.csv
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_rejection_curve(uncertainty, correct, b_hat, knn_dist, d_max,
                            tier_frac, reject_pcts):
    """
    Select tier by uncertainty (lowest = most certain), then reject in order
    of decreasing deployer-risk:
        risk = +inf   if knn_dist > d_max  (distance-flagged → rejected first)
             = b_hat  otherwise
    The rejection rate r% therefore counts distance-flagged samples against
    the deployer's budget (honest accounting).

    Returns list of error rates at [0] + reject_pcts (length N+1).
    """
    N = len(uncertainty)
    k_tier = max(1, int(tier_frac * N))
    sel = uncertainty.argsort()[:k_tier]

    b_in_tier       = b_hat[sel]
    knn_in_tier     = knn_dist[sel]
    correct_in_tier = correct[sel]

    # Composite risk: distance-flagged points get +inf so they are rejected first.
    risk = np.where(knn_in_tier > d_max, np.inf, b_in_tier)

    err_base = (~correct_in_tier).mean() * 100

    errs = [err_base]
    for rp in reject_pcts:
        n_reject = int(rp / 100 * k_tier)
        if n_reject >= k_tier or n_reject == 0:
            errs.append(err_base)
            continue
        # Keep the k_tier - n_reject samples with the smallest risk.
        keep_idx = risk.argsort()[:k_tier - n_reject]
        errs.append((~correct_in_tier[keep_idx]).mean() * 100)
    return errs


def compute_random_rejection_curve(uncertainty, correct, tier_frac, reject_pcts,
                                   n_trials=50, seed=0):
    """
    Random rejection baseline: within the tier, reject samples uniformly
    at random. Returns mean and std of error rates over n_trials.
    """
    N = len(uncertainty)
    k_tier = max(1, int(tier_frac * N))
    sel = uncertainty.argsort()[:k_tier]
    correct_in_tier = correct[sel]
    err_base = (~correct_in_tier).mean() * 100

    rng = np.random.default_rng(seed)
    means = [err_base]
    stds = [0.0]
    for rp in reject_pcts:
        n_reject = int(rp / 100 * k_tier)
        if n_reject >= k_tier or n_reject == 0:
            means.append(err_base)
            stds.append(0.0)
            continue
        trial_errs = []
        for _ in range(n_trials):
            keep = rng.choice(k_tier, size=k_tier - n_reject, replace=False)
            trial_errs.append((~correct_in_tier[keep]).mean() * 100)
        means.append(np.mean(trial_errs))
        stds.append(np.std(trial_errs))
    return np.array(means), np.array(stds)


def compute_uncertainty_rejection_curve(uncertainty, correct, knn_dist, d_max,
                                        tier_frac, reject_pcts):
    """
    Ablation counterpart to compute_rejection_curve: rank within the tier by
    the *same* uncertainty that selected the tier (highest-uncertainty rejected
    first), with the identical distance-flag pre-filter applied for fairness.

    Tests whether "continuing to use uncertainty" can replace B_local.
    """
    N = len(uncertainty)
    k_tier = max(1, int(tier_frac * N))
    sel = uncertainty.argsort()[:k_tier]

    u_in_tier       = uncertainty[sel]
    knn_in_tier     = knn_dist[sel]
    correct_in_tier = correct[sel]

    # Same pre-filter: distance-flagged points rejected first.
    risk = np.where(knn_in_tier > d_max, np.inf, u_in_tier)

    err_base = (~correct_in_tier).mean() * 100
    errs = [err_base]
    for rp in reject_pcts:
        n_reject = int(rp / 100 * k_tier)
        if n_reject >= k_tier or n_reject == 0:
            errs.append(err_base)
            continue
        keep_idx = risk.argsort()[:k_tier - n_reject]
        errs.append((~correct_in_tier[keep_idx]).mean() * 100)
    return errs


# ==================================================================== #
# Ablations A (Stage-2 rejector) and B (Stage-2 information source)    #
# ==================================================================== #

def _aps_set_size(probs, alpha):
    """Uncalibrated APS set size = min #top classes whose cumulative prob
    reaches 1-alpha. Larger = less confident, rejected first."""
    srt = np.sort(probs, axis=-1)[:, ::-1]
    cs = np.cumsum(srt, axis=-1)
    return (cs < (1.0 - alpha)).sum(axis=-1) + 1


def _reject_curve(score_in_tier, correct_in_tier, reject_pcts,
                  pre_filter_in_tier=None):
    """Within-tier rejection curve ordered by a generic score.
    Highest-score points rejected first. `pre_filter_in_tier` (bool mask over
    tier) forces those points to be rejected before anything else."""
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


def _kept_set(score_in_tier, n_reject, pre_filter_in_tier=None):
    score_ = np.asarray(score_in_tier, dtype=float).copy()
    if pre_filter_in_tier is not None:
        score_[pre_filter_in_tier] = np.inf
    k = len(score_)
    return set(np.argsort(score_)[:k - n_reject].tolist())


def _fisher_exclusive(correct_in_tier, keep_A, keep_B):
    """Fisher exact on exclusive-kept 2x2. alt: A kept-set has more correct
    than B kept-set *among the samples they disagree on* (overlap drops out)."""
    excl_A = np.array(sorted(keep_A - keep_B), dtype=int)
    excl_B = np.array(sorted(keep_B - keep_A), dtype=int)
    if len(excl_A) == 0 or len(excl_B) == 0:
        return float("nan"), 0
    a = int(correct_in_tier[excl_A].sum())
    b = int((~correct_in_tier[excl_A]).sum())
    c = int(correct_in_tier[excl_B].sum())
    d = int((~correct_in_tier[excl_B]).sum())
    _, pv = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(pv), len(excl_A)


def _run_ablations(datasets, ds_keys, ds_titles, d_max, reject_pcts, save_dir):
    """Run Ablations A (rejector) and B (component) at tier fracs {0.8,0.7,0.6}.
    One figure per tier per ablation. Writes to save_dir.

    Ablation A (within Coverage-X% tier, tier selected by Predictive Entropy):
        Random / Entropy / MI / APS size / B_local (ours, with dist. pre-filter)

    Ablation B (same tier):
        Stage-1 only / Distance-only / B_local-only / Full audit (ours)
    """
    os.makedirs(save_dir, exist_ok=True)
    ABL_TIER_FRACS = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    ABL_R = 30
    ABL_ALPHA_APS = 0.10

    # APS size precomputed per dataset
    for ds_key in ds_keys:
        datasets[ds_key]["aps_size"] = _aps_set_size(
            datasets[ds_key]["probs"], ABL_ALPHA_APS).astype(float)

    x_vals = np.array([0] + reject_pcts)
    r_idx_30 = list(x_vals).index(ABL_R)

    ablA_methods = [
        ("random",  "Random",                                "#9E9E9E", None, ":"),
        ("entropy", r"Entropy $H[\bar{p}]$",                 "#1E88E5", "o",  "-"),
        ("mi",      "Mutual Info (MI)",                      "#43A047", "^",  "-"),
        ("aps",     fr"APS size ($\alpha={ABL_ALPHA_APS}$)", "#8E24AA", "D",  "-"),
        ("blocal",  r"$\hat{B}_{local}$ (ours)",             "#FB8C00", "s",  "-"),
    ]
    ablB_methods = [
        ("stage1",     "Stage-1 only (no reject)",              "#9E9E9E", None, ":"),
        ("dist",       r"Distance only ($d_{\mathrm{knn}}$)",   "#1976D2", "o",  "--"),
        ("blocal_raw", r"$\hat{B}_{local}$ only (no dist.)",    "#C62828", "^",  "--"),
        ("full",       "Full audit (ours)",                     "#FB8C00", "s",  "-"),
    ]

    ablA_curves, ablA_summary = [], []
    ablB_curves, ablB_summary = [], []

    for tier_frac in ABL_TIER_FRACS:
        tier_pct = int(round(tier_frac * 100))
        print(f"\n{'-' * 70}\n  Coverage {tier_pct}% tier — ablations A & B\n{'-' * 70}")

        fig_A, axes_A = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
        fig_B, axes_B = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

        for ds_idx, ds_key in enumerate(ds_keys):
            ds = datasets[ds_key]
            N = len(ds["correct"])
            k_tier = max(1, int(tier_frac * N))
            sel = ds["pred_entropy"].argsort()[:k_tier]
            correct_in = ds["correct"][sel]
            dist_flag_in = ds["knn_dist"][sel] > d_max
            acc = ds["correct"].mean() * 100
            n_rej_R = int(ABL_R / 100 * k_tier)
            err_base = float((~correct_in).mean() * 100)

            rand_mean, rand_std = compute_random_rejection_curve(
                ds["pred_entropy"], ds["correct"], tier_frac, reject_pcts)

            # -------- Ablation A curves --------
            cA = {
                "random":  rand_mean,
                "entropy": _reject_curve(ds["pred_entropy"][sel], correct_in, reject_pcts),
                "mi":      _reject_curve(ds["mutual_info"][sel],  correct_in, reject_pcts),
                "aps":     _reject_curve(ds["aps_size"][sel],     correct_in, reject_pcts),
                "blocal":  _reject_curve(ds["b_hat"][sel], correct_in, reject_pcts,
                                         pre_filter_in_tier=dist_flag_in),
            }
            kA = {
                "entropy": _kept_set(ds["pred_entropy"][sel], n_rej_R),
                "mi":      _kept_set(ds["mutual_info"][sel],  n_rej_R),
                "aps":     _kept_set(ds["aps_size"][sel],     n_rej_R),
                "blocal":  _kept_set(ds["b_hat"][sel], n_rej_R,
                                     pre_filter_in_tier=dist_flag_in),
            }

            ax = axes_A[0, ds_idx]
            for key, label, color, marker, ls in ablA_methods:
                y = cA[key]
                if key == "random":
                    ax.plot(x_vals, y, color=color, linewidth=1.2, linestyle=ls,
                            alpha=0.75, label=label, zorder=2)
                    ax.fill_between(x_vals, y - rand_std, y + rand_std,
                                    color=color, alpha=0.18, zorder=1)
                else:
                    lw = 2.4 if key == "blocal" else 1.7
                    z = 5 if key == "blocal" else 3
                    ax.plot(x_vals, y, color=color, linewidth=lw, linestyle=ls,
                            marker=marker, markersize=5, markeredgewidth=0.6,
                            markeredgecolor="white", label=label, zorder=z)

            # Star at B_local minimum
            y_b = cA["blocal"]
            baseline = float(y_b[0]); best_i = int(y_b.argmin()); best_e = float(y_b[best_i])
            if baseline > 0 and best_e < baseline:
                rd = (baseline - best_e) / baseline * 100
                ax.plot(x_vals[best_i], best_e, marker="*", color="#FB8C00",
                        markersize=11, zorder=6, markeredgecolor="white",
                        markeredgewidth=0.4)
                ax.annotate(f"$-${rd:.0f}%", xy=(x_vals[best_i], best_e),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=8, fontweight="bold", color="#FB8C00",
                            ha="center", va="bottom", zorder=7)

            ax.set_title(f"{ds_titles[ds_key]} (acc={acc:.1f}%)",
                         fontsize=11, fontweight="bold", pad=6)
            ax.set_xlabel(r"Within-tier Rejection (%)", fontsize=10)
            ax.set_ylabel("Error Rate (%)", fontsize=9.5)
            ax.grid(alpha=0.18, linewidth=0.5)
            ax.set_xlim(-1, 42)
            ax.tick_params(labelsize=8.5)
            ax.legend(fontsize=7, framealpha=0.85, edgecolor="0.8",
                      loc="upper right", handlelength=2.2)

            # Record + stats
            for rp_idx, rp in enumerate(x_vals):
                for key, *_ in ablA_methods:
                    ablA_curves.append({
                        "tier_pct": tier_pct, "dataset": ds_key,
                        "method": key, "reject_pct": int(rp),
                        "err": float(cA[key][rp_idx]),
                    })
            for other in ["entropy", "mi", "aps"]:
                pv, n_ex = _fisher_exclusive(correct_in, kA["blocal"], kA[other])
                delta = float(cA[other][r_idx_30] - cA["blocal"][r_idx_30])
                ablA_summary.append({
                    "tier_pct": tier_pct, "dataset": ds_key,
                    "other_method": other,
                    "err_other_at_r30":  float(cA[other][r_idx_30]),
                    "err_blocal_at_r30": float(cA["blocal"][r_idx_30]),
                    "delta_other_minus_blocal_pp": delta,
                    "n_exclusive_each_side": int(n_ex),
                    "fisher_blocal_gt_p": pv,
                })
                print(f"  A cov{tier_pct} {ds_key:>5} B_local vs {other:>7}: "
                      f"Δerr@r30={delta:+.2f}pp  n_excl={n_ex}  p={pv:.2e}")

            # -------- Ablation B curves --------
            cB = {
                "stage1":     np.full_like(x_vals, err_base, dtype=float),
                "dist":       _reject_curve(ds["knn_dist"][sel], correct_in, reject_pcts),
                "blocal_raw": _reject_curve(ds["b_hat"][sel],    correct_in, reject_pcts),
                "full":       _reject_curve(ds["b_hat"][sel], correct_in, reject_pcts,
                                            pre_filter_in_tier=dist_flag_in),
            }
            kB = {
                "dist":       _kept_set(ds["knn_dist"][sel], n_rej_R),
                "blocal_raw": _kept_set(ds["b_hat"][sel],    n_rej_R),
                "full":       _kept_set(ds["b_hat"][sel], n_rej_R,
                                        pre_filter_in_tier=dist_flag_in),
            }

            ax2 = axes_B[0, ds_idx]
            for key, label, color, marker, ls in ablB_methods:
                y = cB[key]
                if key == "stage1":
                    ax2.plot(x_vals, y, color=color, linewidth=1.1,
                             linestyle=ls, alpha=0.8, label=label, zorder=2)
                else:
                    lw = 2.4 if key == "full" else 1.7
                    z = 5 if key == "full" else 3
                    ax2.plot(x_vals, y, color=color, linewidth=lw, linestyle=ls,
                             marker=marker, markersize=5, markeredgewidth=0.6,
                             markeredgecolor="white", label=label, zorder=z)

            y_f = cB["full"]
            baseline2 = float(y_f[0]); best_i2 = int(y_f.argmin()); best_e2 = float(y_f[best_i2])
            if baseline2 > 0 and best_e2 < baseline2:
                rd = (baseline2 - best_e2) / baseline2 * 100
                ax2.plot(x_vals[best_i2], best_e2, marker="*", color="#FB8C00",
                         markersize=11, zorder=6, markeredgecolor="white",
                         markeredgewidth=0.4)
                ax2.annotate(f"$-${rd:.0f}%", xy=(x_vals[best_i2], best_e2),
                             xytext=(0, 8), textcoords="offset points",
                             fontsize=8, fontweight="bold", color="#FB8C00",
                             ha="center", va="bottom", zorder=7)

            ax2.set_title(f"{ds_titles[ds_key]} (acc={acc:.1f}%)",
                          fontsize=11, fontweight="bold", pad=6)
            ax2.set_xlabel(r"Within-tier Rejection (%)", fontsize=10)
            ax2.set_ylabel("Error Rate (%)", fontsize=9.5)
            ax2.grid(alpha=0.18, linewidth=0.5)
            ax2.set_xlim(-1, 42)
            ax2.tick_params(labelsize=8.5)
            ax2.legend(fontsize=7, framealpha=0.85, edgecolor="0.8",
                       loc="upper right", handlelength=2.2)

            for rp_idx, rp in enumerate(x_vals):
                for key, *_ in ablB_methods:
                    ablB_curves.append({
                        "tier_pct": tier_pct, "dataset": ds_key,
                        "variant": key, "reject_pct": int(rp),
                        "err": float(cB[key][rp_idx]),
                    })
            for other in ["dist", "blocal_raw"]:
                pv, n_ex = _fisher_exclusive(correct_in, kB["full"], kB[other])
                delta = float(cB[other][r_idx_30] - cB["full"][r_idx_30])
                ablB_summary.append({
                    "tier_pct": tier_pct, "dataset": ds_key,
                    "variant_compared": other,
                    "err_other_at_r30": float(cB[other][r_idx_30]),
                    "err_full_at_r30":  float(cB["full"][r_idx_30]),
                    "delta_other_minus_full_pp": delta,
                    "n_exclusive_each_side": int(n_ex),
                    "fisher_full_gt_p": pv,
                })
                print(f"  B cov{tier_pct} {ds_key:>5} Full vs {other:>11}: "
                      f"Δerr@r30={delta:+.2f}pp  n_excl={n_ex}  p={pv:.2e}")

        fig_A.suptitle(f"Ablation A · Stage-2 rejector  (Coverage {tier_pct}% tier)",
                       fontsize=13, fontweight="bold", y=1.005)
        fig_A.tight_layout(h_pad=1.8, w_pad=1.2)
        for ext in ["pdf", "png"]:
            p = os.path.join(save_dir, f"ablation_A_rejectors_cov{tier_pct}.{ext}")
            fig_A.savefig(p, bbox_inches="tight", dpi=200)
            print(f"  saved: {p}")
        plt.close(fig_A)

        fig_B.suptitle(f"Ablation B · Stage-2 information source  (Coverage {tier_pct}% tier)",
                       fontsize=13, fontweight="bold", y=1.005)
        fig_B.tight_layout(h_pad=1.8, w_pad=1.2)
        for ext in ["pdf", "png"]:
            p = os.path.join(save_dir, f"ablation_B_components_cov{tier_pct}.{ext}")
            fig_B.savefig(p, bbox_inches="tight", dpi=200)
            print(f"  saved: {p}")
        plt.close(fig_B)

    # --- CSVs ---
    pd.DataFrame(ablA_curves).to_csv(
        os.path.join(save_dir, "ablation_A_rejectors_curves.csv"),
        index=False, float_format="%.4f")
    pd.DataFrame(ablA_summary).to_csv(
        os.path.join(save_dir, "ablation_A_rejectors_summary.csv"),
        index=False, float_format="%.4f")
    pd.DataFrame(ablB_curves).to_csv(
        os.path.join(save_dir, "ablation_B_components_curves.csv"),
        index=False, float_format="%.4f")
    pd.DataFrame(ablB_summary).to_csv(
        os.path.join(save_dir, "ablation_B_components_summary.csv"),
        index=False, float_format="%.4f")

    # --- LaTeX summary: Err @ r=30% across tier × dataset × method ---
    _meth_a_order = ["random", "entropy", "mi", "aps", "blocal"]
    _meth_a_name = {"random": "Random", "entropy": r"$H[\bar p]$",
                    "mi": "MI", "aps": "APS",
                    "blocal": r"$\hat{B}_{local}$ (ours)"}
    dfA = pd.DataFrame(ablA_curves)
    pivA = dfA[dfA["reject_pct"] == ABL_R].pivot_table(
        index=["tier_pct", "dataset"], columns="method", values="err"
    ).reindex(columns=_meth_a_order)
    tex_a = "\\begin{tabular}{ll|" + "c" * len(pivA.columns) + "}\n\\toprule\n"
    tex_a += "Coverage & Dataset & " + " & ".join(
        _meth_a_name[c] for c in pivA.columns) + " \\\\\n\\midrule\n"
    for (tp, dk), row in pivA.iterrows():
        tex_a += f"{tp}\\% & {ds_titles[dk]} & " + \
                 " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n"
    tex_a += "\\bottomrule\n\\end{tabular}\n"
    with open(os.path.join(save_dir, "ablation_A_err_at_r30.tex"),
              "w", encoding="utf-8") as f:
        f.write(tex_a)

    _var_b_order = ["stage1", "dist", "blocal_raw", "full"]
    _var_b_name = {"stage1": "Stage-1 only", "dist": r"Dist.\ only",
                   "blocal_raw": r"$\hat{B}_{local}$ only",
                   "full": "Full audit (ours)"}
    dfB = pd.DataFrame(ablB_curves)
    pivB = dfB[dfB["reject_pct"] == ABL_R].pivot_table(
        index=["tier_pct", "dataset"], columns="variant", values="err"
    ).reindex(columns=_var_b_order)
    tex_b = "\\begin{tabular}{ll|" + "c" * len(pivB.columns) + "}\n\\toprule\n"
    tex_b += "Coverage & Dataset & " + " & ".join(
        _var_b_name[c] for c in pivB.columns) + " \\\\\n\\midrule\n"
    for (tp, dk), row in pivB.iterrows():
        tex_b += f"{tp}\\% & {ds_titles[dk]} & " + \
                 " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n"
    tex_b += "\\bottomrule\n\\end{tabular}\n"
    with open(os.path.join(save_dir, "ablation_B_err_at_r30.tex"),
              "w", encoding="utf-8") as f:
        f.write(tex_b)

    print(f"\nAll ablation outputs saved to: {save_dir}")


def _run_per_method_tier_ablation(datasets, ds_keys, ds_titles, d_max,
                                  reject_pcts, save_dir):
    """
    Ablation: each Stage-1 ranker selects its OWN tier, then Stage 2 has
    two variants:
        (a) continue with the same metric (homogeneous pipeline)
        (b) swap in B_local as Stage-2 rejector (ours)

    This is the dual of Ablation A (where all rankers share an H[p̄]-selected
    tier): here the tier itself varies per ranker, isolating the value B_local
    adds as a Stage-2 component on top of ANY Stage-1 confidence ranker.

    Outputs:
        per_method_tier_cov{X}.{pdf,png}  — per-coverage 3-panel figure
        per_method_tier_sweep.{pdf,png}   — coverage on x-axis at fixed r=30%
        per_method_tier_curves.csv        — all curves for downstream tables
    """
    os.makedirs(save_dir, exist_ok=True)
    TIER_FRACS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    APS_ALPHA = 0.10
    R_SUMMARY = 30  # rejection budget used for the sweep figure

    for ds_key in ds_keys:
        if "aps_size" not in datasets[ds_key]:
            datasets[ds_key]["aps_size"] = _aps_set_size(
                datasets[ds_key]["probs"], APS_ALPHA).astype(float)

    # (key, label, data-field, color)
    rankers = [
        ("entropy", r"$H[\bar{p}]$",                   "pred_entropy", "#1E88E5"),
        ("mi",      "MI",                               "mutual_info",  "#43A047"),
        ("aps",     fr"APS ($\alpha={APS_ALPHA}$)",     "aps_size",     "#8E24AA"),
    ]
    x_vals = np.array([0] + reject_pcts)
    r_idx_30 = list(x_vals).index(R_SUMMARY)

    curve_rows = []

    for tier_frac in TIER_FRACS:
        tier_pct = int(round(tier_frac * 100))
        print(f"\n{'-' * 70}")
        print(f"  Per-method tier ablation · Coverage {tier_pct}%")
        print(f"{'-' * 70}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

        for ds_idx, ds_key in enumerate(ds_keys):
            ds = datasets[ds_key]
            N = len(ds["correct"])
            k_tier = max(1, int(tier_frac * N))
            acc = ds["correct"].mean() * 100

            ax = axes[0, ds_idx]

            for r_key, r_label, r_field, r_color in rankers:
                score = np.asarray(ds[r_field], dtype=float)
                sel = score.argsort()[:k_tier]
                correct_in = ds["correct"][sel]
                dist_flag = ds["knn_dist"][sel] > d_max
                err_base = float((~correct_in).mean() * 100)

                # Stage-2 variant (a): same metric continues
                curve_same = _reject_curve(score[sel], correct_in, reject_pcts)
                # Stage-2 variant (b): B_local swap (with distance pre-filter)
                curve_swap = _reject_curve(
                    ds["b_hat"][sel], correct_in, reject_pcts,
                    pre_filter_in_tier=dist_flag,
                )

                ax.plot(x_vals, curve_same, color=r_color, linestyle="--",
                        marker="o", markersize=4, linewidth=1.3, alpha=0.7,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=f"{r_label} → {r_label}", zorder=2)
                ax.plot(x_vals, curve_swap, color=r_color, linestyle="-",
                        marker="s", markersize=5, linewidth=2.2,
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=fr"{r_label} → $\hat{{B}}_{{local}}$", zorder=4)

                for rp_idx, rp in enumerate(x_vals):
                    curve_rows.append({
                        "tier_pct": tier_pct, "dataset": ds_key,
                        "ranker": r_key, "stage2": "same",
                        "reject_pct": int(rp),
                        "err": float(curve_same[rp_idx]),
                    })
                    curve_rows.append({
                        "tier_pct": tier_pct, "dataset": ds_key,
                        "ranker": r_key, "stage2": "blocal",
                        "reject_pct": int(rp),
                        "err": float(curve_swap[rp_idx]),
                    })

                delta = float(curve_same[r_idx_30] - curve_swap[r_idx_30])
                print(f"  cov{tier_pct} {ds_key:>5} {r_key:>7}: "
                      f"base={err_base:.2f}%, same@r30={curve_same[r_idx_30]:.2f}, "
                      f"swap@r30={curve_swap[r_idx_30]:.2f}  (Δ={delta:+.2f}pp)")

            ax.set_title(f"{ds_titles[ds_key]} (acc={acc:.1f}%)",
                         fontsize=11, fontweight="bold", pad=6)
            ax.set_xlabel("Within-tier Rejection (%)", fontsize=10)
            ax.set_ylabel("Error Rate (%)", fontsize=9.5)
            ax.grid(alpha=0.18, linewidth=0.5)
            ax.set_xlim(-1, 42)
            ax.tick_params(labelsize=8.5)
            ax.legend(fontsize=6.5, framealpha=0.88, edgecolor="0.8",
                      loc="upper right", handlelength=2.0, ncol=2)

        fig.suptitle(
            f"Per-method tier ablation · Coverage {tier_pct}%  "
            r"(Solid = Stage-2 $\hat{B}_{local}$; Dashed = Stage-2 same metric)",
            fontsize=12, fontweight="bold", y=1.005)
        fig.tight_layout(h_pad=1.8, w_pad=1.2)

        for ext in ["pdf", "png"]:
            p = os.path.join(save_dir, f"per_method_tier_cov{tier_pct}.{ext}")
            fig.savefig(p, bbox_inches="tight", dpi=200)
            print(f"  saved: {p}")
        plt.close(fig)

    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(
        os.path.join(save_dir, "per_method_tier_curves.csv"),
        index=False, float_format="%.4f")

    # ---------- Sweep figure: coverage on x-axis at fixed r=R_SUMMARY ----------
    fig_s, axes_s = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    for ds_idx, ds_key in enumerate(ds_keys):
        ax = axes_s[0, ds_idx]
        ds = datasets[ds_key]
        acc = ds["correct"].mean() * 100

        sub = curves_df[(curves_df["dataset"] == ds_key) &
                        (curves_df["reject_pct"] == R_SUMMARY)]

        for r_key, r_label, _, r_color in rankers:
            for stage2, ls, lw, marker, alpha in [
                ("same",   "--", 1.3, "o", 0.7),
                ("blocal", "-",  2.2, "s", 1.0),
            ]:
                q = (sub[(sub["ranker"] == r_key) & (sub["stage2"] == stage2)]
                     .sort_values("tier_pct"))
                if q.empty:
                    continue
                lbl = (f"{r_label} → {r_label}" if stage2 == "same"
                       else fr"{r_label} → $\hat{{B}}_{{local}}$")
                ax.plot(q["tier_pct"].values, q["err"].values,
                        color=r_color, linestyle=ls, linewidth=lw,
                        marker=marker, markersize=5, alpha=alpha,
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=lbl)

        ax.set_title(f"{ds_titles[ds_key]} (acc={acc:.1f}%)",
                     fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("Coverage Tier (%)", fontsize=10)
        ax.set_ylabel(f"Error Rate @ r={R_SUMMARY}% (%)", fontsize=9.5)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.tick_params(labelsize=8.5)
        ax.legend(fontsize=6.5, framealpha=0.88, edgecolor="0.8",
                  loc="best", handlelength=2.0, ncol=2)

    fig_s.suptitle(
        f"Per-method tier ablation · Error after r={R_SUMMARY}% "
        r"within-tier rejection across coverage tiers  "
        r"(Solid = Stage-2 $\hat{B}_{local}$; Dashed = Stage-2 same metric)",
        fontsize=11, fontweight="bold", y=1.005)
    fig_s.tight_layout(h_pad=1.8, w_pad=1.2)

    for ext in ["pdf", "png"]:
        p = os.path.join(save_dir, f"per_method_tier_sweep.{ext}")
        fig_s.savefig(p, bbox_inches="tight", dpi=200)
        print(f"  saved: {p}")
    plt.close(fig_s)


def _run_id_clean_ablation(datasets, d_max, reject_pcts, save_dir,
                           tier_fracs=(0.80, 0.50)):
    """In-distribution-only ablation on clean CIFAR-10 test.

    On this set knn_dist is near-uniformly small (everything in-dist), so the
    distance signal is muted — a setting where the rejector/component win-loss
    pattern can no longer be explained by distance alone. This isolates
    whether B_local's held-out labelled contrast adds value on hard ID
    samples (label ambiguity, intra-class confusion), which is the scenario
    the method is theoretically designed for.
    """
    os.makedirs(save_dir, exist_ok=True)
    ds = datasets["id_clean"]
    ds["aps_size"] = _aps_set_size(ds["probs"], 0.10).astype(float)

    ABL_R = 30
    x_vals = np.array([0] + reject_pcts)
    r_idx_30 = list(x_vals).index(ABL_R)

    n_tiers = len(tier_fracs)
    fig, axes = plt.subplots(n_tiers, 2, figsize=(11.5, 4.4 * n_tiers),
                             squeeze=False)

    id_curve_rows = []
    id_summary_rows = []

    print(f"\n{'=' * 70}")
    print("  In-distribution ablation · Clean CIFAR-10 test")
    print(f"{'=' * 70}")

    for t_idx, tier_frac in enumerate(tier_fracs):
        tier_pct = int(round(tier_frac * 100))
        N = len(ds["correct"])
        k_tier = max(1, int(tier_frac * N))
        sel = ds["pred_entropy"].argsort()[:k_tier]
        correct_in = ds["correct"][sel]
        knn_in = ds["knn_dist"][sel]
        dist_flag_in = knn_in > d_max
        n_flag = int(dist_flag_in.sum())
        err_base = float((~correct_in).mean() * 100)
        n_rej_R = int(ABL_R / 100 * k_tier)

        print(f"\n  tier {tier_pct}%: N_tier={k_tier}, "
              f"dist-flag={n_flag}/{k_tier} ({100*n_flag/k_tier:.2f}%), "
              f"knn_dist range=[{knn_in.min():.3f}, {knn_in.max():.3f}], "
              f"err_base={err_base:.2f}%")

        rand_mean, rand_std = compute_random_rejection_curve(
            ds["pred_entropy"], ds["correct"], tier_frac, reject_pcts)

        # ---- Ablation A: 5 rejectors ----
        cA = {
            "random":  rand_mean,
            "entropy": _reject_curve(ds["pred_entropy"][sel], correct_in, reject_pcts),
            "mi":      _reject_curve(ds["mutual_info"][sel],  correct_in, reject_pcts),
            "aps":     _reject_curve(ds["aps_size"][sel],     correct_in, reject_pcts),
            "blocal":  _reject_curve(ds["b_hat"][sel], correct_in, reject_pcts,
                                     pre_filter_in_tier=dist_flag_in),
        }
        kA = {
            "entropy": _kept_set(ds["pred_entropy"][sel], n_rej_R),
            "mi":      _kept_set(ds["mutual_info"][sel],  n_rej_R),
            "aps":     _kept_set(ds["aps_size"][sel],     n_rej_R),
            "blocal":  _kept_set(ds["b_hat"][sel], n_rej_R,
                                 pre_filter_in_tier=dist_flag_in),
        }

        axA = axes[t_idx, 0]
        axA.plot(x_vals, cA["random"], color="#9E9E9E", linestyle=":",
                 linewidth=1.2, alpha=0.75, label="Random", zorder=2)
        axA.fill_between(x_vals, cA["random"] - rand_std, cA["random"] + rand_std,
                         color="#9E9E9E", alpha=0.18, zorder=1)
        axA.plot(x_vals, cA["entropy"], color="#1E88E5", marker="o",
                 markersize=5, linewidth=1.7, markeredgecolor="white",
                 markeredgewidth=0.6, label=r"Entropy $H[\bar{p}]$", zorder=3)
        axA.plot(x_vals, cA["mi"], color="#43A047", marker="^",
                 markersize=5, linewidth=1.7, markeredgecolor="white",
                 markeredgewidth=0.6, label="Mutual Info (MI)", zorder=3)
        axA.plot(x_vals, cA["aps"], color="#8E24AA", marker="D",
                 markersize=5, linewidth=1.7, markeredgecolor="white",
                 markeredgewidth=0.6, label=r"APS size ($\alpha$=0.1)", zorder=3)
        axA.plot(x_vals, cA["blocal"], color="#FB8C00", marker="s",
                 markersize=5, linewidth=2.4, markeredgecolor="white",
                 markeredgewidth=0.6, label=r"$\hat{B}_{local}$ (ours)", zorder=5)

        y_b = cA["blocal"]
        baseline = float(y_b[0])
        best_i = int(y_b.argmin()); best_e = float(y_b[best_i])
        if baseline > 0 and best_e < baseline:
            rd = (baseline - best_e) / baseline * 100
            axA.plot(x_vals[best_i], best_e, marker="*", color="#FB8C00",
                     markersize=11, zorder=6, markeredgecolor="white",
                     markeredgewidth=0.4)
            axA.annotate(f"$-${rd:.0f}%", xy=(x_vals[best_i], best_e),
                         xytext=(0, 8), textcoords="offset points",
                         fontsize=8, fontweight="bold", color="#FB8C00",
                         ha="center", va="bottom")

        axA.set_title(f"Ablation A · rejectors (Coverage {tier_pct}%)",
                      fontsize=11, fontweight="bold", pad=6)
        axA.set_xlabel(r"Within-tier Rejection (%)", fontsize=10)
        axA.set_ylabel("Error Rate (%)", fontsize=9.5)
        axA.grid(alpha=0.18, linewidth=0.5)
        axA.set_xlim(-1, 42)
        axA.tick_params(labelsize=8.5)
        axA.legend(fontsize=7.5, framealpha=0.85, edgecolor="0.8",
                   loc="upper right", handlelength=2.2)

        # ---- Ablation B: 4 component variants ----
        cB = {
            "stage1":     np.full_like(x_vals, err_base, dtype=float),
            "dist":       _reject_curve(ds["knn_dist"][sel], correct_in, reject_pcts),
            "blocal_raw": _reject_curve(ds["b_hat"][sel],    correct_in, reject_pcts),
            "full":       _reject_curve(ds["b_hat"][sel], correct_in, reject_pcts,
                                        pre_filter_in_tier=dist_flag_in),
        }
        kB = {
            "dist":       _kept_set(ds["knn_dist"][sel], n_rej_R),
            "blocal_raw": _kept_set(ds["b_hat"][sel],    n_rej_R),
            "full":       _kept_set(ds["b_hat"][sel], n_rej_R,
                                    pre_filter_in_tier=dist_flag_in),
        }

        axB = axes[t_idx, 1]
        axB.plot(x_vals, cB["stage1"], color="#9E9E9E", linestyle=":",
                 linewidth=1.1, alpha=0.8, label="Stage-1 only", zorder=2)
        axB.plot(x_vals, cB["dist"], color="#1976D2", linestyle="--",
                 marker="o", markersize=5, linewidth=1.7,
                 markeredgecolor="white", markeredgewidth=0.6,
                 label=r"Distance only ($d_{\mathrm{knn}}$)", zorder=3)
        axB.plot(x_vals, cB["blocal_raw"], color="#C62828", linestyle="--",
                 marker="^", markersize=5, linewidth=1.7,
                 markeredgecolor="white", markeredgewidth=0.6,
                 label=r"$\hat{B}_{local}$ only (no dist.)", zorder=3)
        axB.plot(x_vals, cB["full"], color="#FB8C00", marker="s",
                 markersize=5, linewidth=2.4, markeredgecolor="white",
                 markeredgewidth=0.6, label="Full audit (ours)", zorder=5)

        y_f = cB["full"]
        baseline2 = err_base
        best_i2 = int(y_f.argmin()); best_e2 = float(y_f[best_i2])
        if baseline2 > 0 and best_e2 < baseline2:
            rd = (baseline2 - best_e2) / baseline2 * 100
            axB.plot(x_vals[best_i2], best_e2, marker="*", color="#FB8C00",
                     markersize=11, zorder=6, markeredgecolor="white",
                     markeredgewidth=0.4)
            axB.annotate(f"$-${rd:.0f}%", xy=(x_vals[best_i2], best_e2),
                         xytext=(0, 8), textcoords="offset points",
                         fontsize=8, fontweight="bold", color="#FB8C00",
                         ha="center", va="bottom")

        axB.set_title(f"Ablation B · components (Coverage {tier_pct}%)",
                      fontsize=11, fontweight="bold", pad=6)
        axB.set_xlabel(r"Within-tier Rejection (%)", fontsize=10)
        axB.set_ylabel("Error Rate (%)", fontsize=9.5)
        axB.grid(alpha=0.18, linewidth=0.5)
        axB.set_xlim(-1, 42)
        axB.tick_params(labelsize=8.5)
        axB.legend(fontsize=7.5, framealpha=0.85, edgecolor="0.8",
                   loc="upper right", handlelength=2.2)

        # Record rows + print deltas
        for rp_idx, rp in enumerate(x_vals):
            for key in ["random", "entropy", "mi", "aps", "blocal"]:
                id_curve_rows.append({
                    "tier_pct": tier_pct, "ablation": "A",
                    "method": key, "reject_pct": int(rp),
                    "err": float(cA[key][rp_idx]),
                })
            for key in ["stage1", "dist", "blocal_raw", "full"]:
                id_curve_rows.append({
                    "tier_pct": tier_pct, "ablation": "B",
                    "method": key, "reject_pct": int(rp),
                    "err": float(cB[key][rp_idx]),
                })

        print(f"    A err@r30: Random={cA['random'][r_idx_30]:.2f}  "
              f"H={cA['entropy'][r_idx_30]:.2f}  MI={cA['mi'][r_idx_30]:.2f}  "
              f"APS={cA['aps'][r_idx_30]:.2f}  B_loc={cA['blocal'][r_idx_30]:.2f}")
        print(f"    B err@r30: Stage1={cB['stage1'][r_idx_30]:.2f}  "
              f"Dist={cB['dist'][r_idx_30]:.2f}  "
              f"B_loc_only={cB['blocal_raw'][r_idx_30]:.2f}  "
              f"Full={cB['full'][r_idx_30]:.2f}")

        for other in ["entropy", "mi", "aps"]:
            pv, n_ex = _fisher_exclusive(correct_in, kA["blocal"], kA[other])
            delta = float(cA[other][r_idx_30] - cA["blocal"][r_idx_30])
            id_summary_rows.append({
                "tier_pct": tier_pct, "ablation": "A",
                "comparison": f"blocal_vs_{other}",
                "err_blocal": float(cA["blocal"][r_idx_30]),
                "err_other":  float(cA[other][r_idx_30]),
                "delta_other_minus_blocal_pp": delta,
                "n_excl": int(n_ex), "fisher_blocal_gt_p": pv,
            })
        for other in ["dist", "blocal_raw"]:
            pv, n_ex = _fisher_exclusive(correct_in, kB["full"], kB[other])
            delta = float(cB[other][r_idx_30] - cB["full"][r_idx_30])
            id_summary_rows.append({
                "tier_pct": tier_pct, "ablation": "B",
                "comparison": f"full_vs_{other}",
                "err_full":   float(cB["full"][r_idx_30]),
                "err_other":  float(cB[other][r_idx_30]),
                "delta_other_minus_full_pp": delta,
                "n_excl": int(n_ex), "fisher_full_gt_p": pv,
            })

    fig.suptitle(
        "In-distribution ablation · Clean CIFAR-10 test  "
        r"(knn distance signal muted; isolates $\hat{B}_{local}$'s labelled contrast)",
        fontsize=12.5, fontweight="bold", y=1.003)
    fig.tight_layout(h_pad=2.2, w_pad=1.4)
    for ext in ["pdf", "png"]:
        p = os.path.join(save_dir, f"ablation_id_clean.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"  saved: {p}")
    plt.close(fig)

    pd.DataFrame(id_curve_rows).to_csv(
        os.path.join(save_dir, "ablation_id_clean_curves.csv"),
        index=False, float_format="%.4f")
    pd.DataFrame(id_summary_rows).to_csv(
        os.path.join(save_dir, "ablation_id_clean_summary.csv"),
        index=False, float_format="%.4f")
    print(f"  In-dist ablation outputs → {save_dir}")


def _run_k_sensitivity(datasets, ds_keys, ds_titles,
                       ho_feats_multi, ho_probs, ho_labels,
                       reject_pcts, save_dir,
                       k_values=(5, 10, 25, 50, 100, 200),
                       tier_frac=0.80, r_pct=30):
    """k-sensitivity ablation for B_local.

    Refits the estimator at each k (features cached), recomputes tau and d_max
    from LOO on the holdout, then measures residual error at Coverage `tier_frac`
    tier, rejection budget `r_pct`, under the full-audit composite
    (distance-flag + B_local ordering). Stability across k demonstrates the
    audit signal is not driven by a single tuned neighbourhood choice.
    """
    from eval.b_local_ablation import BLocalEstimatorV2
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 70}\n  k-sensitivity ablation · Coverage {int(tier_frac*100)}% "
          f"tier, r={r_pct}%\n{'=' * 70}")

    rows = []
    err_base_by_ds = {}

    for k in k_values:
        print(f"\n  fitting B_local with k={k}...")
        est_k = BLocalEstimatorV2(k=k, use_pca=True, pca_dim=64,
                                  local_bandwidth=True)
        est_k.fit(ho_feats_multi, ho_probs, ho_labels)

        loo_k = est_k.estimate_loo()
        tau_p90_k = float(np.percentile(loo_k["b_hat"].numpy(), 90))
        d_max_k   = float(np.percentile(loo_k["knn_dist"].numpy(), 95))
        print(f"    tau_p90={tau_p90_k:.4f}, d_max={d_max_k:.4f}")

        for ds_key in ds_keys:
            ds = datasets[ds_key]
            feats = ds["raw_feats"]
            N = len(ds["correct"])
            k_tier = max(1, int(tier_frac * N))
            n_rej = int(r_pct / 100 * k_tier)

            bres = est_k.estimate(feats)
            b_hat_k = bres["b_hat"].numpy()
            knn_k   = bres["knn_dist"].numpy()

            # Canonical tier by predictive entropy
            sel = ds["pred_entropy"].argsort()[:k_tier]
            correct_in = ds["correct"][sel]
            b_in = b_hat_k[sel]
            dist_in = knn_k[sel]

            err_base = float((~correct_in).mean() * 100)
            err_base_by_ds[ds_key] = err_base

            # Full audit: dist-flag first, then b_hat
            risk = np.where(dist_in > d_max_k, np.inf, b_in)
            keep = np.argsort(risk)[:k_tier - n_rej]
            err_after = float((~correct_in[keep]).mean() * 100)

            rows.append({
                "k": k, "dataset": ds_key,
                "tier_pct": int(tier_frac * 100), "r_pct": r_pct,
                "err_base": err_base,
                "err_after": err_after,
                "delta_pp": err_base - err_after,
                "rel_drop_pct": (err_base - err_after) / err_base * 100
                                 if err_base > 0 else 0.0,
                "tau_p90": tau_p90_k, "d_max": d_max_k,
            })
            print(f"    {ds_key:>6}: err {err_base:.2f}% -> {err_after:.2f}%  "
                  f"(-{err_base - err_after:.2f}pp, "
                  f"rel -{(err_base-err_after)/err_base*100:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "k_sensitivity.csv"),
              index=False, float_format="%.4f")

    # ---- Bar chart: 1×3, one panel per dataset ----
    fig, axes = plt.subplots(1, len(ds_keys), figsize=(5 * len(ds_keys), 4.5),
                             squeeze=False)
    bar_color = "#C8D4E9"   # pale blue fill
    bar_edge  = "#3B5B82"   # darker same-hue for outline + value labels
    ref_color = "#555555"

    for ds_idx, ds_key in enumerate(ds_keys):
        ax = axes[0, ds_idx]
        sub = df[df["dataset"] == ds_key].sort_values("k")
        ks = sub["k"].values.astype(int)
        vals = sub["err_after"].values
        err_base = err_base_by_ds[ds_key]

        x_pos = np.arange(len(ks))
        bars = ax.bar(x_pos, vals, color=bar_color, edgecolor=bar_edge,
                      linewidth=0.9, zorder=3)

        # Reference: err at k=50 (anchor) as dashed horizontal
        if 50 in ks:
            anchor = float(sub[sub["k"] == 50]["err_after"].iloc[0])
            ax.axhline(anchor, color=ref_color, linestyle="--", linewidth=1.2,
                       alpha=0.7, zorder=2, label=f"k=50 anchor ({anchor:.2f}%)")

        # Reference: err_base (Stage-1) as dotted horizontal
        ax.axhline(err_base, color="gray", linestyle=":", linewidth=1.1,
                   alpha=0.7, zorder=2,
                   label=f"Stage-1 baseline ({err_base:.2f}%)")

        # Value labels on bars
        for xp, v in zip(x_pos, vals):
            ax.text(xp, v + (vals.max() * 0.012), f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color=bar_edge, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(int(kv)) for kv in ks])
        ax.set_xlabel(r"Neighbourhood size $k$", fontsize=10)
        ax.set_ylabel("Residual Error Rate (%)", fontsize=9.5)
        ax.set_title(f"{ds_titles[ds_key]} "
                     f"(acc={datasets[ds_key]['correct'].mean()*100:.1f}%)",
                     fontsize=11, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        # headroom for labels
        ax.set_ylim(0, max(vals.max(), err_base) * 1.18)
        ax.tick_params(labelsize=8.5)
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor="0.8",
                  loc="upper right")

    fig.suptitle(
        r"Sensitivity of $\hat{B}_{local}$-guided rejection to the "
        r"neighbourhood size $k$  "
        f"(Coverage {int(tier_frac*100)}% tier, " + r"$r=30\%$)",
        fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=1.4)
    for ext in ["pdf", "png"]:
        p = os.path.join(save_dir, f"k_sensitivity_bar.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"  saved: {p}")
    plt.close(fig)
    print(f"  k-sensitivity outputs → {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path)
    cfg = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from eval.b_local_ablation import (BLocalEstimatorV2,
                                        extract_multilayer_features,
                                        extract_multilayer_from_array)
    from eval.posterior_blindspot import load_cifar10c_severity
    from models.backbone import load_model
    from data.splits import get_loaders

    fig_dir = os.path.join(root, "results", "figures")
    eval_dir = os.path.join(root, "results", "eval")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Load model + fit B_local estimator
    ckpt = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                        "backbone_single.pt")
    model = load_model(ckpt, device)

    ho_result = torch.load(os.path.join(root, "results", "llla", "holdout.pt"),
                           weights_only=False)
    k = cfg.get("k_neighbors", 50)

    print("Extracting multilayer features (holdout)...")
    loaders = get_loaders(cfg, root)
    ho_feats_multi, _ = extract_multilayer_features(model, loaders["holdout"], device)

    est = BLocalEstimatorV2(k=k, use_pca=True, pca_dim=64, local_bandwidth=True)
    est.fit(ho_feats_multi, ho_result["probs"], ho_result["labels"])

    print("LOO on holdout for tau and d_max...")
    loo = est.estimate_loo()
    tau_p90 = float(np.percentile(loo["b_hat"].numpy(), 90))
    d_max   = float(np.percentile(loo["knn_dist"].numpy(), 95))
    print(f"tau_p90 = {tau_p90:.4f}  |  d_max (p95) = {d_max:.4f}")

    # ================================================================ #
    #  Load Bayesian outputs + compute B_local                          #
    # ================================================================ #

    datasets = {}
    llla_dir = os.path.join(root, "results", "llla")
    cifar10c_dir = cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C")
    if not os.path.isabs(cifar10c_dir):
        cifar10c_dir = os.path.join(root, cifar10c_dir)

    for sev in [3, 5]:
        print(f"\nLoading CIFAR-10-C sev {sev} (Bayesian)...")
        llla_data = torch.load(os.path.join(llla_dir, f"cifar10c_sev{sev}.pt"),
                               weights_only=False)
        imgs, labels = load_cifar10c_severity(cifar10c_dir, severity=sev)
        print("  Extracting multilayer features...")
        feats = extract_multilayer_from_array(model, imgs, device)
        print("  Computing B_local...")
        b_res = est.estimate(feats)

        datasets[f"sev{sev}"] = {
            "name": f"CIFAR-10-C Severity {sev}",
            "probs": llla_data["probs"].numpy(),
            "pred_entropy": llla_data["pred_entropy"].numpy(),
            "mutual_info": llla_data["mutual_info"].numpy(),
            "max_prob": llla_data["max_prob"].numpy(),
            "correct": llla_data["correct"].numpy().astype(bool),
            "b_hat": b_res["b_hat"].numpy(),
            "knn_dist": b_res["knn_dist"].numpy(),
            "raw_feats": feats,  # kept for k-sensitivity ablation
        }

    # Mixed: balanced 50% ID + 50% OOD
    print("\nLoading ID (test_id) and OOD (CIFAR-100) Bayesian outputs...")
    id_data = torch.load(os.path.join(llla_dir, "test_id.pt"), weights_only=False)
    ood_data = torch.load(os.path.join(llla_dir, "cifar100.pt"), weights_only=False)

    # B_local for ID
    print("  Extracting multilayer features (test_id)...")
    id_feats, _ = extract_multilayer_features(model, loaders["test_id"], device)
    print("  Computing B_local (test_id)...")
    id_b = est.estimate(id_feats)

    # B_local for OOD
    print("  Extracting multilayer features (CIFAR-100)...")
    from torchvision.datasets import CIFAR100
    ds_c100 = CIFAR100(root=os.path.join(root, "data", "raw"), train=False, download=False)
    ood_feats = extract_multilayer_from_array(model, ds_c100.data, device)
    print("  Computing B_local (CIFAR-100)...")
    ood_b = est.estimate(ood_feats)

    # Balance: take all 5000 ID + random 5000 from 10000 OOD
    n_id = len(id_data["pred_entropy"])
    rng = np.random.default_rng(42)
    ood_sel = rng.choice(len(ood_data["pred_entropy"]), size=n_id, replace=False)

    perm = rng.permutation(2 * n_id)
    mixed_pe = np.concatenate([
        id_data["pred_entropy"].numpy(),
        ood_data["pred_entropy"].numpy()[ood_sel]
    ])[perm]
    mixed_mi = np.concatenate([
        id_data["mutual_info"].numpy(),
        ood_data["mutual_info"].numpy()[ood_sel]
    ])[perm]
    mixed_mp = np.concatenate([
        id_data["max_prob"].numpy(),
        ood_data["max_prob"].numpy()[ood_sel]
    ])[perm]
    mixed_correct = np.concatenate([
        id_data["correct"].numpy().astype(bool),
        np.zeros(n_id, dtype=bool)  # OOD all wrong
    ])[perm]
    mixed_b = np.concatenate([
        id_b["b_hat"].numpy(),
        ood_b["b_hat"].numpy()[ood_sel]
    ])[perm]
    mixed_knn = np.concatenate([
        id_b["knn_dist"].numpy(),
        ood_b["knn_dist"].numpy()[ood_sel]
    ])[perm]
    mixed_probs = np.concatenate([
        id_data["probs"].numpy(),
        ood_data["probs"].numpy()[ood_sel]
    ], axis=0)[perm]
    mixed_feats = np.concatenate(
        [id_feats, ood_feats[ood_sel]], axis=0
    )[perm]

    datasets["mixed"] = {
        "name": "Mixed (50% ID + 50% OOD)",
        "probs": mixed_probs,
        "pred_entropy": mixed_pe,
        "mutual_info": mixed_mi,
        "max_prob": mixed_mp,
        "correct": mixed_correct,
        "b_hat": mixed_b,
        "knn_dist": mixed_knn,
        "raw_feats": mixed_feats,
    }

    # ID-clean (no corruption, no OOD) — for the in-distribution ablation.
    # On this set knn_dist is near-uniformly small, so the distance signal is
    # muted; this isolates whether B_local's labelled contrast adds value
    # beyond model-internal uncertainty on hard ID samples.
    datasets["id_clean"] = {
        "name": "CIFAR-10 test (ID clean)",
        "probs": id_data["probs"].numpy(),
        "pred_entropy": id_data["pred_entropy"].numpy(),
        "mutual_info": id_data["mutual_info"].numpy(),
        "max_prob": id_data["max_prob"].numpy(),
        "correct": id_data["correct"].numpy().astype(bool),
        "b_hat": id_b["b_hat"].numpy(),
        "knn_dist": id_b["knn_dist"].numpy(),
    }

    # ================================================================ #
    #  Uncertainty measures for tier selection                           #
    # ================================================================ #

    # For tier selection: lower = more certain (select lowest)
    # pred_entropy: H[p̄] total uncertainty ✓
    # mutual_info: MI epistemic uncertainty ✓
    # aleatoric: H[p̄] - MI, data (aleatoric) uncertainty ✓
    uncertainty_measures = [
        ("pred_entropy", "Predictive Entropy $H[\\bar{p}]$"),
        ("mutual_info", "Mutual Information (MI)"),
        ("aleatoric", "Aleatoric $H_{\\eta} = H[\\bar{p}] - \\mathrm{MI}$"),
    ]

    tier_pcts = [70, 80, 90]
    reject_pcts = [5, 10, 15, 20, 25, 30, 35, 40]
    tier_fracs = [0.70, 0.80, 0.90]
    tier_names = {0.70: "Coverage 70%", 0.80: "Coverage 80% (main)", 0.90: "Coverage 90%"}
    tier_colors = {0.70: "#E53935", 0.80: "#FB8C00", 0.90: "#1E88E5"}
    tier_markers = {0.70: "o", 0.80: "s", 0.90: "D"}

    table_rows = []
    all_curve_rows = []

    ds_keys = ["sev3", "sev5", "mixed"]
    ds_titles = {
        "sev3": "CIFAR-10-C Sev 3",
        "sev5": "CIFAR-10-C Sev 5",
        "mixed": "Mixed (stress)",
    }

    # ================================================================ #
    #  Table: Per-tier blindspot vs confirmed                           #
    # ================================================================ #

    for ds_key in ds_keys:
        ds = datasets[ds_key]
        correct = ds["correct"]
        b_hat = ds["b_hat"]
        knn_d = ds["knn_dist"]
        N = len(correct)

        print(f"\n{'='*70}")
        print(f"  {ds['name']} (N={N}, acc={correct.mean()*100:.1f}%)")
        print(f"{'='*70}")

        for um_key, um_label in uncertainty_measures:
            if um_key == "aleatoric":
                uncertainty = ds["pred_entropy"] - ds["mutual_info"]
            else:
                uncertainty = ds[um_key]

            print(f"\n  Sorted by: {um_label}")
            print(f"  {'Tier':>8} | {'N':>6} | {'All':>6} | {'Confirmed':>14} | "
                  f"{'Blindspot':>14} | {'Fisher p':>10} | {'AUROC(B)':>9}")
            print(f"  {'-'*78}")

            for tier_pct in tier_pcts:
                k_tier = max(1, int(tier_pct / 100 * N))
                sel = uncertainty.argsort()[:k_tier]

                # Deployer-honest: a point is blindspot if B̂ > tau OR knn_dist > d_max
                is_blind_tier = (b_hat[sel] > tau_p90) | (knn_d[sel] > d_max)
                n_blind = is_blind_tier.sum()
                n_conf = k_tier - n_blind

                err_all = (~correct[sel]).mean() * 100
                err_blind = (~correct[sel[is_blind_tier]]).mean() * 100 if n_blind > 0 else float('nan')
                err_conf = (~correct[sel[~is_blind_tier]]).mean() * 100 if n_conf > 0 else float('nan')

                if n_blind > 0 and n_conf > 0:
                    a = int((~correct[sel[is_blind_tier]]).sum())
                    b_cnt = int(correct[sel[is_blind_tier]].sum())
                    c = int((~correct[sel[~is_blind_tier]]).sum())
                    d = int(correct[sel[~is_blind_tier]].sum())
                    _, pval = fisher_exact([[a, b_cnt], [c, d]], alternative='greater')
                else:
                    pval = float('nan')

                is_wrong_tier = (~correct[sel]).astype(float)
                if len(np.unique(is_wrong_tier)) > 1:
                    auroc_b = roc_auc_score(is_wrong_tier, b_hat[sel])
                else:
                    auroc_b = float('nan')

                print(f"  Top {tier_pct:>3}% | {k_tier:>6} | {err_all:>5.1f}% | "
                      f"{err_conf:>5.1f}% ({n_conf:>5}) | {err_blind:>5.1f}% ({n_blind:>5}) | "
                      f"{pval:>10.1e} | {auroc_b:>9.4f}")

                table_rows.append({
                    "dataset": ds_key, "uncertainty": um_key, "tier_pct": tier_pct,
                    "n_tier": k_tier, "n_confirmed": int(n_conf), "n_blindspot": int(n_blind),
                    "err_all": err_all, "err_confirmed": err_conf, "err_blindspot": err_blind,
                    "fisher_p": pval, "auroc_b_within_tier": auroc_b,
                })

    # ================================================================ #
    #  Figure: 3 rows (datasets) x 3 cols (uncertainty measures)        #
    #  Each panel: 3 tier lines with baseline, shading, annotations     #
    # ================================================================ #

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

    # ================================================================ #
    #  Helper: plot one set of uncertainty measures                     #
    # ================================================================ #

    from matplotlib.lines import Line2D

    def _plot_risk_curves_horizontal(um_list, figsize, save_name, suptitle):
        """Horizontal layout: datasets as columns, uncertainty measures as rows.
        Used for the main figure (single uncertainty measure across 3 datasets)."""
        n_um = len(um_list)
        fig, axes = plt.subplots(n_um, 3, figsize=figsize, squeeze=False)

        for um_idx, (um_key, um_label) in enumerate(um_list):
            for ds_idx, ds_key in enumerate(ds_keys):
                ax = axes[um_idx, ds_idx]
                ds = datasets[ds_key]
                correct = ds["correct"]
                b_hat_ds = ds["b_hat"]
                acc = correct.mean() * 100

                if um_key == "aleatoric":
                    uncertainty = ds["pred_entropy"] - ds["mutual_info"]
                else:
                    uncertainty = ds[um_key]

                for tier_frac in tier_fracs:
                    tier_label = tier_names[tier_frac]
                    color = tier_colors[tier_frac]
                    marker = tier_markers[tier_frac]
                    x_vals = np.array([0] + reject_pcts)

                    errs = compute_rejection_curve(
                        uncertainty, correct, b_hat_ds,
                        ds["knn_dist"], d_max,
                        tier_frac, reject_pcts)
                    errs = np.array(errs)
                    rand_mean, _ = compute_random_rejection_curve(
                        uncertainty, correct, tier_frac, reject_pcts)

                    ax.plot(x_vals, rand_mean, color=color, linewidth=1.2,
                            linestyle="--", alpha=0.50, zorder=2)
                    ax.fill_between(x_vals, errs, rand_mean,
                                    where=(errs <= rand_mean),
                                    color=color, alpha=0.12, zorder=1)

                    baseline = errs[0]
                    ax.plot(x_vals, errs, color=color, linewidth=2.2,
                            marker=marker, markersize=5, markeredgewidth=0.6,
                            markeredgecolor="white",
                            label=f"{tier_label}", zorder=3)

                    best_idx = errs.argmin()
                    best_err = errs[best_idx]
                    if baseline > 0 and best_err < baseline:
                        rel_drop = (baseline - best_err) / baseline * 100
                        best_x = x_vals[best_idx]
                        ax.plot(best_x, best_err, marker="*", color=color,
                                markersize=10, zorder=4, markeredgecolor="white",
                                markeredgewidth=0.4)
                        # Simple placement: every label sits just ABOVE its star.
                        ax.annotate(
                            f"$-${rel_drop:.0f}%",
                            xy=(best_x, best_err),
                            xytext=(0, 8), textcoords="offset points",
                            fontsize=7.5, fontweight="bold", color=color,
                            ha="center", va="bottom", zorder=5,
                        )

                    for rp_idx, rp in enumerate(x_vals):
                        all_curve_rows.append({
                            "dataset": ds_key, "uncertainty": um_key,
                            "tier": tier_label, "reject_pct": int(rp),
                            "err": errs[rp_idx],
                        })

                handles, labels_leg = ax.get_legend_handles_labels()
                handles.append(Line2D([0], [0], color="gray", linewidth=1.3,
                                      linestyle="--", alpha=0.6))
                labels_leg.append("Random (dashed)")
                ax.legend(handles, labels_leg, fontsize=7, framealpha=0.85,
                          edgecolor="0.8", loc="upper right", handlelength=2.0)

                # No top title — caption goes under each panel (see below).
                ax.set_ylabel("Error Rate (%)", fontsize=9.5)
                ax.set_xlabel("$\\hat{B}_{local}$ Rejection (%)", fontsize=10)
                ax.grid(alpha=0.18, linewidth=0.5)
                ax.set_xlim(-1, 42)
                ax.tick_params(labelsize=8.5)

        # "(a) CIFAR-10-C Sev 3 (acc=73.3%)" style captions below each subplot.
        tags = ["(a)", "(b)", "(c)"]
        for col_idx, ds_key in enumerate(ds_keys):
            ax = axes[0, col_idx]
            ds = datasets[ds_key]
            acc = ds["correct"].mean() * 100
            caption = f"{tags[col_idx]} {ds_titles[ds_key]} (acc={acc:.1f}%)"
            ax.text(0.5, -0.22, caption, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", ha="center", va="top")

        fig.tight_layout(h_pad=1.8, w_pad=1.2)
        for ext in ["pdf", "png"]:
            path = os.path.join(fig_dir, f"{save_name}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            print(f"Figure saved: {path}")
        plt.close(fig)


    def _plot_risk_curves(um_list, fig_cols, figsize, save_name, suptitle):
        fig, axes = plt.subplots(3, fig_cols, figsize=figsize)
        if fig_cols == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, ds_key in enumerate(ds_keys):
            ds = datasets[ds_key]
            correct = ds["correct"]
            b_hat_ds = ds["b_hat"]
            N = len(correct)
            acc = correct.mean() * 100

            for col_idx, (um_key, um_label) in enumerate(um_list):
                ax = axes[row_idx, col_idx]

                if um_key == "aleatoric":
                    uncertainty = ds["pred_entropy"] - ds["mutual_info"]
                else:
                    uncertainty = ds[um_key]

                for tier_frac in tier_fracs:
                    tier_label = tier_names[tier_frac]
                    color = tier_colors[tier_frac]
                    marker = tier_markers[tier_frac]
                    x_vals = np.array([0] + reject_pcts)

                    errs = compute_rejection_curve(
                        uncertainty, correct, b_hat_ds,
                        ds["knn_dist"], d_max,
                        tier_frac, reject_pcts)
                    errs = np.array(errs)

                    rand_mean, _ = compute_random_rejection_curve(
                        uncertainty, correct, tier_frac, reject_pcts)
                    ax.plot(x_vals, rand_mean, color=color, linewidth=1.2,
                            linestyle="--", alpha=0.50, zorder=2)

                    ax.fill_between(x_vals, errs, rand_mean,
                                    where=(errs <= rand_mean),
                                    color=color, alpha=0.12, zorder=1)

                    baseline = errs[0]
                    ax.plot(x_vals, errs, color=color, linewidth=2.2,
                            marker=marker, markersize=5, markeredgewidth=0.6,
                            markeredgecolor="white",
                            label=f"{tier_label}",
                            zorder=3)

                    best_idx = errs.argmin()
                    best_err = errs[best_idx]
                    if baseline > 0 and best_err < baseline:
                        rel_drop = (baseline - best_err) / baseline * 100
                        best_x = x_vals[best_idx]
                        ax.plot(best_x, best_err, marker="*", color=color,
                                markersize=10, zorder=4, markeredgecolor="white",
                                markeredgewidth=0.4)
                        ax.annotate(
                            f"$-${rel_drop:.0f}%",
                            xy=(best_x, best_err),
                            xytext=(6, -10), textcoords="offset points",
                            fontsize=8, fontweight="bold", color=color,
                        )

                    for rp_idx, rp in enumerate(x_vals):
                        all_curve_rows.append({
                            "dataset": ds_key, "uncertainty": um_key,
                            "tier": tier_label, "reject_pct": int(rp),
                            "err": errs[rp_idx],
                        })

                handles, labels_leg = ax.get_legend_handles_labels()
                handles.append(Line2D([0], [0], color="gray", linewidth=1.3,
                                      linestyle="--", alpha=0.6))
                labels_leg.append("Random (dashed)")
                ax.legend(handles, labels_leg, fontsize=7, framealpha=0.85,
                          edgecolor="0.8", loc="upper right", handlelength=2.0)

                if row_idx == 0:
                    ax.set_title(um_label, fontsize=11.5, fontweight="bold", pad=10)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"{ds_titles[ds_key]}\n(acc={acc:.1f}%)\n\nError Rate (%)",
                        fontsize=9.5, labelpad=6)
                else:
                    ax.set_ylabel("Error Rate (%)", fontsize=9.5)
                if row_idx == 2:
                    ax.set_xlabel("$\\hat{B}_{local}$ Rejection (%)", fontsize=10)
                ax.grid(alpha=0.18, linewidth=0.5)
                ax.set_xlim(-1, 42)
                ax.tick_params(labelsize=8.5)

        fig.suptitle(suptitle, fontsize=13.5, fontweight="bold", y=1.005)
        fig.tight_layout(h_pad=1.8, w_pad=1.2)
        for ext in ["pdf", "png"]:
            path = os.path.join(fig_dir, f"{save_name}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            print(f"Figure saved: {path}")
        plt.close(fig)

    # ---- Main figure: MI, HORIZONTAL 1x3 (datasets as columns) ----
    _plot_risk_curves_horizontal(
        um_list=[("mutual_info", "Mutual Information (MI)")],
        figsize=(15, 4.5),
        save_name="tier_risk_curves",
        suptitle=(
            "Risk Reduction via $\\hat{B}_{\\ell}$ Rejection "
            "within MI Uncertainty Tiers"
        ),
    )

    # ---- Predictive-entropy figure (parallel, same layout) ----
    _plot_risk_curves_horizontal(
        um_list=[("pred_entropy", "Predictive Entropy $H[\\bar{p}]$")],
        figsize=(15, 4.5),
        save_name="tier_risk_curves_entropy",
        suptitle=(
            "Risk Reduction via $\\hat{B}_{\\ell}$ Rejection "
            "within Predictive-Entropy Tiers"
        ),
    )

    # ---- Appendix: H[p] and Aleatoric curve data still logged to CSV above ----
    # Produce Max.red summary as a compact LaTeX table instead of a figure.
    for um_key, _label in [("pred_entropy", "H[p]"),
                           ("aleatoric", "Aleatoric")]:
        for ds_key in ds_keys:
            ds = datasets[ds_key]
            correct = ds["correct"]
            b_hat_ds = ds["b_hat"]
            if um_key == "aleatoric":
                uncertainty = ds["pred_entropy"] - ds["mutual_info"]
            else:
                uncertainty = ds[um_key]
            for tier_frac in tier_fracs:
                errs = np.array(compute_rejection_curve(
                    uncertainty, correct, b_hat_ds,
                    ds["knn_dist"], d_max,
                    tier_frac, reject_pcts))
                x_vals = np.array([0] + reject_pcts)
                for rp_idx, rp in enumerate(x_vals):
                    all_curve_rows.append({
                        "dataset": ds_key, "uncertainty": um_key,
                        "tier": tier_names[tier_frac], "reject_pct": int(rp),
                        "err": errs[rp_idx],
                    })

    # Build the Max.red summary table from curve data
    df_curves_all = pd.DataFrame(all_curve_rows)
    tex_rows = []
    for ds_key in ds_keys:
        for ti, tier_frac in enumerate(tier_fracs):
            tier_label = tier_names[tier_frac]
            row_cells = [ds_titles[ds_key] if ti == 0 else "",
                         tier_label]
            for um_key in ["pred_entropy", "aleatoric"]:
                sub = df_curves_all[
                    (df_curves_all["dataset"] == ds_key) &
                    (df_curves_all["uncertainty"] == um_key) &
                    (df_curves_all["tier"] == tier_label)
                ].sort_values("reject_pct")
                if len(sub) == 0:
                    row_cells += ["--", "--"]
                    continue
                baseline = sub.iloc[0]["err"]
                i_best = sub["err"].values.argmin()
                best_err = sub.iloc[i_best]["err"]
                best_rp = int(sub.iloc[i_best]["reject_pct"])
                if baseline > 0 and best_err < baseline:
                    maxred = (baseline - best_err) / baseline * 100
                    row_cells += [f"{maxred:.1f}", str(best_rp)]
                else:
                    row_cells += ["--", "--"]
            tex_rows.append(row_cells)

    tex_body = " \\\\\n".join(" & ".join(r) for r in tex_rows) + " \\\\"
    tex = (
        "\\begin{tabular}{ll|cc|cc}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c|}{Predictive Entropy $H[\\bar{p}]$}"
        " & \\multicolumn{2}{c}{Aleatoric $H_{\\eta}$} \\\\\n"
        "Dataset & Tier & Max.red (\\%) & @$r$ (\\%)"
        " & Max.red (\\%) & @$r$ (\\%) \\\\\n"
        "\\midrule\n"
        + tex_body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    tex_path = os.path.join(eval_dir, "tier_risk_maxred_by_uncertainty.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX table saved: {tex_path}")

    # ================================================================ #
    #  ABLATIONS                                                        #
    #  A. Stage-2 rejector     : Random / H / MI / APS / B_local        #
    #  B. Stage-2 info source  : Stage-1 / Dist. / B_local / Full       #
    #  Runs at tier fracs {0.80, 0.70, 0.60}; 1 figure per tier per     #
    #  ablation. Outputs saved to the user's Desktop.                   #
    # ================================================================ #
    desktop_abl_dir = os.path.join(
        os.path.expanduser("~"), "Desktop", "Thesis-Ablations"
    )
    _run_ablations(datasets, ds_keys, ds_titles, d_max, reject_pcts,
                   desktop_abl_dir)
    _run_per_method_tier_ablation(datasets, ds_keys, ds_titles, d_max,
                                  reject_pcts, desktop_abl_dir)
    _run_id_clean_ablation(datasets, d_max, reject_pcts, desktop_abl_dir,
                           tier_fracs=(0.80, 0.50))
    _run_k_sensitivity(
        datasets, ds_keys, ds_titles,
        ho_feats_multi, ho_result["probs"], ho_result["labels"],
        reject_pcts, desktop_abl_dir,
        k_values=(5, 10, 25, 50, 100, 200),
        tier_frac=0.80, r_pct=30,
    )


    # Save CSVs
    df_table = pd.DataFrame(table_rows)
    csv1 = os.path.join(eval_dir, "tier_risk_table.csv")
    df_table.to_csv(csv1, index=False, float_format="%.4f")
    print(f"Table CSV saved: {csv1}")

    df_curves = pd.DataFrame(all_curve_rows)
    csv2 = os.path.join(eval_dir, "tier_risk_curves.csv")
    df_curves.to_csv(csv2, index=False, float_format="%.4f")
    print(f"Curves CSV saved: {csv2}")


if __name__ == "__main__":
    main()
