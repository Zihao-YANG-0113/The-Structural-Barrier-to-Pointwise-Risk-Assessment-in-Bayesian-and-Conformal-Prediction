"""
llm_rejection_curves.py
-----------------------
Tier-risk rejection-curve figure for the LLM experiment
(BoolQ ID / PubMedQA OOD / Mixed) under three Stage-1 coverage tiers
(70% / 80% main / 90%).

Curve values are stored directly on the 5%-grid r ∈ {0,5,...,40}.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------------------------------------------------ #
# Style (copied from eval/tier_risk_analysis.py)                     #
# ------------------------------------------------------------------ #

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

TIER_FRACS   = [0.70, 0.80, 0.90]
TIER_NAMES   = {0.70: "Coverage 70%",
                0.80: "Coverage 80% (main)",
                0.90: "Coverage 90%"}
TIER_COLORS  = {0.70: "#E53935",
                0.80: "#FB8C00",
                0.90: "#1E88E5"}
TIER_MARKERS = {0.70: "o", 0.80: "s", 0.90: "D"}

X_VALS_FINE   = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])

# Font sizes — aligned with rerender_tier_risk_medium.py
FS_CAPTION   = 13.0
FS_LEGEND    = 8.5
FS_XLABEL    = 11.5
FS_YLABEL    = 11.0
FS_TICK      = 10.0
FS_ANNOTATE  = 9.0


# ------------------------------------------------------------------ #
# Data on the 5%-grid r ∈ {0, 5, 10, 15, 20, 25, 30, 35, 40}          #
# ------------------------------------------------------------------ #

DATA = {
    "boolq": {
        "title": "BoolQ (ID)",
        "acc":   88.0,
        0.70: [ 7.82,  6.95,  6.42,  5.88,  5.55,  5.50,  5.55,  5.65,  5.82],
        0.80: [10.02,  9.34,  8.78,  8.18,  7.78,  7.65,  7.58,  7.85,  9.10],
        0.90: [14.21, 13.05, 12.18, 11.32, 10.78, 10.85, 10.95, 11.18, 11.62],
    },
    "pubmedqa": {
        "title": "PubMedQA (OOD)",
        "acc":   62.5,
        0.70: [28.50, 25.20, 22.30, 19.65, 17.85, 16.95, 16.55, 16.78, 17.42],
        0.80: [33.01, 31.62, 28.85, 25.90, 23.05, 22.10, 20.27, 21.05, 22.95],
        0.90: [40.20, 36.65, 33.42, 30.10, 28.05, 27.45, 27.18, 27.50, 28.55],
    },
    "mixed": {
        "title": "Mixed",
        "acc":   73.0,
        0.70: [14.50, 12.10, 10.20,  8.75,  7.85,  7.50,  7.32,  7.55,  7.95],
        0.80: [22.00, 20.32, 16.85, 14.32, 12.18, 11.55, 11.41, 11.72, 12.55],
        0.90: [24.80, 21.70, 18.90, 16.90, 15.42, 14.85, 14.42, 14.95, 15.85],
    },
}


# ------------------------------------------------------------------ #
# Plot                                                               #
# ------------------------------------------------------------------ #

def plot(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ds_keys = ["boolq", "pubmedqa", "mixed"]
    tags = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

    for ds_idx, ds_key in enumerate(ds_keys):
        ax = axes[0, ds_idx]
        ds = DATA[ds_key]

        for tier_frac in TIER_FRACS:
            color  = TIER_COLORS[tier_frac]
            marker = TIER_MARKERS[tier_frac]
            label  = TIER_NAMES[tier_frac]

            errs = np.array(ds[tier_frac], dtype=float)

            # Random rejection baseline = flat at starting err
            baseline = errs[0]
            rand_mean = np.full_like(errs, baseline)

            # Shaded gain
            ax.plot(X_VALS_FINE, rand_mean, color=color, linewidth=1.2,
                    linestyle="--", alpha=0.50, zorder=2)
            ax.fill_between(X_VALS_FINE, errs, rand_mean,
                            where=(errs <= rand_mean),
                            color=color, alpha=0.12, zorder=1)

            # Main curve
            ax.plot(X_VALS_FINE, errs, color=color, linewidth=2.3,
                    marker=marker, markersize=5.5, markeredgewidth=0.65,
                    markeredgecolor="white",
                    label=label, zorder=3)

            # Star at minimum + max-reduction annotation
            best_idx = int(errs.argmin())
            best_err = float(errs[best_idx])
            if baseline > 0 and best_err < baseline:
                rel_drop = (baseline - best_err) / baseline * 100
                best_x = X_VALS_FINE[best_idx]
                ax.plot(best_x, best_err, marker="*", color=color,
                        markersize=11, zorder=4,
                        markeredgecolor="white", markeredgewidth=0.4)
                ax.annotate(
                    f"$-${rel_drop:.0f}%",
                    xy=(best_x, best_err),
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=FS_ANNOTATE, fontweight="bold", color=color,
                    ha="center", va="bottom", zorder=5,
                )

        # Legend
        handles, labels_leg = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="gray", linewidth=1.3,
                              linestyle="--", alpha=0.6))
        labels_leg.append("Random (dashed)")
        ax.legend(handles, labels_leg, fontsize=FS_LEGEND, framealpha=0.9,
                  edgecolor="0.8", loc="upper right", handlelength=2.1)

        ax.set_ylabel("Error Rate (%)", fontsize=FS_YLABEL)
        ax.set_xlabel(r"$\hat{B}_{local}$ Rejection (%)", fontsize=FS_XLABEL)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(-1, 42)
        ax.tick_params(labelsize=FS_TICK)

    # Panel captions below each subplot
    for col_idx, ds_key in enumerate(ds_keys):
        ax = axes[0, col_idx]
        ds = DATA[ds_key]
        caption = f"{tags[col_idx]} {ds['title']} (acc={ds['acc']:.1f}%)"
        ax.text(0.5, -0.22, caption, transform=ax.transAxes,
                fontsize=FS_CAPTION, fontweight="bold", ha="center", va="top")

    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    for ext in ["pdf", "png"]:
        path = os.path.join(save_dir, f"llm_rejection_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"Figure saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    save_dir = os.path.join(root, "results", "figures")
    plot(save_dir)
