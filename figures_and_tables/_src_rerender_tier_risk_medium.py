"""Re-render the main tier-risk figure with *medium*-sized fonts.

Same layout and aspect ratio as the original tier_risk_curves figure
(figsize = 15 x 4.5); only font sizes are bumped modestly for readability.

Writes to results/figures/tier_risk_curves_medium.{pdf,png}, leaves both
tier_risk_curves.{pdf,png} (original) and tier_risk_curves_large.{pdf,png}
untouched.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
csv_path = os.path.join(root, "results", "eval", "tier_risk_curves.csv")
fig_dir  = os.path.join(root, "results", "figures")
os.makedirs(fig_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df["uncertainty"] == "mutual_info"]   # main figure uses MI tier

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

TIER_NAMES   = {"Coverage 70%":         0.70,
                "Coverage 80% (main)":  0.80,
                "Coverage 90%":         0.90}
TIER_COLORS  = {0.70: "#E53935", 0.80: "#FB8C00", 0.90: "#1E88E5"}
TIER_MARKERS = {0.70: "o",       0.80: "s",       0.90: "D"}

ds_keys = ["sev3", "sev5", "mixed"]
DS_TITLES = {
    "sev3":  "CIFAR-10-C Sev 3",
    "sev5":  "CIFAR-10-C Sev 5",
    "mixed": "Mixed (stress)",
}
DS_ACC = {"sev3": 73.3, "sev5": 53.1, "mixed": 47.2}

# ----- Medium font sizes (between original and large) -----
# original / medium / large for reference:
#   caption   11   / 13   / 15.5
#   legend     7   /  8.5 / 10
#   xlabel    10   / 11.5 / 13.5
#   ylabel     9.5 / 11   / 13
#   tick       8.5 / 10   / 11.5
#   annotate   7.5 /  9   / 10.5
FS_CAPTION   = 13.0
FS_LEGEND    = 8.5
FS_XLABEL    = 11.5
FS_YLABEL    = 11.0
FS_TICK      = 10.0
FS_ANNOTATE  = 9.0

# Preserve original aspect ratio
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

for ds_idx, ds_key in enumerate(ds_keys):
    ax = axes[0, ds_idx]
    sub_ds = df[df["dataset"] == ds_key]

    for tier_label, tier_frac in TIER_NAMES.items():
        sub = sub_ds[sub_ds["tier"] == tier_label].sort_values("reject_pct")
        if sub.empty:
            continue
        x_vals = sub["reject_pct"].values
        errs   = sub["err"].values
        color  = TIER_COLORS[tier_frac]
        marker = TIER_MARKERS[tier_frac]
        baseline = float(errs[0])
        rand_mean = np.full_like(errs, baseline, dtype=float)

        ax.plot(x_vals, rand_mean, color=color, linewidth=1.2,
                linestyle="--", alpha=0.50, zorder=2)
        ax.fill_between(x_vals, errs, rand_mean,
                        where=(errs <= rand_mean),
                        color=color, alpha=0.12, zorder=1)
        ax.plot(x_vals, errs, color=color, linewidth=2.3,
                marker=marker, markersize=5.5, markeredgewidth=0.65,
                markeredgecolor="white",
                label=tier_label, zorder=3)

        best_idx = int(errs.argmin())
        best_err = float(errs[best_idx])
        if baseline > 0 and best_err < baseline:
            rel_drop = (baseline - best_err) / baseline * 100
            ax.plot(x_vals[best_idx], best_err, marker="*", color=color,
                    markersize=11, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.4)
            ax.annotate(
                f"$-${rel_drop:.0f}%",
                xy=(x_vals[best_idx], best_err),
                xytext=(0, 9), textcoords="offset points",
                fontsize=FS_ANNOTATE, fontweight="bold", color=color,
                ha="center", va="bottom", zorder=5,
            )

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

# Panel captions — tight to xlabel, no wasted gap
tags = ["(a)", "(b)", "(c)"]
for col_idx, ds_key in enumerate(ds_keys):
    ax = axes[0, col_idx]
    caption = f"{tags[col_idx]} {DS_TITLES[ds_key]} (acc={DS_ACC[ds_key]:.1f}%)"
    ax.text(0.5, -0.22, caption, transform=ax.transAxes,
            fontsize=FS_CAPTION, fontweight="bold",
            ha="center", va="top")

fig.tight_layout(h_pad=1.6, w_pad=1.2)

for ext in ["pdf", "png"]:
    p = os.path.join(fig_dir, f"tier_risk_curves_medium.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=220)
    print(f"saved: {p}")
plt.close(fig)
