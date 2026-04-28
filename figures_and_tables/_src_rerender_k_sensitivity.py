"""Render k-sensitivity bar chart with bar fill #C5DFF4."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

abl_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Thesis-Ablations")
csv_path = os.path.join(abl_dir, "k_sensitivity.csv")
out_dir = os.path.expanduser("~/Desktop")

df = pd.read_csv(csv_path)

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

FS_CAPTION   = 13.0
FS_LEGEND    = 8.5
FS_XLABEL    = 11.5
FS_YLABEL    = 11.0
FS_TICK      = 10.0
FS_ANNOTATE  = 9.0

ds_keys = ["sev3", "sev5", "mixed"]
DS_TITLES = {
    "sev3":  "CIFAR-10-C Sev 3",
    "sev5":  "CIFAR-10-C Sev 5",
    "mixed": "Mixed (stress)",
}
DS_ACC = {"sev3": 73.3, "sev5": 53.1, "mixed": 47.2}

bar_color = "#C5DFF4"
bar_edge  = "#25557F"   # same hue, darker — matches the palette recipe used elsewhere
ref_color = "#555555"
suffix    = "c5dff4"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)

for ds_idx, ds_key in enumerate(ds_keys):
    ax = axes[0, ds_idx]
    sub = df[df["dataset"] == ds_key].sort_values("k")
    ks = sub["k"].values.astype(int)
    vals = sub["err_after"].values
    err_base = float(sub["err_base"].iloc[0])

    x_pos = np.arange(len(ks))
    ax.bar(x_pos, vals, color=bar_color, edgecolor=bar_edge,
           linewidth=0.9, zorder=3)

    if 50 in ks:
        anchor = float(sub[sub["k"] == 50]["err_after"].iloc[0])
        ax.axhline(anchor, color=ref_color, linestyle="--", linewidth=1.2,
                   alpha=0.7, zorder=2,
                   label=f"$k$=50 anchor ({anchor:.2f}%)")

    ax.axhline(err_base, color="gray", linestyle=":", linewidth=1.1,
               alpha=0.7, zorder=2,
               label=f"Stage-1 baseline ({err_base:.2f}%)")

    for xp, v in zip(x_pos, vals):
        ax.text(xp, v + (vals.max() * 0.012), f"{v:.2f}",
                ha="center", va="bottom", fontsize=FS_ANNOTATE,
                color=bar_edge, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(kv)) for kv in ks])
    ax.set_xlabel(r"Neighbourhood size $k$", fontsize=FS_XLABEL)
    ax.set_ylabel("Residual Error Rate (%)", fontsize=FS_YLABEL)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals.max(), err_base) * 1.18)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.9, edgecolor="0.8",
              loc="upper right", handlelength=2.1)

tags = ["(a)", "(b)", "(c)"]
for col_idx, ds_key in enumerate(ds_keys):
    ax = axes[0, col_idx]
    caption = f"{tags[col_idx]} {DS_TITLES[ds_key]} (acc={DS_ACC[ds_key]:.1f}%)"
    ax.text(0.5, -0.22, caption, transform=ax.transAxes,
            fontsize=FS_CAPTION, fontweight="bold",
            ha="center", va="top")

fig.tight_layout(h_pad=1.6, w_pad=1.2)

for ext in ["pdf", "png"]:
    p = os.path.join(out_dir, f"k_sensitivity_bar_{suffix}.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=220)
    print(f"saved: {p}")
plt.close(fig)
