"""
blindspot_map.py
----------------
Visualise WHERE posterior blindspots occur in feature space.

For each of three scenarios (CIFAR-10-C sev3, sev5, Mixed ID+OOD):
  4-panel t-SNE figure: [Class labels | MI | B_local | Flag]
  + highlight an example point: low MI but B_local-flagged

Also: per-class B_local bar chart (ID test set).

Output:
  results/figures/blindspot_map_{scenario}.{png,pdf}
"""

import os, sys, yaml, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_CMAP = plt.cm.tab10

CORRUPTION_TYPES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur",
    "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur",
]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Per-scenario .npz cache                                             #
# ------------------------------------------------------------------ #
#  The expensive parts of this script are feature extraction and t-SNE
#  (~minutes/scenario). We cache every input to plot_scenario so that
#  styling-only re-renders skip both.

def save_scenario_cache(cache_dir, key, Z, labels, pred_entropy,
                        b_hat, flags, correct, tau_b, scenario_name):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, f"{key}.npz"),
             Z=Z, labels=labels, pred_entropy=pred_entropy,
             b_hat=b_hat, flags=np.array(flags),
             correct=correct, tau_b=tau_b,
             scenario_name=scenario_name)


def load_scenario_cache(cache_dir, key):
    p = os.path.join(cache_dir, f"{key}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return dict(
        Z=d["Z"], labels=d["labels"], pred_entropy=d["pred_entropy"],
        b_hat=d["b_hat"], flags=list(d["flags"]),
        correct=d["correct"], tau_b=float(d["tau_b"]),
        scenario_name=str(d["scenario_name"]),
    )


# ------------------------------------------------------------------ #
#  Extract features from raw images                                    #
# ------------------------------------------------------------------ #

def extract_features_from_images(model, images_np, device, batch_size=256):
    """Run model on uint8 images, return 512-d features via avgpool hook."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    model.eval()
    hook_feats = []
    def _hook(module, inp, out):
        hook_feats.append(out.squeeze(-1).squeeze(-1).detach().cpu())
    handle = model.avgpool.register_forward_hook(_hook)

    with torch.no_grad():
        for start in range(0, len(images_np), batch_size):
            batch = images_np[start:start + batch_size]
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            model(x.to(device))
    handle.remove()
    return torch.cat(hook_feats, dim=0).numpy()


def load_cifar10c_subsample(cifar10c_dir, severity, n_subsample=5000, seed=42):
    """Load CIFAR-10-C images at given severity, subsample n points."""
    all_imgs, all_labels = [], []
    for corruption in CORRUPTION_TYPES:
        img_path = os.path.join(cifar10c_dir, f"{corruption}.npy")
        if not os.path.exists(img_path):
            continue
        imgs = np.load(img_path)
        labels = np.load(os.path.join(cifar10c_dir, "labels.npy"))
        start = (severity - 1) * 10000
        all_imgs.append(imgs[start:start + 10000])
        all_labels.append(labels[start:start + 10000])

    imgs = np.concatenate(all_imgs)
    labels = np.concatenate(all_labels)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(imgs), size=min(n_subsample, len(imgs)), replace=False)
    idx.sort()
    return imgs[idx], labels[idx], idx


# ------------------------------------------------------------------ #
#  Plot one scenario                                                   #
# ------------------------------------------------------------------ #

def find_example_point(mi, b_hat, correct, tau_b):
    """Find a point with low MI but high B_local and wrong prediction."""
    mi_np = mi if isinstance(mi, np.ndarray) else mi.numpy()
    b_np  = b_hat if isinstance(b_hat, np.ndarray) else b_hat.numpy()
    c_np  = correct if isinstance(correct, np.ndarray) else correct.numpy()

    # Candidates: MI in bottom 30%, B_local > tau_b, wrong
    mi_thresh = np.percentile(mi_np, 30)
    mask = (mi_np < mi_thresh) & (b_np > tau_b) & (~c_np)
    if mask.sum() == 0:
        # Relax: allow correct predictions too
        mask = (mi_np < mi_thresh) & (b_np > tau_b)
    if mask.sum() == 0:
        return None
    # Pick the one with highest B_local among candidates
    candidates = np.where(mask)[0]
    best = candidates[b_np[candidates].argmax()]
    return best


def plot_scenario(Z, labels, pred_entropy, b_hat, flags, correct, tau_b,
                  scenario_name, fig_dir, n_classes=10, is_cifar10c=True):
    """
    Create 3-panel t-SNE figure for one scenario:
        (a) Predictive entropy  (b) B_local magnitude  (c) Flag category.
    Uses constrained_layout so all three scatter axes share the same size
    despite each having its own colorbar / legend.
    """
    # Font sizes aligned with rerender_tier_risk_medium / k_sensitivity_medium.
    # Colorbar labels are placed ABOVE each colorbar (horizontal) so they can
    # be read at the larger size without crowding the rotated axis.
    FS_CAPTION    = 14.0
    FS_LEGEND     = 12.0
    FS_TICK       = 10.0
    FS_ANNOTATE   = 13.0
    FS_CBAR_LABEL = 11.0

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    s = 4
    alpha = 0.55

    # Convert to numpy
    ent_np = pred_entropy if isinstance(pred_entropy, np.ndarray) else np.array(pred_entropy)
    b_np   = b_hat if isinstance(b_hat, np.ndarray) else np.array(b_hat)
    flags_np = np.array(flags)
    correct_np = correct if isinstance(correct, np.ndarray) else np.array(correct)

    # Find an illustrative point (low entropy but wrong)
    ex_idx = find_example_point(ent_np, b_np, correct_np, tau_b)

    # ---- (a) Predictive entropy ----
    ax = axes[0]
    vmin_e, vmax_e = np.percentile(ent_np, 2), np.percentile(ent_np, 98)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=ent_np, cmap="YlOrRd",
                    vmin=vmin_e, vmax=vmax_e, s=s, alpha=alpha,
                    edgecolors="none", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.set_title(r"$H[\bar{p}]$", fontsize=FS_CBAR_LABEL, pad=6)
    cbar.ax.tick_params(labelsize=FS_TICK)
    ax.set_xticks([]); ax.set_yticks([])
    if ex_idx is not None:
        ax.scatter(Z[ex_idx, 0], Z[ex_idx, 1], s=120, facecolors="none",
                   edgecolors="blue", linewidths=2.0, zorder=10)
        ax.annotate(f"$H[\\bar{{p}}]$={ent_np[ex_idx]:.3f}\n(confident)",
                    xy=(Z[ex_idx, 0], Z[ex_idx, 1]),
                    xytext=(45, 50), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=FS_ANNOTATE, color="blue", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

    # ---- (b) B_local magnitude ----
    ax = axes[1]
    vmin_b, vmax_b = np.percentile(b_np, 2), np.percentile(b_np, 98)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=b_np, cmap="RdYlGn_r",
                    vmin=vmin_b, vmax=vmax_b, s=s, alpha=alpha,
                    edgecolors="none", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.set_title(r"$\hat{B}_{local}$", fontsize=FS_CBAR_LABEL, pad=6)
    cbar.ax.tick_params(labelsize=FS_TICK)
    ax.set_xticks([]); ax.set_yticks([])
    if ex_idx is not None:
        ax.scatter(Z[ex_idx, 0], Z[ex_idx, 1], s=120, facecolors="none",
                   edgecolors="blue", linewidths=2.0, zorder=10)
        ax.annotate(f"$B_l$={b_np[ex_idx]:.3f}\n(blindspot)",
                    xy=(Z[ex_idx, 0], Z[ex_idx, 1]),
                    xytext=(45, 50), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=FS_ANNOTATE, color="blue", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.2))

    # ---- (c) Flag category ----
    ax = axes[2]
    flag_colors = {
        "trustworthy":       "#43A047",
        "Bl_detected":       "#E53935",
        "insufficient_data": "#9E9E9E",
    }
    flag_labels_map = {
        "trustworthy":       "Trustworthy",
        "Bl_detected":       r"$\hat{B}_{local}$-detected",
        "insufficient_data": "Insufficient data",
    }
    for flag_name in ["trustworthy", "Bl_detected", "insufficient_data"]:
        mask = flags_np == flag_name
        if mask.sum() > 0:
            ax.scatter(Z[mask, 0], Z[mask, 1], c=flag_colors[flag_name],
                       s=s, alpha=alpha, edgecolors="none", rasterized=True)
    legend_el = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        markersize=9, label=flag_labels_map[k])
                 for k, c in flag_colors.items()]
    ax.legend(handles=legend_el, loc="upper right", fontsize=FS_LEGEND,
              framealpha=0.9, edgecolor="0.8", handlelength=2.1)
    ax.set_xticks([]); ax.set_yticks([])

    # Subplot captions "(a) Predictive entropy", etc. — centred below each axis.
    subplot_captions = [
        "(a) Predictive entropy",
        r"(b) $\hat{B}_{local}$ magnitude",
        "(c) Flag category",
    ]
    for ax, caption in zip(axes, subplot_captions):
        ax.text(0.5, -0.06, caption, transform=ax.transAxes,
                fontsize=FS_CAPTION, fontweight="bold", ha="center", va="top")
    # No suptitle — the scenario info goes in the figure caption instead.

    safe_name = scenario_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace("%", "pct")
    for ext in ["png", "pdf"]:
        path = os.path.join(fig_dir, f"blindspot_map_{safe_name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: blindspot_map_{safe_name}.png/pdf")
    plt.close(fig)

    # Print example point details
    if ex_idx is not None:
        labels_np = labels if isinstance(labels, np.ndarray) else np.array(labels)
        print(f"  Example point (idx={ex_idx}): H[p̄]={ent_np[ex_idx]:.4f}, "
              f"B_local={b_np[ex_idx]:.4f}, flag={flags_np[ex_idx]}, "
              f"correct={correct_np[ex_idx]}, class={CIFAR10_CLASSES[labels_np[ex_idx]]}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(root, cfg, method="llla", force_recompute=False, rerender_only=False):
    fig_dir = os.path.join(root, "results", "figures")
    cache_dir = os.path.join(root, "results", "blindspot_cache")
    os.makedirs(fig_dir, exist_ok=True)

    scenario_keys = ["sev3", "sev5", "mixed"]

    # ---- Fast path: cache-only rerender ----
    if rerender_only:
        missing = [k for k in scenario_keys
                   if load_scenario_cache(cache_dir, k) is None]
        if missing:
            raise RuntimeError(
                f"--rerender requested but cache missing for: {missing}. "
                f"Run without --rerender once to populate {cache_dir}.")
        for key in scenario_keys:
            c = load_scenario_cache(cache_dir, key)
            print(f"\n[rerender] {key}: {c['scenario_name']}")
            plot_scenario(c["Z"], c["labels"], c["pred_entropy"], c["b_hat"],
                          c["flags"], c["correct"], c["tau_b"],
                          c["scenario_name"], fig_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-fit BLocalEstimator (shared)
    from audit.b_local import BLocalEstimator
    from models.feature_extractor import load_cached_features
    from models.backbone import load_model, build_resnet18

    k = cfg.get("k_neighbors", 50)

    # Load holdout data for B_local fitting (use backbone_single features,
    # matching the original pipeline — backbone features are more robust
    # to corruption than imagenet_pretrained features)
    from audit.b_local import method_to_feature_model
    feat_model = method_to_feature_model(method)  # backbone_single for llla
    bb_holdout = load_cached_features(root, feat_model, "holdout")
    ho_result = torch.load(os.path.join(root, "results", method, "holdout.pt"),
                           weights_only=False)
    estimator = BLocalEstimator(k=k)
    estimator.fit(bb_holdout["features"].numpy(), ho_result["probs"], ho_result["labels"])

    # Load tau_b from flags
    flag_id_data = torch.load(os.path.join(root, "results", f"flags_{method}", "test_id.pt"),
                              weights_only=False)
    tau_b = flag_id_data["tau_b"]
    d_max = flag_id_data["d_max"]
    print(f"tau_b = {tau_b:.4f}, d_max = {d_max:.4f}")

    # Load backbone model for feature extraction (used for both t-SNE and B_local)
    ckpt_dir = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"))
    backbone = load_model(os.path.join(ckpt_dir, "backbone_single.pt"), device)

    def compute_flags(b_hat_np, knn_dist_np, tau_b, d_max):
        flags = []
        for b, d in zip(b_hat_np, knn_dist_np):
            if d > d_max:
                flags.append("insufficient_data")
            elif b > tau_b:
                flags.append("Bl_detected")
            else:
                flags.append("trustworthy")
        return flags

    # =================================================================
    # Scenario 1 & 2: CIFAR-10-C severity 3 and 5
    # =================================================================
    cifar10c_dir = cfg.get("cifar10c_dir", "./data/raw/CIFAR-10-C")
    if not os.path.isabs(cifar10c_dir):
        cifar10c_dir = os.path.join(root, cifar10c_dir)

    n_sub = 5000

    for severity in [3, 5]:
        key = f"sev{severity}"
        print(f"\n=== CIFAR-10-C severity {severity} ===")

        cached_inputs = None if force_recompute else load_scenario_cache(cache_dir, key)
        if cached_inputs is not None:
            print(f"  [cache hit] {key}.npz")
            plot_scenario(cached_inputs["Z"], cached_inputs["labels"],
                          cached_inputs["pred_entropy"], cached_inputs["b_hat"],
                          cached_inputs["flags"], cached_inputs["correct"],
                          cached_inputs["tau_b"], cached_inputs["scenario_name"],
                          fig_dir)
            continue

        # Load cached LLLA results (190k samples)
        cached_path = os.path.join(root, "results", method, f"cifar10c_sev{severity}.pt")
        cached = torch.load(cached_path, weights_only=False)

        # Subsample
        imgs_sub, labels_sub, sub_idx = load_cifar10c_subsample(
            cifar10c_dir, severity, n_subsample=n_sub)
        print(f"  Subsampled {len(sub_idx)} from {len(cached['labels'])} total")

        # Get cached MI, entropy, correct for subsampled indices
        mi_sub      = cached["mutual_info"][sub_idx].numpy()
        entropy_sub = cached["pred_entropy"][sub_idx].numpy()
        correct_sub = cached["correct"][sub_idx].numpy()
        labels_sub_t = cached["labels"][sub_idx].numpy().astype(int)

        # Extract backbone features (for both t-SNE layout AND B_local estimation)
        print(f"  Extracting backbone features...")
        bb_feats = extract_features_from_images(backbone, imgs_sub, device)

        # Compute B_local using backbone features (same space as holdout)
        print(f"  Computing B_local...")
        b_result = estimator.estimate(bb_feats)
        b_hat_np = b_result["b_hat"].numpy()
        knn_dist_np = b_result["knn_dist"].numpy()
        flags_sub = compute_flags(b_hat_np, knn_dist_np, tau_b, d_max)

        # t-SNE
        print(f"  Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=50, random_state=42,
                    learning_rate="auto", init="pca", max_iter=2000)
        Z = tsne.fit_transform(bb_feats)

        acc = correct_sub.mean() * 100
        scenario_name = f"CIFAR-10-C severity {severity} (N={n_sub}, acc={acc:.1f}%)"
        save_scenario_cache(cache_dir, key, Z, labels_sub_t, entropy_sub,
                            b_hat_np, flags_sub, correct_sub, tau_b, scenario_name)
        plot_scenario(Z, labels_sub_t, entropy_sub, b_hat_np, flags_sub,
                      correct_sub, tau_b, scenario_name, fig_dir)

    # =================================================================
    # Scenario 3: Mixed (ID + OOD)
    # =================================================================
    print(f"\n=== Mixed (ID + OOD) ===")

    cached_inputs = None if force_recompute else load_scenario_cache(cache_dir, "mixed")
    if cached_inputs is not None:
        print(f"  [cache hit] mixed.npz")
        plot_scenario(cached_inputs["Z"], cached_inputs["labels"],
                      cached_inputs["pred_entropy"], cached_inputs["b_hat"],
                      cached_inputs["flags"], cached_inputs["correct"],
                      cached_inputs["tau_b"], cached_inputs["scenario_name"],
                      fig_dir)
        return

    # Load cached data
    feat_id  = load_cached_features(root, "imagenet_pretrained", "test_id")
    feat_ood = load_cached_features(root, "imagenet_pretrained", "test_ood")
    bb_id    = load_cached_features(root, "backbone_single", "test_id")
    bb_ood   = load_cached_features(root, "backbone_single", "test_ood")

    pred_id  = torch.load(os.path.join(root, "results", method, "test_id.pt"),
                          weights_only=False)
    pred_ood = torch.load(os.path.join(root, "results", method, "test_ood.pt"),
                          weights_only=False)
    flag_ood_data = torch.load(os.path.join(root, "results", f"flags_{method}", "test_ood.pt"),
                               weights_only=False)

    # Combine
    labels_mix  = np.concatenate([pred_id["labels"].numpy(),
                                  pred_ood["labels"].numpy().astype(int) % 10])
    entropy_mix = np.concatenate([pred_id["pred_entropy"].numpy(),
                                  pred_ood["pred_entropy"].numpy()])
    correct_mix = np.concatenate([pred_id["correct"].numpy(),
                                  np.zeros(len(pred_ood["correct"]), dtype=bool)])
    b_mix       = np.concatenate([flag_id_data["b_hat"].numpy(),
                                  flag_ood_data["b_hat"].numpy()])
    flags_mix   = np.concatenate([np.array(flag_id_data["flags"]),
                                  np.array(flag_ood_data["flags"])])
    bb_mix      = np.concatenate([bb_id["features"].numpy(),
                                  bb_ood["features"].numpy()])

    N_mix = len(labels_mix)
    print(f"  Total: {N_mix}")

    # t-SNE
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42,
                learning_rate="auto", init="pca", max_iter=2000)
    Z_mix = tsne.fit_transform(bb_mix)

    acc_mix = correct_mix.mean() * 100
    scenario_name = f"Mixed ID+OOD (N={N_mix}, acc={acc_mix:.1f}%)"
    save_scenario_cache(cache_dir, "mixed", Z_mix, labels_mix, entropy_mix,
                        b_mix, list(flags_mix), correct_mix, tau_b, scenario_name)
    plot_scenario(Z_mix, labels_mix, entropy_mix, b_mix, flags_mix,
                  correct_mix, tau_b, scenario_name, fig_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla")
    parser.add_argument("--rerender", action="store_true",
                        help="Skip all computation; replot from cache only.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force full recompute, ignoring any existing cache.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path)
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(root, cfg, args.method,
         force_recompute=args.no_cache, rerender_only=args.rerender)
