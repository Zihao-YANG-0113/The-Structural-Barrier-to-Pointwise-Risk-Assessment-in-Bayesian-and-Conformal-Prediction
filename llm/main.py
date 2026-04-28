"""
End-to-end pipeline for the posterior blindness experiment on LLMs.

Usage:
  python main.py

Produces Table 4.4:
   Scenario         0%     10%    20%    30%    40%   Max.red.
   BoolQ (ID)       ...    ...    ...    ...    ...   ...%
   PubMedQA (OOD)   ...    ...    ...    ...    ...   ...%

Workflow:
  1. Build model with LoRA adapters
  2. Fine-tune on BoolQ training split
  3. Freeze LoRA, fit LLLA on classification head
  4. Extract features + LLLA predictions on:
       - held-out portion of BoolQ train (for k-NN of B_ell)
       - BoolQ validation set (in-distribution eval)
       - PubMedQA (OOD eval)
  5. Run blindspot rejection protocol on both eval sets
  6. Print the table
"""

import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW

from config import get_config, STAGE
from data import make_dataloaders
from model import (
    build_model_for_training,
    train_one_epoch,
    extract_features_and_labels,
    fit_llla,
    predict_with_llla,
    predict_with_llla_probit,
    optimize_prior_precision_ml,
)
from estimator import compute_z_ell, compute_blindspot_table


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cfg = get_config()
    print(f"=== STAGE: {STAGE} ===")
    print(f"Model: {cfg.model_name}")

    # ---- Auto-fallback device ----
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
        print("CUDA not available, falling back to CPU. This will be slow.")
    print(f"Device: {cfg.device}")

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Step 1: Build model & tokenizer ----
    t0 = time.time()
    model, tokenizer = build_model_for_training(cfg)
    model.to(cfg.device)
    print(f"Model built in {time.time()-t0:.1f}s")

    # ---- Step 2: Load data ----
    t0 = time.time()
    loaders = make_dataloaders(cfg, tokenizer)
    for name, loader in loaders.items():
        print(f"  {name}: {len(loader.dataset)} examples")
    print(f"Data loaded in {time.time()-t0:.1f}s")

    # ---- Step 3: LoRA fine-tuning ----
    print("\n=== Training ===")
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    for epoch in range(cfg.num_train_epochs):
        t0 = time.time()
        loss, acc = train_one_epoch(model, loaders["train"], optimizer, cfg.device)
        dt = time.time() - t0
        print(f"  Epoch {epoch+1}: loss={loss:.4f}  acc={acc:.3f}  time={dt:.1f}s")

    # ---- Step 4: Freeze everything, extract features ----
    print("\n=== Feature extraction ===")
    for p in model.parameters():
        p.requires_grad = False

    results = {}
    for name in ["heldout", "val", "ood"]:
        t0 = time.time()
        feats, labels, logits = extract_features_and_labels(
            model, loaders[name], cfg.device
        )
        results[name] = {"features": feats, "labels": labels, "logits": logits}
        print(f"  {name}: {feats.shape[0]} examples, D={feats.shape[1]}, "
              f"acc={(logits.argmax(-1) == labels).float().mean().item():.3f}  "
              f"({time.time()-t0:.1f}s)")

    # ---- Step 5: Optimize prior precision via marginal likelihood ----
    if cfg.optimize_prior_precision:
        print("\n=== Optimizing prior precision (Laplace model evidence) ===")
        t0 = time.time()
        prior_lam = optimize_prior_precision_ml(
            results["heldout"]["features"],
            results["heldout"]["labels"],
            model.classifier,
            init=cfg.laplace_prior_precision,
            n_steps=cfg.prior_opt_steps,
            lr=cfg.prior_opt_lr,
        )
        print(f"  lambda* = {prior_lam:.2f}  (init {cfg.laplace_prior_precision:.2f})  "
              f"({time.time()-t0:.1f}s)")
    else:
        prior_lam = cfg.laplace_prior_precision

    # ---- Step 6: Fit LLLA on classification head ----
    print("\n=== Fitting LLLA ===")
    t0 = time.time()
    W_map, b_map, precision_diag = fit_llla(
        results["heldout"]["features"],
        results["heldout"]["labels"],
        model.classifier,
        prior_precision=prior_lam,
    )
    print(f"  Fitted in {time.time()-t0:.1f}s. "
          f"Mean precision: {precision_diag.mean().item():.2f}")

    # ---- Step 7: Posterior predictive ----
    print(f"\n=== Posterior predictive ({cfg.predict_method}) ===")
    for name in ["heldout", "val", "ood"]:
        if cfg.predict_method == "probit":
            mean_probs, entropy, mi = predict_with_llla_probit(
                results[name]["features"], W_map, b_map, precision_diag,
            )
        else:
            mean_probs, entropy, mi = predict_with_llla(
                results[name]["features"], W_map, b_map, precision_diag,
                n_samples=cfg.n_mc_samples,
            )
        results[name]["mean_probs"] = mean_probs
        results[name]["entropy"] = entropy
        results[name]["mutual_info"] = mi
        acc = (mean_probs.argmax(-1) == results[name]["labels"]).float().mean().item()
        print(f"  {name}: LLLA acc={acc:.3f}, mean MI={mi.mean().item():.4f}")

    # ---- Step 7: Compute z_ell on held-out set ----
    heldout_z = compute_z_ell(
        results["heldout"]["mean_probs"],
        results["heldout"]["labels"],
    )
    print(f"\nHeld-out z_ell: mean={heldout_z.mean():.4f}, std={heldout_z.std():.4f}")

    # ---- Step 8: Blindspot rejection table ----
    print("\n=== Table 4.4: Blindspot rejection ===")
    print(f"{'Scenario':<15} {'0%':>7} {'10%':>7} {'20%':>7} {'30%':>7} {'40%':>7} {'Max.red':>10}")

    for scenario, name in [("BoolQ (ID)", "val"), ("PubMedQA (OOD)", "ood")]:
        table = compute_blindspot_table(
            val_features=results[name]["features"],
            val_labels=results[name]["labels"],
            val_probs=results[name]["mean_probs"],
            val_entropy=results[name]["entropy"],
            heldout_features=results["heldout"]["features"],
            heldout_z=heldout_z,
            k=cfg.k_nn,
        )
        errs = table["residual_errors"]
        row = f"{scenario:<15}"
        for e in errs:
            row += f" {e*100:6.2f}%"
        row += f"  {table['max_reduction_pct']:7.1f}%"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
