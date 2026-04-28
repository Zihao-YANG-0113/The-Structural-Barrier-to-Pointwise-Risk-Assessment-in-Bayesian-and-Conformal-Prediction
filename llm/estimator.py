"""
Held-out k-NN estimator for the pointwise risk gap B_ell(x_0),
and the blindspot-rejection pipeline used to produce Table 4.4.

Key formula (Theorem 3.3):
    B_hat_local(x_0) = (1/k) * sum_{j in N_k(x_0)} z_ell(X'_j, Y'_j; theta_hat)
where
    z_ell(x, y; theta) = ell(p_theta(.|x), y) - E_{y~p_theta(.|x)}[ell(p_theta(.|x), y)]
is the one-step contrast: "actual loss at held-out label" minus
                          "model's self-assessed expected loss".

Under theta = theta*, E[z_ell | X=x] = B_ell(x).
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_z_ell(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the one-step contrast z_ell(x, y; theta_hat) under log loss,
    given predictive probabilities directly.

    Args:
      probs:  (N, K)  — predictive probabilities at each point
                        (e.g. LLLA posterior-predictive mean)
      labels: (N,)    — observed labels from held-out set

    Returns:
      z:      (N,)    — actual NLL minus self-assessed expected NLL
    """
    probs = probs.clamp(min=1e-10)
    log_probs = probs.log()                                         # (N, K)
    # actual loss at observed label
    actual_loss = -log_probs[torch.arange(len(labels)), labels]     # (N,)
    # model's self-assessed expected loss: E_{y~p}[ -log p(y)] = H[p]
    expected_loss = -(probs * log_probs).sum(-1)                    # (N,) — entropy
    return actual_loss - expected_loss


def knn_estimator(query_features: torch.Tensor,
                  heldout_features: torch.Tensor,
                  heldout_z: torch.Tensor,
                  k: int,
                  metric: str = "euclidean") -> torch.Tensor:
    """
    For each query x_0, find its k nearest neighbors in the held-out feature
    space and average their z_ell values.

    Returns:
      B_hat_ell:  (Nq,)  — estimator of B_ell(x_0) for each query point
    """
    if metric == "cosine":
        q_norm = F.normalize(query_features, dim=-1)
        h_norm = F.normalize(heldout_features, dim=-1)
        sim = q_norm @ h_norm.T
        dist = 1.0 - sim
    else:
        # Squared Euclidean is fine; nearest neighbors don't depend on sqrt
        dist = torch.cdist(query_features, heldout_features, p=2)

    _, idx = dist.topk(k=k, largest=False, dim=-1)                  # (Nq, k)
    # Gather heldout_z at those indices
    z_neighbors = heldout_z[idx]                                    # (Nq, k)
    return z_neighbors.mean(dim=-1)                                 # (Nq,)


def compute_blindspot_table(
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_probs: torch.Tensor,
    val_entropy: torch.Tensor,
    heldout_features: torch.Tensor,
    heldout_z: torch.Tensor,
    k: int,
    rejection_rates: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4),
    top_confident_frac: float = 0.5,
) -> Dict:
    """
    Reproduces the protocol of Table 4.3 in the thesis:

      1. Select the top (top_confident_frac) most confident samples by
         predictive entropy (smaller entropy => more confident).
      2. Compute B_hat_ell for each of them using the held-out set.
      3. For each rejection rate r%, remove the r% with largest B_hat_ell
         and report the residual error rate.

    Returns a dict with keys ready to be formatted into a table.
    """
    N = len(val_features)
    N_confident = int(N * top_confident_frac)

    # Step 1: rank by predictive entropy, keep the most confident half
    conf_order = val_entropy.argsort()              # ascending (most confident first)
    kept_idx = conf_order[:N_confident]

    kept_feats = val_features[kept_idx]
    kept_labels = val_labels[kept_idx]
    kept_probs = val_probs[kept_idx]
    kept_preds = kept_probs.argmax(-1)

    # Step 2: compute B_hat_ell for these confident points
    B_hat = knn_estimator(kept_feats, heldout_features, heldout_z, k=k)

    # Step 3: progressive rejection
    # Higher B_hat = more suspected of being a blindspot (model underestimates risk)
    # Sort by B_hat descending: highest B_hat first (these are the rejected ones)
    reject_order = B_hat.argsort(descending=True)

    results = {
        "n_kept": N_confident,
        "rejection_rates": [],
        "residual_errors": [],
        "residual_counts": [],
    }

    # Also compute the baseline error on the full kept set
    baseline_err = (kept_preds != kept_labels).float().mean().item()

    for r in rejection_rates:
        n_reject = int(N_confident * r)
        # Indices to KEEP (i.e., not rejected): the N_confident - n_reject with smallest B_hat
        keep_after_reject = reject_order[n_reject:]
        if len(keep_after_reject) == 0:
            results["rejection_rates"].append(r)
            results["residual_errors"].append(float("nan"))
            results["residual_counts"].append(0)
            continue

        preds_kept = kept_preds[keep_after_reject]
        labels_kept = kept_labels[keep_after_reject]
        err = (preds_kept != labels_kept).float().mean().item()

        results["rejection_rates"].append(r)
        results["residual_errors"].append(err)
        results["residual_counts"].append(len(keep_after_reject))

    # Max relative reduction vs 0% rejection
    e0 = results["residual_errors"][0]
    emin = min(e for e in results["residual_errors"] if not np.isnan(e))
    results["max_reduction_pct"] = (e0 - emin) / max(e0, 1e-10) * 100

    return results
