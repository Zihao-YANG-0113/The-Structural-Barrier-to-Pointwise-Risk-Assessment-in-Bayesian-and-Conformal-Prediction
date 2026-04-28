"""
Figure 4.3 — Sensitivity to the neighbourhood size k.
======================================================

Dissertation reference (Chapter 4, §4.2.5):
    "Error rate after B̂_local-guided rejection across k, under the
     Stage-1 tier at 80% coverage with LLLA predictive entropy."

The audit estimator B̂_local depends on a single hyperparameter — the
neighbourhood size k. This figure refits the estimator at
k ∈ {5, 10, 25, 50, 100, 200}, recomputes τ_B / d_max from a fresh LOO
on the held-out set, and measures the residual within-tier error after
r = 30% B̂_local-guided rejection on three scenarios:

    sev3   : CIFAR-10-C severity 3
    sev5   : CIFAR-10-C severity 5
    mixed  : Mixed (stress, 50/50 CIFAR-10 + CIFAR-100)

Stage-1 tier is fixed at 80% coverage selected by LLLA H[p̄].

Outputs
-------
  ~/Desktop/Thesis-Ablations/k_sensitivity_bar_c5dff4.png
  (this is the file referenced as `k_sensitivity_bar_c5dff4.png` in the
   dissertation source)

Numerical inputs (`k_sensitivity.csv`) are produced by the
`_run_k_sensitivity` block in `_src_tier_risk_analysis.py`.

Run
---
  python figures_and_tables/fig4_3_k_sensitivity.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    runpy.run_path(
        os.path.join(HERE, "_src_rerender_k_sensitivity.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
