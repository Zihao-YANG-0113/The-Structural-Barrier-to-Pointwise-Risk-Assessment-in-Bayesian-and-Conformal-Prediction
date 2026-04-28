"""
Figure 4.2 — Stage-2 ranker comparison within the Top-30% confident tier.
=========================================================================

Dissertation reference (Chapter 4, §4.2.4):
    "Confident-tier error rate against rejection budget for B̂_local,
     predictive entropy, mutual information, APS prediction-set size,
     and random rejection. Stars mark the lowest-error operating point
     along each curve."

This figure isolates the *complementary* value of B̂_local against
within-tier rankers that share the Stage-1 information source. The
Stage-1 tier is the Top-30% most-confident subset selected by predictive
entropy; the five Stage-2 rejectors compete on the same tier:

    • Random              (baseline)
    • Predictive entropy  H[p̄]
    • Mutual information  MI
    • APS set size        (α = 0.10)
    • B̂_local (ours, with insufficient-data pre-filter via d_max)

The realistic-mix rendering (panel "Mixed (realistic)" at 80/20 ID/OOD)
is produced by `_src_render_ablationA_realistic.py`, which loads the
multilayer-feature B̂_local pipeline and renders the three-panel figure.

Outputs
-------
  ~/Desktop/Thesis-Ablations/ablation_A_rejectors_cov30_realistic_muted.png
  (this is the file referenced as `ablation_A_rejectors_cov30.png` in
   the dissertation source)

Run
---
  python figures_and_tables/fig4_2_ablation_A.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    runpy.run_path(
        os.path.join(HERE, "_src_render_ablationA_realistic.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
