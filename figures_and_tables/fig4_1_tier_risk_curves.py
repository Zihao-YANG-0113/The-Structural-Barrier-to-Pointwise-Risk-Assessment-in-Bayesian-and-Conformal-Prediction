"""
Figure 4.1 + Table 4.3 — Tier risk curves and risk-reduction by Stage-2 rejection.
==================================================================================

Dissertation references
-----------------------
* Figure 4.1 (Chapter 4, §4.2.3): "Rejection curves across Stage-1 coverage
  levels" — within-tier error vs Stage-2 rejection budget r ∈ {0,5,...,40}%
  at coverage 70%, 80%, 90%, on three scenarios (Mixed stress, CIFAR-10-C
  severity 3, severity 5). Stars mark each curve's minimum; dashed lines
  are random-rejection baselines.
* Table 4.3 (Chapter 4, §4.2.3): "Risk reduction by Stage-2 rejection".
  Within-tier error rate at r ∈ {0,10,20,30,40}% under three Stage-1
  rankers — LLLA H[p̄], LLLA MI, APS prediction-set size — for the same
  three scenarios.

Pipeline
--------
1. `_src_tier_risk_analysis.py::main` runs the full Stage-1 + Stage-2
   pipeline on (sev3 / sev5 / mixed-stress) and writes:
        results/eval/tier_risk_curves.csv   (curve points)
        results/eval/tier_risk_table.csv    (table cells)
        results/figures/tier_risk_curves.png/.pdf
        results/figures/tier_risk_curves_entropy.png/.pdf
2. `_src_rerender_tier_risk_medium.py` re-renders the main figure with
   medium-sized fonts that match the dissertation typography:
        results/figures/tier_risk_curves_medium.png/.pdf  ← Fig 4.1

Run
---
  python figures_and_tables/fig4_1_tier_risk_curves.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    # Step 1: run the full pipeline (also produces Table 4.3 CSV).
    runpy.run_path(
        os.path.join(HERE, "_src_tier_risk_analysis.py"),
        run_name="__main__",
    )
    # Step 2: re-render the main MI-tier figure at the dissertation typography.
    runpy.run_path(
        os.path.join(HERE, "_src_rerender_tier_risk_medium.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
