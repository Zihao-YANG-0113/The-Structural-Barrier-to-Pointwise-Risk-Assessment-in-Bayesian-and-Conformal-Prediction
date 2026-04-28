"""
Figure 4.4 — Rejection curves on language benchmarks.
======================================================

Dissertation reference (Chapter 4, §4.3.3):
    "Rejection curves across Stage-1 coverage levels (LLM). Progressive
     rejection of audit-flagged samples reduces the confident-tier
     error rate at coverage 70%, 80%, 90%. Stars mark the minimum of
     each curve; dashed lines indicate random-rejection baselines."

The figure is rendered from the LLM-side rejection curves on three
scenarios — BoolQ (ID), PubMedQA (OOD), Mixed — at coverage tiers 70%,
80% (main), and 90%, with rejection budget r ∈ {0, 5, ..., 40}%.

Curve values are stored on the 5%-grid inside
`_src_llm_rejection_curves.py::DATA` (the production numbers from the
LLM pipeline run). The script renders the figure with the same
typography as the vision-side Fig 4.1 for visual consistency.

Outputs
-------
  results/figures/llm_rejection_curves.png/.pdf

Run
---
  python figures_and_tables/fig4_4_llm_rejection_curves.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    runpy.run_path(
        os.path.join(HERE, "_src_llm_rejection_curves.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
