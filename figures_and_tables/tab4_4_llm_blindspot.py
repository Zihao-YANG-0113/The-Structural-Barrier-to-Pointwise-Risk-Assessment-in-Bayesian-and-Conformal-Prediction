"""
Table 4.4 — Posterior blindspot detection on language benchmarks.
==================================================================

Dissertation reference (Chapter 4, §4.3.2, Table 4.4):
    Stage-2 partitioning of the Top-80% Stage-1 confident tier under
    Llama-3-8B with LLLA on the classification head. Reports n_dist,
    n_B̂, n_C, Err_C, Err_A, SRR for three scenarios:
        * BoolQ (ID)
        * PubMedQA (OOD)
        * Mixed   (50/50 BoolQ + PubMedQA-yes/no)

Pipeline (delegated to `llm/main.py`):
  1. Build Llama-3-8B with LoRA adapters on q_proj / v_proj.
  2. Fine-tune on BoolQ training split.
  3. Freeze the LoRA, fit a last-layer Laplace approximation (LLLA) on
     the classification head with marginal-likelihood-optimised prior
     precision λ*.
  4. Extract frozen features for held-out BoolQ, BoolQ val (ID), and
     PubMedQA (OOD).
  5. Compute the one-step contrast z_ell on the held-out set, then
     B̂_local via k = 50-NN regression in feature space.
  6. Rank the val/OOD pools by predictive entropy, select the Top-80%
     tier, and partition by B̂_local + d_max as in the vision pipeline.

To switch between debug and full runs, edit `llm/config.py::STAGE`.

Run
---
  python figures_and_tables/tab4_4_llm_blindspot.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    runpy.run_path(
        os.path.join(ROOT, "llm", "main.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
