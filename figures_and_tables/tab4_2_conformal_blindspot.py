"""
Table 4.2 — Conformal blindspot detection on vision benchmarks (APS).
======================================================================

Dissertation reference (Chapter 4, §4.2.2, Table 4.2):
    Stage-2 partitioning of the Top-80% Stage-1 confident tier under
    Adaptive Prediction Sets (APS). The Stage-1 confident tier is here
    selected by APS prediction-set size (smaller set → higher confidence).
    Stage-2 still uses B̂_local (k = 50) with the same data-driven τ_B,
    d_max as in Table 4.1.

This wrapper calls `_src_conformal_blindspot.py`, which:
  * Calibrates split conformal (APS) on the calibration split at α = 0.1.
  * For each scenario (Mixed stress, Mixed realistic, CIFAR-10-C sev 3,
    sev 5), partitions the 80% smallest-set tier by the audit flag
    derived from B̂_local.

Outputs
-------
  results/eval/conformal_blindspot.csv  (full table data)
  console table per scenario

Run
---
  # 50/50 (stress) row
  python figures_and_tables/tab4_2_conformal_blindspot.py --ood_fraction 0.5

  # 80/20 (realistic) row
  python figures_and_tables/tab4_2_conformal_blindspot.py --ood_fraction 0.2
"""

import argparse
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figures_and_tables import _src_conformal_blindspot as backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="llla",
                        help="Method whose B̂_local flags drive Stage-2 (LLLA in the paper).")
    parser.add_argument("--ood_fraction", type=float, default=0.5)
    parser.add_argument("--ood_dataset", default="cifar100",
                        help="CIFAR-100 to match the Mixed scenarios in the dissertation.")
    args = parser.parse_args()

    cfg_path = os.path.join(ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    backend.main(ROOT, cfg, method=args.method,
                 ood_fraction=args.ood_fraction, ood_dataset=args.ood_dataset)


if __name__ == "__main__":
    main()
