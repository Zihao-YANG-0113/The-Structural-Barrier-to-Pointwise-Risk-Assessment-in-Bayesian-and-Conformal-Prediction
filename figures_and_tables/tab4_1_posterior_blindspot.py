"""
Table 4.1 — Posterior blindspot detection on vision benchmarks.
================================================================

Dissertation reference (Chapter 4, §4.2.2, Table 4.1):
    Stage-2 partitioning of the Top-80% Stage-1 confident tier, under LLLA
    and SWAG. Reports n_dist, n_B̂, n_C, Err_C, Err_A, SRR for four
    scenarios:
        * Mixed (stress)         : 50/50 CIFAR-10 + CIFAR-100
        * Mixed (realistic)      : 80/20 CIFAR-10 + CIFAR-100
        * CIFAR-10-C severity 3
        * CIFAR-10-C severity 5

Stage-1 tier is selected by the predictive entropy H[p̄] of the Bayesian
posterior. Stage-2 partitions that tier by B̂_local (k = 50, k-NN in
ResNet-20 backbone feature space) using the data-driven thresholds
(τ_B from Youden's J on the held-out LOO ROC; d_max from the 95th
percentile of held-out LOO k-NN distances).

This wrapper calls the partitioning experiment in
`_src_posterior_blindspot.py` for each (method, ood_fraction) pair to
reproduce the four rows of the table.

Outputs
-------
  results/eval/posterior_blindspot.csv  (the full table data)
  console table per scenario / method

Run
---
  # LLLA rows of Table 4.1
  python figures_and_tables/tab4_1_posterior_blindspot.py --method llla

  # SWAG rows of Table 4.1 (re-run with method=swag)
  python figures_and_tables/tab4_1_posterior_blindspot.py --method swag
"""

import argparse
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figures_and_tables import _src_posterior_blindspot as backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="llla", choices=["llla", "swag"],
                        help="Bayesian posterior approximation: LLLA (default) or SWAG.")
    parser.add_argument("--ood_fraction", type=float, default=0.5,
                        help="0.5 → Mixed (stress), 0.2 → Mixed (realistic).")
    parser.add_argument("--ood_dataset", default="cifar100",
                        help="Use CIFAR-100 to match the Mixed scenarios in the dissertation.")
    args = parser.parse_args()

    cfg_path = os.path.join(ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    backend.run_posterior_blindspot(
        ROOT, cfg, method=args.method,
        ood_fraction=args.ood_fraction,
        ood_dataset=args.ood_dataset,
    )


if __name__ == "__main__":
    main()
