"""
Figure 3.1 — Audit signal recovers posterior blindness on CIFAR-10-C.
=====================================================================

Dissertation reference (Chapter 3, §3.4.1):
    "t-SNE of ResNet-20 features on CIFAR-10-C (severity 3), coloured by
     (a) predictive entropy H[p̄]; (b) the audit estimator B̂_local;
     (c) the downstream Stage-2 flag."

This script is a thin wrapper that calls the underlying experiment in
`_src_blindspot_map.py`. It:

  1. Loads the cached LLLA posterior outputs on CIFAR-10-C severity 3,
     (predictive entropy, mutual information, correctness).
  2. Runs the k-NN B̂_local estimator on backbone features (k = 50).
  3. Fits t-SNE (perplexity = 50) on those features.
  4. Generates the three-panel figure with an illustrative point that has
     low entropy but is flagged as a blindspot by B̂_local.

Outputs
-------
  results/figures/blindspot_map_CIFAR-10-C_severity_3_*.png

Run
---
  python figures_and_tables/fig3_1_blindspot_map.py
"""

import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from figures_and_tables import _src_blindspot_map as backend


def main():
    cfg_path = os.path.join(ROOT, "configs", "default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    backend.main(ROOT, cfg, method="llla", force_recompute=False, rerender_only=False)


if __name__ == "__main__":
    main()
