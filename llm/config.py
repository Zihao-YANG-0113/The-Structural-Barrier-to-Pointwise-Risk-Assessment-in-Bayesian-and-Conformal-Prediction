"""
Configuration for posterior blindness experiments on LLMs.

Dissertation §4.3.1 reports results on Llama-3-8B (`meta-llama/Meta-Llama-3-8B`).
Stage A (local debug):  MODEL_NAME = "bert-base-uncased", small subset.
Stage B (real runs):    MODEL_NAME = "meta-llama/Meta-Llama-3-8B", full datasets.

Change only the STAGE variable below to switch between them.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, List

STAGE: Literal["debug", "full"] = "debug"  # Change this to "full" for real runs


@dataclass
class Config:
    # ---------- Model ----------
    model_name: str = ""
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Which layer to put LoRA on. For LLLA variant: output layer only.
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])
    use_llla: bool = True  # If True, freeze LoRA after training and only do Laplace on classification head

    # ---------- Data ----------
    train_dataset: str = "boolq"
    ood_dataset: str = "pubmed_qa"
    train_subset_size: int = 1000       # For debug. Use -1 for full.
    eval_subset_size: int = 500
    heldout_subset_size: int = 500      # Held-out set for B_hat_ell estimation
    max_seq_len: int = 384

    # ---------- Training ----------
    num_train_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 42

    # ---------- Laplace ----------
    laplace_prior_precision: float = 1.0     # initial value; optimized via ML if enabled
    optimize_prior_precision: bool = True    # type-II MLE via Laplace model evidence
    prior_opt_steps: int = 300
    prior_opt_lr: float = 0.1
    laplace_hessian_structure: Literal["full", "kron", "diag"] = "diag"
    predict_method: Literal["probit", "mc"] = "probit"
    n_mc_samples: int = 20                   # only used when predict_method == "mc"

    # ---------- B_ell estimator ----------
    # Dissertation §4.1: "uses k = 50 in the main experiments". Same value
    # used on the vision side (configs/default.yaml: k_neighbors: 50).
    k_nn: int = 50                      # k in k-NN
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"

    # ---------- Paths ----------
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"

    # ---------- Device ----------
    device: str = "cuda"  # Auto-fallback to cpu in debug
    torch_dtype: Literal["float32", "bfloat16", "float16"] = "float32"


def get_config() -> Config:
    cfg = Config()
    if STAGE == "debug":
        cfg.model_name = "bert-base-uncased"
        cfg.lora_target_modules = ["query", "value"]
        cfg.train_subset_size = 2000
        cfg.eval_subset_size = 400
        cfg.heldout_subset_size = 500
        cfg.batch_size = 8
        cfg.num_train_epochs = 5
    elif STAGE == "full":
        # Llama-3-8B as in the dissertation (Chapter 4, §4.3.1).
        cfg.model_name = "meta-llama/Meta-Llama-3-8B"
        cfg.lora_target_modules = ["q_proj", "v_proj"]
        cfg.train_subset_size = -1
        cfg.eval_subset_size = -1
        cfg.heldout_subset_size = 2000
        cfg.batch_size = 4  # Larger model, smaller batch
        cfg.num_train_epochs = 3
        cfg.torch_dtype = "bfloat16"  # Required to fit 8B on 40GB A100
    return cfg
