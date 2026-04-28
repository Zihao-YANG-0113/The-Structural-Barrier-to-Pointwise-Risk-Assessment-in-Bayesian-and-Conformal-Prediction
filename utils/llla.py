"""
llla.py
-------
Last-Layer Laplace Approximation (LLLA) inference.

Uses `laplace-torch` to fit a last-layer Laplace approximation on the
trained CIFAR-10 backbone (ResNet-20 in the dissertation main pipeline).
The prior precision is selected by marginal likelihood optimisation, as
in Dissertation §4.2.1.

Outputs the uncertainty measures used in Chapter 4:
  - predictive_entropy : H[p̄(y|x)]
  - mutual_information : MI = H[p̄] - E_π[H[p_θ]]
  - max_prob_mean      : max class probability under the predictive mean

Usage:
    python utils/llla.py --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.splits import get_loaders, get_ood_loader
from models.backbone import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  不确定性计算工具                                                    #
# ------------------------------------------------------------------ #

def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算每个样本的预测熵。probs: (N, C)，返回 (N,)。"""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def mutual_information(
    mean_probs: torch.Tensor,
    probs_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MI = H[p̄] - E[H[p_θ]]
    mean_probs:    (N, C)
    probs_samples: (N, S, C)  S 个近似样本
    """
    h_mean   = entropy(mean_probs, eps)
    h_per    = -(probs_samples * (probs_samples + eps).log()).sum(dim=-1)  # (N, S)
    e_h_per  = h_per.mean(dim=1)                                            # (N,)
    return h_mean - e_h_per


# ------------------------------------------------------------------ #
#  LLLA 推理                                                           #
# ------------------------------------------------------------------ #

def fit_llla(model, train_loader, device="cuda"):
    """
    用 laplace-torch 拟合 Last-Layer Laplace，并优化先验精度。
    返回 Laplace 对象。
    """
    try:
        from laplace import Laplace
    except ImportError:
        raise ImportError("请安装 laplace-torch: pip install laplace-torch")

    model = model.to(device)
    model.eval()

    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    print("拟合 Laplace 近似（last-layer Kron Hessian）...")
    la.fit(train_loader)
    print("优化先验精度（marginal likelihood）...")
    la.optimize_prior_precision(method="marglik")
    print(f"最优先验精度: {la.prior_precision.item():.4f}")
    return la


@torch.no_grad()
def predict_llla(la, loader, n_samples: int = 100, device="cuda"):
    """
    对 loader 中所有样本进行 LLLA 预测。
    返回 dict，包含：
      probs          : (N, C)  均值预测概率
      pred_entropy   : (N,)    预测熵
      mutual_info    : (N,)    互信息
      max_prob       : (N,)    最大概率
      labels         : (N,)    真实标签
      correct        : (N,)    bool，是否预测正确
    """
    all_probs, all_labels = [], []

    for x, y in tqdm(loader, desc="LLLA 预测", leave=False):
        x = x.to(device)
        # la() 返回均值预测概率 (B, C)，link_approx='probit' 适合分类
        with torch.no_grad():
            p = la(x, link_approx="probit")
        all_probs.append(p.cpu())
        all_labels.append(y)

    probs  = torch.cat(all_probs,  dim=0)   # (N, C)
    labels = torch.cat(all_labels, dim=0)   # (N,)

    # 用蒙特卡洛采样估计 MI（从后验采样权重）
    # laplace-torch 支持 la.predictive_samples
    print("从后验采样估计互信息（MC 采样）...")
    sample_probs_list = []
    for x, _ in tqdm(loader, desc="LLLA MC 采样", leave=False):
        x = x.to(device)
        # 采样 n_samples 次，每次返回 (B, C)
        s = la.predictive_samples(x, pred_type="glm", n_samples=n_samples)
        # s: (n_samples, B, C) 或 (B, n_samples, C)，取决于 laplace 版本
        if s.shape[0] == n_samples:
            s = s.permute(1, 0, 2)  # → (B, n_samples, C)
        sample_probs_list.append(s.cpu())

    probs_samples = torch.cat(sample_probs_list, dim=0)  # (N, n_samples, C)

    pred_ent = entropy(probs)
    mi       = mutual_information(probs, probs_samples)
    max_prob = probs.max(dim=-1).values
    correct  = (probs.argmax(dim=-1) == labels)

    return {
        "probs":        probs,
        "pred_entropy": pred_ent,
        "mutual_info":  mi,
        "max_prob":     max_prob,
        "labels":       labels,
        "correct":      correct,
    }


# ------------------------------------------------------------------ #
#  保存结果                                                            #
# ------------------------------------------------------------------ #

def save_results(results: dict, root: str, split: str):
    save_dir = os.path.join(root, "results", "llla")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{split}.pt")
    torch.save(results, path)
    print(f"LLLA 结果已保存: {path}")


def load_results(root: str, split: str) -> dict:
    path = os.path.join(root, "results", "llla", f"{split}.pt")
    return torch.load(path, weights_only=False)


# ------------------------------------------------------------------ #
#  主入口                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(root, cfg.get("checkpoints_dir", "./checkpoints"),
                              "backbone_single.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}。请先训练 backbone。")

    model   = load_model(ckpt_path, device)
    loaders = get_loaders(cfg, root)

    # 拟合 Laplace
    la = fit_llla(model, loaders["train"], device)

    # 保存 Laplace 对象
    la_save = os.path.join(root, "checkpoints", "llla.pt")
    torch.save(la, la_save)
    print(f"Laplace 对象已保存: {la_save}")

    # 在各 split 上预测
    for split in ["val", "calibration", "test_id"]:
        print(f"\n=== LLLA 预测 [{split}] ===")
        results = predict_llla(la, loaders[split], device=device)
        save_results(results, root, split)

    # OOD
    print("\n=== LLLA 预测 [test_ood] ===")
    ood_loader = get_ood_loader(cfg, root)
    results    = predict_llla(la, ood_loader, device=device)
    save_results(results, root, "test_ood")

    print("\nLLLA 推理完成！")


if __name__ == "__main__":
    main()
