"""
conformal.py
------------
Split Conformal Prediction，使用 APS（Adaptive Prediction Sets）评分。

APS 评分：对于样本 (x, y)，
  score = cumulative softmax probability up to and including true class y
  (按概率从大到小排序后的累积和)

校准：在 calibration set 计算所有 APS score，取 ⌈(n+1)(1-α)⌉/n 分位数 q̂。
预测：预测集 = {y : APS_score(x, y) ≤ q̂}

用法：
    python methods/conformal.py --config configs/default.yaml --method llla
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
#  APS 评分                                                            #
# ------------------------------------------------------------------ #

def aps_scores(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算每个样本的 APS 评分。
    probs:  (N, C) softmax 概率
    labels: (N,)   真实标签
    返回:  (N,)   APS 分数 ∈ [0, 1]
    """
    N, C = probs.shape
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)  # (N, C)
    cumsum = sorted_probs.cumsum(dim=-1)                              # (N, C)

    scores = torch.zeros(N)
    for i in range(N):
        # 找到真实标签在排序中的位置
        rank = (sorted_idx[i] == labels[i]).nonzero(as_tuple=True)[0].item()
        scores[i] = cumsum[i, rank].item()
    return scores


def aps_scores_batch(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    向量化版 APS 评分（更快）。
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)

    # 对每个样本找真实标签的排名
    label_expand = labels.unsqueeze(1).expand_as(sorted_idx)    # (N, C)
    match_mask   = (sorted_idx == label_expand)                  # (N, C) bool
    # 取第一个匹配位置
    ranks = match_mask.float().argmax(dim=-1)                    # (N,)
    scores = cumsum[torch.arange(len(labels)), ranks]            # (N,)
    return scores


def prediction_sets_from_scores(probs: torch.Tensor, quantile: float) -> list[list[int]]:
    """
    根据量化分位数生成预测集。
    对每个样本，收集所有满足 cumsum ≤ quantile 的类别
    （但至少包含第一个类别，即最高概率类别）。
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    pred_sets = []
    for i in range(probs.shape[0]):
        included = (cumsum[i] <= quantile).nonzero(as_tuple=True)[0]
        # 找第一个超过分位数的位置（以确保真实类包含进来的概率 ≥ 1-α）
        exceed = (cumsum[i] > quantile).nonzero(as_tuple=True)[0]
        if len(exceed) == 0:
            cutoff = probs.shape[1] - 1
        else:
            cutoff = exceed[0].item()
        set_indices = sorted_idx[i, :cutoff + 1].tolist()
        pred_sets.append(set_indices)
    return pred_sets


# ------------------------------------------------------------------ #
#  校准 & 预测                                                         #
# ------------------------------------------------------------------ #

class SplitConformal:
    """
    Split Conformal Predictor（APS 评分）。
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha    = alpha
        self.quantile = None

    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor):
        """
        在 calibration set 上估计分位数 q̂。
        cal_probs:  (n, C)
        cal_labels: (n,)
        """
        scores = aps_scores_batch(cal_probs, cal_labels)
        n      = len(scores)
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        level  = min(level, 1.0)
        self.quantile = float(torch.quantile(scores, level))
        print(f"Conformal 分位数 q̂ = {self.quantile:.4f}  (α={self.alpha}, n={n})")
        return self.quantile

    def predict(self, test_probs: torch.Tensor) -> list[list[int]]:
        """返回每个测试样本的预测集（类别索引列表）。"""
        if self.quantile is None:
            raise RuntimeError("请先调用 calibrate()。")
        return prediction_sets_from_scores(test_probs, self.quantile)

    def coverage(self, pred_sets: list[list[int]], labels: torch.Tensor) -> float:
        """计算经验覆盖率。"""
        covered = sum(
            1 for ps, y in zip(pred_sets, labels.tolist()) if y in ps
        )
        return covered / len(labels)

    def avg_set_size(self, pred_sets: list[list[int]]) -> float:
        return sum(len(ps) for ps in pred_sets) / len(pred_sets)


# ------------------------------------------------------------------ #
#  加载各方法的预测概率                                                #
# ------------------------------------------------------------------ #

def load_probs(root: str, method: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从已保存的方法结果中加载 probs 和 labels。"""
    path = os.path.join(root, "results", method, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}，请先运行对应方法。")
    data = torch.load(path, weights_only=False)
    return data["probs"], data["labels"]


# ------------------------------------------------------------------ #
#  运行 & 保存                                                         #
# ------------------------------------------------------------------ #

def run_conformal(root: str, cfg: dict, method: str = "llla"):
    """
    对指定方法运行 split conformal，保存预测集和覆盖率。
    """
    alpha = cfg.get("alpha", 0.1)
    cp    = SplitConformal(alpha=alpha)

    # 校准
    cal_probs, cal_labels = load_probs(root, method, "calibration")
    cp.calibrate(cal_probs, cal_labels)

    results = {}
    for split in ["test_id"]:
        test_probs, test_labels = load_probs(root, method, split)
        pred_sets = cp.predict(test_probs)
        cov       = cp.coverage(pred_sets, test_labels)
        avg_size  = cp.avg_set_size(pred_sets)
        print(f"  [{method}|{split}] 覆盖率={cov:.4f}  平均集合大小={avg_size:.2f}")

        # APS 分数（用于后续 B_local 校正）
        aps = aps_scores_batch(test_probs, test_labels)
        results[split] = {
            "pred_sets":  pred_sets,
            "coverage":   cov,
            "avg_size":   avg_size,
            "aps_scores": aps,
            "quantile":   cp.quantile,
            "labels":     test_labels,
            "probs":      test_probs,
        }

    save_dir = os.path.join(root, "results", f"conformal_{method}")
    os.makedirs(save_dir, exist_ok=True)
    for split, res in results.items():
        path = os.path.join(save_dir, f"{split}.pt")
        torch.save(res, path)
        print(f"Conformal 结果已保存: {path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="llla",
                        choices=["llla", "ensemble", "mc_dropout"],
                        help="使用哪个方法的预测概率进行 conformal")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"=== Split Conformal (APS) with {args.method} ===")
    run_conformal(root, cfg, args.method)
    print("\nConformal 推理完成！")


if __name__ == "__main__":
    main()
