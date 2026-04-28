"""
b_local.py
----------
B̂_local 估计器：用 hold-out 数据估计逐点风险差距（Pointwise Risk Gap）。

核心算法：
  1. 默认在主模型（被审计模型）的倒数第二层特征空间中找 k-NN 邻居
     （消融时可传 --feature_model imagenet_pretrained 切换为 ImageNet 预训练特征）
  2. actual_loss   = 邻域内加权平均 cross-entropy（使用 hold-out 真实标签）
  3. believed_loss = 邻域内加权平均预测熵（模型"相信"的损失）
  4. B̂(x) = actual_loss - believed_loss
  5. D(x)  = 到 k 个邻居的平均距离（密度信号）

hold-out 上的 B̂ 用 Leave-One-Out（LOO）估计：
  每个 hold-out 点排除自身，从其余 N-1 个点中取 k 个最近邻。
  LOO 结果用于 flag.py 自动确定 τ_B（Youden's J）和 d_max（95th 分位数）。

用法：
    python audit/b_local.py --config configs/default.yaml --method llla
    # 消融：用 ImageNet 预训练特征
    python audit/b_local.py --method llla --feature_model imagenet_pretrained
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feature_extractor import load_cached_features


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def method_to_feature_model(method: str) -> str:
    """返回被审计方法对应的特征缓存名称（主模型倒数第二层）。"""
    return {
        "llla":       "backbone_single",
        "ensemble":   "ensemble_mean",
        "mc_dropout": "backbone_mcdropout",
        "swag":       "backbone_single",
    }.get(method, "backbone_single")


# ------------------------------------------------------------------ #
#  B̂_local 核心计算                                                  #
# ------------------------------------------------------------------ #

class BLocalEstimator:
    """
    逐点风险差距估计器。
      fit()         : 在 hold-out 集上拟合 k-NN 索引，存储预测概率和标签。
      estimate()    : 对新测试点估计 B̂(x) 和 D(x)。
      estimate_loo(): 对 hold-out 点做 Leave-One-Out 估计（排除自身邻居）。
    """

    def __init__(self, k: int = 50, kernel_bandwidth: float = None):
        self.k = k
        self.h = kernel_bandwidth
        self.knn = None
        self._holdout_features_np = None
        self._model_probs         = None
        self._holdout_labels      = None

    def fit(
        self,
        holdout_features: np.ndarray,       # (n_ho, d)，特征向量
        model_holdout_probs: torch.Tensor,  # (n_ho, C)，被审计模型预测概率
        holdout_labels: torch.Tensor,       # (n_ho,)
    ):
        """构建 k-NN 索引，存储 hold-out 预测。"""
        self._holdout_features_np = holdout_features
        self._model_probs         = model_holdout_probs.float()
        self._holdout_labels      = holdout_labels

        self.knn = NearestNeighbors(
            n_neighbors=self.k, algorithm="auto",
            metric="euclidean", n_jobs=-1,
        )
        self.knn.fit(holdout_features)

        if self.h is None:
            # 自动带宽：k 邻居距离中位数
            sample_dists, _ = self.knn.kneighbors(holdout_features[:500])
            self.h = float(np.median(sample_dists[:, -1])) + 1e-6

        print(
            f"BLocalEstimator 拟合完成。"
            f"  k={self.k}  h={self.h:.4f}  "
            f"hold-out 大小={len(holdout_labels):,}"
        )

    def _compute_b(self, idx: np.ndarray, dist: np.ndarray) -> tuple:
        """对单个点计算加权 B̂ 和平均距离。返回 (b_value, mean_dist)。"""
        w = np.exp(-0.5 * (dist / self.h) ** 2)
        w = w / (w.sum() + 1e-12)
        w_t = torch.from_numpy(w.astype(np.float32))

        k_actual = len(idx)
        p  = self._model_probs[idx]           # (k, C)
        y  = self._holdout_labels[idx]        # (k,)

        actual_loss   = -(p[torch.arange(k_actual), y].clamp(min=1e-8).log())   # (k,)
        believed_loss = -(p * p.clamp(min=1e-8).log()).sum(dim=-1)               # (k,)

        b = ((w_t * actual_loss).sum() - (w_t * believed_loss).sum()).item()
        return b, float(dist.mean())

    def estimate(
        self,
        test_features: np.ndarray,
        batch_size: int = 256,
    ) -> dict:
        """
        对新测试点估计 B̂(x) 和 D(x)。
        返回 {"b_hat": Tensor(N,), "knn_dist": Tensor(N,), "k": int}
        """
        if self.knn is None:
            raise RuntimeError("请先调用 fit()。")

        b_hat_list, knn_dist_list = [], []
        for start in tqdm(range(0, len(test_features), batch_size),
                          desc="估计 B̂_local", leave=False):
            batch = test_features[start:start + batch_size]
            dists, indices = self.knn.kneighbors(batch)
            for i in range(len(batch)):
                b, d = self._compute_b(indices[i], dists[i])
                b_hat_list.append(b)
                knn_dist_list.append(d)

        return {
            "b_hat":    torch.tensor(b_hat_list),
            "knn_dist": torch.tensor(knn_dist_list),
            "k":        self.k,
        }

    def estimate_loo(self, batch_size: int = 256) -> dict:
        """
        对 hold-out 点做 Leave-One-Out B̂ 估计：每个点排除自身。

        方法：用 k+1 邻居（包含自身），然后排除全局索引等于自身的那个点，
              取其余 k 个最近邻做 Nadaraya-Watson 加权估计。

        返回 {"b_hat": Tensor(N,), "knn_dist": Tensor(N,), "k": int, "loo": True}
        """
        if self._holdout_features_np is None:
            raise RuntimeError("请先调用 fit()。")

        N = len(self._holdout_features_np)
        knn_loo = NearestNeighbors(
            n_neighbors=self.k + 1, algorithm="auto",
            metric="euclidean", n_jobs=-1,
        )
        knn_loo.fit(self._holdout_features_np)

        b_hat_list, knn_dist_list = [], []

        for start in tqdm(range(0, N, batch_size), desc="LOO B̂_local", leave=False):
            batch = self._holdout_features_np[start:start + batch_size]
            dists, indices = knn_loo.kneighbors(batch)  # (B, k+1)

            for i in range(len(batch)):
                global_i = start + i
                all_idx  = indices[i]   # (k+1,)
                all_dist = dists[i]     # (k+1,)

                # 排除自身（全局索引 == global_i）
                mask = all_idx != global_i
                idx  = all_idx[mask][:self.k]
                dist = all_dist[mask][:self.k]

                b, d = self._compute_b(idx, dist)
                b_hat_list.append(b)
                knn_dist_list.append(d)

        return {
            "b_hat":    torch.tensor(b_hat_list),
            "knn_dist": torch.tensor(knn_dist_list),
            "k":        self.k,
            "loo":      True,
        }


# ------------------------------------------------------------------ #
#  运行 B̂_local 估计                                                  #
# ------------------------------------------------------------------ #

def run_b_local(
    root: str,
    cfg: dict,
    method: str = "llla",
    feature_model: str = None,
):
    """
    为指定方法估计 B̂_local，并保存结果。

    method:        被审计模型名称（llla/ensemble/mc_dropout）
    feature_model: k-NN 所用特征缓存的名称。
                   默认（None）= 被审计方法自身的主模型特征：
                     llla      → backbone_single
                     ensemble  → ensemble_mean
                     mc_dropout→ backbone_mcdropout
                   传 "imagenet_pretrained" 可做消融实验。
    """
    k = cfg.get("k_neighbors", 50)
    estimator = BLocalEstimator(k=k)

    if feature_model is None:
        feature_model = method_to_feature_model(method)

    print(f"\n[b_local | {method}] 使用特征缓存: {feature_model}")

    # 加载 hold-out 集特征
    ref_holdout  = load_cached_features(root, feature_model, "holdout")
    ref_feats_np = ref_holdout["features"].numpy()

    # 加载被审计模型在 hold-out 上的预测概率
    method_holdout_path = os.path.join(root, "results", method, "holdout.pt")
    if not os.path.exists(method_holdout_path):
        raise FileNotFoundError(
            f"未找到 {method_holdout_path}。"
            f"请先对 {method} 在 holdout 上运行 predict()。"
        )
    holdout_result = torch.load(method_holdout_path, weights_only=False)
    model_probs    = holdout_result["probs"]    # (n_ho, C)
    holdout_labels = holdout_result["labels"]   # (n_ho,)

    estimator.fit(ref_feats_np, model_probs, holdout_labels)

    save_dir = os.path.join(root, "results", f"b_local_{method}")
    os.makedirs(save_dir, exist_ok=True)

    # holdout：Leave-One-Out 估计（用于 flag.py 确定 τ_B 和 d_max）
    print(f"\n估计 B̂_local [{method}|holdout] (LOO) ...")
    result_loo = estimator.estimate_loo()
    path = os.path.join(save_dir, "holdout.pt")
    torch.save(result_loo, path)
    b = result_loo["b_hat"]
    print(
        f"  LOO B̂ 均值={b.mean():.4f}  std={b.std():.4f}  "
        f"  正值比例={( b > 0 ).float().mean():.4f}"
    )

    # 其他分割：标准 k-NN 估计
    for split in ["test_id", "calibration"]:
        ref_data   = load_cached_features(root, feature_model, split)
        test_feats = ref_data["features"].numpy()
        print(f"\n估计 B̂_local [{method}|{split}] ...")
        result = estimator.estimate(test_feats)
        path   = os.path.join(save_dir, f"{split}.pt")
        torch.save(result, path)
        b = result["b_hat"]
        print(
            f"  B̂ 均值={b.mean():.4f}  std={b.std():.4f}  "
            f"  正值比例={( b > 0 ).float().mean():.4f}"
        )

    # OOD
    try:
        ref_ood   = load_cached_features(root, feature_model, "test_ood")
        ood_feats = ref_ood["features"].numpy()
        print(f"\n估计 B̂_local [{method}|test_ood] ...")
        result = estimator.estimate(ood_feats)
        path   = os.path.join(save_dir, "test_ood.pt")
        torch.save(result, path)
        b = result["b_hat"]
        print(f"  B̂ 均值={b.mean():.4f}  std={b.std():.4f}")
    except FileNotFoundError:
        print("  [跳过] test_ood 特征尚未提取。")

    return estimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/default.yaml")
    parser.add_argument("--method",        default="llla",
                        choices=["llla", "ensemble", "mc_dropout", "swag"])
    parser.add_argument("--feature_model", default=None,
                        help="k-NN 特征缓存名称。None=自动用主模型特征；"
                             "'imagenet_pretrained'=消融用 ImageNet 预训练特征")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            config_path,
        )
    cfg  = load_config(config_path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_b_local(root, cfg, method=args.method, feature_model=args.feature_model)
    print("\nB̂_local 估计完成！")


if __name__ == "__main__":
    main()
