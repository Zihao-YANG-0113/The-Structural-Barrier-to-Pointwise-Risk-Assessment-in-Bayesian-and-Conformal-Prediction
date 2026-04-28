"""
Model setup, LoRA fine-tuning, and Last-Layer Laplace approximation (LLLA).

Core decisions:
  1. We add a 2-class classification head on top of pooled hidden states.
     (Simpler than the "label token logit" trick; mathematically equivalent
      for our purposes since both give a K=2 softmax exponential family.)
  2. LoRA is applied to attention matrices only.
  3. LLLA variant: after LoRA training, we freeze *everything* except the
     classification head, and fit a Laplace approximation N(theta_hat, H^-1)
     over just the 2*D classification head weights.
     This is the frozen-backbone, finite-p setting where Assumption 3.1 holds.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer


class BayesianLastLayerClassifier(nn.Module):
    """
    Frozen backbone (with LoRA during training) + 2-class linear head.

    After training, we treat the backbone (including LoRA) as a fixed feature
    extractor phi(x), and do Bayesian inference only over the linear head.
    """

    def __init__(self, model_name: str, num_labels: int = 2,
                 torch_dtype: Optional[torch.dtype] = None):
        super().__init__()
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
        self.hidden_dim = self.backbone.config.hidden_size
        self.num_labels = num_labels
        # Classifier kept in float32 for numerical stability of LLLA
        self.classifier = nn.Linear(self.hidden_dim, num_labels)

    def get_features(self, input_ids, attention_mask):
        """Return phi(x) in R^D. Uses mean pooling over non-padding tokens."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                     # (B, T, D)
        mask = attention_mask.unsqueeze(-1).float()        # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # (B, D)
        return pooled

    def forward(self, input_ids, attention_mask, return_features: bool = False):
        phi = self.get_features(input_ids, attention_mask)
        phi_for_head = phi.to(self.classifier.weight.dtype)
        logits = self.classifier(phi_for_head)
        if return_features:
            return logits, phi_for_head
        return logits


_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_model_for_training(cfg):
    """Attach LoRA adapters for fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _DTYPE_MAP.get(getattr(cfg, "torch_dtype", "float32"), torch.float32)
    model = BayesianLastLayerClassifier(cfg.model_name, num_labels=2, torch_dtype=dtype)

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # Apply LoRA only to backbone; classifier stays as a regular nn.Linear
    model.backbone = get_peft_model(model.backbone, lora_cfg)
    return model, tokenizer


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n += len(labels)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def extract_features_and_labels(model, loader, device):
    """
    Extract phi(x) for every example in loader. Returns (features, labels, logits).
    Backbone including LoRA is fixed during this step.
    """
    model.eval()
    feats_list, labels_list, logits_list = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, feats = model(input_ids, attention_mask, return_features=True)
        feats_list.append(feats.cpu())
        labels_list.append(labels.cpu())
        logits_list.append(logits.cpu())

    return (
        torch.cat(feats_list, dim=0),
        torch.cat(labels_list, dim=0),
        torch.cat(logits_list, dim=0),
    )


def _compute_diag_fisher(features: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Diagonal GGN Fisher for a K-class softmax linear head."""
    K, D = W.shape
    with torch.no_grad():
        logits = features @ W.T + b
        probs = torch.softmax(logits, dim=-1)
    diag_fisher = torch.zeros(K, D)
    for k in range(K):
        weights_k = probs[:, k] * (1 - probs[:, k])
        diag_fisher[k] = (weights_k.unsqueeze(-1) * features ** 2).sum(0)
    return diag_fisher


def optimize_prior_precision_ml(features: torch.Tensor, labels: torch.Tensor,
                                classifier: nn.Linear,
                                init: float = 1.0,
                                n_steps: int = 300,
                                lr: float = 0.1) -> float:
    """
    Type-II MLE for the scalar prior precision lambda via the Laplace model
    evidence (Daxberger et al. 2021; Yang et al. 2024):

        log p(D|lambda) ~= -NLL(W_MAP) - lambda/2 ||W||^2
                          - 1/2 log det(H + lambda I) + P/2 log lambda

    We optimize log(lambda) by gradient ascent. Terms independent of lambda
    are dropped. Works on diagonal Fisher over the classifier weight matrix.
    """
    W = classifier.weight.data.detach().cpu()
    b = classifier.bias.data.detach().cpu()
    K, D = W.shape
    P = K * D

    features = features.cpu()
    diag_fisher = _compute_diag_fisher(features, W, b)     # (K, D)
    theta_sq = (W ** 2).sum().item()                       # ||W||^2 (bias excluded from prior)

    log_lam = torch.tensor(float(torch.tensor(init).log()), requires_grad=True)
    optim = torch.optim.Adam([log_lam], lr=lr)
    for _ in range(n_steps):
        lam = log_lam.exp()
        log_det = (diag_fisher + lam).log().sum()
        neg_log_ev = 0.5 * lam * theta_sq + 0.5 * log_det - 0.5 * P * log_lam
        optim.zero_grad()
        neg_log_ev.backward()
        optim.step()
    return log_lam.exp().item()


def fit_llla(features: torch.Tensor, labels: torch.Tensor,
             classifier: nn.Linear, prior_precision: float = 1.0):
    """
    Fit a diagonal-Fisher Laplace approximation over the classifier weights.

    Posterior precision: precision_diag[k,j] = H_diag[k,j] + lambda
    Returns: (W_map, b_map, precision_diag)
    """
    W = classifier.weight.data.detach().cpu()
    b = classifier.bias.data.detach().cpu()
    features = features.cpu()
    diag_fisher = _compute_diag_fisher(features, W, b)
    precision_diag = diag_fisher + prior_precision
    return W, b, precision_diag


def predict_with_llla(features: torch.Tensor, W_map: torch.Tensor, b_map: torch.Tensor,
                      precision_diag: torch.Tensor, n_samples: int = 20):
    """
    Draw weight samples from N(W_map, diag(1/precision)) and average softmax.
    Returns (mean_probs, posterior_entropy, mutual_info, mc_logits_list).

    - mean_probs:       posterior predictive mean over classes, (N, K)
    - posterior_entropy:H[bar p(.|x)], total uncertainty, (N,)
    - mutual_info:      MI(theta; Y|x) = H[bar p] - E[H[p_theta]], epistemic, (N,)
    """
    K, D = W_map.shape
    N = features.shape[0]
    std_diag = precision_diag.rsqrt()  # (K, D)

    all_probs = torch.zeros(n_samples, N, K)
    for s in range(n_samples):
        noise = torch.randn(K, D) * std_diag
        W_s = W_map + noise                                 # (K, D)
        logits_s = features @ W_s.T + b_map                 # (N, K)
        all_probs[s] = torch.softmax(logits_s, dim=-1)

    mean_probs = all_probs.mean(0)                                  # (N, K)
    # Total entropy of the predictive mean
    H_predmean = -(mean_probs * mean_probs.clamp(min=1e-10).log()).sum(-1)
    # E[H[p_theta]]
    per_sample_ent = -(all_probs * all_probs.clamp(min=1e-10).log()).sum(-1)  # (S, N)
    E_H = per_sample_ent.mean(0)
    mutual_info = H_predmean - E_H
    return mean_probs, H_predmean, mutual_info


def predict_with_llla_probit(features: torch.Tensor,
                             W_map: torch.Tensor, b_map: torch.Tensor,
                             precision_diag: torch.Tensor):
    """
    Closed-form probit (MacKay) approximation to the Laplace posterior
    predictive (Daxberger et al. 2021, Eq. 15):

        p(y=k|x) ~= softmax_k( mu_k / sqrt(1 + pi/8 * sigma_k^2(x)) )

    where mu_k = W_k^MAP . phi(x) + b_k  and  sigma_k^2(x) = sum_j phi_j^2 * Cov(W_k)_jj.

    Deterministic, no MC variance, O(ND) cost.
    Returns (mean_probs, entropy, mutual_info). MI is undefined in the
    marginalized form; we return zeros as a clear sentinel.
    """
    features = features.cpu()
    mu = features @ W_map.T + b_map                                 # (N, K)
    var_diag = precision_diag.reciprocal()                          # (K, D)
    sigma_sq = features.pow(2) @ var_diag.T                         # (N, K)
    kappa = (1.0 + (torch.pi / 8.0) * sigma_sq).sqrt()
    scaled = mu / kappa
    mean_probs = torch.softmax(scaled, dim=-1)
    entropy = -(mean_probs * mean_probs.clamp(min=1e-10).log()).sum(-1)
    mutual_info = torch.zeros(mean_probs.shape[0])
    return mean_probs, entropy, mutual_info
