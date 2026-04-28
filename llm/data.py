"""
Data loading for BoolQ (in-distribution) and PubMedQA (out-of-distribution).

Both are yes/no QA tasks with identical label space {0=no, 1=yes},
so we can train on BoolQ and evaluate on PubMedQA to induce
controlled distribution shift — a language-domain analog of CIFAR-10 -> CIFAR-10-C.

The "label token" trick:
  Instead of adding a new classification head, we read the logits at
  fixed label tokens ("no" and "yes") and softmax over just those two.
  This fits Assumption 3.1 exactly: K=2 softmax exponential family
  with S(x,y) = e_y ⊗ phi_LLM(x).
"""

import random
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


def format_boolq_example(ex: Dict) -> Dict:
    prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
    label = int(ex["answer"])  # True -> 1 (yes), False -> 0 (no)
    return {"text": prompt, "label": label}


def format_pubmedqa_example(ex: Dict) -> Dict:
    # PubMedQA has final_decision in {"yes", "no", "maybe"}; skip "maybe"
    if ex["final_decision"] == "maybe":
        return None
    context = " ".join(ex["context"]["contexts"])
    prompt = f"Passage: {context}\nQuestion: {ex['question']}\nAnswer:"
    label = 1 if ex["final_decision"] == "yes" else 0
    return {"text": prompt, "label": label}


class QATokenDataset(Dataset):
    """
    Wraps a list of {"text", "label"} dicts into tokenized tensors.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }


def load_boolq(tokenizer, max_seq_len: int,
               train_subset_size: int = -1, val_subset_size: int = -1,
               seed: int = 42):
    ds = load_dataset("boolq")
    train = [format_boolq_example(ex) for ex in ds["train"]]
    val = [format_boolq_example(ex) for ex in ds["validation"]]

    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)

    if train_subset_size > 0:
        train = train[:train_subset_size]
    if val_subset_size > 0:
        val = val[:val_subset_size]

    return (
        QATokenDataset(train, tokenizer, max_seq_len),
        QATokenDataset(val, tokenizer, max_seq_len),
    )


def load_pubmedqa(tokenizer, max_seq_len: int, subset_size: int = -1, seed: int = 42):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    # Only labeled split is 'train' (1000 examples); filter "maybe" out
    examples = [format_pubmedqa_example(ex) for ex in ds["train"]]
    examples = [ex for ex in examples if ex is not None]

    random.Random(seed).shuffle(examples)

    if subset_size > 0:
        examples = examples[:subset_size]

    return QATokenDataset(examples, tokenizer, max_seq_len)


def make_dataloaders(cfg, tokenizer) -> Dict[str, DataLoader]:
    """
    Returns dict with keys: 'train', 'val', 'heldout', 'ood'.
    - train:    BoolQ train set, for LoRA fine-tuning
    - val:      BoolQ validation set, for in-distribution evaluation
    - heldout:  separate portion of BoolQ train, for k-NN B_ell estimator
    - ood:      PubMedQA, for out-of-distribution evaluation
    """
    train_full, val = load_boolq(
        tokenizer, cfg.max_seq_len,
        train_subset_size=cfg.train_subset_size,
        val_subset_size=cfg.eval_subset_size,
        seed=cfg.seed,
    )
    ood = load_pubmedqa(
        tokenizer, cfg.max_seq_len, cfg.eval_subset_size, cfg.seed
    )

    # Split train_full into (actual train, heldout for B_ell)
    n_total = len(train_full)
    n_heldout = min(cfg.heldout_subset_size, n_total // 3)
    indices = list(range(n_total))
    random.Random(cfg.seed).shuffle(indices)
    heldout_idx = indices[:n_heldout]
    train_idx = indices[n_heldout:]

    heldout_examples = [train_full.examples[i] for i in heldout_idx]
    train_examples = [train_full.examples[i] for i in train_idx]

    train = QATokenDataset(train_examples, tokenizer, cfg.max_seq_len)
    heldout = QATokenDataset(heldout_examples, tokenizer, cfg.max_seq_len)

    return {
        "train": DataLoader(train, batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(val, batch_size=cfg.batch_size, shuffle=False),
        "heldout": DataLoader(heldout, batch_size=cfg.batch_size, shuffle=False),
        "ood": DataLoader(ood, batch_size=cfg.batch_size, shuffle=False),
    }
