"""
dataset.py
──────────
PyTorch Dataset class for multi-label ICD-10 classification.
Reads JSONL files produced by preprocess.py.
"""

import json
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ICD10Dataset(Dataset):
    """
    Multi-label ICD-10 dataset for BioClinical ModernBERT.

    Args:
        jsonl_path:       Path to .jsonl file (train/val/test)
        tokenizer_name:   HuggingFace tokenizer id
        label_binarizer:  Fitted sklearn MultiLabelBinarizer
        max_length:       Max token length (default 2048 for Colab T4)
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str,
        label_binarizer,
        max_length: int = 2048,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mlb = label_binarizer
        self.max_length = max_length
        self.records = self._load(jsonl_path)
        self.num_labels = len(self.mlb.classes_)

    def _load(self, path: str) -> List[Dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        text = record["text"]
        labels = record["labels"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Multi-hot encode labels
        label_vector = self.mlb.transform([labels])[0]  # shape: (num_labels,)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_vector, dtype=torch.float),
            "hadm_id":        torch.tensor(record["hadm_id"], dtype=torch.long),
        }


def load_label_binarizer(path: str):
    """Load a pickled MultiLabelBinarizer."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_class_weights(jsonl_path: str, mlb, device: str = "cpu") -> torch.Tensor:
    """
    Compute inverse-frequency class weights for focal loss / BCE weighting.
    Returns tensor of shape (num_labels,).
    """
    label_counts = np.zeros(len(mlb.classes_))
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            vec = mlb.transform([record["labels"]])[0]
            label_counts += vec
            total += 1

    # Inverse frequency, clipped to avoid extreme weights
    weights = total / (label_counts + 1e-6)
    weights = np.clip(weights, 1.0, 10.0)
    return torch.tensor(weights, dtype=torch.float).to(device)
