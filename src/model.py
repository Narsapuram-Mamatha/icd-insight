"""
model.py
────────
Model loading utilities for BioClinical ModernBERT + PEFT (QLoRA / LoRA).

Supports:
  - QLoRA: 4-bit NF4 quantized base + LoRA adapters  (default, Colab T4)
  - LoRA:  FP16 base + LoRA adapters                 (fallback)
  - Full:  Standard FP32/FP16 fine-tune              (requires A100+)
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Literal, Optional

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# `NLP4Science/BioClinical-ModernBERT-base` is not publicly accessible (HTTP 401);
# use the canonical public BioClinical ModernBERT (MIT license, same 150M model).
BASE_MODEL_ID = "thomas-sounack/BioClinical-ModernBERT-base"

# LoRA target modules for ModernBERT attention + FFN layers.
# ModernBERT names its linears Wqkv (fused QKV) / Wo (attn out) / Wi (MLP in).
MODERNBERT_LORA_TARGETS = [
    "Wqkv",
    "Wo",
]


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-label sigmoid focal loss.
    Reduces the relative loss for well-classified examples,
    focusing training on hard / rare codes.

    gamma=2 is standard (Lin et al., 2017).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).mean()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_qlora_model(
    num_labels: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    model_id: str = BASE_MODEL_ID,
):
    """
    Load BioClinical ModernBERT in 4-bit NF4 quantization and attach LoRA adapters.
    Designed for Google Colab T4 (16 GB VRAM).

    Returns: (model, tokenizer)
    """
    print(f"  Loading {model_id} in 4-bit NF4 …")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 bits/param
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        problem_type="multi_label_classification",
    )

    # Prepare model for k-bit training (freezes base, adds gradient checkpointing)
    from peft import prepare_model_for_kbit_training
    base_model = prepare_model_for_kbit_training(base_model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=MODERNBERT_LORA_TARGETS,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def build_lora_model(
    num_labels: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    model_id: str = BASE_MODEL_ID,
):
    """
    Fallback: FP16 base + LoRA adapters (no 4-bit quantization).
    Use if bitsandbytes is unavailable.
    """
    print(f"  Loading {model_id} in FP16 (LoRA fallback) …")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        problem_type="multi_label_classification",
    )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=MODERNBERT_LORA_TARGETS,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def load_adapter_model(
    adapter_path: str,
    label_binarizer_path: str,
    strategy: Literal["qlora", "lora"] = "qlora",
    model_id: str = BASE_MODEL_ID,
):
    """
    Load a trained PEFT adapter from disk for inference.

    Args:
        adapter_path:          Directory containing adapter_model.safetensors
        label_binarizer_path:  Path to label_binarizer.pkl
        strategy:              "qlora" or "lora"
        model_id:              Base HuggingFace model id

    Returns: (model, tokenizer, mlb)
    """
    with open(label_binarizer_path, "rb") as f:
        mlb = pickle.load(f)

    num_labels = len(mlb.classes_)

    if strategy == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            problem_type="multi_label_classification",
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            problem_type="multi_label_classification",
        )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer, mlb
