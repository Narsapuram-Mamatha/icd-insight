"""
inference.py
────────────
Inference pipeline: load model + adapters → tokenize text → predict ICD-10 codes → extract evidence.

Usage:
    from src.inference import ICD10Predictor
    predictor = ICD10Predictor(adapter_path="checkpoints/", strategy="qlora")
    results = predictor.predict(text)
"""

from __future__ import annotations

import csv
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 description loader
# ─────────────────────────────────────────────────────────────────────────────

def load_icd_descriptions(tsv_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load ICD-10-CM code → description mapping.
    Falls back to empty dict if file not found.
    """
    descriptions = {}
    if tsv_path and os.path.exists(tsv_path):
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                code = row.get("icd_code", "").strip()
                desc = row.get("description", "").strip()
                if code:
                    descriptions[code] = desc
    return descriptions


# ─────────────────────────────────────────────────────────────────────────────
# Evidence extractor (attention-based)
# ─────────────────────────────────────────────────────────────────────────────

def extract_evidence_snippets(
    text: str,
    tokenizer,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_idx: int,
    top_k_snippets: int = 3,
    snippet_window: int = 80,
) -> List[str]:
    """
    Extract top-k text snippets as evidence for a predicted label
    using attention weight attribution.

    Args:
        text:            Original (clean) note text
        tokenizer:       HuggingFace tokenizer
        model:           Loaded PEFT model (with output_attentions=True)
        input_ids:       Tokenized input [1, seq_len]
        attention_mask:  Attention mask  [1, seq_len]
        label_idx:       Index of the predicted label to explain
        top_k_snippets:  Number of evidence snippets to return
        snippet_window:  Character window around each evidence token

    Returns:
        List of text snippet strings
    """
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Average attention across all heads and layers
        # Each attention layer: [batch, heads, seq, seq]
        attn_layers = outputs.attentions  # tuple of tensors
        if not attn_layers:
            return []

        # Mean across layers and heads → shape [seq_len, seq_len]
        attn = torch.stack(attn_layers).mean(dim=0).mean(dim=1).squeeze(0)

        # CLS token attention to each token → shape [seq_len]
        cls_attn = attn[0, :].cpu().numpy()

        # Get top-k token positions by attention score
        top_positions = np.argsort(cls_attn)[::-1][:top_k_snippets * 5]

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        snippets = set()

        for pos in top_positions:
            if pos == 0 or pos >= len(tokens):  # skip CLS/PAD
                continue
            token = tokens[pos]
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
                continue

            # Find approximate character offset in original text
            prefix_tokens = tokenizer.convert_tokens_to_string(tokens[1:pos])
            char_start = max(0, len(prefix_tokens) - snippet_window // 2)
            char_end = min(len(text), char_start + snippet_window)
            snippet = text[char_start:char_end].strip()

            if len(snippet) > 20 and snippet not in snippets:
                snippets.add(snippet)

            if len(snippets) >= top_k_snippets:
                break

        return list(snippets)

    except Exception as e:
        # Evidence extraction is best-effort; never crash inference
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ─────────────────────────────────────────────────────────────────────────────

class ICD10Predictor:
    """
    End-to-end ICD-10 prediction from clinical text.

    Args:
        adapter_path:          Path to saved PEFT adapter directory
        icd_descriptions_path: Path to icd10_descriptions.tsv (optional)
        strategy:              "qlora" or "lora"
        threshold:             Sigmoid threshold for positive label (default 0.5)
        max_length:            Max tokenization length
        base_model_id:         HuggingFace base model id
    """

    def __init__(
        self,
        adapter_path: str = "checkpoints",
        icd_descriptions_path: Optional[str] = None,
        strategy: str = "qlora",
        threshold: float = 0.5,
        max_length: int = 2048,
        base_model_id: str = "thomas-sounack/BioClinical-ModernBERT-base",
    ):
        from src.model import load_adapter_model

        label_binarizer_path = os.path.join(adapter_path, "label_binarizer.pkl")
        self.model, self.tokenizer, self.mlb = load_adapter_model(
            adapter_path=adapter_path,
            label_binarizer_path=label_binarizer_path,
            strategy=strategy,
            model_id=base_model_id,
        )
        self.threshold = threshold
        self.max_length = max_length
        self.descriptions = load_icd_descriptions(icd_descriptions_path)
        self.device = next(self.model.parameters()).device

        print(f"  Predictor ready — {len(self.mlb.classes_)} codes, threshold={threshold}")

    def predict(self, text: str, extract_evidence: bool = True) -> List[Dict]:
        """
        Predict ICD-10 codes for a clinical note.

        Args:
            text:             Clean clinical note text
            extract_evidence: Whether to extract attention-based evidence snippets

        Returns:
            List of dicts: [{icd_code, description, confidence, evidence}, ...]
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(0)           # shape: (num_labels,)
            probs  = torch.sigmoid(logits).cpu().numpy()

        # Apply threshold
        predicted_indices = np.where(probs >= self.threshold)[0]

        results = []
        for idx in predicted_indices:
            code = self.mlb.classes_[idx]
            confidence = float(probs[idx])

            # Evidence extraction
            evidence = []
            if extract_evidence:
                evidence = extract_evidence_snippets(
                    text=text,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label_idx=idx,
                )

            results.append({
                "icd_code":    code,
                "description": self.descriptions.get(code, f"ICD-10: {code}"),
                "confidence":  round(confidence, 4),
                "evidence":    evidence,
            })

        # Sort by confidence descending
        results.sort(key=lambda x: -x["confidence"])
        return results
