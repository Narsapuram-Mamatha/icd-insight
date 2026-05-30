# ICD-Insight: Updated Implementation Plan
### M.Tech Final Year Project — Clinical ICD-10 Prediction from Discharge Summaries

> **Strategy**: Parameter-Efficient Fine-Tuning (PEFT) on Google Colab Free Tier  
> **Techniques**: LoRA / QLoRA · birgermoell/icd10-clinical-notes · BioClinical ModernBERT  
> **Output**: Colab Notebooks → LoRA Adapters on GitHub → Inference PDF Report

---

## Project Phases

```
Phase 1 (Current)
─────────────────
  Notebook A: 01_QLoRA_Training.ipynb
    └─ Fine-tune BioClinical ModernBERT + QLoRA on public HF dataset
    └─ Save LoRA adapter weights (~20–50 MB) → Google Drive → GitHub
    └─ Save evaluation metrics and figures

  Notebook B: 02_Inference_New_Summary.ipynb
    └─ Load adapters from GitHub
    └─ Upload any discharge summary PDF (interactive Colab prompt)
    └─ Predict ICD-10 codes + generate PDF report → auto-download

Phase 2 (Next)
──────────────
  Minimal Web App (FastAPI + HTML/CSS/JS)
    └─ Upload PDF → predict → display codes + confidence → download PDF
```

---

## Hardware: Google Colab Free Tier

| Resource | Colab Free |
|----------|-----------|
| GPU | T4 (16 GB VRAM) |
| RAM | ~12 GB |
| Disk | ~77 GB |
| Session | ~4–6 hrs |

**Strategy: QLoRA** loads BioClinical ModernBERT in 4-bit NF4 (~2 GB VRAM),  
attaches trainable LoRA adapters (~4–8 M trainable params out of 150 M total),  
and saves only the adapter delta (~20–50 MB) to GitHub.

---

## Chosen Configuration

```python
# 4-bit quantization
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

# LoRA adapters
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj"],
           task_type=TaskType.SEQ_CLS)

# Training
batch_size=4, gradient_accumulation=8  # effective batch=32
lr=2e-4, epochs=5, max_length=2048
focal_loss(gamma=2)  # handles class imbalance
```

---

## Repository Structure

```
icd-insight/
├── PLAN.md
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_QLoRA_Training.ipynb        ← Run on Colab to train
│   └── 02_Inference_New_Summary.ipynb ← Run on Colab to predict
├── checkpoints/                       ← LoRA adapter (git-tracked, ~50 MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer_config.json
│   ├── label_binarizer.pkl
│   └── training_results.json
├── results/
│   ├── metrics_summary.json
│   ├── per_code_f1.csv
│   └── figures/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── inference.py
│   ├── pdf_parser.py
│   └── pdf_generator.py
└── docs/
```

---

## Expected Metrics (QLoRA on T4)

| Metric | Target | Full FT Reference |
|--------|--------|------------------|
| Micro-F1 | ≥ 0.72 | 0.79 |
| Macro-F1 | ≥ 0.48 | 0.56 |
| AUC-ROC | ≥ 0.84 | 0.88 |
| Training time | ~1.5–2 hrs | ~4 hrs (A100) |
| Adapter size | ~20–50 MB | N/A |

---

*Plan v2.0 — Colab Free Tier · QLoRA/LoRA PEFT · Phase 1 Notebooks*
