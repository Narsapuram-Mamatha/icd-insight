# Clinical ICD-10 Prediction from Discharge Summaries

> **M.Tech Final Year Project** | Computer Science | 2025–2026  
>
> LLM-Augmented Multi-label ICD-10 Classification with RAG + Privacy-Preserving PDF Output

[![Open Training Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/icd-insight/blob/main/notebooks/01_QLoRA_Training.ipynb)
[![Open Inference Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/icd-insight/blob/main/notebooks/02_Inference_New_Summary.ipynb)

---

## Project Overview

This project builds a clinical NLP system that:

1. **Reads** a hospital discharge summary PDF
2. **Removes PHI** (Protected Health Information) via Microsoft Presidio
3. **Predicts ICD-10-CM codes** using BioClinical ModernBERT fine-tuned with **QLoRA**
4. **Explains** predictions with attention-based evidence snippets (XAI)
5. **Generates a clean PDF report** with codes, confidence scores, and evidence

**Dataset**: birgermoell/icd10-clinical-notes (Hugging Face) — Synthetic public dataset  
**Base Model**: [BioClinical ModernBERT](https://huggingface.co/NLP4Science/BioClinical-ModernBERT-base) — 8,192 token context  
**Strategy**: QLoRA (4-bit NF4 quantization + LoRA adapters) — runs on **Colab Free T4**

---

## Research Gaps Addressed

| Gap | Prior Work | This Project |
|-----|-----------|-------------|
| 512-token truncation loses critical text | All BERT-based prior work | BioClinical ModernBERT — 2,048–8,192 tokens |
| No explainability | Most ICD coding papers | Attention-weight evidence highlighting |
| Requires A100+ GPU | Full fine-tuning papers | QLoRA — fits on free T4 (16 GB) |
| No end-user application | All reviewed papers | PDF upload → prediction → PDF download |
| Rare code F1 < 0.30 | FL study (PMC9693720) | Focal loss (γ=2) for class imbalance |

---

## Phase 1 — Colab Notebooks

### Notebook A: QLoRA Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/icd-insight/blob/main/notebooks/01_QLoRA_Training.ipynb)

**File**: `notebooks/01_QLoRA_Training.ipynb`

Trains BioClinical ModernBERT with QLoRA on `birgermoell/icd10-clinical-notes`.  
Saves LoRA adapter weights (~20–50 MB) to Google Drive for GitHub upload.

**Steps inside:**
| Section | Description | Est. Time |
|---------|-------------|-----------|
| A.0 | Hardware check (T4 GPU verification) | < 1 min |
| A.1 | Install dependencies | ~5 min |
| A.2 | Mount Google Drive for checkpoints | < 1 min |
| A.3 | Public dataset preprocessing | ~15 min |
| A.4 | Tokenization + Dataset class | ~5 min |
| A.5 | Load model in 4-bit QLoRA | ~5 min |
| A.6 | Configure trainer + focal loss | < 1 min |
| A.7 | **Training** (5 epochs) | ~90 min |
| A.8 | Evaluation + plots | ~10 min |
| A.9 | Save adapter artifacts | ~5 min |
| A.10 | Download zip for GitHub | < 1 min |

---

### Notebook B: Inference on New Discharge Summaries
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/icd-insight/blob/main/notebooks/02_Inference_New_Summary.ipynb)

**File**: `notebooks/02_Inference_New_Summary.ipynb`

Runs ICD-10 prediction on any new discharge summary PDF.  
Loads adapters from this GitHub repo — no retraining needed.

**User flow:**
```
1. Open notebook in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (Ctrl+F9)
4. Cell B.3 shows file picker → upload your discharge PDF
5. Wait ~2–3 minutes for inference
6. PDF report auto-downloads to your computer
```

---

## QLoRA Configuration

```python
# 4-bit NF4 quantization (base model: ~2 GB VRAM vs ~6 GB FP32)
BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
)

# LoRA adapters (~4–8 M trainable params out of 150 M total)
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj"],
           task_type=TaskType.SEQ_CLS)

# Training (effective batch = 4 × 8 = 32)
batch_size=4, gradient_accumulation=8, lr=2e-4, epochs=5
focal_loss(gamma=2)  # handles rare code imbalance
```

---

## Repository Structure

```
icd-insight/
├── PLAN.md                              ← Implementation plan
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 01_QLoRA_Training.ipynb          ← Colab training notebook ← RUN THIS FIRST
│   └── 02_Inference_New_Summary.ipynb   ← Colab inference notebook
│
├── checkpoints/                         ← LoRA adapter weights (git-tracked)
│   ├── adapter_model.safetensors        ← ~20–50 MB (added after training)
│   ├── adapter_config.json
│   ├── tokenizer_config.json
│   ├── label_binarizer.pkl
│   └── training_results.json
│
├── results/
│   ├── metrics_summary.json
│   ├── per_code_f1.csv
│   └── figures/
│
├── src/                                 ← Reusable Python modules
│   ├── preprocess.py                    ← MIMIC-IV preprocessing
│   ├── dataset.py                       ← PyTorch Dataset class
│   ├── model.py                         ← QLoRA/LoRA model builders
│   ├── inference.py                     ← Prediction pipeline
│   ├── pdf_parser.py                    ← PyMuPDF + Presidio PHI removal
│   └── pdf_generator.py                 ← ReportLab PDF report builder
│
└── docs/
    └── MTech_Project_Plan.docx
```

---

## Dataset Access

### birgermoell/icd10-clinical-notes (Public)
This project uses a 100% public, synthetic Hugging Face dataset. No credentials or DUA required. The pipeline handles downloading it automatically.

---

## Expected Results (QLoRA on T4)

| Metric | Target | Full FT Reference |
|--------|--------|------------------|
| Micro-F1 | ≥ 0.72 | 0.79 |
| Macro-F1 | ≥ 0.48 | 0.56 |
| AUC-ROC | ≥ 0.84 | 0.88 |
| Training time | ~1.5–2 hrs | ~4 hrs (A100) |
| Adapter size | ~20–50 MB | N/A (full weights ~600 MB) |

---

## Workflow

```
Step 1: Open 01_QLoRA_Training.ipynb in Colab → set GPU T4 → Run All
Step 2: Download checkpoint zip → extract to checkpoints/ → git push
Step 3: Open 02_Inference_New_Summary.ipynb → Run All → upload PDF → get report
```

---

## Phase 2 (Planned): Web Application

A minimal web app where users can upload a discharge PDF and view/download ICD predictions directly in the browser.

```
Upload PDF → FastAPI backend → QLoRA inference → Display codes + confidence bars → Download PDF
```

Deployment options: Hugging Face Spaces, Render.com, Railway.app

---

## System Requirements (Colab Free Tier)

| Component | Value |
|-----------|-------|
| GPU | T4 16 GB (free via Colab) |
| RAM | ~12 GB |
| Storage | ~77 GB |
| Session | ~4–6 hrs (save to Drive mid-training) |

---

## Citation

```bibtex
@misc{yourname2026clinicalicd,
  title={LLM-Augmented ICD-10 Code Prediction from Discharge Summaries
         with Privacy-Preserving PDF Output},
  author={Your Name},
  year={2026},
  institution={Your University},
  note={M.Tech Final Year Project}
}
```

> Möller, B. (2023). icd10-clinical-notes. Hugging Face.
> https://huggingface.co/datasets/birgermoell/icd10-clinical-notes

---

*Built with PyTorch · HuggingFace Transformers · PEFT · BitsAndBytes · PyMuPDF · Presidio · ReportLab*
