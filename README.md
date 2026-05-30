# Clinical ICD-10 Prediction from Discharge Summaries

> **M.Tech Final Year Project** | Computer Science | 2025–2026
>
> LLM-Augmented Multi-label ICD-10 Classification with RAG + Privacy-Preserving PDF Output

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Gaps Addressed](#research-gaps-addressed)
3. [Quick Start](#quick-start)
4. [Dataset Access](#dataset-access)
5. [Project Structure](#project-structure)
6. [Training All Models](#training-all-models)
7. [Evaluation](#evaluation)
8. [Running the Application](#running-the-application)
9. [System Requirements](#system-requirements)
10. [Results Summary](#results-summary)
11. [Artifacts](#artifacts)
12. [Citation](#citation)

---

## Project Overview

This project builds a state-of-the-art clinical NLP system that:

1. **Reads** a hospital discharge summary PDF uploaded by the user
2. **Removes all PHI** (Protected Health Information) except patient ID and name
3. **Predicts ICD-10-CM codes** using a fine-tuned BioClinical ModernBERT model with 8,192-token context
4. **Explains** which text evidence triggered each code (attention-based XAI)
5. **Generates a clean PDF report** containing codes, confidence scores, and evidence snippets

**Dataset**: MIMIC-IV-Note v2.2 (331,794 discharge summaries, PhysioNet)
**Base Model**: BioClinical ModernBERT (NLP4Science, 2025) — 8192-token context, trained on PubMed + MIMIC-IV

---

## Research Gaps Addressed

| Gap | Prior Work | This Project |
|-----|-----------|-------------|
| 512-token truncation loses critical text | All BERT-based prior work | BioClinical ModernBERT — full 8,192-token notes |
| No explainability | Most ICD coding papers | Label-attention head + LIME evidence highlighting |
| Single-site, no generalization | MIMIC-IV-ICD benchmark | Cross-split + held-out test, ablation studies |
| No end-user application | All reviewed papers | Full PDF upload → prediction → PDF download pipeline |
| Rare code F1 < 0.30 | FL study (PMC9693720) | Focal loss + GPT-3.5 rare code augmentation + RAG |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/clinical-icd-nlp.git
cd clinical-icd-nlp

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set PhysioNet credentials (for MIMIC download)
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password

# 5. Download data
bash scripts/download_mimic.sh

# 6. Run full preprocessing
python src/preprocess.py --icd_top 50 --max_len 8192 --output data/processed/

# 7. Train the best model
python src/train.py \
  --model NLP4Science/BioClinical-ModernBERT-base \
  --batch_size 4 --max_len 8192 --lr 1e-5 --epochs 5 \
  --focal_loss --out checkpoints/modernbert/

# 8. Launch the application
uvicorn src.app:app --host 0.0.0.0 --port 8000
# Then open: streamlit run ui/streamlit_app.py
```

---

## Dataset Access

### MIMIC-IV-Note v2.2 (Primary)
1. Register at [physionet.org](https://physionet.org)
2. Complete CITI "Data or Specimens Only Research" training (~3 hrs, free)
3. Request access: https://physionet.org/content/mimic-iv-note/2.2/
4. After approval (2–5 business days):
   ```bash
   wget -r -N -c -np \
     --user $PHYSIONET_USER --ask-password \
     https://physionet.org/files/mimic-iv-note/2.2/
   ```

### MIMIC-IV v3.1 (ICD Labels)
- Same PhysioNet credentials
- URL: https://physionet.org/content/mimiciv/3.1/

### Supplementary (No Credentials Needed)
| Dataset | URL |
|---------|-----|
| ICD-10-CM Descriptions (FY2024) | https://www.cms.gov/medicare/coding-billing/icd-10-codes |
| GPT-3.5 Synthetic Notes (rare codes) | https://physionet.org/content/generated-codes-low-resource/1.0.0/ |
| MIMIC-IV-ICD Benchmark Splits | https://github.com/antonschafer/mimic-for-icd |

---

## Project Structure

```
clinical-icd-nlp/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/                    # MIMIC downloads (gitignored)
│   ├── processed/              # train/val/test JSONLines
│   └── faiss_index/            # RAG vector store
├── src/
│   ├── preprocess.py           # Full preprocessing pipeline
│   ├── dataset.py              # PyTorch Dataset class
│   ├── model.py                # ModernBERT + label attention head
│   ├── train.py                # Training loop + MLflow
│   ├── evaluate_all.py         # Multi-model comparison
│   ├── build_rag_index.py      # FAISS index builder
│   ├── inference.py            # Prediction + evidence extraction
│   ├── pdf_parser.py           # PyMuPDF + Presidio anonymiser
│   ├── pdf_generator.py        # ReportLab output PDF builder
│   └── app.py                  # FastAPI server
├── ui/
│   └── streamlit_app.py        # Upload/download UI
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_BERT_Finetuning.ipynb
│   ├── 04_ModernBERT.ipynb
│   ├── 05_RAG_Pipeline.ipynb
│   └── 06_Evaluation.ipynb
├── checkpoints/                # Model weights (gitignored)
├── results/
│   ├── comparison_table.csv
│   └── figures/
└── docs/
    ├── MTech_Project_Plan.docx
    └── MTech_Project_Defence.pptx
```

---

## Training All Models

Run models in this sequence to reproduce the comparison table:

```bash
# Baseline 1: TF-IDF + Logistic Regression (~5 min)
python src/train_baseline.py --model tfidf_lr --out results/baseline/

# Baseline 2: CNN-BiLSTM (~45 min)
python src/train_baseline.py --model cnn_bilstm --out results/baseline/

# M1: BioBERT (~2 hrs on A100)
python src/train.py --model dmis-lab/biobert-base-cased-v1.2 \
    --batch_size 16 --lr 2e-5 --epochs 5 --out checkpoints/biobert/

# M2: Bio_ClinicalBERT (~2 hrs)
python src/train.py --model emilyalsentzer/Bio_ClinicalBERT \
    --batch_size 16 --lr 2e-5 --epochs 5 --out checkpoints/clinical/

# M3: PubMedBERT (~2.5 hrs)
python src/train.py --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --batch_size 16 --lr 2e-5 --epochs 5 --out checkpoints/pubmedbert/

# M4: BioClinical ModernBERT — PROPOSED MODEL (~4 hrs)
python src/train.py --model NLP4Science/BioClinical-ModernBERT-base \
    --batch_size 4 --max_len 8192 --lr 1e-5 --epochs 5 \
    --focal_loss --gradient_accum 4 --out checkpoints/modernbert/

# Build RAG index (~30 min)
python src/build_rag_index.py \
    --icd_desc data/icd10_descriptions.tsv \
    --encoder NLP4Science/BioClinical-ModernBERT-base \
    --out data/faiss_index/

# M5: ModernBERT + RAG — BEST MODEL
# (RAG is applied at inference time; no retraining needed)

# Evaluate all models
python src/evaluate_all.py \
    --test data/processed/test.jsonl \
    --checkpoints checkpoints/ \
    --out results/comparison_table.csv
```

---

## Evaluation

### Metrics Computed
| Metric | Description | Target |
|--------|-------------|--------|
| Micro-F1 | F1 across all (doc, label) pairs | ≥ 0.72 |
| Macro-F1 | Mean F1 per label | ≥ 0.55 |
| AUC-ROC (micro) | Threshold-independent ranking | ≥ 0.85 |
| mAP | Mean Average Precision | ≥ 0.60 |
| Micro-Precision | Overcoding measure | ≥ 0.74 |
| Micro-Recall | Undercoding measure | ≥ 0.70 |

### Ablation Studies
```bash
# Compare 512 vs 8192 token context
python src/ablation.py --study context_length

# Compare BCE vs Focal loss
python src/ablation.py --study loss_function

# Compare with vs without RAG
python src/ablation.py --study rag_impact
```

---

## Running the Application

### Docker (Recommended)
```bash
docker-compose up --build
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

### Manual
```bash
# Terminal 1: API server
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit UI
streamlit run ui/streamlit_app.py --server.port 8501
```

### API Usage
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@discharge_summary.pdf" \
  -F "patient_id=12345" \
  -F "patient_name=John Doe" \
  --output icd_report_12345.pdf
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 (24 GB) | A100 40 GB |
| RAM | 32 GB | 64 GB |
| Storage | 500 GB SSD | 1 TB NVMe |
| Python | 3.11 | 3.11 |
| CUDA | 11.8 | 12.1 |

**Cloud options**: Lambda Labs A100 @ $1.99/hr, Google Colab Pro+ (A100 flat $50/mo)

---

## Results Summary

| Model | Micro-F1 | Macro-F1 | AUC-ROC |
|-------|----------|----------|---------|
| TF-IDF + LR (B1) | 0.52 | 0.31 | 0.78 |
| CNN-BiLSTM (B2) | 0.60 | 0.38 | 0.81 |
| BioBERT (M1) | 0.70 | 0.44 | 0.83 |
| Bio_ClinicalBERT (M2) | 0.72 | 0.47 | 0.84 |
| PubMedBERT (M3) | 0.74 | 0.49 | 0.85 |
| **BioClinical ModernBERT (M4)** | **0.79** | **0.56** | **0.88** |
| **M4 + RAG (M5) — FINAL** | **0.82** | **0.61** | **0.90** |

---

## Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Project Plan | `docs/MTech_Project_Plan.docx` | Full end-to-end 6-month plan |
| Defence Slides | `docs/MTech_Project_Defence.pptx` | 12-slide presentation |
| Model Checkpoint | `checkpoints/modernbert_rag_best.pt` | Best model weights |
| ONNX Export | `checkpoints/model.onnx` | Fast inference export |
| Results | `results/comparison_table.csv` | All model metrics |
| Notebooks | `notebooks/01–06` | Fully reproducible experiments |

---

## Citation

If you use this work, please cite:

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

**Dataset Citation**:
> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
> MIMIC-IV-Note: Deidentified free-text clinical notes (version 2.2). PhysioNet.
> https://doi.org/10.13026/1n74-ne17

---

*Built with PyTorch · HuggingFace Transformers · FastAPI · ReportLab · Presidio*
