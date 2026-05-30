"""
preprocess.py
─────────────
Preprocessing pipeline for the public Hugging Face dataset:
birgermoell/icd10-clinical-notes

This script downloads synthetic clinical notes, filters for English,
extracts the top K ICD-10 codes, and prepares train/val/test splits
in JSONL format for the ICD-Insight pipeline.

Usage (from repo root):
    python src/preprocess.py \
        --output_dir  data/processed/ \
        --top_k_codes 50 \
        --max_notes   60000

Outputs:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl
    data/processed/label_binarizer.pkl
    data/processed/icd10_descriptions.tsv
"""

import argparse
import json
import os
import pickle
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("▶ Loading dataset from Hugging Face: birgermoell/icd10-clinical-notes …")
    # The dataset has a 'train' split which we will use as our full corpus
    ds = load_dataset("birgermoell/icd10-clinical-notes", split="train")
    df = ds.to_pandas()
    
    print(f"  Total records downloaded: {len(df):,}")

    print("▶ Filtering and cleaning …")
    # Optional language filter. NOTE: this dataset is a small multilingual synthetic
    # set (34 languages x 53 rows = 1,802). The English subset is only 53 rows (one
    # per ICD code), which cannot be split into train/val/test. Default keeps all
    # languages so the model has enough examples per class to train and evaluate.
    if "language" in df.columns and args.language and args.language.lower() != "all":
        df = df[df["language"] == args.language].copy()
        print(f"  Records after language filter ({args.language}): {len(df):,}")
    else:
        print(f"  Keeping all languages: {len(df):,} records")

    # Ensure required columns exist and drop empty text
    df = df[["id", "code", "name", "journal_note"]].dropna(subset=["code", "journal_note"])
    df = df[df["journal_note"].str.len() > 50].copy()
    
    # Standardize column names for our pipeline
    df.rename(columns={"id": "hadm_id", "journal_note": "text", "code": "icd_code"}, inplace=True)
    
    # Strip whitespace from codes
    df["icd_code"] = df["icd_code"].str.strip()
    
    print(f"  Notes after filtering empty text: {len(df):,}")

    # Select top-K codes
    code_counts = Counter(df["icd_code"])
    top_codes = {code for code, _ in code_counts.most_common(args.top_k_codes)}
    print(f"  Top-{args.top_k_codes} codes cover {len(top_codes)} unique codes")

    df = df[df["icd_code"].isin(top_codes)].copy()
    
    # Format labels as a list (to support multi-label classification architecture)
    # Even though this synthetic dataset has 1 label per note, our pipeline supports multiple.
    df["labels"] = df["icd_code"].apply(lambda x: [x])

    print(f"  Records with at least 1 top-{args.top_k_codes} code: {len(df):,}")

    # Subsample if needed
    if args.max_notes and len(df) > args.max_notes:
        df = df.sample(n=args.max_notes, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {args.max_notes:,} notes")

    # Fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=sorted(top_codes))
    mlb.fit(df["labels"])

    # Stratified split based on the primary ICD-10 code
    df["strat_key"] = df["icd_code"]
    
    # We might have classes with only 1 sample after filtering, which breaks stratify.
    # We only stratify on classes with >= 2 samples.
    class_counts = df["strat_key"].value_counts()
    valid_strat_classes = class_counts[class_counts >= 2].index
    
    # Fallback to random split if stratification isn't viable
    try:
        train_val, test = train_test_split(
            df, test_size=0.083, random_state=42, stratify=df["strat_key"]
        )
        train, val = train_test_split(
            train_val, test_size=0.091, random_state=42, stratify=train_val["strat_key"]
        )
    except ValueError:
        print("  ⚠ Stratification failed (likely due to rare classes). Falling back to random split.")
        train_val, test = train_test_split(df, test_size=0.083, random_state=42)
        train, val = train_test_split(train_val, test_size=0.091, random_state=42)
        
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    # Save splits
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                # Hash the string hadm_id to an integer to match PyTorch dataset expectations
                hadm_id_int = abs(hash(row["hadm_id"])) % (10 ** 8)
                record = {
                    "subject_id": hadm_id_int,
                    "hadm_id": hadm_id_int,
                    "text": row["text"],
                    "labels": row["labels"],
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Saved {out_path}")

    # Save label binarizer
    lb_path = os.path.join(args.output_dir, "label_binarizer.pkl")
    with open(lb_path, "wb") as f:
        pickle.dump(mlb, f)
    print(f"  Saved {lb_path}")

    # Save ICD descriptions mapping
    desc_path = os.path.join(args.output_dir, "icd10_descriptions.tsv")
    # Extract unique code to name mapping from the dataset
    code_to_name = dict(zip(df["icd_code"], df["name"]))
    
    desc_df = pd.DataFrame([
        {"icd_code": code, "description": code_to_name.get(code, "")}
        for code in sorted(top_codes)
    ])
    desc_df.to_csv(desc_path, sep="\t", index=False)
    print(f"  Saved descriptions to {desc_path}")

    print("✅ Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Dataset preprocessing")
    parser.add_argument("--output_dir",  default="data/processed/")
    parser.add_argument("--top_k_codes", type=int, default=50)
    parser.add_argument("--max_notes",   type=int, default=60000)
    parser.add_argument("--language",     default="all",
                        help="Language code to filter (e.g. 'en'), or 'all' to keep every language.")
    args = parser.parse_args()
    main(args)
