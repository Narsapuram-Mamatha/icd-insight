import json
import os

path = "c:/Users/Mamatha/Desktop/Projects/icd-insight/notebooks/02_Inference_New_Summary.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        # Update code references
        src = cell["source"]
        for i in range(len(src)):
            src[i] = src[i].replace("Dataset: MIMIC-IV-Note", "Dataset: birgermoell/icd10-clinical-notes (Public HF)")
        cell["source"] = src
    elif cell["cell_type"] == "markdown":
        # Update references to dataset
        src = cell["source"]
        for i in range(len(src)):
            src[i] = src[i].replace("Dataset: MIMIC-IV-Note", "Dataset: birgermoell/icd10-clinical-notes (Public HF)")
        cell["source"] = src

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Updated Notebook B successfully.")
