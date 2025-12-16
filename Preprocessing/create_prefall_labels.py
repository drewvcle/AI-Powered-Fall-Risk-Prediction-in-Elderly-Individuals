import os
import pandas as pd
import numpy as np

# === CONFIG ===
BASE_DIR = r"C:\Users\andre\OneDrive\Desktop\School and Work\Year 4\COE70A-Capstone\Datasets\data\MobiAct_Dataset_v2.0\MobiAct_Dataset_v2.0\Annotated Data"
OUT_DIR  = r"C:\Users\andre\OneDrive\Desktop\School and Work\Year 4\COE70A-Capstone\Datasets\data_labeled_with_prefall"

os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COL = "label"
TIME_COL  = "rel_time"

# seconds before fall to relabel as "PRE_FALL"
PRE_FALL_SECONDS = 3.0    # try 3–5 seconds per guideline
FS_TARGET = 20.0          # Hz after resampling
PRE_FALL_SAMPLES = int(PRE_FALL_SECONDS * FS_TARGET)

# all labels considered "fall"
FALL_LABELS = ["FOL", "FKL", "BSC", "SDL"]

def add_prefall_labels(df):
    # normalize label strings
    labels = df[LABEL_COL].astype(str).str.strip().str.upper().values

    # all labels considered "fall"
    FALL_LABELS = ["FOL", "FKL", "BSC", "SDL"]

    is_fall = np.isin(labels, FALL_LABELS)
    labels[is_fall] = "FALL"

    df[LABEL_COL] = labels
    return df


# === PROCESS ALL FILES ===
csv_files = []
for root, _, files in os.walk(BASE_DIR):
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(os.path.join(root, f))

print(f"Found {len(csv_files)} CSVs")

for i, path in enumerate(csv_files, start=1):
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        print(f"Skipping {path}: no '{LABEL_COL}' column.")
        continue

    df_out = add_prefall_labels(df)

    # preserve subfolder structure
    rel_path = os.path.relpath(path, BASE_DIR)
    out_path = os.path.join(OUT_DIR, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)

    if i % 10 == 0 or i == len(csv_files):
        print(f"[{i}/{len(csv_files)}] Processed {path}")

print("\n✅ Done. New labels added:")
print(f"   - All fall types unified into 'FALL'")
print(f"   - {PRE_FALL_SECONDS:.1f}s (~{PRE_FALL_SAMPLES} samples) before each fall marked as 'PRE_FALL'")
print(f"   - Output saved to: {OUT_DIR}")
