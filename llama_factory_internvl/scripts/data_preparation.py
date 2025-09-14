import pandas as pd
import json
import os
from ast import literal_eval
from pathlib import Path

# Config
DATASET_PATH = "/path/to/training.csv"
OUTPUT_PATH = "llama_factory_internvl/data/internvl_train.jsonl"

# Load CSV
df = pd.read_csv(DATASET_PATH)

# Parse image list column
df["ImageName"] = df["ImageName"].apply(literal_eval)  # Convert string list to list

# Explode to one row per image
df_exploded = df.explode("ImageName")

# Full image path
df_exploded["FullImagePath"] = df_exploded.apply(
    lambda row: os.path.join(row["DirectoryPath"], row["ImageName"]), axis=1
)

# Filter out missing image files
df_exploded = df_exploded[df_exploded["FullImagePath"].apply(os.path.exists)].reset_index(drop=True)

# Convert to JSONL
records = []
for _, row in df_exploded.iterrows():
    prompt = f"<image> {row['Question']}"
    response = row["Answer"]

    record = {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ],
        "images": [row["FullImagePath"]]
    }
    records.append(record)

# Write JSONL
with open(OUTPUT_PATH, 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

print(f"✅ Done! Wrote {len(records)} single-image examples to: {OUTPUT_PATH}")

