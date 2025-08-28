# import pandas as pd
# import json
# import os
# from ast import literal_eval
# from pathlib import Path

# # Config
# ROOT_DIR = "/l/users/amal.saqib/flare"
# DATASET_PATH = f"{ROOT_DIR}/datasets/training/training.csv"
# OUTPUT_PATH = "/home/amal.saqib/LLaMA-Factory/data/internvl_train.jsonl"

# # Load CSV
# df = pd.read_csv(DATASET_PATH)

# # Parse image list column
# df["ImageName"] = df["ImageName"].apply(literal_eval)  # Convert string list to list

# # Explode to one row per image
# df_exploded = df.explode("ImageName")

# # Full image path
# df_exploded["FullImagePath"] = df_exploded.apply(
#     lambda row: os.path.join(row["DirectoryPath"], row["ImageName"]), axis=1
# )

# # Filter out missing image files
# df_exploded = df_exploded[df_exploded["FullImagePath"].apply(os.path.exists)].reset_index(drop=True)

# # Convert to JSONL
# records = []
# for _, row in df_exploded.iterrows():
#     prompt = f"<image> {row['Question']}"
#     response = row["Answer"]

#     record = {
#         "conversations": [
#             {"from": "human", "value": prompt},
#             {"from": "gpt", "value": response}
#         ],
#         "images": [row["FullImagePath"]]
#     }
#     records.append(record)

# # Write JSONL
# with open(OUTPUT_PATH, 'w') as f:
#     for r in records:
#         f.write(json.dumps(r) + '\n')

# print(f"✅ Done! Wrote {len(records)} single-image examples to: {OUTPUT_PATH}")


# import csv
# import json
# import ast

# input_csv = "/l/users/amal.saqib/flare/datasets/training/training.csv"
# output_json = "converted.json"
# output = []

# with open(input_csv, newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         _, _, images_str, question, answer, *_ = row

#         # Convert string list to actual list
#         image_list = ast.literal_eval(images_str)

#         # Add one <image> token per image
#         image_tokens = "<image>" * len(image_list)
#         question_with_tokens = image_tokens + " " + question.strip()

#         entry = {
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": question_with_tokens
#                 },
#                 {
#                     "from": "gpt",
#                     "value": answer.strip()
#                 }
#             ],
#             "images": image_list
#         }

#         output.append(entry)

# # Save to JSON
# with open(output_json, "w") as f:
#     json.dump(output, f, indent=2)
import csv
import json
import ast
from pathlib import Path

input_csv = "/l/users/amal.saqib/flare/datasets/training/training.csv"
output_json = "converted.json"
output = []

with open(input_csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # Parse fields
        # TaskType,Modality,ImageName,Question,Answer,Split,FileName,DirectoryPath
        type, modality, images_str, question, answer, _,_, base_path = row

        # Step 1: Parse the list of images safely
        try:
            image_list = ast.literal_eval(images_str)
        except Exception as e:
            print("❌ Could not parse image list:", images_str)
            continue

        # Step 2: Prepend full path
        full_image_paths = [str(Path(base_path) / image) for image in image_list]

        # Step 3: Add matching <image> tokens
        image_tokens = " ".join(["<image>"] * len(full_image_paths))
        full_question = f"{image_tokens} {question.strip()}"

        # Step 4: Add to final JSON structure
        output.append({
            "conversations": [
                {"from": "human", "value": full_question},
                {"from": "gpt", "value": answer.strip()}
            ],
            "images": full_image_paths
        })

# Write JSON
with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ JSON saved to {output_json} with {len(output)} entries.")
