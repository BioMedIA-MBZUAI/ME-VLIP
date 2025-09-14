import csv
import json
import ast
from pathlib import Path

input_csv = "/path/to/training.csv"
output_json = "llama_factory_internvl/data/converted.json"
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
