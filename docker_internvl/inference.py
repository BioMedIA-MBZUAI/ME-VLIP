import os
import json
import argparse
import glob
import re
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import PeftModel

from gliclass import GLiClassModel, ZeroShotClassificationPipeline

# ================================================================================
# ARGUMENT PARSING
# ================================================================================


def parse_args():
    """Parse command line arguments for prediction script."""
    parser = argparse.ArgumentParser(
        description="Prediction script for medical image analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_id",
        type=str,
        default="OpenGVLab/InternVL3-8B-hf",
        help="Model identifier for base model",
    )

    # Data Configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--base_dataset_path",
        type=str,
        default="/app/input",
        help="Base path to dataset directory (e.g., /app/input for test-fake)",
    )
    data_group.add_argument(
        "--output_dir",
        type=str,
        default="/app/output",
        help="Output directory for prediction results",
    )
    data_group.add_argument(
        "--output_filename",
        type=str,
        default="predictions.json",
        help="Filename for the output predictions file",
    )

    # Inference Configuration
    inference_group = parser.add_argument_group("Inference Configuration")
    inference_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    inference_group.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run inference on"
    )

    return parser.parse_args()


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================


def find_json_files(base_path):
    """Find JSON files based on validation type in the test-fake directory structure."""
    json_files = []

    # TODO: Adjust suffix for test data
    # suffix = "_test.json"
    suffix = ".json"

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(suffix):
                json_files.append(os.path.join(root, file))

    return json_files


def validate_paths(dataset_path):
    """Validate that required paths exist."""
    # Check dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Find JSON files
    json_files = find_json_files(dataset_path)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dataset_path}")

    return json_files


# ================================================================================
# ANSWER PARSING FUNCTIONS
# ================================================================================


def parse_answer(output, task_type=None):
    """Parse model output based on task type to extract the final answer."""
    output = output.strip()

    # Remove common prefixes
    if "Please provide a clear and concise answer." in output:
        try:
            output = output.split("Please provide a clear and concise answer.")[
                -1
            ].strip()
        except:
            pass

    # Remove leading newlines
    if "\n" in output:
        output = output.split("\n", 1)[-1].strip()

    # Task-specific parsing
    task_type = (task_type or "").strip().lower()

    if task_type == "classification":
        return _parse_classification(output)
    elif task_type == "multi-label classification":
        return _parse_multi_label_classification(output)
    elif task_type in ["detection", "instance_detection"]:
        return _parse_detection(output)
    elif task_type in ["cell counting", "regression", "counting"]:
        return _parse_numeric(output)
    elif task_type == "report generation":
        return output
    else:
        return output


def _parse_classification(output):
    """Parse classification task output."""
    lines = output.splitlines()
    if len(lines) >= 1:
        last_line = lines[-1].strip()
        return last_line
    return output


def _parse_multi_label_classification(output):
    """Parse multi-label classification task output."""
    lines = output.splitlines()
    labels = []
    for line in lines:
        for part in re.split(r"[;]", line):
            label = part.strip()
            if label:
                labels.append(label)
    return "; ".join(labels)


def _parse_detection(output):
    """Parse detection task output (JSON format expected)."""
    match = re.search(r"\{.*\}|\[.*\]", output, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return json.dumps(parsed)
        except:
            return match.group()
    return output


def _parse_numeric(output):
    """Parse numeric task output (counting, regression)."""
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", output)
    if match:
        return match.group()
    return "0"


# ================================================================================
# MODEL LOADING FUNCTIONS
# ================================================================================


def verify_offline_mode():
    """Verify that we're running in offline mode with local models only."""
    # Set offline environment variables
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("🔒 Offline mode enabled - no internet access required")
    print("📦 All models will be loaded from local files only")


def load_model_and_processor(model_id, checkpoint_path=None, device="cuda:0"):
    """Load the base model and FLARE 2025 fine-tuned adapters from local directories."""

    # Enable offline mode first
    verify_offline_mode()

    # Define local model paths
    models_dir = "/app/models"
    base_model_path = os.path.join(models_dir, "InternVL3-8B-hf")
    adapter_model_path = os.path.join(models_dir, "FLARE-InternVL3-8B-hf")
    question_classifier_model_path = os.path.join(
        models_dir, "FLARE-gliclass-small-v1.0"
    )

    print(f"🔄 Loading models from local directories (offline mode)...")
    print(f"  Base model path: {base_model_path}")
    print(f"  Adapter path: {adapter_model_path}")
    print(f"  Question classifier path: {question_classifier_model_path}")

    # Check if local models exist
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. Models should be pre-downloaded during Docker build."
        )

    if not os.path.exists(adapter_model_path):
        raise FileNotFoundError(
            f"Adapter model not found at {adapter_model_path}. Models should be pre-downloaded during Docker build."
        )
    if not os.path.exists(question_classifier_model_path):
        raise FileNotFoundError(
            f"Question classifier model not found at {question_classifier_model_path}. Models should be pre-downloaded during Docker build."
        )

    # Load processor from local directory (offline mode)
    print(f"Loading processor from: {base_model_path}")
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            local_files_only=True,
        )
        print(f"Processor loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load processor from local files: {e}")

    # Configure 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load base model from local directory with quantization (offline mode)
    print(f"Loading 4-bit quantized base model from: {base_model_path}")
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        print(f"Base model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load base model from local files: {e}")

    # Load FLARE 2025 fine-tuned adapter from local directory (offline mode)
    print(f"Loading FLARE adapter from: {adapter_model_path}")
    adapter_paths = [
        f"{adapter_model_path}/internvl-all/lora/sft",
        f"{adapter_model_path}/classification",
        f"{adapter_model_path}/detection",
        f"{adapter_model_path}/regression",
        f"{adapter_model_path}/counting",
        f"{adapter_model_path}/instance_detection",
        f"{adapter_model_path}/multi_label_classification",
        f"{adapter_model_path}/report_generation",
    ]
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_paths[0],
            local_files_only=True,
            adapter_name=f"adapter_general",
        )
        print(f"FLARE adapter loaded successfully")

        # Load remaining adapters
        for i, adapter_path in enumerate(adapter_paths[1:], start=1):
            adapter_name = adapter_path.split("/")[-1]
            adapter_name = f"adapter_{adapter_name}"
            model.load_adapter(adapter_path, adapter_name=adapter_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load adapter from local files: {e}")

    # Load question classifier model and tokenizer from local directory (offline mode)
    try:
        question_classifier_model = GLiClassModel.from_pretrained(
            question_classifier_model_path,
            local_files_only=True,
        )
        question_classifier_tokenizer = AutoTokenizer.from_pretrained(
            question_classifier_model_path,
            local_files_only=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load question classifier model from local files: {e}"
        )

    print(f"Successfully loaded all models from local directories (offline mode)!")

    return model, processor, question_classifier_model, question_classifier_tokenizer


# ================================================================================
# PREDICTION FUNCTIONS
# ================================================================================


def classify_question_type(
    question_classifier_model, question_classifier_tokenizer, question, device
):
    """Classify the type of question using GLiClass"""

    pipeline = ZeroShotClassificationPipeline(
        question_classifier_model,
        question_classifier_tokenizer,
        classification_type="single-label",
        device=device,
    )

    labels = [
        "classification",
        "detection",
        "regression",
        "counting",
        "instance_detection",
        "multi-label classification",
        "report_generation",
    ]

    results = pipeline(question, labels, threshold=0.1)[0]
    return results[0]["label"], results[0]["score"]


def predict_on_file(
    input_file,
    model,
    processor,
    question_classifier_model,
    question_classifier_tokenizer,
    max_new_tokens=1024,
    device="cuda:0",
):
    """Perform predictions on a single JSON file containing questions and images."""
    IMAGE_TOKEN = "<image>"

    # Load data
    with open(input_file) as f:
        val_data = json.load(f)

    print(f"Processing {len(val_data)} samples from {os.path.basename(input_file)}")

    # Process each sample
    for sample in tqdm(val_data, desc=f"Predicting {os.path.basename(input_file)}"):
        try:
            # Handle image loading
            img_field = sample["ImageName"]
            if isinstance(img_field, list):
                img_paths = img_field[:1]  # Limit to 1 images max
            else:
                img_paths = [img_field]

            # Load and validate images
            imgs = []
            for img_path in img_paths:
                full_path = os.path.join(os.path.dirname(input_file), img_path)
                try:
                    img = Image.open(full_path).convert("RGB")
                    imgs.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue

            if not imgs:
                print(f"Warning: No valid images for sample, skipping")
                sample["Answer"] = "Error: No valid images"
                continue

            # Prepare input
            system_prompt = ""
            image_tokens = "\n".join([IMAGE_TOKEN] * len(imgs))
            user_content = f"{image_tokens}\n{sample['Question']}"
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Classify question type
            question_type, question_score = classify_question_type(
                question_classifier_model,
                question_classifier_tokenizer,
                sample["Question"],
                model.device,
            )
            if question_score >= 0.3:
                if question_type == "classification":
                    model.set_adapter("adapter_classification")
                elif question_type == "detection":
                    model.set_adapter("adapter_detection")
                elif question_type == "regression":
                    model.set_adapter("adapter_regression")
                elif question_type == "counting":
                    model.set_adapter("adapter_counting")
                elif question_type == "instance_detection":
                    model.set_adapter("adapter_instance_detection")
                elif question_type == "multi-label classification":
                    model.set_adapter("adapter_multi_label_classification")
                elif question_type == "report_generation":
                    model.set_adapter("adapter_report_generation")
                else:
                    model.set_adapter("adapter_general")
            else:
                model.set_adapter("adapter_general")

            # Preprocess input
            inputs = processor(
                text=[prompt],
                images=imgs,
                padding=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            # Generate prediction
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

            # Decode output
            output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Parse answer based on task type
            parsed_answer = parse_answer(output, sample.get("TaskType", ""))
            sample["Answer"] = parsed_answer

        except Exception as e:
            print(f"Error processing sample: {e}")
            sample["Answer"] = f"Error: {str(e)}"

    return val_data


# ================================================================================
# MAIN PROCESSING FUNCTION
# ================================================================================


def run_predictions(args):
    """
    Main function to run predictions on all JSON files in the dataset directory.

    Args:
        args: Parsed command line arguments

    Returns:
        int: Number of predictions made
    """
    # Use base dataset path directly (test-fake directory)
    dataset_path = args.base_dataset_path

    # Validate paths and find JSON files
    print("Validating paths and discovering files...")
    input_files = validate_paths(dataset_path)

    print(f"Found {len(input_files)} JSON files in {dataset_path}:")
    for file in input_files:
        print(f"  - {os.path.relpath(file, dataset_path)}")

    # Load model and processor
    print("\nLoading model and processor...")
    model, processor, question_classifier_model, question_classifier_tokenizer = (
        load_model_and_processor(
            args.model_id, None, args.device  # checkpoint_path not used anymore
        )
    )

    # Run predictions on all files
    print(f"\nRunning predictions...")
    all_predictions = []

    for input_file in input_files:
        predictions = predict_on_file(
            input_file,
            model,
            processor,
            question_classifier_model,
            question_classifier_tokenizer,
            args.max_new_tokens,
            args.device,
        )
        all_predictions.extend(predictions)

    # Save results
    print(f"\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_filename)

    with open(output_file, "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Results saved to: {output_file}")
    print(f"Total predictions: {len(all_predictions)}")

    return len(all_predictions)


# ================================================================================
# SCRIPT ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    args = parse_args()

    print("Medical Image Prediction")
    print("=" * 50)
    print(f"Base dataset path: {args.base_dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    try:
        prediction_count = run_predictions(args)
        print(f"\nSuccessfully completed {prediction_count} predictions")
    except Exception as e:
        print(f"\nError: {e}")
        raise
