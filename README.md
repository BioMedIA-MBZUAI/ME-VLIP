
This repository is the official implementation of [ME-VLIP: A Modular and Efficient Vision-Language Framework for Generalizable Medical Image Parsing](https://openreview.net/forum?id=0LYAs4b8T6). 

## Environments and Requirements

| Component            | Setting                                                       |
| :------------------- | :------------------------------------------------------------ |
| System               | Ubuntu 22.04.4 LTS                                            |
| Programming language | Python 3.10                                                   |
| Dependencies         | torch 2.3.1, torchvision 0.18.1, transformers 4.52.dev0       |
| GPU                  | 1x NVIDIA A100-SXM4                                           |
| VRAM                 | 40GB                                                          |
| CPU                  | 64 cores                                                      |

To install requirements:

```setup
cd docker_internvl
conda create -n flare25-internvl python=3.10
conda activate flare25-internvl
pip install -r requirements.txt
```


## Dataset

- **Source:** [FLARE-MedFM/FLARE-Task5-MLLM-2D](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task5-MLLM-2D)
- **Description:** 19 medical datasets, 8 imaging modalities, 50,996 images, 58,112 Q&A pairs
- **Structure:**
```
original_dataset/
├── training/
├── validation-public/
└── validation-hidden/
```
- **Download:**
```bash
huggingface-cli login
huggingface-cli download FLARE-MedFM/FLARE-Task5-MLLM-2D --repo-type dataset --local-dir ./original_dataset
find original_dataset -name "*.zip" -exec unzip -o "{}" -d "$(dirname "{}")" \;
```

---

## Preprocessing

- **Purpose:**
  - Convert to a single JSON file for LLaMA-Factory.

- **Command:**
```bash
cd llama_factory_internvl/scripts
python data_preparation.py
```

---
## Training

We follow the instructions of this repo: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
1. To fine-tune the model(s) in the paper, run this command:

```bash
cd llama_factory_internvl
llamafactory-cli train examples/train_qlora/internvl_lora_sft_bnb.yaml
```

**QLoRA fine-tuning**
| Component | Setting |
| :--- | :--- |
| Base model | InternVL3 [1] |
| Number of parameters | 8B |
| Framework | LLaMA-Factory [2] |
| Method | QLoRA (4-bit quantization) |
| LoRA rank | 8 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj |
| Fine-tuning approach | SFT |
| Epochs | 3 |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 steps |
| Effective batch size | 8 |
| Sequence length | 2048 tokens |
| Optimizer | AdamW |
| Precision | BF16 |
| Learning rate schedule | Linear |
| Initial learning rate | 2e-4 |
| Warm-up ratio | 3% |
| Training time | ~60 hours |
| Inference VRAM | ~9GB |
| Number of trainable parameters | ~20M |

**TC configurations**
| Component | Setting |
| :--- | :--- |
| Model | GLiClass [3] |
| Encoder | DeBERTa-v3-small |
| Label model | BGE-small |
| Epochs | 3 |
| Optimizer | AdamW |
| Loss | Focal Loss (α=1, γ=1) |
| Learning rate | 1e-5 |
| Precision | FP16 (mixed) |

You can download trained models here:

- [InternVL3-8B](https://huggingface.co/AmalSaqib/flare-internvl) trained using the procedure explained above.
- [GLiClass](https://huggingface.co/MaiAShaaban/flare-gliclass-small-v1.0) trained using the [GLiClass](https://github.com/Knowledgator/GLiClass) repo.

## Inference

1. To infer the testing cases, run this command:

```
cd docker_internvl
python inference.py --base_dataset_path <path to test dataset> --output_dir <path to output folder> --output_filename predictions.json --max_new_tokens 512 --device cuda:0
```

2. Docker containers on [DockerHub](https://hub.docker.com/r/maiahmed95/biomedia/)

```bash
docker container run --gpus "device=0" -m 28G --name biomedia --rm -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/biomedia_outputs/:/workspace/outputs/ biomedia:latest /bin/bash -c "sh predict.sh"
```

## Evaluation

To compute the evaluation metrics, run:

```eval
cd evaluation
python evaluation.py --base_dataset_path <path>/original_dataset --prediction_file predictions.json --output_dir evaluation_results --output_filename metrics_predictions.json
```


## Results

Model performance comparison on validation sets (Public | Hidden scores).

| Task & Metric                | InternVL3-8B | InternVL3-8B (w/ TC) |
| :-------------------------- | :------------------- | :--------------------------- |
| **Classification**          |                      |                              |
| Balanced Accuracy ↑         | 0.52 \| 0.71         | 0.53 \| 0.74                 |
| **Multi-label Classification** |                  |                              |
| F1 Score ↑                  | 0.46 \| 0.56         | 0.46 \| 0.57                 |
| **Detection**               |                      |                              |
| F1 Score ↑                  | 0.37 \| 0.82         | 0.26 \| 0.82                 |
| **Instance Detection**      |                      |                              |
| F1 Score ↑                  | - \| 0.00            | - \| 0.00                    |
| **Cell Counting**           |                      |                              |
| MAE ↓                       | 301.4 \| -           | 251.6 \| -                   |
| **Regression**              |                      |                              |
| MAE ↓                       | - \| 18.67           | - \| 11.84                   |
| **Report Generation**       |                      |                              |
| GREEN Score ↑               | 0.75 \| -            | 0.71 \| -                    |

## Citation
```bibtex
@inproceedings{
shaaban2025mevlip,
title={{ME}-{VLIP}: A Modular and Efficient Vision-Language Framework for Generalizable Medical Image Parsing},
author={Mai A. Shaaban and Amal Saqib and Shahad Emad Hardan and Darya Taratynova and Tausifa Jan Saleem and Mohammad Yaqub},
booktitle={Submitted to MICCAI 2025 FLARE Challenge},
year={2025},
url={https://openreview.net/forum?id=0LYAs4b8T6},
note={under review}
}
```

## Contributing

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/BioMedIA-MBZUAI/FLARE2025-Task5-2D-biomedia/blob/main/LICENSE) file for details.

## Acknowledgement

> We thank the contributors of public datasets. 
