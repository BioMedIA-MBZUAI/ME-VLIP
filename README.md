
This repository is the official implementation of [ME-VLIP: A Modular and Efficient Vision-Language Framework for Generalizable Medical Image Parsing]([TBA](https://openreview.net/forum?id=0LYAs4b8T6)). 

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
  - Prepare for LLaMA-Factory.

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
| Component | Setting |
| :--- | :--- |
| **QLoRA fine-tuning** | |
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
| **TC configurations** | |
| Model | GLiClass [3] |
| Encoder | DeBERTa-v3-small |
| Label model | BGE-small |
| Epochs | 3 |
| Optimizer | AdamW |
| Loss | Focal Loss (α=1, γ=1) |
| Learning rate | 1e-5 |
| Precision | FP16 (mixed) |
You can download trained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on the above dataset with the above code. 

>Give a link to where/how the trained models can be downloaded.

## Inference

1. To infer the testing cases, run this command:

```python
python inference.py --input-data <path_to_data> --model_path <path_to_trained_model> --output_path <path_to_output_data>
```

> Describe how to infer testing cases with the trained models.

2. [Colab](https://colab.research.google.com/) jupyter notebook

3. Docker containers on [DockerHub](https://hub.docker.com/)

```bash
docker container run --gpus "device=0" -m 28G --name algorithm --rm -v $PWD/CellSeg_Test/:/workspace/inputs/ -v $PWD/algorithm_results/:/workspace/outputs/ algorithm:latest /bin/bash -c "sh predict.sh"
```

## Evaluation

To compute the evaluation metrics, run:

```eval
python eval.py --seg_data <path_to_inference_results> --gt_data <path_to_ground_truth>
```

>Describe how to evaluate the inference results and obtain the reported results in the paper.



## Results

Our method achieves the following performance on [Brain Tumor Segmentation (BraTS) Challenge](https://www.med.upenn.edu/cbica/brats2020/)

| Model name       |  DICE  | 95% Hausdorff Distance |
| ---------------- | :----: | :--------------------: |
| My awesome model | 90.68% |         32.71          |

>Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>Pick a license and describe how to contribute to your code repository. 

## Acknowledgement

> We thank the contributors of public datasets. 
