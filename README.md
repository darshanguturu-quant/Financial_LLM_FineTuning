# Financial_LLM_FineTuning

### Financial Domain Fine-Tuning with LoRA

FinChat is a lightweight financial-domain language model created by fine-tuning `HuggingFaceTB/SmolLM2-360M-Instruct` on a filtered subset of the `sujet-ai/Sujet-Finance-Instruct-177k` dataset.

The project focuses on parameter-efficient fine-tuning, controlled data preprocessing, leakage-safe evaluation, and deployment-ready model export.

---

## Project Overview

General-purpose language models can answer many financial questions, but their responses can be improved for domain-specific financial question answering and conversational tasks.

This project explores whether a relatively small instruction-tuned language model can be adapted to financial-domain data using **LoRA (Low-Rank Adaptation)** rather than full model fine-tuning.

### Objectives

- Adapt a compact language model to financial-domain QA.
- Use LoRA to reduce the number of trainable parameters.
- Build a reproducible data preprocessing pipeline.
- Prevent train/test leakage during augmentation.
- Evaluate the fine-tuned model using validation loss and perplexity.
- Compare the base model with the fine-tuned model.
- Export a merged model and tokenizer for downstream use.

---

## Model

**Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct`

The project uses the instruction-tuned 360M-parameter SmolLM2 model as the starting point.

### Why this model?

A compact model makes experimentation practical on consumer/cloud GPUs such as the NVIDIA T4 while keeping the fine-tuning workflow lightweight.

---

## Dataset

**Dataset:** `sujet-ai/Sujet-Finance-Instruct-177k`

The notebook loads the dataset from Hugging Face and filters it to the following task types:

- `qa`
- `qa_conversation`

Additional preprocessing restricts prompt and answer lengths and removes examples containing excessive emojis or highly informal punctuation.

### Data split

The filtered dataset is split into:

- **90% training**
- **10% evaluation**

The split is performed **before training-only augmentation** to avoid duplicated examples leaking into the evaluation set.

---

## Data Pipeline

```text
Sujet-Finance-Instruct-177k
            │
            ▼
      Task Filtering
       QA / QA Chat
            │
            ▼
     Length Filtering
            │
            ▼
      Content Cleaning
            │
            ▼
       Train / Test Split
          90% / 10%
            │
            ▼
 Training-only Augmentation
            │
            ▼
      Chat Template
            │
            ▼
      Tokenization
       Max Length 512
            │
            ▼
          LoRA
            │
            ▼
      Fine-Tuning
            │
            ▼
        Evaluation
            │
            ▼
      Merge LoRA Adapter
            │
            ▼
       Financial Inference
            │
            ▼
     Hugging Face Export
```

---

## Training Configuration

| Component | Configuration |
|---|---|
| Base Model | SmolLM2-360M-Instruct |
| Dataset | Sujet-Finance-Instruct-177k |
| Fine-Tuning Method | LoRA |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Maximum Sequence Length | 512 |
| Epochs | 1 |
| Learning Rate | 1.5e-4 |
| Effective Batch Size | 16 |
| Micro Batch Size | 4 |
| Gradient Accumulation | 4 |
| Weight Decay | 0.005 |
| Scheduler | Cosine |
| Optimizer | AdamW |
| Evaluation Split | 10% |
| NVIDIA T4 Precision | FP16 |

---

## Why LoRA?

Full fine-tuning updates all model parameters and can require substantially more GPU memory.

LoRA instead introduces trainable low-rank matrices into selected model layers while keeping the majority of the pretrained model frozen.

In this project, LoRA is applied to the attention projections:

```text
q_proj
k_proj
v_proj
o_proj
```

This provides a parameter-efficient way to specialize the model for financial-domain data.

---

## Training Results

The training run completed successfully:

```text
Training steps: 887 / 887
Epochs: 1
```

Observed training/validation loss during the run:

| Step | Training Loss | Validation Loss |
|---:|---:|---:|
| 100 | 0.7763 | 0.7380 |
| 200 | 0.7380 | 0.7183 |
| 300 | 0.7206 | 0.7113 |
| 400 | 0.7099 | 0.7067 |
| 500 | 0.7040 | 0.7040 |
| 600 | 0.7023 | 0.7022 |
| 700 | 0.6946 | 0.7012 |
| 800 | 0.7211 | 0.7009 |

The final evaluation cell calculates:

- Evaluation Loss
- Perplexity

> **Note:** The final `trainer.evaluate()` output should be recorded here after the evaluation cell is executed. The values above are the logged training-run checkpoints, not a replacement for the final evaluation result.

---

## Qualitative Evaluation

The notebook tests the fine-tuned model on financial prompts such as:

```text
What is the difference between a stock and a bond?

What is diversification in investing?

Explain market capitalization.

What is the difference between revenue and profit?

What is a dividend?
```

It also includes a direct **base-model vs FinChat-XS comparison** using:

```text
Explain the concept of price-to-earnings ratio and why investors use it.
```

This provides a qualitative view of how financial-domain adaptation changes the model's responses.

---

## Base Model vs Fine-Tuned Model

The notebook evaluates the same financial prompt with:

```text
SmolLM2-360M-Instruct
        vs
FinChat-XS
```

This comparison is useful for determining whether fine-tuning produces a meaningful domain-specific improvement rather than simply measuring the model in isolation.

---

## Hardware

The notebook is designed for Google Colab GPU runtimes.

The training configuration automatically selects mixed precision based on GPU capability:

```text
NVIDIA T4
    ↓
FP16

BF16-capable GPU
    ↓
BF16
```

The training run was performed on an **NVIDIA Tesla T4**.

---

## Environment

Core dependencies:

```text
torch
torchvision
transformers==4.47.1
datasets==3.2.0
peft==0.14.0
```

The notebook uses a matching PyTorch/TorchVision CUDA build for the Colab environment.

---

## Running the Project

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Open the notebook

Open:

```text
Finetuning_Fin.ipynb
```

in Google Colab.

### 3. Select a GPU runtime

Recommended:

```text
NVIDIA T4 or better
```

### 4. Run the notebook

The notebook performs:

1. Environment setup
2. Dataset loading
3. Dataset filtering
4. Train/test splitting
5. Training-only augmentation
6. Tokenization
7. LoRA configuration
8. Fine-tuning
9. Evaluation
10. Model merging
11. Financial inference
12. Base-vs-fine-tuned comparison
13. Hugging Face export

---

## Model Export

After training, the LoRA adapter is merged into the base model.

The merged model and tokenizer are saved locally as:

```text
FinChat-XS/
```

The notebook also includes Hugging Face Hub upload cells for:

- The merged model
- The tokenizer
- The LoRA adapter

---

## Project Structure

```text
.
├── Finetuning_Fin.ipynb
├── README.md
└── FinChat-XS/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

The `FinChat-XS/` directory is generated after running the notebook and should only be committed if you intentionally want to store model artifacts in the repository. For large model files, Hugging Face Hub or Git LFS is preferable.

---

## Key Engineering Decisions

### Leakage-safe augmentation

The dataset is split before short conversational examples are duplicated.

This keeps augmented training examples out of the evaluation set.

### Fixed context length

A maximum sequence length of **512 tokens** is used instead of deriving the context length from the longest example.

This provides more predictable GPU memory usage.

### Parameter-efficient fine-tuning

LoRA is used instead of updating the full model.

### Mixed precision

FP16 is used on the NVIDIA T4 to reduce memory usage and improve training efficiency.

### Merged model

After training, the LoRA adapter is merged with the base model to produce a standalone model that can be loaded without separately applying the adapter.

---

## Limitations

This project is a fine-tuning experiment and should not be treated as a financial advisory system.

Important limitations include:

- The model can generate incorrect financial information.
- Financial knowledge may be incomplete or outdated.
- Validation loss does not directly measure financial factual accuracy.
- The evaluation set is derived from the same source dataset as the training data.
- The qualitative evaluation uses a limited number of prompts.
- No dedicated financial benchmark is included yet.
- One training epoch does not establish that the selected hyperparameters are optimal.

---

## Future Improvements

Potential extensions include:

- Add a dedicated financial benchmark.
- Evaluate factual accuracy and hallucination rate.
- Add ROUGE/BLEU or task-specific evaluation where appropriate.
- Compare multiple LoRA ranks.
- Perform learning-rate and context-length ablations.
- Compare LoRA against QLoRA.
- Add a larger high-quality financial reasoning dataset.
- Evaluate responses with a structured financial QA test set.
- Add inference latency and memory benchmarks.
- Deploy FinChat-XS through an API or lightweight Gradio interface.

---

## Tech Stack

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT
LoRA
Google Colab
CUDA
```

---

## Disclaimer

FinChat-XS is an experimental machine-learning project for research and educational purposes. It is not a substitute for professional financial, investment, tax, or legal advice.

---

## Author

**Darshan Guturu**

Built as a financial-domain LLM fine-tuning project using parameter-efficient adaptation.
