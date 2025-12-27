# FP-NLP: Final Project Natural Language Processing

This repository contains the end-to-end development, training, and evaluation of various Large Language Models (LLMs) for a specific NLP task (likely a medical or health-related Q&A, based on the dataset names).

## Project Overview

**Comparative Performance Analysis of Medical LLMs through QLoRA Adaptation**

This project aims to develop a reliable and safe Indonesian-language medical chatbot specifically focused on digestive system health. We analyzed and fine-tuned 7 open-source Large Language Models (LLMs) using **QLoRA (Quantized Low-Rank Adaptation)** on a dataset of **21,376 doctor-patient Q&A pairs** sourced from Alodokter.

**Contributors:** - @[Fadhil](https://github.com/Yaoimng) - @[Farhan](https://github.com/farhanwew) - @[Efan](https://github.com/Aeroo11)

**Key Highlights:** - **Goal:** Create an medical chatbot while minimizing hallucinations. - **Models:** Mistral, Llama 3, Sahabat AI (Gemma 2 & Llama 3), and Komodo 7B. - **Evaluation:** A dual approach using traditional metrics (BERTScore, ROUGE, BLEU) and "LLM-as-a-Judge" (GPT-4o mini) to assess relevance, safety, and hallucinations. - **Conclusion:** The fine-tuned **Komodo-7b-base** emerged as the most balanced model, delivering relevant and informative responses with minimal hallucinations compared to others.

## Repository Structure

``` text
├── Notebook Final Projek NLP (End to End).ipynb  # Main pipeline
├── Result.xlsx                                   # Compiled results and metrics
├── Notebook Evaluation/                          # Metric-specific evaluation notebooks
├── Notebook Preprocessing & Analysis/            # Data cleaning and EDA
├── Notebook Training/                            # Model-specific training scripts
└── Retrieval/                                    # Retrieval logic and notebooks
```

## Model Configurations {#model-configurations}

| No | Model Name | Optimizer | Device | Training Time | Weights / Logs |
|------------|------------|------------|------------|------------|------------|
| 1 | unsloth/mistral-7b-v0.3 | adamw_8bit | \- | 1:21:42 | [Kaggle Adapter](https://www.kaggle.com/datasets/farhanwew/lora-adapter-1) |
| 2 | unsloth/mistral-7b-instruct-v0.2 | adamw_8bit | \- | 37:50 | [Google Drive](https://drive.google.com/file/d/1sqVp1FinYwrqss-S1i6chSK2r9UN9Jx0/view?usp=drive_link) |
| 3 | unsloth/llama-3-8b-bnb-4bit | adamw_8bit | H-100 | 31:00 | [Hugging Face](https://huggingface.co/farwew/DoctorsAnswerTextDataset-in-IndonesianUnsloth-llama-3-8b-bnb-4bit) |
| 4 | Sahabat AI Llama3-8b-instruct | adamw_8bit | H-100 | 31:58 | [Hugging Face](https://huggingface.co/farwew/lora_model) |
| 5 | Sahabat AI Gemma2-9b-instruct | adamw_8bit | A-100 | \- | [Hugging Face](https://huggingface.co/farwew/Med-QA) |
| 6 | Yellow-AI-NLP/komodo-7b-base (1 epoch) | adamw_8bit | H-100 | 26:02 | [Hugging Face](https://huggingface.co/farwew/Med-QA-komodo) |
| 7 | Yellow-AI-NLP/komodo-7b-base (2 epochs) | adamw_8bit | H-100 | 51:48 | [Hugging Face](https://huggingface.co/farwew/Med-QA-komodo-2) |

## Model Weights & Datasets

-   **Main Dataset**: [farwew/DoctorsAnswerTextDataset-in-Indonesian](https://huggingface.co/datasets/farwew/DoctorsAnswerTextDataset-in-Indonesian)
-   **Weights & Logs**: The specific weights and training logs for each experiment are available directly in the [Model Configurations](#model-configurations) table above.

## Evaluation Results

### 1. Traditional Metrics (Base vs Finetuning)

| Model Name | Method | BERT Precision | BERT Recall | BERT F1 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-LSum |
|-------|------|------|------|------|------|------|------|------|------|------|------|------|
| mistral-7b-v0.3 | Base | 0.621 | 0.589 | 0.603 | 8.294 | 2.776 | 1.418 | 0.949 | 11.423 | 1.422 | 7.027 | 6.924 |
| mistral-7b-v0.3 | Finetuning | 68.896 | 68.592 | 68.691 | 26.407 | 12.636 | 6.892 | 4.184 | 30.294 | 6.696 | 15.461 | 15.461 |
| mistral-7b-instruct-v0.2-bnb-4bit | Base | 0.651 | 0.631 | 0.641 | 12.96 | 4.302 | 2.145 | 1.413 | 13.847 | 1.864 | 7.207 | 7.849 |
| mistral-7b-instruct-v0.2-bnb-4bit | Finetuning | 69.342 | 69.161 | 69.213 | 27.092 | 13.047 | 7.174 | 4.485 | 31.162 | 6.961 | 16.031 | 16.031 |
| llama-3-8b-bnb-4bit | Base | 0.628 | 0.600 | 0.613 | 8.522 | 2.88 | 1.473 | 0.98 | 12.311 | 1.299 | 7.692 | 7.344 |
| llama-3-8b-bnb-4bit | Finetuning | 69.596 | 69.194 | 69.348 | 26.152 | 12.577 | 6.848 | 4.149 | 30.238 | 6.806 | 15.668 | 15.668 |
| llama3-8b-cpt-sahabatai-v1-instruct | Base | 0.681 | 0.652 | 0.666 | 18.572 | 7.586 | 3.901 | 2.448 | 25.707 | 4.68 | 13.407 | 16.23 |
| llama3-8b-cpt-sahabatai-v1-instruct | Finetuning | 69.754 | 67.469 | 68.567 | 23.161 | 11.119 | 6.203 | 3.837 | 27.61 | 6.217 | 14.467 | 14.684 |
| gemma2-9b-cpt-sahabatai-v1-instruct | Base | 0.681 | 0.669 | 0.675 | 22.639 | 9.46 | 4.627 | 2.704 | 29.472 | 5.41 | 14.844 | 18.232 |
| gemma2-9b-cpt-sahabatai-v1-instruct | Finetuning | 69.13 | 69.106 | 69.082 | 27.269 | 13.117 | 7.26 | 4.473 | 32.109 | 7.441 | 15.99 | 15.99 |
| komodo-7b-base (Set 6) | Base | 0.634 | 0.637 | 0.635 | 15.79 | 6.193 | 3.071 | 1.874 | 20.217 | 3.355 | 10.552 | 13.15 |
| komodo-7b-base (Set 6) | Finetuning | 68.538 | 68.222 | 68.323 | 24.445 | 11.257 | 6.068 | 3.769 | 29.041 | 6.365 | 15.211 | 15.211 |
| komodo-7b-base (Set 7) | Base | 0.634 | 0.637 | 0.635 | 15.79 | 6.193 | 3.071 | 1.874 | 20.217 | 3.355 | 10.552 | 13.15 |
| komodo-7b-base (Set 7) | Finetuning | 69.752 | 69.737 | 69.694 | 28.367 | 13.878 | 7.906 | 5.103 | 32.215 | 7.605 | 16.798 | 16.798 |

### 2. LLM as a Judge Metrics

Evaluation performed using metrics such as Hallucination, Answer Relevance, Moderation, and Usefulness.

| No | Model Name | Method | Hallucination (avg) | Answer Relevance (avg) | Moderation (avg) | Usefulness (avg) |
|----------|-------------|----------|----------|----------|----------|----------|
| 1 | mistral-7b-v0.3 | Base | 7.525 | 4.665 | 0.230 | 2.561 |
| 1 | mistral-7b-v0.3 | Finetuning | 5.020 | 8.291 | 0.020 | 6.871 |
| 2 | mistral-7b-instruct-v0.2-bnb-4bit | Base | 7.069 | 7.718 | 0.010 | 6.257 |
| 2 | mistral-7b-instruct-v0.2-bnb-4bit | Finetuning | 4.870 | 8.265 | 0.080 | 7.051 |
| 3 | llama-3-8b-bnb-4bit | Base | 7.510 | 5.640 | 0.180 | 3.400 |
| 3 | llama-3-8b-bnb-4bit | Finetuning | 4.820 | 8.320 | 0.050 | 6.990 |
| 4 | llama3-8b-cpt-sahabatai-v1-instruct | Base | 4.171 | 8.543 | 0.019 | 7.327 |
| 4 | llama3-8b-cpt-sahabatai-v1-instruct | Finetuning | 4.950 | 8.147 | 0.000 | 6.408 |
| 5 | gemma2-9b-cpt-sahabatai-v1-instruct | Base | 3.284 | 8.860 | 0.000 | 7.931 |
| 5 | gemma2-9b-cpt-sahabatai-v1-instruct | Finetuning | 5.120 | 8.505 | 0.120 | 7.260 |
| 6 | Yellow-AI-NLP/komodo-7b-base | Base | 6.365 | 7.218 | 0.136 | 4.904 |
| 6 | Yellow-AI-NLP/komodo-7b-base | Finetuning | 5.770 | 8.215 | 0.070 | 6.800 |
| 7 | Yellow-AI-NLP/komodo-7b-base | Base | 6.433 | 7.250 | 0.117 | 4.788 |
| 7 | Yellow-AI-NLP/komodo-7b-base | Finetuning | 4.667 | 8.318 | 0.020 | 7.080 |