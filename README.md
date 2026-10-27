# DravidianCodeMix 2026: Multilingual and Code-Mixed Text Classification

This repository contains the implementation for three NLP classification tasks associated with the [DravidianCodeMix competition on Codabench](https://www.codabench.org/competitions/15232/).

The approach uses multilingual Transformer encoders, task-specific classification heads, class-weighted training, validation-based model selection, and ensemble strategies to improve classification performance on code-mixed and regional-language social media text.

## Tasks

### Task 1: Hierarchical Gender-Aware Tamil Hate Speech Detection

A hierarchical multi-level classification approach for Tamil text.

- Processes user-generated Tamil text with code-mixed expressions.
- Predicts labels across five hierarchical levels.
- Uses a Transformer encoder with separate classification heads for the label hierarchy.
- Incorporates hierarchical loss weighting and label normalization.
- Evaluates individual models and ensemble predictions.

### Task 2: Multi-Task Hope Speech Detection

A multi-task learning approach for hope speech analysis.

- Predicts two targets: `tone` and `hope_type`.
- Shares a multilingual Transformer encoder between both classification tasks.
- Uses task-specific output heads and configurable loss weights.
- Compares different loss-weight configurations.
- Evaluates individual models and ensembles for each target.

### Task 3: Binary Kannada Hate Speech Detection

A binary text classification approach for Kannada social media text.

- Classifies text into `Hate` and `Not-hate`.
- Uses a multilingual Transformer encoder with mean pooling and a classification head.
- Supports class-weighted loss and threshold optimization.
- Compares individual models and ensemble strategies.

## Methodology

The general workflow is:

1. Load the training, validation, and test datasets.
2. Normalize Unicode text and clean URLs, HTML tags, mentions, repeated characters, and punctuation.
3. Normalize task-specific labels and encode them for training.
4. Tokenize text using the selected pretrained Transformer tokenizer.
5. Fine-tune multilingual Transformer models for each task.
6. Evaluate validation predictions using macro-F1 and other classification metrics.
7. Compare individual models and ensemble strategies.
8. Generate predictions and save submission files for the respective tasks.
9. Perform error analysis on labeled data.

## Models

The notebook defines the following pretrained model registry:

| Model | Hugging Face model identifier |
|---|---|
| XLM-RoBERTa Base | `xlm-roberta-base` |
| MuRIL Base | `google/muril-base-cased` |
| Multilingual BERT | `bert-base-multilingual-cased` |
| Distil Multilingual BERT | `distilbert-base-multilingual-cased` |
| XLM-RoBERTa Large | `xlm-roberta-large` |

The default training configuration uses XLM-RoBERTa Base, MuRIL Base, and Multilingual BERT. Other models are available in the registry and can be selected by modifying the configuration.

## Training Configuration

The task-specific configurations in the notebook use the following common settings:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 192 tokens |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Gradient accumulation steps | 4 |
| Maximum epochs | 5 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Dropout | 0.1 |
| Learning-rate scheduler | Linear |
| Loss function | Weighted cross-entropy |
| Early stopping patience | 2 |
| Random seed | 42 |

Mixed-precision training is enabled when CUDA is available. Gradient checkpointing, gradient clipping, checkpoint saving, and memory management are used to support model training.

## Ensemble Strategies

The notebook compares multiple prediction strategies, including:

- Hard voting
- Uniform soft voting
- Rank averaging
- Logit averaging
- Optimized weighted soft voting
- Individual model predictions

For binary Kannada hate speech detection, validation-based threshold optimization is also implemented. The selected strategy depends on the task and validation results.

## Evaluation

The notebook supports the following evaluation methods:

- Macro-F1 score
- Accuracy
- Precision and recall
- Per-class classification reports
- Confusion matrices
- Training and validation loss curves
- Prediction confidence analysis
- Misclassification and error-pattern analysis

Macro-F1 is used for model comparison and validation-based selection. Final performance should be reported using the actual evaluation results obtained from the notebook or official competition leaderboard.


## Requirements

The notebook uses Python and the following main libraries:

- PyTorch
- Transformers
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Optuna (optional)

Install the main dependencies using:

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn optuna
```

The exact PyTorch installation command may depend on your operating system and CUDA version.



## Competition

- **Platform:** [Codabench](https://www.codabench.org/competitions/15232/)
- **Project:** DravidianCodeMix 2026
- **Tasks covered:** Hierarchical Tamil hate speech detection, multi-task hope speech detection, and binary Kannada hate speech detection.


Add a license file before redistributing the code. Ensure that the code, pretrained model checkpoints, and datasets comply with their respective licenses and competition policies.
