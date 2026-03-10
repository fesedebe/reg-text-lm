# Methods

## Motivation

Legal text is written for lawyers. Sentences run long and the information that actually matters hides inside nested clauses. For the people these documents affect (tenants facing eviction, applicants trying to parse a benefits form) the text is inaccessible. Automated simplification could help, but you also can't sacrifice precision for readability. Every condition and exception must survive the rewrite, making legal simplification a different problem from generic summarization. 

This repo tests if small open-source models (7–8B parameters), fine-tuned with QLoRA on a modest dataset, can do this well enough to beat frontier API models used out of the box. Is fine-tuning worth the effort, or can you get the same results with a system prompt and an API call?

## Methodology & Evaluation Setup

The pipeline consists of four stages: data preparation, QLoRA fine-tuning, model serving, and evaluation. Two base models are trained and compared, Qwen 2.5-7B-Instruct and Llama 3.1-8B-Instruct, alongside API baselines from OpenAI (GPT-4o-mini, GPT-5.4) in both zero-shot and few-shot configurations.

| Variant | Model | Method | Cost |
|---|---|---|---|
| Base Qwen 2.5-7B | Qwen/Qwen2.5-7B-Instruct | System prompt only | Local GPU |
| Base Llama 3.1-8B | meta-llama/Meta-Llama-3.1-8B-Instruct | System prompt only | Local GPU |
| FT Qwen 2.5-7B | Qwen + QLoRA adapter | Fine-tuned on 350 examples | Local GPU |
| FT Llama 3.1-8B | Llama + QLoRA adapter | Fine-tuned on 350 examples | Local GPU |
| Zero-shot GPT-4o-mini | gpt-4o-mini | System prompt only | API |
| Few-shot GPT-4o-mini | gpt-4o-mini | 5-shot prompting | API |
| Zero-shot GPT-5.4 | gpt-5.4 | System prompt only | API |
| Few-shot GPT-5.4 | gpt-5.4 | 5-shot prompting | API |

### Data Preparation

Training and test data are drawn from the Lex-Simple dataset (Cemri et al., 2022), a benchmark for legal text simplification containing 500 U.S. Supreme Court opinion sentences (github.com/koc-lab/uslt-lex-simple). The original sentences were sampled from the Caselaw Access Project (Harvard Law School) and paired with three independently produced simplified references written by faculty and students at Bilkent University Faculty of Law. This pipeline uses the original sentences (supreme_org_full.txt from the full directory) paired with the third reference set (ref3), which contained the most simplification after inspection.

Raw data is processed in two stages. First, `prepare_data.py` extracts complex-simple pairs from the source spreadsheet and applies a 90/10 train/test split (seed=42). Second, `filter_data.py` applies quality filters to remove noise:

- Pairs with empty input or output fields are removed.
- Identical input-output pairs are removed.
- Near-identical pairs (>93.5% similarity via Python's `SequenceMatcher`) are removed.
- Outputs containing known typos or bad terms ("suers," "subpeana," "younglings") are corrected or removed.
- Truncated outputs (less than 60% of input length) are removed.

After filtering, 350 training examples and 30 test examples remain from the original 500. A small number of high-quality test examples are manually migrated to the training set during curation to improve coverage of underrepresented legal text styles.

### Fine-Tuning

Both base models are fine-tuned using QLoRA (Dettmers et al., 2023) via TRL's `SFTTrainer`. The models are quantized to 4-bit NF4 precision with bfloat16 compute dtype, enabling 7–8B parameter models to train on a single L4 GPU (24GB VRAM). LoRA adapters are applied to all attention and MLP projection layers (q/k/v/o_proj + gate/up/down_proj) with rank 32, alpha 64, and dropout 0.1. Training runs for 3 epochs with a learning rate of 1e-4, 10% linear warmup, and the `paged_adamw_8bit` optimizer. Per-device batch size is 2 with 4 gradient accumulation steps, yielding an effective batch size of 8. Gradient checkpointing is enabled to reduce memory usage. Two checkpoints are saved (one per epoch after the first), and training completes in approximately 15 minutes per model.

Both Qwen and Llama use identical LoRA configurations and training hyperparameters. Hyperparameters were set once and used identically for both models with no hyperparameter search. At this data scale (350 examples, 3 epochs), the overhead of tuning would likely exceed the gains. Training data is formatted as multi-turn chat using each model's native chat template via `data_loader.py`. Each example consists of a system message, a user message containing the legal text, and an assistant message containing the simplified output. The system prompt used during both training and inference is:

```
You rewrite legal text in plain English that non-lawyers can understand. Preserve ALL information, dates, names, citations, conditions, and exceptions. Use simpler words but keep the full legal meaning intact. Do not soften requirements, omit edge cases, or add explanations.
```

After training, the LoRA adapter is merged into the base model weights via `merge.py`, producing a standalone model for serving.

### API Baselines

Four API baselines are evaluated using `run_external_api.py`:

- **GPT-4o-mini** (zero-shot and 5-shot)
- **GPT-5.4** (zero-shot and 5-shot)

All API calls use the same system prompt as the fine-tuned models. Few-shot variants include 5 hardcoded complex-simple pairs drawn from the training set as in-context examples. API calls use temperature 0.7 and max tokens 512, matching the fine-tuned model generation parameters.

### Inference

Fine-tuned models are served via vLLM (`serve.py`) as an OpenAI-compatible API at `localhost:8000/v1`, configured to use 90% of available GPU memory. Inference is run through `inference.py`, which calls this local API using the OpenAI Python client.

Generation parameters: temperature 0.7, top-p 0.9, max output tokens 512. Base model variants (without the LoRA adapter) are run for comparison using the same generation parameters. All predictions are written to JSONL files in `output/`, with one file per model variant, enabling direct multi-model comparison via `eval.py`.

### Evaluation

Three primary metrics are computed per prediction using `eval.py`:

- **Flesch-Kincaid Grade Level (FKGL)** is computed using the `textstat` library. It estimates the U.S. school grade level required to understand the text based on average sentence length and average syllables per word. Both absolute output FKGL and delta (output FKGL minus input FKGL) are reported. Lower output FKGL and more negative delta indicate more readable output.
- **Change rate** measures how much the model actually rewrote, computed as `1 - SequenceMatcher.ratio()` from Python's `difflib`. A change rate of 0% means the output is identical to the input; higher values indicate more extensive rewriting. Outputs with change rate below 10% are classified as near-copies.
- **Length ratio** is the character count of the output divided by the character count of the input. Values between 0.75 and 1.15 are considered acceptable. Ratios below 0.75 suggest information loss; above 1.15 suggests bloat.
- A prediction is classified as a **success** if all three conditions hold: change rate > 10%, length ratio between 0.75 and 1.15, and output FKGL lower than input FKGL.
- **Adjusted FKGL** is a composite metric that replaces a model's output FKGL with the input FKGL on any near-copy example (change rate < 10%), giving no credit for unchanged text.

Head-to-head comparisons between models are computed per-example on output FKGL: for each test example, the model with the lower output FKGL wins. Ties are excluded. All metrics are reference-free. Gold outputs exist but contain known errors, so they are not used for scoring.

### Limitations

FKGL measures word length and sentence length, not comprehension. The test set is small (n=30), so margins between top models are directional not statistically significant. No semantic faithfulness metric is included. See [eval_report.md Section 5](eval_report.md#5-metric-limitations).

### Pipeline

```
1. DATA PREP          2. TRAINING           3. MERGE              4. SERVING            5. EVAL
─────────────         ──────────            ─────                 ───────               ────
Excel (raw)           JSONL → HF Dataset    LoRA adapter          Merged model          Predictions
  │                     │                    + base model           │                     │
  ▼                     ▼                      │                    ▼                     ▼
prepare_data.py       train.py                 ▼                  serve.py              eval.py
  │                   (QLoRA SFTTrainer)      merge.py            (vLLM server)         (readability,
  ▼                     │                      │                    │                    similarity,
filter_data.py          ▼                      ▼                    ▼                    length ratio)
  │                   final_adapter/          merged_model/        localhost:8000/v1
  ▼                                                                 │
qlora_{train,test}                                                  ▼
  _filtered.jsonl                                              inference.py
                                                               (local or API mode)
                                                                    │
                                                                    ▼
                                                               predictions.jsonl
```

### Reproduction

```bash
# Fine-tuned model predictions (requires trained adapters)
python src/inference.py --model qwen --output-dir output/comparison --output-name ft-qwen
python src/inference.py --model llama --output-dir output/comparison --output-name ft-llama

# Base model predictions
python src/inference.py --model qwen --base-only --output-dir output/comparison --output-name base-qwen
python src/inference.py --model llama --base-only --output-dir output/comparison --output-name base-llama

# API predictions
python scripts/run_external_api.py --provider openai --model gpt-4o-mini --shots 0
python scripts/run_external_api.py --provider openai --model gpt-4o-mini --shots 5
python scripts/run_external_api.py --provider openai --model gpt-5.4 --shots 0
python scripts/run_external_api.py --provider openai --model gpt-5.4 --shots 5

# Combine and evaluate
python scripts/combine_predictions.py
```