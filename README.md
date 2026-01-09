
Fine-tuned LLM that rewrites legal text in plain English.

## Overview

Repo fine-tunes open-source models (Qwen2.5-7B or Llama 3.1 8B) using QLoRA to simplify legal language. The model is trained to:

- Simplify complex legal terminology
- Preserve all information (dates, names, citations, conditions, exceptions)
- Maintain accuracy while improving readability for non-legal folks

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
python src/merge_adapter.py
PLAINLAW_API_KEY="your-secret-key" python src/serve.py
```

## API Usage

The model is served via vLLM with an OpenAI-compatible API. After obtaining credentials (server IP & API key), it can be called like this:

```python
from openai import OpenAI

client = OpenAI(base_url="http://:8000/v1", api_key="")

response = client.chat.completions.create(
    model="legal-simplifier",
    messages=[{"role": "user", "content": "Simplify: The defendant shall comply..."}]
)
print(response.choices[0].message.content)
```

## Project Structure

```
├── src/           
│   ├── config.py          # Shared configuration
│   ├── data_loader.py     # Dataset loading and formatting
│   ├── train.py           # QLoRA fine-tuning
│   ├── inference.py       # Batch inference (local or API)
│   ├── merge_adapter.py   # Merge LoRA adapter with base model
│   └── serve.py           # vLLM API server
├── scripts/       
├── tests/         
├── data/          # Training data
└── output/        # Model checkpoints
```