#push_to_hub.py
#Push fine-tuned model (adapter or merged) to Hugging Face Hub

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from huggingface_hub import HfApi

from src.config import (
    OUTPUT_DIR,
    MERGED_MODEL_DIR,
    SELECTED_MODEL,
    model_config,
    lora_config,
    data_config,
)

def build_model_card():
    #Generate a basic model card README
    return f"""---
base_model: {model_config.model_name}
tags:
  - legal
  - text-simplification
  - qlora
---

# Legal Text Simplifier — {SELECTED_MODEL}

Fine-tuned from **{model_config.model_name}** using QLoRA to rewrite complex legal
language into plain English.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `{model_config.model_name}` |
| LoRA rank (r) | {lora_config.r} |
| LoRA alpha | {lora_config.lora_alpha} |
| Quantization | 4-bit NF4 |
| Target modules | {', '.join(lora_config.target_modules)} |

## System Prompt

```
{data_config.system_prompt}
```
"""

def upload(repo_id, folder, private):
    #Create repo (if needed) and upload folder
    if not folder.exists():
        print(f"Error: directory does not exist: {folder}")
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    # Write a temporary model card into the folder if one doesn't exist
    card_path = folder / "README.md"
    card_existed = card_path.exists()
    if not card_existed:
        card_path.write_text(build_model_card())

    try:
        print(f"Uploading {folder} -> https://huggingface.co/{repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model")
        print("Done.")
    finally:
        # Clean up generated card
        if not card_existed and card_path.exists():
            card_path.unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Push fine-tuned model to Hugging Face Hub"
    )
    parser.add_argument("--repo-id", required=True,
                        help="HF repo id, e.g. user/model-name")
    parser.add_argument("--model", choices=["qwen", "llama"],
                        default=SELECTED_MODEL,
                        help="Model key (default: %(default)s). Also set via src.config.")
    parser.add_argument("--type", choices=["adapter", "merged", "both"],
                        default="merged", dest="upload_type",
                        help="What to upload (default: merged)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repo")
    args = parser.parse_args()

    adapter_dir = OUTPUT_DIR / "final_adapter"
    merged_dir = MERGED_MODEL_DIR

    if args.upload_type == "adapter":
        upload(args.repo_id, adapter_dir, args.private)
    elif args.upload_type == "merged":
        upload(args.repo_id, merged_dir, args.private)
    else:
        upload(f"{args.repo_id}-adapter", adapter_dir, args.private)
        upload(f"{args.repo_id}-merged", merged_dir, args.private)

if __name__ == "__main__":
    main()
