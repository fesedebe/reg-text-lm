#merge.py
#Merges QLoRA adapter with base model for vLLM serving.
#
#Usage: "python merge.py" (qwen default) or "python merge.py --model llama"  

import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import (
    model_config,
    OUTPUT_DIR,
    MERGED_MODEL_DIR,
)

def merge_adapter():
    adapter_path = OUTPUT_DIR / "final_adapter"
    output_path = MERGED_MODEL_DIR
    
    print(f"Model: {model_config.model_name} ({model_config.model_key})")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {output_path}")
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer from adapter
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )
    
    # Load base model in full precision
    print("Tokenizer loaded. Loading base model (this may take a few minutes)")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load & merge adapter
    print("Loading adapter and merging")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"\nMerged model saved to: {output_path}")

if __name__ == "__main__":
    merge_adapter()