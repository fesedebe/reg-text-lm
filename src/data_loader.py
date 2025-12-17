#Transforms JSONL data into Llama 3.1 chat format

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

def load_hf_dataset(path: str) -> Dataset:
    #Load JSONL as HF Dataset
    if not Path(path).exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    
    ds = load_dataset("json", data_files={"train": path})
    return ds["train"]

def format_chat_message(
    input_text: str,
    output_text: Optional[str] = None,
    instruction: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    #Format input/output into Llama 3.1 chat message format with system prompt
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    if instruction:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = input_text
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    if output_text is not None:
        messages.append({
            "role": "assistant",
            "content": output_text
        })
    
    return messages

def format_llama_chat(
    example: Dict[str, Any],
    tokenizer: Any,
    system_prompt: str
) -> str:
    #Format example using Llama 3.1 chat template
    messages = format_chat_message(
        input_text=example.get("input", "").strip(),
        output_text=example.get("output", "").strip(),
        instruction=example.get("instruction", "").strip() or None,
        system_prompt=system_prompt
    )
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def prepare_inference_prompt(
    input_text: str,
    tokenizer: Any,
    system_prompt: str,
    instruction: Optional[str] = None
) -> str:
    #Prepare a prompt for inference (without the assistant response)
    messages = format_chat_message(
        input_text=input_text,
        output_text=None,
        instruction=instruction,
        system_prompt=system_prompt
    )
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt

if __name__ == "__main__":
    from config import data_config
    
    ds = load_hf_dataset(data_config.train_file)
    print(f"Loaded {len(ds)} training examples")
    
    if Path(data_config.test_file).exists():
        test_ds = load_hf_dataset(data_config.test_file)
        print(f"Loaded {len(test_ds)} test examples")
    
    if len(ds) > 0:
        print("\nSample example:")
        sample = ds[0]
        print(f"  Input: {sample['input'][:100]}")
        print(f"  Output: {sample['output'][:100]}")
