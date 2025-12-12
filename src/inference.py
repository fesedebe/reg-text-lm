import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

# Config
@dataclass
class InferenceConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path: str = "output/qwen25b-qlora"
    test_path: str = "data/processed/qlora_test.jsonl"
    output_path: str = "output/test_predictions.jsonl"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    system_prompt: str = (
        "You restate legal and regulatory obligations in clear plain language for "
        "non-legal audiences, preserving ALL meaning, conditions, exceptions, and citations. "
        "Do not soften requirements or omit edge cases. Do not add explanations, interpretations, "
        "summaries, or new sentences. Rewrite only what is present in the original text, using "
        "simpler wording while keeping the full legal meaning intact."
    )

# Helper: Chat Formatting
def build_prompt(instruction: str, text: str, tokenizer, config: InferenceConfig):
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": instruction + "\n\nOriginal obligation:\n" + text},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# Load Model, LoRA adapters, and Tokenizer
def load_model_and_tokenizer(config: InferenceConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.lora_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, config.lora_path)
    model.eval()

    return model, tokenizer

def run_inference():
    config = InferenceConfig()

    ds = load_dataset("json", data_files={"test": config.test_path})["test"]
    model, tokenizer = load_model_and_tokenizer(config)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    out_f = open(config.output_path, "w")

    print(f"Running inference on {len(ds)} test examples\n")

    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        original = ex.get("input", "").strip()
        gold = ex.get("output", "").strip()

        # Build prompt
        prompt = build_prompt(instruction, original, tokenizer, config)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=(config.temperature > 0),
            )

        decoded = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        # Extract model output after the last assistant tag
        if "<|im_start|>assistant" in decoded:
            model_output = decoded.split("<|im_start|>assistant")[-1].strip()
        else:
            model_output = decoded

        # Write JSONL record
        record = {
            "instruction": instruction,
            "input": original,
            "gold_output": gold,
            "model_output": model_output,
        }
        out_f.write(json.dumps(record) + "\n")

    out_f.close()
    print(f"Saved test predictions to: {config.output_path}")

if __name__ == "__main__":
    run_inference()