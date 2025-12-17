import os
import json
from dataclasses import dataclass

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from config import OUTPUT_DIR

# Config
@dataclass
class InferenceConfig:
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    lora_path: str = str(OUTPUT_DIR / "final_adapter")
    test_path: str = "data/processed/qlora_test.jsonl"
    output_path: str = str(OUTPUT_DIR / "test_predictions.jsonl")
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

    # Must match training system prompt
    system_prompt: str = (
        "You rewrite legal text in plain English that non-lawyers can understand. "
        "Preserve ALL information, dates, names, citations, conditions, and exceptions exactly. "
        "Use simpler words but keep the full legal meaning intact. "
        "Do not soften requirements, omit edge cases, or add explanations."
    )

# Helper: Chat Formatting
def build_prompt(instruction: str, text: str, tokenizer, config: InferenceConfig):
    # Match training format exactly
    if instruction:
        user_content = f"{instruction}\n\n{text}"
    else:
        user_content = text

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content},
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

    # 4-bit quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # FP16 for T4
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, config.lora_path)
    model.eval()

    return model, tokenizer

def run_inference(save_excel: bool = False):
    config = InferenceConfig()

    ds = load_dataset("json", data_files={"test": config.test_path})["test"]
    model, tokenizer = load_model_and_tokenizer(config)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    out_f = open(config.output_path, "w")
    records = []

    print(f"Running inference on {len(ds)} examples\n")

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
                pad_token_id=tokenizer.eos_token_id,
            )

        # Only decode newly generated tokens (not the prompt)
        new_tokens = gen_tokens[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        record = {
            "instruction": instruction,
            "input": original,
            "gold_output": gold,
            "model_output": model_output,
        }
        records.append(record)
        out_f.write(json.dumps(record) + "\n")

    out_f.close()
    print(f"Saved predictions to: {config.output_path}")

    if save_excel:
        excel_path = config.output_path.replace(".jsonl", ".xlsx")
        pd.DataFrame(records).to_excel(excel_path, index=False)
        print(f"Saved predictions to: {excel_path}")

if __name__ == "__main__":
    run_inference(save_excel=True)