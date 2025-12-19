#inference.py
#Inference script for fine-tuned Llama 3.1 8B or Qwen2.5-7B models.
#
#Usage: "python inference.py --model" (qwen default) or "python inference.py --model llama"  

import os
import json

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from config import (
    model_config,
    data_config,
    inference_config,
    OUTPUT_DIR,
)

# Helper: Chat Formatting
def build_prompt(instruction: str, text: str, tokenizer):
    # Match training format exactly
    if instruction:
        user_content = f"{instruction}\n\n{text}"
    else:
        user_content = text

    messages = [
        {"role": "system", "content": data_config.system_prompt},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# Load Model, LoRA adapters, and Tokenizer
def load_model_and_tokenizer():
    lora_path = str(OUTPUT_DIR / "final_adapter")
    
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 4-bit quantization config (same as training)
    compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return model, tokenizer

def run_inference(save_excel: bool = False):
    test_path = data_config.test_file
    output_path = str(OUTPUT_DIR / "test_filtered_predictions.jsonl")

    print(f"Model: {model_config.model_name} ({model_config.model_key})")
    print(f"Adapter: {OUTPUT_DIR / 'final_adapter'}")
    print(f"Test data: {test_path}")

    ds = load_dataset("json", data_files={"test": test_path})["test"]
    model, tokenizer = load_model_and_tokenizer()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_f = open(output_path, "w")
    records = []

    print(f"Running inference on {len(ds)} examples\n")

    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        original = ex.get("input", "").strip()
        gold = ex.get("output", "").strip()

        # Build prompt
        prompt = build_prompt(instruction, original, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate output
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=inference_config.max_new_tokens,
                temperature=inference_config.temperature,
                top_p=inference_config.top_p,
                do_sample=(inference_config.temperature > 0),
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
    print(f"Saved predictions to: {output_path}")

    if save_excel:
        excel_path = output_path.replace(".jsonl", ".xlsx")
        pd.DataFrame(records).to_excel(excel_path, index=False)
        print(f"Saved predictions to: {excel_path}")

if __name__ == "__main__":
    run_inference(save_excel=True)