#inference.py
#Inference script for (fine-tuned) Qwen2.5-7B or Llama 3.1 8B models.
#Supports both local model loading and API calls to served model.
#
#Usage: python inference.py --model qwen (or llama) [--base-only]

import argparse
import os
import json

import pandas as pd
from datasets import load_dataset

from config import (
    model_config,
    data_config,
    inference_config,
    OUTPUT_DIR,
)

API_URL = "http://localhost:8000/v1"
SAVE_EXCEL = True

def build_messages(instruction: str, text: str) -> list:
    #Build chat messages for both local and API inference
    if instruction:
        user_content = f"{instruction}\n\n{text}"
    else:
        user_content = text

    return [
        {"role": "system", "content": data_config.system_prompt},
        {"role": "user", "content": user_content},
    ]

def load_model_and_tokenizer(base_only=False):
    #Load model locally, optionally without LoRA adapter
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    lora_path = str(OUTPUT_DIR / "final_adapter")
    tokenizer_source = model_config.model_name if base_only else lora_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if not base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")

    model.eval()
    return model, tokenizer

def generate_local(messages: list, model, tokenizer) -> str:
    #Generate using locally loaded model
    import torch
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_tokens = model.generate(
            **inputs,
            max_new_tokens=inference_config.max_new_tokens,
            temperature=inference_config.temperature,
            top_p=inference_config.top_p,
            do_sample=(inference_config.temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = gen_tokens[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_api(messages: list) -> str:
    #Generate using served model API
    from openai import OpenAI
    
    client = OpenAI(base_url=API_URL, api_key="unused")
    
    response = client.chat.completions.create(
        model="legal-simplifier",
        messages=messages,
        temperature=inference_config.temperature,
        max_tokens=inference_config.max_new_tokens,
    )
    
    return response.choices[0].message.content.strip()

def run_inference():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["local", "api"], default="local",
                        help="local: load model directly, api: call served model")
    parser.add_argument("--base-only", action="store_true",
                        help="Skip LoRA adapter, run base model only")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Override output filename (without extension)")
    args, _ = parser.parse_known_args()

    use_api = args.mode == "api"
    test_path = data_config.test_file

    # Build output path
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = str(OUTPUT_DIR)
    if args.output_name:
        output_path = os.path.join(out_dir, f"{args.output_name}.jsonl")
    else:
        output_path = os.path.join(out_dir, "test_filtered_predictions.jsonl")

    print(f"Model: {model_config.model_name} ({model_config.model_key})")
    print(f"Mode: {'API' if use_api else 'local'}")
    print(f"Base only: {args.base_only}")
    print(f"Test data: {test_path}")
    print(f"Output: {output_path}")

    if not use_api:
        if not args.base_only:
            print(f"Adapter: {OUTPUT_DIR / 'final_adapter'}")
    else:
        print(f"API URL: {API_URL}")

    ds = load_dataset("json", data_files={"test": test_path})["test"]

    # Load model only if running locally
    model, tokenizer = None, None
    if not use_api:
        model, tokenizer = load_model_and_tokenizer(base_only=args.base_only)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    records = []

    print(f"\nRunning inference on {len(ds)} examples\n")

    with open(output_path, "w") as out_f:
        for i, ex in enumerate(ds):
            instruction = ex.get("instruction", "").strip()
            original = ex.get("input", "").strip()
            gold = ex.get("output", "").strip()

            messages = build_messages(instruction, original)

            if use_api:
                model_output = generate_api(messages)
            else:
                model_output = generate_local(messages, model, tokenizer)

            record = {
                "instruction": instruction,
                "input": original,
                "gold_output": gold,
                "model_output": model_output,
            }
            records.append(record)
            out_f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(ds)} examples")

    print(f"\nSaved predictions to: {output_path}")

    if SAVE_EXCEL:
        excel_path = output_path.replace(".jsonl", ".xlsx")
        pd.DataFrame(records).to_excel(excel_path, index=False)
        print(f"Saved predictions to: {excel_path}")

if __name__ == "__main__":
    run_inference()