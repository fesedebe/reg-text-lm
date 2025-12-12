import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Config
@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path: str = "data/processed/qlora_train.jsonl"
    output_dir: str = "output/qwen25b-qlora"
    max_seq_len: int = 512
    num_train_epochs: int = 4
    learning_rate: float = 2e-4
    gradient_accum: int = 4

    # System prompt
    system_prompt: str = (
        "You restate legal and regulatory obligations clearly while preserving ALL legal meaning, "
        "conditions, exceptions, and citations. Do not soften requirements or omit edge cases."
    )

# Data & Tokenization
def load_training_dataset(path: str):
    #Load JSONL as HF Dataset
    if not os.path.exists(path):
        raise FileNotFoundError(path)
        
    ds = load_dataset("json", data_files={"train": path})
    return ds["train"]

def format_chat(ex: Dict[str, Any], tokenizer, config: TrainConfig) -> str:
    #Qwen chat template format
    inst = ex.get("instruction", "").strip()
    inp = ex.get("input", "").strip()
    out = ex.get("output", "").strip()

    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": inst + "\n\nOriginal obligation:\n" + inp},
        {"role": "assistant", "content": out},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def tokenize_function(ex, tokenizer, config: TrainConfig):
    #Tokenize formatted training examples
    text = format_chat(ex, tokenizer, config)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=config.max_seq_len,
        padding="max_length",
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    }

# Model & Lora Functions
def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for causal LM
    return tokenizer

def load_qlora_model(model_name: str):
    #Load base model in 4-bit quantization and apply QLoRA adapters
    print(f"Loading 4-bit model: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    model.print_trainable_parameters()
    return model

def build_training_args(config: TrainConfig):
    #Construct TrainingArguments separately for clarity and easy tuning
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accum,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
        report_to=[],
    )

# Training Orchestration
def fine_tune_model():
    config = TrainConfig()
    train_ds = load_training_dataset(config.data_path)

    tokenizer = build_tokenizer(config.model_name)
    tokenized_ds = train_ds.map(
        lambda ex: tokenize_function(ex, tokenizer, config),
        remove_columns=train_ds.column_names,
    )
    print("Training dataset tokenized")

    model = load_qlora_model(config.model_name)
    args = build_training_args(config)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Training model")
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Model & tokenizer saved")

if __name__ == "__main__":
    fine_tune_model()
