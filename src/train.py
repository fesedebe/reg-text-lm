#train.py
#QLoRA fine-tuning script for Llama 3.1 8B Instruct or Qwen2.5-7B-Instruct,
#trained to simplify legal text to plain English.
#
#Usage: "python train.py --model" (qwen default) or "python train.py --model llama"  

import os
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import (
    model_config,
    lora_config,
    training_config,
    data_config,
    OUTPUT_DIR,
)
from data_loader import load_hf_dataset, format_chat

def build_tokenizer(model_name: str):
    #Initialize tokenizer with proper padding
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" #"right"
    return tokenizer

def setup_quantization_config() -> BitsAndBytesConfig:
    #Create BitsAndBytes config for 4-bit quantization
    compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,  
        bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
    )

def load_qlora_model(model_name: str):
    #Load base model in 4-bit quantization and apply QLoRA adapters
    print(f"Loading 4-bit model: {model_name}")
    
    bnb_config = setup_quantization_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=getattr(torch, model_config.torch_dtype),
    )
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    
    model.print_trainable_parameters()
    return model

def build_training_args() -> SFTConfig:
    output_dir = str(OUTPUT_DIR)
    
    return SFTConfig(
        output_dir=output_dir,
        
        # Batch size settings
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        
        num_train_epochs=training_config.num_train_epochs,
        max_steps=training_config.max_steps,
        
        learning_rate=training_config.learning_rate,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        
        # Precision
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        
        # Gradient checkpointing for memory
        gradient_checkpointing=training_config.gradient_checkpointing,
        
        # Logging and saving
        logging_steps=training_config.logging_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy=training_config.save_strategy,
        save_total_limit=training_config.save_total_limit,
        
        # Evaluation
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        
        # SFT-specific
        max_length=training_config.max_seq_length,
        completion_only_loss=False,  
        dataset_text_field="text", 
        
        # Misc
        seed=training_config.seed,
        report_to=training_config.report_to,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
    )

def train():
    # Training orchestration
    print(f"Model: {model_config.model_name} ({model_config.model_key})")
    print(f"Data: {data_config.train_file}")
    print(f"Output: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer = build_tokenizer(model_config.model_name)
    system_prompt = data_config.system_prompt
    
    # Load and pre-format datasets
    train_ds = load_hf_dataset(data_config.train_file)
    train_ds = train_ds.map(
        lambda ex: {"text": format_chat(ex, tokenizer, system_prompt)},
        remove_columns=train_ds.column_names,
    )
    
    # Load validation set if exists
    val_ds = None
    if Path(data_config.val_file).exists():
        val_ds = load_hf_dataset(data_config.val_file)
        val_ds = val_ds.map(
            lambda ex: {"text": format_chat(ex, tokenizer, system_prompt)},
            remove_columns=val_ds.column_names,
        )
    
    model = load_qlora_model(model_config.model_name)
    training_args = build_training_args()
    print("\nDataset and model loaded. Starting training.")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    
    # Saving model and tokenizer
    final_output_dir = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"\nTraining complete. Model & tokenizer saved to: {final_output_dir}")
    
    return trainer

if __name__ == "__main__":
    train()