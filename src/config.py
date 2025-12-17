#Configuration for QLoRA fine-tuning of Llama 3.1 8B Instruct
#Optimized for NVIDIA T4 GPU (16GB VRAM)

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MERGED_MODEL_DIR = PROJECT_ROOT / "output/merged_model"

@dataclass
class ModelConfig:
    #Model architecture and quantization settings
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    torch_dtype: str = "float16"
    
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoraConfig:
    #LoRA adapter settings
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1 
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainingConfig:
    #Training hyperparameters optimized for T4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8 
    
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1  
    
    num_train_epochs: int = 3
    max_steps: int = -1  
    
    optim: str = "paged_adamw_8bit"  
    weight_decay: float = 0.05  
    
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    
    eval_strategy: str = "steps"
    eval_steps: int = 50
    
    gradient_checkpointing: bool = True
    fp16: bool = False  
    bf16: bool = False  
    dataloader_pin_memory: bool = False  
    
    max_seq_length: int = 768
    
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

@dataclass
class DataConfig:
    train_file: str = str(DATA_DIR / "qlora_train_filtered.jsonl")
    test_file: str = str(DATA_DIR / "qlora_test.jsonl")
    val_file: str = str(DATA_DIR / "qlora_val.jsonl")
    
    # System prompt
    system_prompt: str = (
        "You rewrite legal text in plain English that non-lawyers can understand. "
        "Preserve ALL information, dates, names, citations, conditions, and exceptions exactly. "
        "Use simpler words but keep the full legal meaning intact. "
        "Do not soften requirements, omit edge cases, or add explanations."
    )

@dataclass
class InferenceConfig:
    #Inference and serving settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    model_path: str = str(MERGED_MODEL_DIR)

model_config = ModelConfig()
lora_config = LoraConfig()
training_config = TrainingConfig()
data_config = DataConfig()
inference_config = InferenceConfig()
