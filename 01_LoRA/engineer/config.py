"""
Cloud GPU training config for LoRA engineer track.
Target: A100/H100 with CUDA, Qwen2.5-7B-Instruct.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "bfloat16"
    use_flash_attention_2: bool = True

    # LoRA (PEFT)
    lora_r: int = 16
    lora_alpha: int = 32              # alpha = 2r
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = 32
    max_seq_len: int = 2048
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Data
    dataset_belle: str = "BelleGroup/train_1M_CN"
    dataset_alpaca: str = "tatsu-lab/alpaca"
    max_train_samples: int = 50000
    belle_ratio: float = 0.5

    # Logging & saving
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    log_steps: int = 10
    save_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "lora-engineer"
    run_name: str = "qwen2.5-7b-lora-r16"

    # Throughput
    dataloader_num_workers: int = 4
    group_by_length: bool = True
    packing: bool = False

    def __post_init__(self):
        if self.use_wandb and os.environ.get("WANDB_API_KEY") is None:
            print("Warning: WANDB_API_KEY not set; disabling W&B logging.")
            self.use_wandb = False
