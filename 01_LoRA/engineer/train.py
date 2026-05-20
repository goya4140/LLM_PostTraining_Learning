"""
LoRA fine-tuning with TRL SFTTrainer + PEFT — engineer track (cloud GPU).

Usage:
    python train.py
    python train.py --dry-run        # 10 synthetic steps, no model download
    export WANDB_API_KEY=...         # enable W&B logging
"""
import argparse
import dataclasses
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from config import Config
from dataset import build_dataset, get_formatting_func
from utils import print_memory_stats, print_trainable_parameters, WandbLogger


def build_model_and_tokenizer(cfg: Config):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.torch_dtype, torch.bfloat16)

    model_kwargs = {
        "torch_dtype":      torch_dtype,
        "device_map":       "auto",
        "trust_remote_code": True,
    }
    if cfg.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            print("flash-attn not installed — using default attention.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return tokenizer, model


def build_peft_model(model, cfg: Config):
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def train(cfg: Config, dry_run: bool = False):
    print(f"Model: {cfg.model_name}  |  r={cfg.lora_r}, alpha={cfg.lora_alpha}")
    print_memory_stats("Before load: ")

    tokenizer, model = build_model_and_tokenizer(cfg)
    model = build_peft_model(model, cfg)
    print_trainable_parameters(model, "After LoRA: ")
    print_memory_stats("After load: ")

    if dry_run:
        from datasets import Dataset as HFDataset
        train_dataset = HFDataset.from_list(
            [{"instruction": f"What is {i}+{i}?", "output": str(2*i), "input": ""}
             for i in range(20)]
        )
    else:
        train_dataset = build_dataset(cfg)

    formatting_func = get_formatting_func(tokenizer)

    wandb_logger = None
    if cfg.use_wandb and not dry_run:
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=dataclasses.asdict(cfg),
        )

    sft_config = SFTConfig(
        output_dir=cfg.save_dir,
        num_train_epochs=1 if dry_run else cfg.num_epochs,
        per_device_train_batch_size=2 if dry_run else cfg.batch_size,
        gradient_accumulation_steps=1 if dry_run else cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=0.0 if dry_run else cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16 and not dry_run,
        max_seq_length=128 if dry_run else cfg.max_seq_len,
        logging_steps=1 if dry_run else cfg.log_steps,
        save_steps=99999 if dry_run else cfg.save_steps,
        report_to="wandb" if (cfg.use_wandb and not dry_run) else "none",
        run_name=cfg.run_name,
        dataloader_num_workers=0 if dry_run else cfg.dataloader_num_workers,
        group_by_length=False if dry_run else cfg.group_by_length,
        max_steps=10 if dry_run else -1,
        packing=cfg.packing,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print_memory_stats("After train: ")

    if not dry_run:
        os.makedirs(cfg.save_dir, exist_ok=True)
        trainer.save_model(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        print(f"Saved adapter to {cfg.save_dir}")

    if wandb_logger:
        wandb_logger.finish()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="10-step smoke test with tiny synthetic data (no download)")
    args = parser.parse_args()
    train(Config(), dry_run=args.dry_run)
