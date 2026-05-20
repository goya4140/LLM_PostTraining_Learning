"""
LoRA 训练（使用 PEFT 库）

与 train_scratch.py 实现等价的功能，直接使用 HuggingFace PEFT 库。
相同的超参数和数据，对比两种实现方式的训练曲线和最终效果应基本一致。

主要差异：
  train_scratch.py -> 手动实现 LoRALinear 和注入逻辑
  train_peft.py   -> peft.get_peft_model() + LoraConfig 自动处理
"""

import os
import math
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from config import CONFIG
from model import load_base_model
from dataset import get_dataloaders
from utils import get_device, memory_stats, print_trainable_parameters, set_seed, ExperimentLogger


def main():
    config = CONFIG()
    set_seed(config.seed)
    device = get_device(config.device)

    print("=" * 60)
    print("LoRA 微调训练（PEFT 库）")
    print("=" * 60)

    # 1. 加载基座模型
    print("[步骤 1] 加载基座模型...")
    model, tokenizer = load_base_model(config)

    # 2. 配置 LoRA（与 train_scratch.py 的参数完全对应）
    print("\n[步骤 2] 应用 PEFT LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",                     # 不训练 bias
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    print("\n[步骤 3] 可训练参数统计:")
    print_trainable_parameters(model)
    model.print_trainable_parameters()   # PEFT 库自带的显示（对比验证）

    # 3. 准备数据
    print("\n[步骤 4] 加载数据集...")
    train_loader, eval_loader = get_dataloaders(config, tokenizer)

    # 4. 优化器和调度器（与 train_scratch.py 完全相同）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # 5. 训练
    os.makedirs(config.save_dir.replace("lora_local", "lora_peft_local").replace("lora_cloud", "lora_peft_cloud"), exist_ok=True)
    save_dir = config.save_dir.replace("lora_local", "lora_peft_local").replace("lora_cloud", "lora_peft_cloud")
    best_eval_loss = float("inf")
    global_step = 0

    print("\n" + "=" * 60)
    print("开始训练（PEFT 库）")
    print("=" * 60)

    with ExperimentLogger(config.log_dir.replace("lora_local", "lora_peft_local")) as logger:
        for epoch in range(1, config.num_epochs + 1):
            print(f"\n[Epoch {epoch:02d}/{config.num_epochs}]")
            model.train()
            total_loss, total_tokens = 0.0, 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="  训练", leave=False)):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()

                n_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.log_interval == 0:
                        logger.log(global_step, train_loss=total_loss / max(total_tokens, 1), lr=scheduler.get_last_lr()[0])
                        total_loss, total_tokens = 0.0, 0

            # 验证
            model.eval()
            eval_loss, eval_tokens = 0.0, 0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="  验证", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    n_tokens = (labels != -100).sum().item()
                    eval_loss += outputs.loss.item() * n_tokens
                    eval_tokens += n_tokens

            eval_loss = eval_loss / max(eval_tokens, 1)
            print(f"  验证 loss: {eval_loss:.4f} | PPL: {math.exp(eval_loss):.2f}")
            logger.log(global_step, eval_loss=eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.save_pretrained(save_dir)
                print(f"  ✓ 保存最优 PEFT adapter 至: {save_dir}")

    print("\n训练完成!")


if __name__ == "__main__":
    main()
