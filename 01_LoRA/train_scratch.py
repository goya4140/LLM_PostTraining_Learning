"""
LoRA 训练（从零实现）

流程：
  1. 加载基座模型（Qwen2.5-0.5B 本地 / Qwen2.5-7B 云端）
  2. 将指定线性层注入 LoRALinear（手动实现）
  3. 冻结非 LoRA 参数
  4. 训练，只更新 A 和 B 矩阵
  5. 定期保存 LoRA adapter 权重（轻量级）

运行方式：
  TRAIN_ENV=local python train_scratch.py    # Mac M4 Pro
  TRAIN_ENV=cloud python train_scratch.py    # A100/H100
"""

import os
import math
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import CONFIG
from model import load_base_model, inject_lora, freeze_non_lora, get_lora_state_dict
from dataset import get_dataloaders
from utils import get_device, memory_stats, print_trainable_parameters, set_seed, ExperimentLogger


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_accum_steps: int,
    max_grad_norm: float,
    logger: ExperimentLogger,
    global_step: int,
    log_interval: int,
) -> tuple[float, int]:
    """训练一个 epoch，返回 (平均 loss, 最新的 global_step)"""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()
    start_time = time.time()

    for step, batch in enumerate(tqdm(loader, desc="  训练", leave=False)):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        # 有效 token 数（用于计算平均 loss）
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

        # 梯度累积 + 参数更新
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # 日志记录
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                current_lr = scheduler.get_last_lr()[0]
                logger.log(
                    global_step,
                    train_loss=total_loss / max(total_tokens, 1),
                    lr=current_lr,
                    tokens_per_sec=tokens_per_sec,
                )
                total_loss = 0.0
                total_tokens = 0
                start_time = time.time()

    return global_step


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """计算验证集平均 loss"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in tqdm(loader, desc="  验证", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main():
    config = CONFIG()
    set_seed(config.seed)
    device = get_device(config.device)

    print("=" * 60)
    print("LoRA 微调训练（从零实现）")
    print("=" * 60)
    print(f"  训练环境 : {os.environ.get('TRAIN_ENV', 'local')}")
    print(f"  模型     : {config.model_name}")
    print(f"  设备     : {device}")
    print(f"  LoRA r    : {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  target    : {config.target_modules}")
    print()

    # 1. 加载基座模型
    print("[步骤 1] 加载基座模型...")
    model, tokenizer = load_base_model(config)
    print(f"  模型加载完成：{config.model_name}")
    print(f"  {memory_stats(device)}")

    # 2. 注入 LoRA 层
    print("\n[步骤 2] 注入 LoRA 层...")
    model = inject_lora(
        model,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )

    # 3. 冻结非 LoRA 参数
    freeze_non_lora(model)
    print("\n[步骤 3] 可训练参数统计:")
    print_trainable_parameters(model)

    # 4. 准备数据
    print("\n[步骤 4] 加载数据集...")
    train_loader, eval_loader = get_dataloaders(config, tokenizer)

    # 5. 优化器和学习率调度
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    # 6. 训练循环
    os.makedirs(config.save_dir, exist_ok=True)
    best_eval_loss = float("inf")
    global_step = 0

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    with ExperimentLogger(config.log_dir) as logger:
        for epoch in range(1, config.num_epochs + 1):
            print(f"\n[Epoch {epoch:02d}/{config.num_epochs}]")

            global_step = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                config.gradient_accumulation_steps, config.max_grad_norm,
                logger, global_step, config.log_interval,
            )

            eval_loss = evaluate(model, eval_loader, device)
            print(f"  验证 loss: {eval_loss:.4f} | PPL: {math.exp(eval_loss):.2f}")
            logger.log(global_step, eval_loss=eval_loss, eval_ppl=math.exp(eval_loss))

            # 保存最优 LoRA adapter
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                adapter_path = os.path.join(config.save_dir, "best_adapter.pt")
                torch.save(get_lora_state_dict(model), adapter_path)
                print(f"  ✓ 保存最优 adapter （eval loss: {best_eval_loss:.4f}）")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"  最优验证 loss: {best_eval_loss:.4f}")
    print(f"  Adapter 保存于: {config.save_dir}/best_adapter.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
