"""
指令微调数据集加载（BELLE 中文 + Alpaca 英文）

支持数据集：
  - belle:        BelleGroup/train_1M_CN（中文指令数据）
  - alpaca:       tatsu-lab/alpaca（英文指令数据）
  - belle+alpaca: 同时加载中英双语数据，随机混合

数据格式转换：
  原始格式 -> ChatML 格式 -> tokenize -> label mask
关键点：
  label mask: prompt 部分（system + user）的 token label 设为 -100，
             只对 assistant 回复部分计算损失。
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def format_belle(example: dict) -> dict:
    """将 BELLE 数据转换为 ChatML 格式

    BELLE 原始格式: {"instruction": ..., "input": ..., "output": ...}
    输出格式: {"prompt": <系统+用户部分>, "response": <助手回复>}
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}"

    return {"prompt": user_content, "response": output}


def format_alpaca(example: dict) -> dict:
    """将 Alpaca 数据转换为 ChatML 格式

    Alpaca 原始格式: {"instruction": ..., "input": ..., "output": ...}
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}"

    return {"prompt": user_content, "response": output}


def load_instruction_data(
    dataset_name: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """加载并合并指定的指令数据集

    Args:
        dataset_name: "belle" | "alpaca" | "belle+alpaca"
        max_samples:  最大样本数（None 为全量）
        seed:         随机种子（切片和混合时使用）

    Returns:
        统一格式的样本列表，每个元素为 {"prompt": ..., "response": ...}
    """
    samples = []

    if "belle" in dataset_name:
        print("  正在加载 BELLE 数据集...")
        belle_ds = load_dataset(
            "BelleGroup/train_1M_CN",
            split="train",
            trust_remote_code=True,
        )
        if max_samples and "alpaca" not in dataset_name:
            belle_ds = belle_ds.shuffle(seed=seed).select(range(min(max_samples, len(belle_ds))))
        elif max_samples:
            half = max_samples // 2
            belle_ds = belle_ds.shuffle(seed=seed).select(range(min(half, len(belle_ds))))
        samples.extend([format_belle(ex) for ex in belle_ds])
        print(f"    BELLE: {len(belle_ds)} 条样本")

    if "alpaca" in dataset_name:
        print("  正在加载 Alpaca 数据集...")
        alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        if max_samples and "belle" not in dataset_name:
            alpaca_ds = alpaca_ds.shuffle(seed=seed).select(range(min(max_samples, len(alpaca_ds))))
        elif max_samples:
            half = max_samples - len(samples)   # 和 BELLE 凑齐到 max_samples
            alpaca_ds = alpaca_ds.shuffle(seed=seed).select(range(min(half, len(alpaca_ds))))
        samples.extend([format_alpaca(ex) for ex in alpaca_ds])
        print(f"    Alpaca: {len(alpaca_ds)} 条样本")

    # 混合并打乱（双语数据避免中英文样本块状分布）
    random.seed(seed)
    random.shuffle(samples)
    print(f"  合并后共 {len(samples)} 条指令样本")
    return samples


class InstructionDataset(Dataset):
    """指令微调数据集

    将 (prompt, response) 对应用 Qwen2.5 的 chat template 进行格式化，
    并对 prompt 部分的 label 设为 -100（不计入 loss）。
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        response = sample["response"]

        # 应用 Qwen2.5 的 ChatML 格式
        # 完整对话（prompt + response）用于计算 label
        messages_full = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 只有 prompt 部分（用于确定 label mask 边界）
        messages_prompt = [
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,   # 加上 <|im_start|>assistant\n
        )

        # Tokenize
        full_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids[0]

        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=False,
            return_tensors="pt",
        ).input_ids[0]

        prompt_len = min(len(prompt_ids), self.max_length)

        # 构建 label：prompt 部分 = -100，assistant 部分 = token id
        labels = full_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """自定义 collate 函数：对每个 batch 内的序列进行右填充"""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids_list, labels_list, mask_list = [], [], []

    for x in batch:
        seq_len = x["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        labels_list.append(
            torch.cat([x["labels"], torch.full((pad_len,), -100)])
        )
        mask_list.append(
            torch.cat([x["attention_mask"], torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def get_dataloaders(
    config,
    tokenizer: PreTrainedTokenizer,
) -> tuple[DataLoader, DataLoader]:
    """创建训练集和验证集 DataLoader"""
    samples = load_instruction_data(
        config.dataset_name,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )

    # 9:1 划分训练集和验证集
    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    eval_samples = samples[split:split + (config.max_eval_samples or len(samples) - split)]

    train_ds = InstructionDataset(train_samples, tokenizer, config.max_seq_len)
    eval_ds = InstructionDataset(eval_samples, tokenizer, config.max_seq_len)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,   # MPS 和多进程存在兼容问题
        pin_memory=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
    )

    print(f"  训练集: {len(train_ds)} 条 | 验证集: {len(eval_ds)} 条")
    return train_loader, eval_loader
