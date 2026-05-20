"""
Dataset loading for LoRA learn track.
BELLE (Chinese) + Alpaca (English) with ChatML formatting and manual label masking.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict
from config import Config


BELLE_DATASET = "BelleGroup/train_0.5M_CN"
ALPACA_DATASET = "tatsu-lab/alpaca"


def load_belle_samples(n: int) -> List[Dict]:
    """Load n samples from BELLE Chinese instruction dataset."""
    ds = load_dataset(BELLE_DATASET, split="train", streaming=True)
    samples = []
    for item in ds:
        if len(samples) >= n:
            break
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if instruction and output:
            samples.append({"instruction": instruction, "output": output})
    return samples


def load_alpaca_samples(n: int) -> List[Dict]:
    """Load n samples from Alpaca English instruction dataset."""
    ds = load_dataset(ALPACA_DATASET, split="train")
    samples = []
    for item in ds:
        if len(samples) >= n:
            break
        instruction = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        output = (item.get("output") or "").strip()
        if instruction and output:
            full_inst = f"{instruction}\n{inp}" if inp else instruction
            samples.append({"instruction": full_inst, "output": output})
    return samples


def format_chatml(instruction: str, response: str) -> tuple:
    """
    Returns (full_text, prompt_only) in ChatML format.

    full_text:   complete input+output string for tokenization
    prompt_only: the prompt prefix (used to compute label mask boundary)

    ChatML format (Qwen2.5 standard):
        <|im_start|>system\n{system}<|im_end|>\n
        <|im_start|>user\n{instruction}<|im_end|>\n
        <|im_start|>assistant\n{response}<|im_end|>
    """
    system = "You are a helpful assistant."
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt + response + "<|im_end|>"
    return full_text, prompt


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction fine-tuning.
    Labels are masked to -100 on the prompt portion so loss is computed
    only on the assistant response tokens.
    """

    def __init__(self, samples: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        full_text, prompt = format_chatml(item["instruction"], item["output"])

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
        labels = input_ids.clone()

        # Mask prompt tokens: only compute loss on the assistant response
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch to the same length with right-padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        mask_list.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long),
                       torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def build_dataloader(tokenizer: AutoTokenizer, cfg: Config,
                     belle_ratio: float = 0.5) -> DataLoader:
    """Build training dataloader mixing BELLE (zh) and Alpaca (en)."""
    n_belle = int(cfg.max_train_samples * belle_ratio)
    n_alpaca = cfg.max_train_samples - n_belle

    print(f"Loading {n_belle} BELLE samples (Chinese)...")
    belle = load_belle_samples(n_belle)
    print(f"Loading {n_alpaca} Alpaca samples (English)...")
    alpaca = load_alpaca_samples(n_alpaca)

    all_samples = belle + alpaca
    print(f"Total samples: {len(all_samples)}")

    dataset = InstructionDataset(all_samples, tokenizer, max_length=cfg.max_seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
        pin_memory=False,
    )
