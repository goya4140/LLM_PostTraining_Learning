"""
Manual LoRA training loop — learn track (local, Mac M4 Pro).

Usage:
    python train.py
    python train.py --dry-run    # 5 synthetic steps, no model download
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from dataset import build_dataloader
from model import inject_lora, freeze_non_lora, get_lora_state_dict
from utils import ExperimentLogger, get_device, memory_stats, count_parameters, set_seed


def load_model(cfg: Config, device: torch.device):
    kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # bitsandbytes handles device placement internally
    else:
        kwargs["device_map"] = {"" : device}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    return tokenizer, model


def train(cfg: Config, dry_run: bool = False):
    set_seed(42)
    device = get_device(cfg.device)
    print(f"Device: {device}")
    print(memory_stats(device))

    print("Loading model and tokenizer...")
    tokenizer, model = load_model(cfg, device)

    print("Injecting LoRA layers...")
    inject_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout, target_modules=cfg.target_modules)
    freeze_non_lora(model)

    total, trainable = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total "
          f"({100*trainable/total:.3f}%)")
    print(memory_stats(device))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    if not dry_run:
        print("Building dataloader...")
        dataloader = build_dataloader(tokenizer, cfg)
    else:
        # Minimal synthetic data: 8 sequences of length 64
        from torch.utils.data import DataLoader, TensorDataset
        seqs = torch.randint(100, 1000, (8, 64))
        labels = seqs.clone()
        labels[:, :32] = -100     # mask first half as "prompt"
        masks = torch.ones_like(seqs)
        dataloader = DataLoader(TensorDataset(seqs, labels, masks), batch_size=2)

    logger = ExperimentLogger(cfg.log_dir)
    model.train()

    global_step = 0
    grad_accum = cfg.gradient_accumulation_steps
    optimizer.zero_grad()
    max_steps = 5 if dry_run else None

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_steps is not None and global_step >= max_steps:
                break

            if dry_run:
                input_ids, labels, attention_mask = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
            else:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    mem = memory_stats(device)
                    step_loss = loss.item() * grad_accum
                    print(f"Step {global_step} | Loss: {step_loss:.4f} | {mem}")
                    logger.log({"step": global_step, "loss": step_loss, "epoch": epoch})

            epoch_loss += loss.item() * grad_accum
            n_batches += 1

        if n_batches > 0:
            avg = epoch_loss / n_batches
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Avg Loss: {avg:.4f}")

        if not dry_run:
            os.makedirs(cfg.save_dir, exist_ok=True)
            ckpt = os.path.join(cfg.save_dir, f"lora_epoch{epoch+1}.pt")
            torch.save(get_lora_state_dict(model), ckpt)
            print(f"Saved LoRA checkpoint: {ckpt}")

    logger.plot()
    print("Training complete.")
    print(memory_stats(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="5-step smoke test with synthetic data (no model download)")
    args = parser.parse_args()
    cfg = Config()
    train(cfg, dry_run=args.dry_run)
