"""
Dataset loading for LoRA engineer track.
Produces HuggingFace Dataset objects compatible with TRL SFTTrainer.
"""
from datasets import load_dataset, concatenate_datasets, Dataset
from config import Config


def get_belle_dataset(n_samples: int) -> Dataset:
    ds = load_dataset("BelleGroup/train_1M_CN", split="train")
    return ds.select(range(min(n_samples, len(ds))))


def get_alpaca_dataset(n_samples: int) -> Dataset:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    return ds.select(range(min(n_samples, len(ds))))


def get_formatting_func(tokenizer):
    """
    Returns a formatting function compatible with TRL SFTTrainer.
    Applies the model's chat template to produce ready-to-train text.
    """
    def formatting_func(examples):
        texts = []
        instructions = examples.get("instruction", [])
        outputs      = examples.get("output", [])
        inputs_field = examples.get("input", ["" ] * len(instructions))

        for i in range(len(instructions)):
            inst = (instructions[i] or "").strip()
            inp  = (inputs_field[i]  or "").strip() if isinstance(inputs_field, list) else ""
            out  = (outputs[i]       or "").strip()
            if inp:
                inst = f"{inst}\n{inp}"
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system",    "content": "You are a helpful assistant."},
                    {"role": "user",      "content": inst},
                    {"role": "assistant", "content": out},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts
    return formatting_func


def build_dataset(cfg: Config) -> Dataset:
    """Build combined BELLE (zh) + Alpaca (en) training dataset."""
    n_belle  = int(cfg.max_train_samples * cfg.belle_ratio)
    n_alpaca = cfg.max_train_samples - n_belle

    print(f"Loading {n_belle} BELLE (CN) + {n_alpaca} Alpaca (EN) samples...")
    belle_ds  = get_belle_dataset(n_belle)
    alpaca_ds = get_alpaca_dataset(n_alpaca)

    # Normalise to shared columns
    keep = ["instruction", "input", "output"]
    belle_ds  = belle_ds.select_columns([c for c in keep if c in belle_ds.column_names])
    alpaca_ds = alpaca_ds.select_columns([c for c in keep if c in alpaca_ds.column_names])

    combined = concatenate_datasets([belle_ds, alpaca_ds]).shuffle(seed=42)
    print(f"Total samples: {len(combined)}")
    return combined
