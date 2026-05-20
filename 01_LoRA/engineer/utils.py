"""
Utilities for LoRA engineer track: W&B logging, GPU memory monitoring, throughput.
"""
import time
import torch
from typing import Dict, Any, Optional


def get_memory_stats() -> Dict[str, str]:
    """Current GPU memory usage as human-readable strings."""
    if not torch.cuda.is_available():
        return {"note": "no CUDA device"}
    allocated = torch.cuda.memory_allocated()  / 1024**3
    reserved  = torch.cuda.memory_reserved()   / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return {
        "allocated_GB": f"{allocated:.2f}",
        "reserved_GB":  f"{reserved:.2f}",
        "total_GB":     f"{total:.2f}",
        "util_pct":     f"{100*allocated/total:.1f}%",
    }


def print_memory_stats(prefix: str = ""):
    s = get_memory_stats()
    if "note" in s:
        print(f"{prefix}Memory: {s['note']}")
    else:
        print(f"{prefix}Memory: {s['allocated_GB']}GB / {s['total_GB']}GB ({s['util_pct']})")


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_trainable_parameters(model, label: str = ""):
    total, trainable = count_parameters(model)
    pct = 100 * trainable / total if total > 0 else 0
    print(f"{label}Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")


class WandbLogger:
    """Thin W&B wrapper that falls back to a no-op if W&B is unavailable."""

    def __init__(self, project: str, name: str, config: Dict[str, Any]):
        self.enabled = False
        try:
            import wandb
            wandb.init(project=project, name=name, config=config)
            self.enabled = True
            print(f"W&B run: {project}/{name}")
        except Exception as e:
            print(f"W&B disabled: {e}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass

    def finish(self):
        if not self.enabled:
            return
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


class ThroughputLogger:
    """Tracks tokens/second generation throughput."""

    def __init__(self):
        self._start  = time.time()
        self._tokens = 0

    def update(self, n_tokens: int):
        self._tokens += n_tokens

    def get(self) -> float:
        elapsed = time.time() - self._start
        return self._tokens / elapsed if elapsed > 0 else 0.0

    def reset(self):
        self._start  = time.time()
        self._tokens = 0
