"""
共用工具函数（日志、可视化、设备检测）
"""

import os
import csv
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():           return torch.device("cuda")
        if torch.backends.mps.is_available():  return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def memory_stats(device: torch.device) -> str:
    if device.type == "cuda":
        a = torch.cuda.memory_allocated(device) / 1e9
        r = torch.cuda.memory_reserved(device)  / 1e9
        return f"CUDA 显存 — 已分配: {a:.2f} GB / 已预留: {r:.2f} GB"
    if device.type == "mps":
        return "MPS 统一内存，请用系统监控器查看实际占用"
    return "CPU 模式"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable, total = count_parameters(model)
    print(f"  可训练参数: {trainable:,}")
    print(f"  总参数数量: {total:,}")
    print(f"  可训练比例: {100 * trainable / total:.4f}%")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentLogger:
    """CSV 日志 + 训练曲线可视化"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"metrics_{ts}.csv"
        self.records: list[dict] = []
        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = None
        self._t0 = time.time()

    def log(self, step: int, **kwargs) -> None:
        row = {"step": step, "elapsed_sec": f"{time.time()-self._t0:.1f}", **kwargs}
        self.records.append(row)
        if self._writer is None:
            self._writer = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._f.flush()
        parts = [f"step {step}"] + [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in kwargs.items()
        ]
        print("  " + " | ".join(parts))

    def plot(self, save_name: str = "training_curves") -> None:
        if not self.records:
            return
        steps = [r["step"] for r in self.records]
        plot_keys = [k for k in self.records[0] if k not in ("step", "elapsed_sec")]
        fig, axes = plt.subplots(1, len(plot_keys), figsize=(5 * len(plot_keys), 4))
        if len(plot_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, plot_keys):
            vals = []
            vsteps = []
            for r in self.records:
                try:
                    vals.append(float(r.get(key, "")))
                    vsteps.append(r["step"])
                except (TypeError, ValueError):
                    pass
            if vals:
                ax.plot(vsteps, vals, linewidth=1.5)
                ax.set(xlabel="Step", ylabel=key, title=key)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.log_dir / f"{save_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  训练曲线已保存至: {path}")

    def close(self) -> None:
        self._f.close()
        self.plot()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
