"""
共用工具函数

提供训练过程中的设备检测、日志记录、显存监控和可视化功能。
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
matplotlib.use("Agg")             # 无头环境兼容
import matplotlib.pyplot as plt


def get_device(preference: str = "auto") -> torch.device:
    """自动选择训练设备：CUDA > MPS > CPU

    Args:
        preference: "auto" | "cuda" | "mps" | "cpu"
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def memory_stats(device: torch.device) -> str:
    """返回当前显存占用信息（人可读字符串）"""
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        return f"CUDA 已分配: {alloc:.2f} GB / 已预留: {reserved:.2f} GB"
    if device.type == "mps":
        # MPS 无法直接查询已分配显存，返回或略说明
        return "MPS 统一内存，请用系统活动监控器查看实际占用"
    return "CPU 模式，无 GPU 显存"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """统计模型可训练参数量和总参数量

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """打印可训练参数比例（LoRA 效果验证关键指标）"""
    trainable, total = count_parameters(model)
    ratio = 100 * trainable / total if total > 0 else 0
    print(f"  可训练参数: {trainable:,}")
    print(f"  总参数数量: {total:,}")
    print(f"  可训练比例: {ratio:.4f}%")


def set_seed(seed: int) -> None:
    """固定随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentLogger:
    """实验记录器：同时输出到控制台、CSV 和 TensorBoard

    与 DL_Learning 中的 ExperimentLogger 保持相同的 CSV 格式，
    延伸了 LLM 特有的 tokens/sec 、VRAM 、reward 等指标字段。
    """

    def __init__(self, log_dir: str, exp_name: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name or timestamp
        self.csv_path = self.log_dir / f"metrics_{self.exp_name}.csv"
        self.records = []
        self._start_time = time.time()

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None              # 第一条记录时初始化

    def log(self, step: int, **kwargs) -> None:
        """记录一条指标。kwargs 支持任意命名的指标字段。"""
        elapsed = time.time() - self._start_time
        record = {"step": step, "elapsed_sec": f"{elapsed:.1f}", **kwargs}
        self.records.append(record)

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(record.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(record)
        self._csv_file.flush()

        # 控制台输出
        parts = [f"step {step}"]
        for k, v in kwargs.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print("  " + " | ".join(parts))

    def plot(self, keys: list[str] = None, save_name: str = "training_curves") -> None:
        """绘制训练曲线并保存为 PNG"""
        if not self.records:
            return
        steps = [r["step"] for r in self.records]
        plot_keys = keys or [
            k for k in self.records[0] if k not in ("step", "elapsed_sec")
        ]

        fig, axes = plt.subplots(1, len(plot_keys), figsize=(6 * len(plot_keys), 4))
        if len(plot_keys) == 1:
            axes = [axes]

        for ax, key in zip(axes, plot_keys):
            vals = []
            valid_steps = []
            for r in self.records:
                if key in r and r[key] is not None:
                    try:
                        vals.append(float(r[key]))
                        valid_steps.append(r["step"])
                    except (TypeError, ValueError):
                        pass
            if vals:
                ax.plot(valid_steps, vals, linewidth=1.5)
                ax.set_xlabel("Step")
                ax.set_ylabel(key)
                ax.set_title(key)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.log_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  训练曲线已保存至: {save_path}")

    def close(self) -> None:
        self._csv_file.close()
        self.plot()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
