"""
LoRA 层从零实现

核心公式：
  W = W₀ + ΔW = W₀ + B · A
  B ∈ ℝ^{d×r}， A ∈ ℝ^{r×k}，r << min(d, k)

实现要点：
  - W₀ 冻结（requires_grad=False），不参与梯度更新
  - A 初始化为高斯分布，B 初始化为零（確保训练开始时 ΔW = 0）
  - 输出缩放因子 α/r，控制适配器对原始权重的干预强度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """LoRA 增强的线性层

    封装一个冻结的 nn.Linear，并加入可训练的低秩旁路。
    训练期间：只更新 A 和 B
    推理期间：可选择将 BA 内合进 W₀ 以消除额外延迟

    Args:
        in_features:  输入维度 k
        out_features: 输出维度 d
        r:            LoRA 秩（常用 4〘16）
        alpha:        缩放因子 α（实际缩放 = α/r）
        dropout:      LoRA 旁路内部的 dropout 概率
        bias:         是否保留原始偏置项
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert r > 0, f"LoRA 秩 r 必须 > 0，当前传入: {r}"

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r          # 缩放因子：控制 ΔW 对输出的干预强度

        # 冻结的原始预训练权重 W₀
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias else None
        )

        # 可训练的 LoRA 旁路：A 和 B
        self.lora_A = nn.Parameter(torch.empty(r, in_features))   # A ∈ ℝ^{r×k}
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B ∈ ℝ^{d×r}

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """A 用 Kaiming 均匀分布初始化，B 初始化为零

        这样设计保证训练开始时 ΔW = B·A = 0，
        模型行为与原始预训练模型完全一致。
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        # 原始权重的线性变换（冻结）
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA 旁路的线性变换（可训练）
        # x -> dropout -> A·x -> B·(A·x) -> α/r 缩放
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # (*, r)
            self.lora_B                                    # (d, r) -> (*, d)
        ) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """将 LoRA 权重内合入 W₀，返回标准 nn.Linear（用于推理加速）

        内合后：
          W_merged = W₀ + α/r * B·A
          预测时无额外的数据流，推理延迟与原始模型相同
        """
        merged_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
        has_bias = self.bias is not None
        new_linear = nn.Linear(
            self.weight.shape[1], self.weight.shape[0], bias=has_bias,
            device=self.weight.device, dtype=self.weight.dtype
        )
        new_linear.weight = nn.Parameter(merged_weight)
        if has_bias:
            new_linear.bias = nn.Parameter(self.bias.clone())
        return new_linear

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )
