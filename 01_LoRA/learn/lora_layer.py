"""
LoRA 层从零实现

核心公式：
  W = W₀ + ΔW = W₀ + B · A
  B ∈ ℝ^{d×r}， A ∈ ℝ^{r×k}，r << min(d, k)

实现要点：
  - W₀ 冻结（requires_grad=False），不参与梯度更新
  - A 初始化为 Kaiming 均匀分布，B 初始化为零（训练开始时 ΔW = 0）
  - 输出缩放因子 α/r，控制适配器对原始权重的干预强度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA 增强的线性层

    封装一个冻结的 nn.Linear，并加入可训练的低秩旁路。
    训练期间：只更新 A 和 B
    推理期间：可选择将 BA 内合进 W₀ 以消除额外延迟
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

        self.r       = r
        self.alpha   = alpha
        self.scaling = alpha / r   # 缩放因子：控制 ΔW 对输出的干预强度

        # 冻结的原始预训练权重 W₀
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias   = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        )

        # 可训练的 LoRA 旁路
        self.lora_A = nn.Parameter(torch.empty(r, in_features))    # A ∈ ℝ^{r×k}
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # B ∈ ℝ^{d×r}——全零
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))      # A 用 Kaiming 均匀
        # B 已在初始化时设为零（确保训练开始时 ΔW = 0）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., k) -> (..., d)
        base_out = F.linear(x, self.weight, self.bias)              # W₀ x
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),            # A x: (..., r)
            self.lora_B                                             # B(Ax): (..., d)
        ) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """将 LoRA 内合入 W₀，返回标准 nn.Linear（用于推理加速）

        W_merged = W₀ + (α/r) · B·A
        """
        merged = self.weight + (self.lora_B @ self.lora_A) * self.scaling
        new_layer = nn.Linear(
            self.weight.shape[1], self.weight.shape[0],
            bias=(self.bias is not None),
            device=self.weight.device, dtype=self.weight.dtype
        )
        new_layer.weight = nn.Parameter(merged)
        if self.bias is not None:
            new_layer.bias = nn.Parameter(self.bias.clone())
        return new_layer

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, α={self.alpha}, scaling={self.scaling:.2f}")
