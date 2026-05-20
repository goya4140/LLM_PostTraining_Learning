# 量化（Quantization）

> 首次出现：01_LoRA（本地 M4 Pro 运行 0.5B 模型）

## 1. 为什么需要量化

```
模型        FP32 大小   BF16 大小   INT8 大小   NF4 大小
─────────────────────────────────────────────────────
Qwen2.5-0.5B   2 GB      1 GB      0.5 GB    0.25 GB
Qwen2.5-7B    28 GB     14 GB       7 GB      3.5 GB
Qwen2.5-72B  288 GB    144 GB      72 GB      36 GB
```

量化是用更低比特数表示权重（和/或激活值），以换取更小的显存占用，代价是轻微的精度损失。

---

## 2. 数值格式速查

| 格式 | 位宽 | 范围 | 精度 | 典型用途 |
|------|------|------|------|----------|
| **FP32** | 32 bit | ±3.4×10³⁸ | 最高 | 训练基准、梯度累积 |
| **BF16** | 16 bit | ±3.4×10³⁸（同FP32） | 中 | A100/H100 训练首选 |
| **FP16** | 16 bit | ±65504 | 中 | V100、MPS 训练 |
| **INT8** | 8 bit | -128～127 | 低 | 推理加速（bitsandbytes） |
| **NF4** | 4 bit | 正态分位数 | 最低 | QLoRA 训练 / 本地推理 |

### BF16 vs FP16 的关键区别

```
FP16: 1 位符号 + 5 位指数 + 10 位尾数  → 动态范围小，易溢出
BF16: 1 位符号 + 8 位指数 +  7 位尾数  → 与 FP32 相同动态范围，不易溢出
```

**结论**：A100/H100 支持 BF16 硬件加速，优先使用 `torch_dtype=torch.bfloat16`。  
V100/T4/MPS 不支持 BF16，使用 `torch_dtype=torch.float16`（注意梯度溢出风险）。

---

## 3. NF4（Normal Float 4-bit）原理

标准 INT4 将值均匀映射到 -8～7 区间，但 LLM 权重服从**正态分布**，均匀映射浪费了大量表示能力。

NF4 将 4 bit 的 16 个码字按正态分布的分位数分配：

```
传统 INT4（均匀）：-8, -7, -6, ..., 6, 7   ← 极端值很少用到
NF4（正态分位数）：-1.00, -0.69, -0.53, -0.40, -0.28, -0.17, -0.07,  0
                    0.07,  0.17,  0.28,  0.40,  0.53,  0.69,  1.00   ← 集中在 ±1 范围
```

同等比特数下，NF4 对正态分布权重的量化误差比 INT4 低约 20-30%。

---

## 4. QLoRA：4-bit 基础模型 + 高精度 LoRA

QLoRA 的关键思想（[Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)）：

```
基础模型权重  → NF4 量化（冻结，节省显存）
LoRA 适配器  → BF16/FP16（可训练，梯度精确）
前向计算     → 权重反量化到 BF16 后与输入相乘（Double Quantization 再压一级）
```

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True,      # Double Quantization（再省 ~0.4 bits）
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## 5. 推理精度选择指南

```python
# ① A100 / H100（BF16 硬件加速）
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)

# ② V100 / T4 / RTX 系列
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)

# ③ Mac M4 Pro（MPS，不支持 BF16）
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.float16, device_map={"" : "mps"}
)

# ④ 显存极度受限：INT8 推理（精度损失 < 1%）
model = AutoModelForCausalLM.from_pretrained(name, load_in_8bit=True)

# ⑤ 显存最小化：NF4 推理
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)
```

### 精度 vs 显存速查表（Qwen2.5-7B）

| 模式 | 显存 | 速度（A100） | 生成质量 |
|------|------|------------|----------|
| FP32 | 28 GB | 慢 | 基准 |
| BF16 | 14 GB | 快（硬件加速） | ≈ FP32 |
| FP16 | 14 GB | 快 | ≈ FP32 |
| INT8 | 7 GB | 中 | -0.2% PPL |
| NF4 (4-bit) | 3.5 GB | 中 | -1% PPL |

---

## 6. 训练中的精度设置

```python
# SFTConfig / TrainingArguments
bf16=True,   # A100/H100 首选
fp16=False,
# 两者不能同时为 True

# 梯度累积精度（默认 FP32，更稳定）
optim="adamw_torch",        # FP32 优化器
optim="adamw_torch_fused", # 融合版（更快，A100+）
```

### 常见混合精度陷阱

- FP16 训练时 loss 突然变 `nan`：梯度溢出，换用 BF16 或加 `fp16_full_eval=False`
- 4-bit 训练时无法直接调用 `.backward()`：需通过 `prepare_model_for_kbit_training()` 启用

```python
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
```
