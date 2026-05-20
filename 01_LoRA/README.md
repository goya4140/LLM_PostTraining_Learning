# LoRA：低秩适配（Low-Rank Adaptation）

> 任务：Qwen2.5 指令微调（中英双语）
> 本地模型：Qwen2.5-0.5B-Instruct（M4 Pro + 4-bit）
> 云端模型：Qwen2.5-7B-Instruct（A100 + BF16）
> 目标：理解 LoRA 为什么能用不到 1% 的可训练参数达到全量微调近似的效果

---

## 目录

1. [为什么需要 LoRA：全参数微调的代价](#1-为什么需要-lora全参数微调的代价)
2. [LoRA 的数学原理](#2-lora-的数学原理)
3. [层选择策略：哪些层加 LoRA](#3-层选择策略哪些层加-lora)
4. [架构总览（ASCII 图）](#4-架构总览)
5. [LoRALinear 源码解析](#5-loralinear-源码解析)
6. [inject\_lora 与 freeze\_non\_lora](#6-inject_lora-与-freeze_non_lora)
7. [Quick Start（M4 Pro）](#7-quick-start-m4-pro)
8. [Quick Start（A100）](#8-quick-start-a100)
9. [超参数调优指南](#9-超参数调优指南)
10. [PEFT 库对比](#10-peft-库对比)
11. [LoRA vs 其它 PEFT 方法](#11-lora-vs-其它-peft-方法)
12. [常见问题排查](#12-常见问题排查)

---

## 1. 为什么需要 LoRA：全参数微调的代价

### 1.1 全参数微调的内存负担

训练期间显存占用来自四个部分：

```
显存占用 ≈ 参数占内存 + 梯度占内存 + 优化器占内存 + 激活値占内存

以 Qwen2.5-7B 为例（BF16 精度）：
  参数内存： 7B × 2 bytes = 14 GB
  梯度内存： 7B × 2 bytes = 14 GB
  Adam 优化器 : 7B × 8 bytes = 56 GB  (一阶和二阶矩阵各一份)
  激活値内存： 随 batch_size 和序列长度变化
  ----------------------------------
  总计最少： 84 GB+（远超单张 A100 40G 的上限）
```

### 1.2 PEFT 的核心思路

大语言模型的知识存储在预训练权重 W₀ 中。微调的本质是将任务相关的知识
“叠加”到 W₀ 上。PEFT 的关键洞察是：

> 微调引入的权重变化 ΔW 的内在秩远低于 W₀ 本身。

Li et al. (2018) 和 Aghajanyan et al. (2021) 的实验证明：**微调后的权重变化 ΔW 居于一个很低维的子空间中**。
这意味着我们可以用一个低秩矩阵来住近似 ΔW，而无需存储完整的高维变化。

---

## 2. LoRA 的数学原理

### 2.1 核心公式

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   W = W₀ + ΔW = W₀ + B · A                            ║
║                                                          ║
║   B ∈ ℝ^{d×r}， A ∈ ℝ^{r×k}                              ║
║   r << min(d, k)                                         ║
║                                                          ║
║   前向传播：h = W₀x + (α/r) · BAx                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### 2.2 参数量密度对比

以 Qwen2.5-7B 的 `q_proj` 层为例（d = k = 3584）：

```
全量微调：W ∈ ℝ^{3584×3584} → 12,845,056 个参数

LoRA (r=8)：
  A ∈ ℝ^{8×3584}   →    28,672 个参数
  B ∈ ℝ^{3584×8} →    28,672 个参数
  共计           →    57,344 个参数

压缩比：57,344 / 12,845,056 ≈ 0.45% —— 仅需全量的 1/224 !
```

### 2.3 初始化设计

```
A ~ N(0, σ²)    「高斯分布」
B = 0             「全零」

训练开始时： ΔW = B·A = 0
即： LoRA 注入后的模型输出与冻结的预训练模型完全一致
读出一个一个重要属性：LoRA 不引入额外的初始化噪声。
```

### 2.4 缩放因子 α/r 的意义

```
h = W₀x + (α/r) · BAx

缩放因子 s = α/r 控制 LoRA 旁路对最终输出的干预强度。

常用配置：
  r=8,  α=16  → s = 2.0
  r=16, α=32  → s = 2.0
  r=64, α=128 → s = 2.0

保持 α/r = 2.0 不变，改变 r 时无需重新调达学习率。
这是一个实用技巧：大多数实验中直接设置 alpha = 2*r。
```

### 2.5 内合公式（推理加速）

```
训练完成后，可将 LoRA 权重内合入 W₀：

W_merged = W₀ + (α/r) · B·A

内合后的模型与原始模型具有完全相同的结构，
推理时无额外计算延迟，与全量微调的模型完全等价。
```

---

## 3. 层选择策略：哪些层加 LoRA

### 3.1 Transformer 内线性层分类

```
Qwen2.5 的每个 Transformer 块包含：

  注意力模块 (Attention)：
  ├── q_proj   (查询投影)   ← 常用
  ├── k_proj   (键投影)     ← 常用
  ├── v_proj   (値投影)     ← 常用  本地小资源建议只加 q_proj + v_proj
  └── o_proj   (输出投影)   ← 常用

  前馈网络 (FFN, Qwen2.5 用 SwiGLU)：
  ├── gate_proj   ← 可选
  ├── up_proj     ← 可选
  └── down_proj   ← 可选

  其它：
  └── lm_head     × 不建议（词表头层）
```

### 3.2 选择原则

| 场景 | 建议的 target_modules | 说明 |
|------|------------------------|------|
| 本地快速验证 | `["q_proj", "v_proj"]` | 最少的可训练参数 |
| 标准微调 | `["q_proj", "k_proj", "v_proj", "o_proj"]` | 不影响 FFN |
| 全量替代 | 所有 7 个投影 | 效果最强，但显存要求更高 |
| 任务特化实验 1 | `["q_proj", "v_proj", "gate_proj"]` | 加入一个 FFN 层 |

---

## 4. 架构总览

```
输入 x: (..., d_model)
│
│  Qwen2.5 Transformer Block (28 层)
│  一个层内部结构（以 q_proj 为例）：
│  ┌────────────────────────────────────────────────────────┐
│  │  x                                                    │
│  │  │                   │ (LoRA 旁路)                │
│  │  │                   │                              │
│  │  │   [W₀, 击结]      A ∈ ℝ^{r×k}  [可训练]       │
│  │  ├───────────────┼   ↓                              │
│  │  │  W₀ x             A x ∈ ℝ^{r}                    │
│  │  │                   ↓                              │
│  │  │               B ∈ ℝ^{d×r}  [可训练]            │
│  │  │                   ↓                              │
│  │  │               B(Ax) ∈ ℝ^d                        │
│  │  │                   ↓ × (α/r)                       │
│  │  │                   ↓                              │
│  │  └───────────────┼───────────────────────────────┘
│  │             ↓ (+加和)
│  │             h = W₀x + (α/r)·BAx
│  └────────────────────────────────────────────────────────┘
│
│  [所有其他参数：层归一、FFN(gate/up/down)、其余注意力头均冻结]
│
↓
输出: (..., d_model)
```

**标识说明：**
- `[冻结]` —— `requires_grad=False`，不参与梯度更新
- `[可训练]` —— `requires_grad=True`，反向传播更新 A 和 B

---

## 5. LoRALinear 源码解析

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16.0, ...):
        # 缩放因子：固定为 alpha/r
        self.scaling = alpha / r          # 本例中 = 16/8 = 2.0

        # 原始权重——冻结
        self.weight = nn.Parameter(..., requires_grad=False)

        # LoRA 旁路——可训练
        self.lora_A = nn.Parameter(torch.empty(r, in_features))   # A
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B = 0 初始

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))     # A 用 Kaiming 均匀
        nn.init.zeros_(self.lora_B)                                 # B 初始化为零

    def forward(self, x):
        base_out = F.linear(x, self.weight, self.bias)             # W₀ x
        lora_out = F.linear(F.linear(self.lora_dropout(x),        # A·x
                                      self.lora_A),
                             self.lora_B) * self.scaling            # B·(A·x)×s
        return base_out + lora_out                                 # 加和
```

**几个常见问题：**

**Q: 为什么 B 初始化为零而不是 A?**
如果 A=0，训练初期所有 LoRA 输入都为零，导致 A 无法利用输入分布自动初始化。
A 用随机初始化吸收不同的输入方向， B=0 保证训练初期 ΔW=0。

**Q: lora_dropout 加在哪个位置?**
在 x 输入进入 A 之前，即对原始输入进行 dropout。这与 Hu et al. 2021 原文一致。

**Q: merge_weights() 的使用时机?**
推理阶段调用，将 LoRA 内合进 W₀ 后，模型结构与全量微调完全相同，推理速度无损失。

---

## 6. inject_lora 与 freeze_non_lora

```python
def inject_lora(model, r, alpha, dropout, target_modules):
    """遍历所有模块，将匹配的 nn.Linear 替换为 LoRALinear"""
    for name, module in list(model.named_modules()):
        if module_name not in target_modules: continue
        if not isinstance(module, nn.Linear): continue

        # 将原始线性层替换为 LoRALinear
        lora_layer = LoRALinear(
            module.in_features, module.out_features,
            r=r, alpha=alpha, dropout=dropout,
        )
        # 复制原始权重（保持量化状态）
        lora_layer.weight = module.weight  # 共享张量，非拷贝

        setattr(parent, attr_name, lora_layer)  # in-place 替换


def freeze_non_lora(model):
    """将所有非 LoRA 参数冻结"""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True    # LoRA 参数开放梯度
        else:
            param.requires_grad = False   # 所有其他层全部冻结
```

**为什么共享张量而非拷贝?**
- 拷贝指针——不占用额外显存（对量化模型尤其重要）
- 反向传播时 LoRALinear 不会为 W₀ 计算梯度（requires_grad=False）

---

## 7. Quick Start（M4 Pro）

### 版本兼容性

| 包 | 版本 | 说明 |
|----|------|-|
| Python | 3.11 | 推荐版本 |
| torch | ≥2.5.1 | MPS 支持 |
| transformers | ≥4.46.0 | Qwen2.5-0.5B 支持 |
| peft | ≥0.13.0 | LoraConfig |
| bitsandbytes | ≥0.44.0 | 4-bit 量化（仅权重加载） |

```bash
# 在项目根目录创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements_local.txt

# 进入 LoRA 目录
cd 01_LoRA

# 运行从零实现（默认本地模式）
python train_scratch.py

# 对比 PEFT 库实现
python train_peft.py

# 效果评估
python evaluate.py
```

### 预期训练输出

```
============================================================
LoRA 微调训练（从零实现）
============================================================
  训练环境 : local
  模型     : Qwen/Qwen2.5-0.5B-Instruct
  设备     : mps
  LoRA r    : 8
  LoRA alpha: 16
  target    : ['q_proj', 'v_proj']

[步骤 1] 加载基座模型...
  MPS 统一内存，请用系统活动监控器查看实际占用

[步骤 2] 注入 LoRA 层...
  LoRA 注入完成：共替换 56 个线性层

[步骤 3] 可训练参数统计:
  可训练参数: 1,146,880
  总参数数量: 494,383,616
  可训练比例: 0.2320%

[步骤 4] 加载数据集...
  BELLE 数据集: 5000 条样本
  训练集: 4500 条 | 验证集: 500 条

============================================================
开始训练
============================================================
[Epoch 01/03]
  step 10 | train_loss=2.1543 | lr=0.000020 | tokens_per_sec=1250.3
  step 20 | train_loss=1.9832 | lr=0.000050 | tokens_per_sec=1280.1
  ...
  验证 loss: 1.7234 | PPL: 5.60
  ✓ 保存最优 adapter（eval loss: 1.7234）
```

---

## 8. Quick Start（A100）

```bash
# 环境准备
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cloud.txt
# 可选但强烈推荐： Flash Attention 2
pip install flash-attn --no-build-isolation

# 云端模式训练
cd 01_LoRA
TRAIN_ENV=cloud python train_scratch.py

# 云端对比 PEFT 库
TRAIN_ENV=cloud python train_peft.py
```

### 预期 A100 显存占用

```
模型: Qwen2.5-7B-Instruct
预训练层权重（BF16）: ~14 GB
LoRA 适配器（r=16, 7 个模块）: ~250 MB
梯度（仅 LoRA）:  ~250 MB
Adam 状态（仅 LoRA）:  ~500 MB
激活値（batch=8, len=2048）: ~8 GB
-----------------------------------------
合计概算： ~23 GB（A100 40G 内容达 58%）
```

---

## 9. 超参数调优指南

### 9.1 核心超参数影响

| 参数 | 默认小/云 | 调整建议 | 说明 |
|------|---------|---------|------|
| `lora_r` | 8 / 16 | 4, 8, 16, 64 | 秩越高容量越强，参数量越多 |
| `lora_alpha` | 16 / 32 | 固定为 2×r | 保持 alpha/r = 2.0 简化调参 |
| `lora_dropout` | 0.05 | 0.0~0.1 | 数据小时用 0.05，大时用 0.0 |
| `target_modules` | qv / qkvo+ffn | 见层选择策略 | 加的层越多效果越强，显存越高 |
| `learning_rate` | 2e-4 / 1e-4 | 5e-5 ~ 5e-4 | 比全量微调高 10倍 |
| `batch_size` (eff.) | 32 | 16~128 | 大 batch 收敛更稳定 |
| `max_seq_len` | 512 / 2048 | 和任务需求对齐 | 越长显存占用越高 |

### 9.2 秩 r 的选择指南

```
r=4   — 极少参数，适合简单领域适配或将来展示
k
r=8   — [推荐起点] 大多数指令微调任务的默认选择
r=16  — 较小数据集上提升效果显著；云端训练的默认
r=64  — 对于复杂任务（如数学推理）有帮助
r=128 — 极少情况下使用，接近全量微调

实验结论（Hu et al. 2021）：大多数任务中 r=8 和 r=64 效果相近，
比高秩的收益递减得很快。
```

### 9.3 学习率选择原则

```
LoRA 小参数论证：最优学习率 ∝ 参数量

经验值：
  0.5B 模型 ⇒ lr = 2e-4 ~ 5e-4
  7B 模型  ⇒ lr = 5e-5 ~ 2e-4
  查询类任务 ⇒ 偏大 lr
  创作类任务 ⇒ 偏小 lr
```

### 9.4 常见故障识别

| 现象 | 可能原因 | 解决方案 |
|------|---------|----------|
| 训练 loss 不下降 | lr 太小或 r 太小 | 调大 lr，增大 r |
| 训练忘记印象划
| 训练 loss 越高越快 | lr 太大导致闪烁 | 加 warmup，降低 lr |
| PPL 円形高但生成质量差 | 数据质量差 | 过滤短于 30 字的样本 |
| 产出中英文混杂 | 数据分布不均 | 各语言样本比例平衡 |
| MPS OOM | batch 过大 | 减小 batch_size，增大 gradient_accumulation_steps |

---

## 10. PEFT 库对比

### 10.1 主要差异

| | `train_scratch.py`（手动） | `train_peft.py`（PEFT 库） |
|--|---|---|
| **LoRA 层类型** | `LoRALinear`（自定义） | `peft.Linear`（PEFT 内部） |
| **注入方式** | `inject_lora()`（手动遍历） | `get_peft_model()`（自动） |
| **冻结方式** | `freeze_non_lora()`（手动） | `get_peft_model()`内部自动处理 |
| **保存格式** | `.pt` 字典（自定义） | HuggingFace `adapter_model.bin` |
| **内合方式** | `merge_weights()`（自定义） | `model.merge_and_unload()` |
| **学习效果** | 完全了解内部机制 | 一行代码完成配置 |

### 10.2 PEFT 库实际调用方式

```python
# 应用 LoRA 只需 3 行核心代码
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_r,               # 与手动版完全对应
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 1,146,880 || all params: 494,383,616 || trainable%: 0.2320%

# 保存 adapter
model.save_pretrained("./adapter_output")

# 内合 LoRA 并保存完整模型
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

---

## 11. LoRA vs 其它 PEFT 方法

| 特性 | LoRA | Adapter | Prefix Tuning | IA³ |
|------|------|---------|---------------|------|
| **大概思路** | 低秩 ΔW = BA | 在层间插入小 MLP | 在 KV 前添加可训练前缀 | 对 K、V、FFN 进行元素级缩放 |
| **推理延迟** | 内合后 = 0 | 增加额外层 | 序列变长 | 几乎 0 |
| **参数量** | r×(d+k) | 中间维度两倍 | 前缀长度×d | 每层 3 个向量 |
| **任务适应性** | 强 | 较强 | 中等 | 强 |
| **库支持** | PEFT 第一公民 | 支持 | 支持 | 支持 |
| **推荐场景** | 大多数微调任务 | 多任务并行 | 岑注射类任务 | 高效推理 |

**结论：** LoRA 是目前实验验证最全面、社区最成熟的 PEFT 方法，也是后续所有模块的基础。

---

## 12. 常见问题排查

| 错误 / 现象 | 原因 | 解决方案 |
|------------|------|----------|
| `NotImplementedError: Could not run 'aten::...' on 'mps'` | 某些操作 MPS 不支持 | 在 config.py 将 `device` 改为 `"cpu"` 婡过 |
| `CUDA out of memory` | 云端 batch 过大 | 减小 `batch_size`，增大 `gradient_accumulation_steps` |
| `adapter_state` 加载后 loss 没有下降 | 加载的 adapter 权重与当前模型不匹配 | 确认 `target_modules` 一致 |
| 中英文级微调后英文能力下降 | 训练数据几乎全部是中文 | 加入 10%-20% 的英文数据 |
| 生成重复内容 | 没有使用 `repetition_penalty` | 添加 `repetition_penalty=1.2` |
| LoRA 层注入后 `print_trainable_parameters` 显示 0% | `freeze_non_lora` 调用错误 | 确认 `inject_lora` 在 `freeze_non_lora` 之前调用 |

---

## 参考论文

- **LoRA 原始论文**：Hu et al. (2021) [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **内在维度研究**：Aghajanyan et al. (2021) [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- **QLoRA**：Dettmers et al. (2023) [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **PEFT 库**：https://github.com/huggingface/peft
