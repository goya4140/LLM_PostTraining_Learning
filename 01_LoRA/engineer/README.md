# LoRA — 工程实践版

> **定位**：掉用业界标准工具完成 LoRA 训练，在 A100/H100 云端 GPU 运行。  
> 数学原理版见 [`../learn/`](../learn/README.md)。

---

## 目录

1. [PEFT 库快速上手](#1-peft-库快速上手)
2. [TRL SFTTrainer 参数详解](#2-trl-sfttrainer-参数详解)
3. [超参数选择指南](#3-超参数选择指南)
4. [显存优化技巧](#4-显存优化技巧)
5. [W\&B 监控设置](#5-wb-监控设置)
6. [Quick Start](#6-quick-start)
7. [常见报错与修复](#7-常见报错与修复)

---

## 1. PEFT 库快速上手

### 1.1 安装

```bash
pip install peft trl transformers datasets accelerate
pip install flash-attn --no-build-isolation   # A100/H100 必装
pip install wandb                             # 可选，实验追踪
```

### 1.2 核心 API

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                          # 秩
    lora_alpha=32,                 # 缩放因子（alpha/r = 2）
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,245,537,280 || trainable%: 1.1576
```

### 1.3 保存与加载 adapter

```python
# 保存 adapterï¼只存 LoRA 权重，~100MB）
trainer.save_model("./checkpoints")

# 加载并合并（推理零延迟）
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./checkpoints")
model = model.merge_and_unload()   # 合并进基础模型
```

---

## 2. TRL SFTTrainer 参数详解

```python
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    # 基本训练
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # 有效 batch = 32

    # 优化器
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,               # 带 warmup 的 cosine 调度
    lr_scheduler_type="cosine",

    # 精度（A100+ 推荐 BF16）
    bf16=True,
    fp16=False,

    # 序列处理
    max_seq_length=2048,
    packing=False,                   # True 可提升吞吐量，但需要注意位置编码

    # 记录
    logging_steps=10,
    save_steps=500,
    report_to="wandb",

    # 内存
    gradient_checkpointing=True,     # 用重计算换显存，约节射0-60%
    dataloader_num_workers=4,
    group_by_length=True,            # 一个批内尽量相同长度，减少 padding
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    formatting_func=formatting_func, # 将样本转成单字符串
    tokenizer=tokenizer,
)
```

### `formatting_func` 的作用

SFTTrainer 需要将数据集转为字符串格式。`formatting_func` 是一个函数，接受批数据并返回字符串列表：

```python
def formatting_func(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": inst},
             {"role": "assistant", "content": out}],
            tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
    return texts
```

SFTTrainer 内部自动处理 label mask（通过 `DataCollatorForCompletionOnlyLM`）。

---

## 3. 超参数选择指南

### LoRA 超参数

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `r` | 8–32 | 指令遵从用 8–16，代码/数学用 16–32 |
| `lora_alpha` | `= 2r` | 固定比例，等效于学习率×2 |
| `lora_dropout` | 0.05–0.1 | 数据少时用 0.1，数据充足用 0.05 |
| `target_modules` | q+k+v+o+MLP | 覆盖越多效果越好，但显存更多 |

### 训练超参数

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `learning_rate` | 1e-4–2e-4 | LoRA 可用比全量微调更大的 lr |
| `batch_size` × `grad_accum` | 32–64 | 有效 batch 过小会不稳定 |
| `warmup_ratio` | 0.03–0.1 | ~100-500 步 warmup |
| `num_epochs` | 1–3 | 多于 3 epoch 易过拟合 |
| `max_seq_len` | 1024–2048 | 按数据分布选择，超长无益 |

### 学习率调试经验法则

```
loss 不下降        → lr 太小，试 lr × 2
loss 展震           → lr 太大，试 lr ÷ 2
loss 下降后直线  → 缺少 warmup「始点」收敛
val loss 上升     → 过拟合，减少 epoch / 增大 dropout
```

---

## 4. 显存优化技巧

### A100 80GB 上 Qwen2.5-7B 的典型占用

```
基础模型（BF16）        ~14 GB
LoRA 参数（r=16）        ~0.7 GB
AdamW 优化器状态（×2）  ~1.4 GB
激活值（语序长 2048）  ~8 GB
────────────────────────
Total                   ~24 GB    ✓ 80GB GPU 很宽裕
```

### 如果显存不够

```python
# 1. 梯度检查点（重计算急进※馀，小一半激活值）
model.gradient_checkpointing_enable()
model.config.use_cache = False    # 必须同时关闭

# 2. 缩小 seq_length，显存 ∝ O(seq^2)
max_seq_length=1024    # 2048 → 1024：显存 ×0.25

# 3. FlashAttention 2
attn_implementation="flash_attention_2"    # A100+ 显存下降 40%

# 4. 进一步用 4-bit 基础模型（QLoRA）
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)
```

### 多 GPU 训练（DeepSpeed / FSDP）

```bash
# DeepSpeed ZeRO-2
deepspeed --num_gpus=4 train.py \
    --deepspeed ds_config_zero2.json

# 或者用 accelerate
accelerate launch --config_file fsdp_config.yaml train.py
```

---

## 5. W&B 监控设置

```bash
# 登录
export WANDB_API_KEY=your_key
wandb login

# 在 SFTConfig 中开启
report_to="wandb"
run_name="qwen2.5-7b-lora-r16"
```

### 必要监控的指标

| 指标 | 训练中期期望 | 异常情况 |
|------|-----------|--------|
| `train/loss` | 单调下降 | 震荡或反升 → lr 问题 |
| `train/grad_norm` | 稳定在 1–5 | 高于 10 → 梯度爆炸 |
| `train/learning_rate` | 先升后降 | 平坦说明 warmup 缺失 |
| GPU 内存 | 尽量高 | OOM → 小 batch / GC |

### 感知训练进展的实用质求

```bash
# 定期检查生成质量（而不是只看 loss）
python test.py --adapter checkpoints/checkpoint-500
```

---

## 6. Quick Start

```bash
# 1. 安装依赖
conda create -n lora python=3.11
conda activate lora
pip install -r ../../requirements_cloud.txt

# 2. 设置璯境变量
export WANDB_API_KEY=your_key         # 可选
export HF_TOKEN=your_hf_token         # 如果模型需要授权

# 3. 冒烟起周测试（无需下载模型）
python test.py --dry-run

# 4. 完整训练
python train.py

# 5. 训练后评估
python test.py --adapter checkpoints/ --max-new 200
```

### 训练时长预估（A100 80GB）

| 配置 | 样本量 | 预估时长 |
|------|--------|--------|
| r=16, bs=32, seq=2048, 1 epoch | 50K | ~2.5 小时 |
| r=8, bs=32, seq=1024, 3 epoch | 10K | ~40 分钟 |

---

## 7. 常见报错与修复

**`RuntimeError: Expected all tensors to be on the same device`**  
因：`device_map="auto"` 将模型切分到多 GPU，不应手动 `.to(device)`。解：将所有手动的 tensor 移动删除，依靠 `device_map` 自动安排。

**`CUDA out of memory`**  
手顺对策：
1. `gradient_checkpointing=True`
2. `max_seq_length` 调小一半
3. `per_device_train_batch_size=4`，`gradient_accumulation_steps=8`
4. 换用 QLoRA（`load_in_4bit=True`）

**`TypeError: SFTTrainer.__init__() got unexpected keyword argument`**  
媁生于 trl 版本升级后 API 变化。运行 `pip show trl` 确认版本，本项目需要 `trl>=0.12.0`。

**`ValueError: You should supply an encoding or a list of encodings to this method`**  
因：`formatting_func` 返回了内嵌列表而非平幂列表。确保返回类型为 `List[str]`。
