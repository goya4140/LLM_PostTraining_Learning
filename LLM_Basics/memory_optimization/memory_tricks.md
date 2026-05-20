# 显存优化技巧

> 首次出现：02_SFT（梯度检查点）；04_PPO（多模型显存挑战）

---

## 1. 训练显存来源分析

```
组成部分                     占比说明
──────────────────────────────────────────────────
模型权重（BF16）              2 bytes × 参数量
梯度                         2 bytes × 参数量（与权重等大）
AdamW 优化器状态（fp32）     8 bytes × 参数量（m + v 两个状态 × fp32）
激活值（前向中间结果）        取决于 batch_size × seq_len²
```

**规律**：对于 7B 模型 BF16 训练，权重+梯度+优化器 ≈ 14+14+56 = **84 GB**（不含激活值）。

---

## 2. 梯度检查点（Gradient Checkpointing）

**原理**：不保存所有中间激活值，反向传播时重新计算。

```
正常训练：  前向 → 保存全部激活值 → 反向（直接读取）
           显存占用：O(L × d)，L = 层数

梯度检查点：前向 → 只保存部分激活值（如每隔 √L 层保存一次） → 反向（重新计算被丢弃的部分）
           显存占用：O(√L × d)  ← 减少约 60%
           计算开销：增加约 30%（一次额外前向）
```

```python
model.gradient_checkpointing_enable()
model.config.use_cache = False   # 必须同时关闭 KV Cache（两者冲突）
```

---

## 3. Flash Attention 2

标准 Attention 的显存瓶颈在于需要存储完整的 Q×K^T 矩阵（O(L²) 大小）。

Flash Attention 2（[Dao 2023](https://arxiv.org/abs/2307.08691)）利用分块计算（tiling）和重计算，将 Attention 的显存从 O(L²) 降至 O(L)：

```
标准 Attention：materialise N×N 矩阵到 HBM → O(L²) 读写
Flash Attention：分块在 SRAM 中计算 → 只需 O(L) HBM 读写，10-40% 提速
```

```python
# 安装（需要 CUDA，A100/H100 首选）
pip install flash-attn --no-build-isolation

# 启用
model = AutoModelForCausalLM.from_pretrained(
    name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,   # FA2 要求 BF16 或 FP16
)
```

**限制**：不支持 MPS（Mac）；需要 Ampere 及以上 GPU（A100, H100, A6000 等）。

---

## 4. 梯度累积（Gradient Accumulation）

在显存不足以支撑大 batch 时，通过多步小 batch 累积梯度模拟大 batch 效果。

```python
# 等效 batch_size = per_device_batch × accumulation_steps
batch_size = 8
gradient_accumulation_steps = 4
# 等效 batch = 32

# 手动实现
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / gradient_accumulation_steps
    loss.backward()
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**注意**：`loss` 要除以 `accumulation_steps`，否则梯度被放大。

---

## 5. 序列打包（Packing）

不同长度的样本拼接后一起训练，消除 padding 浪费。

```
正常 batch（batch_size=2, max_len=512）：
  [seq1: 100 tokens] [PAD × 412]
  [seq2: 300 tokens] [PAD × 212]
  利用率：（100+300）/ (512×2) = 39%

Packing：
  [seq1: 100 tokens][seq2: 300 tokens][seq3: 112 tokens 的开头]
  利用率：~100%
```

```python
# TRL SFTTrainer 中启用
SFTConfig(packing=True, max_seq_length=2048)
```

**注意**：打包时需要确保 position_id 重置，否则跨样本的位置编码会出错（TRL 自动处理）。

---

## 6. DeepSpeed ZeRO 多 GPU 策略

```
ZeRO-1：分片优化器状态（显存减少 4x）
ZeRO-2：分片优化器状态 + 梯度（显存减少 8x）
ZeRO-3：分片优化器状态 + 梯度 + 参数（显存减少 N × 8x，N = GPU 数）

实用选择：
  2 GPU    → ZeRO-2 通常足够
  4–8 GPU  → ZeRO-2 或 ZeRO-3
  >8 GPU   → ZeRO-3 + CPU offload
```

```bash
# accelerate 配置
accelerate config  # 交互式配置
accelerate launch train.py
```

---

## 7. 显存优化优先级建议

```
成本     优化手段                  收益
──────────────────────────────────────────
免费     梯度检查点                 ~60% 激活值显存
免费     减小 max_seq_length        O(seq²) 降低
低       Flash Attention 2          ~40% 注意力显存
中       QLoRA（4-bit 基础模型）    ~75% 模型权重显存
高       DeepSpeed ZeRO-3          近线性随 GPU 数扩展
```
