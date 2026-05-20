# 生成策略（Decoding Strategies）

> 首次出现：05_GRPO（采样多个响应用于组内对比）

---

## 1. 自回归生成基本流程

```
输入：prompt token ids  [t₁, t₂, ..., tₙ]
          ↓ 模型前向
输出：词表上的 logits  [v₁, v₂, ..., v_vocab]
          ↓ 解码策略
选择下一个 token tₙ₊₁
          ↓ 追加到序列，循环直到 <eos>
```

---

## 2. 贪心解码（Greedy Decoding）

```python
next_token = logits.argmax()  # 每步选最高概率 token
```

- **优点**：确定性，可复现，生成最「自信」的路径
- **缺点**：容易陷入重复、局部最优
- **适用**：评估基准（固定输出方便对比）

---

## 3. 温度采样（Temperature Sampling）

```python
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

```
temperature = 1.0  → 原始分布
temperature < 1.0  → 分布更尖锐，更保守（确定性↑）
temperature > 1.0  → 分布更平坦，更随机（多样性↑）
temperature → 0    → 等价于贪心
temperature → ∞    → 均匀分布
```

```python
# generate() 中的设置
model.generate(
    input_ids,
    do_sample=True,      # 必须 True 才启用采样
    temperature=0.7,     # 推理推荐 0.6~0.9
)
```

---

## 4. Top-k 采样

```python
# 只从概率最高的 k 个 token 中采样
top_k_probs, top_k_ids = logits.topk(k)
probs = F.softmax(top_k_probs / temperature)
next_token = top_k_ids[torch.multinomial(probs, 1)]
```

- `k=1`：等价于贪心
- `k=50`：常用默认值
- **缺点**：k 是固定的，不管实际分布有多集中

---

## 5. Top-p（Nucleus Sampling）

动态选择最小的 token 集合，使其概率之和 ≥ p：

```python
# 按概率降序排列，累积概率超过 p 时截断
sorted_probs = probs.sort(descending=True)
cumsum = sorted_probs.cumsum()
nucleus = sorted_probs[cumsum <= p]  # 保留这些 token
```

```
分布集中时：nucleus 可能只有 3~5 个 token（避免随机性）
分布分散时：nucleus 可能有 50+ 个 token（保留多样性）
```

- `p=0.9`：推理推荐默认值
- Top-p 通常比 Top-k 效果更稳定

---

## 6. 束搜索（Beam Search）

维护 num_beams 条候选序列，最终选择整体概率最高的：

```python
model.generate(input_ids, num_beams=4, early_stopping=True)
```

- **优点**：生成「全局最优」序列，PPL 更低
- **缺点**：不适合开放域生成（输出单调），速度慢
- **适用**：翻译、摘要等有标准答案的任务

---

## 7. RL 训练中的生成策略

不同阶段对生成策略的要求不同：

| 阶段 | 策略 | 原因 |
|------|------|------|
| **GRPO 组采样** | `do_sample=True, temperature=0.7~1.0` | 需要多样化的响应用于组内对比 |
| **PPO rollout** | `do_sample=True, temperature=1.0` | 保持策略分布，不偏置采样 |
| **DPO 评估** | `do_sample=False`（贪心） | 固定输出，方便对比 SFT 基线 |
| **最终推理** | `do_sample=True, temperature=0.6, top_p=0.9` | 平衡质量与多样性 |

```python
# Qwen2.5 官方推荐推理参数
model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
)
```
