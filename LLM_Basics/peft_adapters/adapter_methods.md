# PEFT 适配器方法全景

> 首次出现：01_LoRA  
> 问题：为什么 LoRA 成为主流？其他方法有哪些权衡？

---

## 1. 方法速览

| 方法 | 可训练参数位置 | 推理开销 | 参数量 | 论文 |
|------|-------------|---------|--------|------|
| **Full FT** | 所有参数 | 无 | 100% | — |
| **Adapter** | 每层后插入小 MLP | +2 层延迟 | 0.5–3% | Houlsby 2019 |
| **Prefix-Tuning** | 输入前置可训练向量 | 额外 KV Cache | 0.01–0.1% | Li & Liang 2021 |
| **Prompt Tuning** | 输入 token embedding | 极小 | 0.001% | Lester et al. 2021 |
| **LoRA** | 注意力/MLP 权重低秩矩阵 | 合并后为零 | 0.1–1% | Hu et al. 2021 |
| **IA³** | 激活值缩放向量 | 合并后为零 | 0.01% | Liu et al. 2022 |
| **DoRA** | LoRA + 方向分解 | 合并后为零 | 0.1–1% | Liu et al. 2024 |

---

## 2. Adapter（Houlsby 2019）

```
原始结构：  Self-Attn → Add&Norm → FFN → Add&Norm
Adapter 后：Self-Attn → Add&Norm → [Adapter] → FFN → Add&Norm → [Adapter]

Adapter 内部：
  Linear(d → r)  [瓶颈层，r << d]
  激活函数
  Linear(r → d)  [还原维度]
  残差连接
```

**优点**：模块化，可为不同任务保留独立 Adapter，切换无需重载基础模型  
**缺点**：每次前向要额外计算 2 个线性层，无法合并，有推理延迟

---

## 3. Prefix-Tuning（Li & Liang 2021）

在每一层的 K、V 序列前面拼接可训练的「前缀」向量，模型只看到这些软提示，不改变任何权重。

```python
# PEFT 实现
from peft import PrefixTuningConfig
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,   # 前缀 token 数量
)
```

**优点**：参数极少，适合生成任务；不修改模型结构  
**缺点**：推理时需维护额外 KV Cache；对长序列效果下降

---

## 4. LoRA（Hu et al. 2021）★ 主流选择

见 [01_LoRA/learn/README.md](../../01_LoRA/learn/README.md) 完整推导。

核心优势：
- 训练完后可将 ΔW = BA 合并进 W₀，**推理零开销**
- 支持多 LoRA 动态切换（不同任务加载不同 BA，共享 W₀）
- 参数量最优比：0.1% 参数量，接近全量微调效果

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model  = get_peft_model(model, config)
```

---

## 5. IA³（Liu et al. 2022）

在注意力的 K、V 和 FFN 的激活值上各乘一个可训练的**缩放向量** `l`（element-wise）：

```
K_new = l_k ⊙ K
V_new = l_v ⊙ V
FFN_out = l_ff ⊙ FFN(x)
```

**优点**：参数量极少（约 0.01%），Few-Shot 场景效果好  
**缺点**：表达能力有限，复杂任务效果不如 LoRA

---

## 6. 选择建议

```
场景                            推荐方法
────────────────────────────────────────────
通用指令微调（SFT）              LoRA（r=8~16）
偏好对齐（DPO/PPO）             LoRA（r=16~32）
多任务共享基础模型               Adapter
极少参数（prompt 调整）          Prefix-Tuning / IA³
追求最终精度                     Full Fine-tuning
本地 4-bit 设备                  QLoRA（NF4 + LoRA）
```
