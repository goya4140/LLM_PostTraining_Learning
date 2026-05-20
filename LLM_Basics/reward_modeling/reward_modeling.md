# 奖励建模（Reward Modeling）

> 首次出现：04_PPO（Bradley-Terry 奖励模型训练）；05_GRPO（可验证奖励替代）

---

## 1. 为什么需要奖励模型

RLHF 的核心是将「人类偏好」转化为可微的标量奖励信号。

```
人类标注：(prompt x, 回复 y_w, 回复 y_l) → 人类更喜欢 y_w
           ↓
奖励模型：r(x, y) ∈ ℝ  ← 标量，能对任意回复打分
           ↓
PPO：最大化 E[r(x, y)] - β·KL(π || π_ref)
```

---

## 2. Bradley-Terry 模型

将「A 比 B 好」的偏好建模为概率：

```
p(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中 σ 是 sigmoid 函数，r 是奖励函数（标量）。
```

**训练目标**（最大化偏好对的正确概率）：

```
L_BT = -E[ log σ(r(x, y_w) - r(x, y_l)) ]
```

直觉：模型应给 preferred response 打更高分，两者差距越大越好。

---

## 3. 奖励模型架构

在 SFT 模型基础上加一个线性「Value Head」：

```
基础模型（SFT checkpoint）：  token → hidden states
Value Head：                  last_hidden_state → Linear(d, 1) → scalar reward
```

```python
from transformers import AutoModel
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # 取最后一个 token
        reward = self.value_head(last_hidden).squeeze(-1)   # [batch]
        return reward
```

---

## 4. 奖励 Hacking（Reward Overoptimization）

PPO 训练时模型可能利用奖励模型的漏洞获得高分，而实际质量下降（Goodhart's Law）：

```
策略发现：用大量重复词能欺骗 RM 打高分
实际输出："The answer is great great great great great..."
```

**缓解方法**：
- **KL 惩罚**：`L_total = r(x, y) - β × KL(π_θ || π_ref)`，防止偏离太远
- **KL 上限**：当 KL > threshold 时截断训练
- **定期更新 RM**（Iterative RLHF）

```python
# PPO 中的 KL 惩罚
kl_penalty = log_prob_policy - log_prob_ref  # per token KL
reward = reward_from_rm - beta * kl_penalty
```

---

## 5. 可验证奖励（Verifiable Rewards）

GRPO 和 DeepSeek-R1 的关键创新：用**可程序验证**的答案替代奖励模型。

```python
# 数学题：答案完全可验证
def math_reward(response: str, ground_truth: str) -> float:
    predicted = extract_answer(response)  # 抽取 \boxed{...} 内容
    return 1.0 if predicted == ground_truth else 0.0

# 代码题：执行测试用例验证
def code_reward(response: str, test_cases: list) -> float:
    code = extract_code_block(response)
    passed = run_tests(code, test_cases)
    return passed / len(test_cases)

# 格式奖励（鼓励思维链）
def format_reward(response: str) -> float:
    has_think = "<think>" in response and "</think>" in response
    return 0.1 if has_think else 0.0
```

**可验证奖励 vs 奖励模型**：

| | 奖励模型 | 可验证奖励 |
|--|---------|----------|
| 适用场景 | 主观质量（写作、对话） | 有客观答案（数学、代码） |
| 奖励 Hacking 风险 | 高 | 极低 |
| 需要标注数据 | 需要大量偏好对 | 只需题目+答案 |
| 奖励稠密性 | 连续值 | 通常稀疏（0/1） |
