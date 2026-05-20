# RLHF 流水线总览

> 首次出现：03_DPO（理解 DPO 是 RLHF 的简化版）；04_PPO（完整实现）

---

## 1. 完整 RLHF 三阶段

```
阶段 1: 监督微调（SFT）
  预训练模型 → 用高质量指令数据微调 → SFT 模型
  目标：让模型学会遵循指令格式

阶段 2: 奖励建模（RM Training）
  SFT 模型 → 加 Value Head → 用偏好对数据训练 → 奖励模型
  数据格式：(prompt, chosen, rejected)  
  目标：学习 r(x, y_w) > r(x, y_l)

阶段 3: 强化学习微调（RL Fine-tuning）
  SFT 模型（策略）+ 奖励模型 → PPO 优化
  目标：max E[r(x,y)] - β·KL(π_θ || π_ref)
```

```
四个模型同时存在于 PPO 中：
┌─────────────────┐    ┌─────────────────┐
│  Policy Model   │    │  Reference Model│
│  (π_θ, 训练)   │    │  (π_ref, 冻结)  │
└────────┬────────┘    └────────┬────────┘
         │  生成回复            │  计算 KL
         └───────────┬──────────┘
                     ↓
┌─────────────────┐    ┌─────────────────┐
│  Reward Model   │    │  Value Model    │
│  (r, 冻结)      │    │  (V, 与policy   │
│  打分           │    │   共享主干)     │
└─────────────────┘    └─────────────────┘
```

---

## 2. PPO 训练循环

```
每次 rollout:
  1. 用 Policy 对 prompt 采样 y ~ π_θ(·|x)
  2. 用 Reward Model 打分：r = RM(x, y)
  3. 加入 KL 惩罚：r_adjusted = r - β·KL_token
  4. 用 GAE 估计优势：A = Σ (γλ)^t · δ_t
  5. 用 PPO-Clip 更新 Policy：
     L = E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)]
  6. 更新 Value Head（降低方差）
```

**GAE（Generalized Advantage Estimation）**：
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)    ← TD 误差
A_t = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}   ← 加权平均

λ=0：纯 TD（低方差，高偏差）
λ=1：纯 MC（高方差，低偏差）
λ=0.95：常用默认值
```

---

## 3. DPO 如何简化 RLHF

DPO 的关键洞察：RLHF 最优策略有解析解，可以直接从偏好数据学习。

```
RLHF 最优解（理论上）：
  π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)

代入 Bradley-Terry，消去 r：
  L_DPO = -E[log σ(β · log (π_θ(y_w|x)/π_ref(y_w|x))
                       - β · log (π_θ(y_l|x)/π_ref(y_l|x)))]
```

```
RLHF (PPO)                    DPO
──────────────────────────────────────
需要 SFT + RM + Policy + Value    只需 SFT + Policy（π_ref=SFT）
在线采样（expensive）             离线训练（efficient）
4 个模型同时在 GPU 上             2 个模型（policy + frozen ref）
训练不稳定                        训练稳定
可以持续改进                      受限于偏好数据质量上限
```

---

## 4. GRPO 进一步简化

```
DPO → 还需要 reference model + 偏好对数据
GRPO → 用组内统计替代 Value model，用可验证奖励替代 RM

GRPO 只需要：Policy 模型 + 可验证奖励函数
      消去了：Value Model + Reward Model + Reference Model（optional）
```

```
GRPO 对同一 prompt 采样 G 个响应：
  r = [r₁, r₂, ..., r_G]
  A_i = (r_i - mean(r)) / std(r)   ← 用组内统计做归一化
```

---

## 5. 对齐方法演进时间线

```
2017  PPO（Schulman et al.）         → 通用 RL 算法
2020  InstructGPT（OpenAI）         → PPO + 人类反馈 = RLHF
2022  RLHF + ChatGPT                → 大规模商业化
2023  DPO（Rafailov et al.）        → 绕开奖励模型
2023  QLoRA（Dettmers et al.）      → 消费级 GPU 运行 65B
2024  GRPO（DeepSeek-R1）           → 组采样替代 Value 网络
2024  On-Policy Distillation        → 教师模型替代奖励模型
```

---

## 6. 关键超参数 β 的含义

`β` 出现在所有对齐方法中，含义一致：**KL 散度惩罚强度**。

```
β = 0   → 完全不惩罚 KL，策略随意偏离 SFT
         → 模型崩溃（reward hacking）

β = ∞   → KL 惩罚无穷大，策略不更新
         → 等价于 SFT 基线

β ≈ 0.1 → DPO 常用值，适度保守
β ≈ 0.01→ PPO 中的 KL 系数（token 级别，比 sequence 级别小）
```
