# LLM_PostTraining_Learning

大语言模型后训练技术系统学习（LoRA / SFT / DPO / PPO / GRPO / On-Policy Distillation）。

每个模块分为两条轨道并行推进：

| | `learn/` | `engineer/` |
|--|---|---|
| **目标** | 从零理解原理，手写核心算法 | 掌握业界工具，完成真实训练 |
| **运行环境** | Mac M4 Pro（MPS / CPU） | A100/H100（CUDA） |
| **实现方式** | 纯 PyTorch，无高阶依赖 | PEFT + TRL + W&B |
| **README 重点** | 数学推导、梯度流分析 | 超参数选择、显存优化、工具调用 |

---

## 一、工程实践基础

### 1.1 学习路线

```
LoRA ──► SFT ──► DPO ──► PPO ──► GRPO ──► On-Policy Distillation
  │        │       │       │        │              │
参数高效  指令跟随  偏好对齐  完整RLHF  可验证奖励    在线策略蒸馏
微调基础  数据构造  无需RM   三模型    组采样优势    KL教师引导
```

### 1.2 项目结构

```
LLM_PostTraining_Learning/
├── README.md
├── requirements_local.txt     # Mac M4 Pro 依赖
├── requirements_cloud.txt     # 云 GPU 依赖（含 FlashAttention、DeepSpeed）
├── LLM_Basics/                # 跨模块基础技术文档（量化、tokenizer、评估指标等）
├── 01_LoRA/
│   ├── README.md              # 导航：learn/ vs engineer/ 对比
│   ├── learn/                 # 本地从零实现（Mac M4 Pro）
│   │   ├── config.py          # Qwen2.5-0.5B, MPS, 4-bit
│   │   ├── lora_layer.py      # 手写 LoRALinear
│   │   ├── model.py           # inject_lora / freeze_non_lora
│   │   ├── dataset.py         # BELLE + Alpaca，手动 ChatML + label mask
│   │   ├── train.py           # 手动训练循环
│   │   ├── test.py            # 无外部依赖的单元测试（15 cases）
│   │   ├── utils.py           # ExperimentLogger, memory_stats
│   │   └── README.md          # 数学推导、梯度流分析
│   └── engineer/              # 云端集成框架（A100/H100）
│       ├── config.py          # Qwen2.5-7B, CUDA, BF16, r=16
│       ├── dataset.py         # HuggingFace datasets + formatting_func
│       ├── train.py           # TRL SFTTrainer + PEFT LoraConfig
│       ├── test.py            # PPL + 生成质量 + 吞吐量评估
│       ├── utils.py           # WandbLogger, ThroughputLogger
│       └── README.md          # PEFT 用法、超参数指南、W&B 监控
├── 02_SFT/  （学完 LoRA 后开启）
├── 03_DPO/
├── 04_PPO/
├── 05_GRPO/
└── 06_OnPolicyDistillation/
```

### 1.3 模型选择

| 环境 | 模型 | 参数量 | 精度 | 显存需求 |
|------|------|--------|------|----------|
| Mac M4 Pro（learn/） | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | float16 + LoRA | ~2 GB |
| Mac M4 Pro（learn/） | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | float16 + LoRA | ~4 GB |
| 云端 GPU A100（engineer/） | Qwen/Qwen2.5-7B-Instruct | 7B | BF16 + LoRA | ~16 GB |

> Qwen2.5 系列原生支持中英双语，所有模块无需切换模型即可处理中英文任务。

### 1.4 环境配置

#### 本地（Mac M4 Pro）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_local.txt
```

主要依赖：`torch≥2.5.1`（MPS）、`transformers≥4.46.0`、`peft≥0.13.0`、`trl≥0.12.0`、`bitsandbytes≥0.44.0`

#### 云端（A100/H100，CUDA 12.1）

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cloud.txt
pip install flash-attn --no-build-isolation   # 强烈推荐，显存节省 40%
```

主要额外依赖：`flash-attn≥2.6.0`、`deepspeed≥0.15.0`、`wandb≥0.18.0`

### 1.5 各模块快速运行

```bash
# ──── learn 轨道（本地，无需下载模型即可运行测试）────
cd 01_LoRA/learn
python test.py -v                # 15 个单元测试，约 0.3 秒
python train.py --dry-run        # 5 步合成数据冒烟测试
python train.py                  # 完整训练（需下载 Qwen2.5-0.5B）

# ──── engineer 轨道（云端）────
cd 01_LoRA/engineer
export WANDB_API_KEY=your_key
python test.py --dry-run                        # 验证脚本逻辑
python train.py --dry-run                       # 10 步合成数据测试
python train.py                                 # 完整训练
python test.py --adapter ./checkpoints          # 评估已训练 adapter
```

### 1.6 核心工具速查

| 工具 | 用途 | 首次使用模块 |
|------|------|--------------|
| `peft.get_peft_model()` + `LoraConfig` | LoRA 注入 | 01_LoRA/engineer |
| `trl.SFTTrainer` | 指令微调 | 01_LoRA/engineer, 02_SFT |
| `trl.DPOTrainer` | 偏好对齐 | 03_DPO |
| `trl.PPOTrainer` | 强化学习 | 04_PPO |
| `trl.GRPOTrainer` | 组采样强化学习 | 05_GRPO |
| `model.merge_and_unload()` | 合并 LoRA 权重，推理零延迟 | 01_LoRA |
| `gradient_checkpointing_enable()` | 重计算换显存 | 所有 engineer/ |
| `flash_attention_2` | 高效注意力，降显存 40% | 所有 engineer/ |
| W&B `wandb.log()` | 实验追踪 | 所有 engineer/ |

---

## 二、原理记录

### 2.1 学习模块总览

| 模块 | 核心问题 | 解决方案 | 前置知识 |
|------|---------|---------|----------|
| **LoRA** | 全量微调显存/时间代价太高 | 低秩分解 ΔW = BA，只训练 A, B | Transformer 基础 |
| **SFT** | 预训练模型不能跟随指令 | ChatML 格式 + assistant-only loss mask | LoRA |
| **DPO** | 需要奖励模型的 RLHF 太复杂 | 直接从偏好数据导出策略损失，消去 RM | SFT |
| **PPO** | 无法处理复杂奖励信号 | GAE + 裁剪代理目标 + KL 惩罚 | SFT, DPO |
| **GRPO** | PPO 需要 Value 网络，内存重 | 组内平均奖励做基线，消去 Critic | PPO |
| **On-Policy Distillation** | 没有奖励函数时无法用 RL | 教师模型 KL 散度替代奖励 | GRPO |

### 2.2 核心公式一览

**LoRA**（低秩自适应）
```
ΔW = B·A，B ∈ ℝ^{d×r}，A ∈ ℝ^{r×k}，scaling = α/r
初始化：B=0 → 训练开始时 ΔW=0，与预训练模型等价
```

**SFT**（监督微调）
```
只在 assistant token 上计算 loss：
labels[:, :prompt_len] = -100    # mask 掉 instruction 部分
loss = CrossEntropyLoss(logits, labels, ignore_index=-100)
```

**DPO**（直接偏好优化）
```
L_DPO = -E[ log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x)
                      - log π_θ(y_l|x)/π_ref(y_l|x))) ]
从 RLHF 最优解出发，代入 Bradley-Terry 模型，消去奖励 r 推导得到
```

**PPO**（近端策略优化）
```
L_CLIP = E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]
L_total = L_CLIP - β·KL(π_θ || π_ref)
ratio = π_θ(a|s) / π_θ_old(a|s)
```

**GRPO**（组相对策略优化）
```
对同一 prompt 采样 G 个响应：A_i = (r_i - mean(r)) / std(r)
消去 Value 网络，组内统计做基线
```

**On-Policy Distillation**（在线策略蒸馏）
```
L_total = L_NLL + λ · L_KD
L_KD = Σ_t KL(p_teacher(·|x,y<t) || p_student(·|x,y<t))
学生自生成 → 教师打分 → 最小化 token 级 KL 散度
```

### 2.3 各模块 README 内容预览

每个模块的 `learn/README.md` 包含：

1. **动机**：为什么上一个方法不够用
2. **数学推导**：完整公式推导（含中间步骤）
3. **实现解析**：关键代码片段的数学对应关系
4. **梯度流分析**：反向传播路径，哪些参数更新
5. **参数选择直觉**：超参数背后的数学含义
6. **常见失败模式**：理论上解释为什么会崩溃

### 2.4 基础知识索引

详见 [LLM_Basics/README.md](./LLM_Basics/README.md)：

| 技术 | 文档位置 | 首次出现 |
|------|---------|----------|
| BPE Tokenizer | LLM_Basics/tokenization/ | 01_LoRA |
| ChatML / 指令格式 | LLM_Basics/data_formats/ | 02_SFT |
| 4-bit 量化（QLoRA） | LLM_Basics/memory_optimization/ | 01_LoRA |
| FlashAttention / 梯度检查点 | LLM_Basics/memory_optimization/ | 04_PPO |
| PPL / Win Rate 等评估指标 | LLM_Basics/evaluation/ | 02_SFT |
| PEFT 方法全景对比 | LLM_Basics/peft_overview/ | 01_LoRA |
| Bradley-Terry 奖励模型 | LLM_Basics/reward_modeling/ | 03_DPO |
| KL 散度与策略约束 | LLM_Basics/rl_basics/ | 04_PPO |

---

## 三、学习计划

> 板块制：**学完一个模块再开启下一个**，每个模块给出充分的反馈和改进建议。

| 阶段 | 模块 | 状态 | 完成标志 |
|------|------|------|----------|
| 1 | **01_LoRA** | 🔨 构建中 | `python test.py` 15 tests pass；理解 B=0 初始化的意义 |
| 2 | **02_SFT** | ⏳ 待开启 | 0.5B 模型能跟随中英文指令；理解 loss mask 的必要性 |
| 3 | **03_DPO** | ⏳ 待开启 | chosen log-prob > rejected 趋势；从 RLHF 推导出 DPO |
| 4 | **04_PPO** | ⏳ 待开启 | 奖励分数单调上升，KL < 10 |
| 5 | **05_GRPO** | ⏳ 待开启 | GSM8K 准确率从基线提升 |
| 6 | **06_OPD** | ⏳ 待开启 | 学生 PPL 向教师收敛 |
