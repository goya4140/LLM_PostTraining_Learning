# LLM_PostTraining_Learning

大语言模型后训练技术系统学习（LoRA / SFT / DPO / PPO / GRPO / On-Policy Distillation），每个算法独立成项目，配套完整数学推导、PyTorch 从零实现与 TRL 库对比、中英双语任务支持。

同时支持**本地 Mac M4 Pro**（Qwen2.5-0.5B + 4-bit 量化）和**云端 GPU**（Qwen2.5-7B + full precision）双轨运行。

## 学习路线

```
LoRA → SFT → DPO → PPO → GRPO → On-Policy Distillation
  ↓      ↓     ↓      ↓      ↓           ↓
参数   指令   偏好   完整  可验证       在线策略
高效   跟随   对齐   RLHF  奖励 RL      蒸馏
微调        （无RM） 流水线
```

**为什么这个顺序？**
- **LoRA**：后续所有模块都依赖参数高效微调，必须先掌握
- **SFT**：对齐训练的起点，构建指令跟随能力（DPO/PPO 的前置步骤）
- **DPO**：偏好优化中最简单的形式（无需奖励模型），建立直觉
- **PPO**：完整 RLHF 流水线（SFT → 奖励模型 → PPO），理解 RL 在 LLM 的完整应用
- **GRPO**：PPO 的高效变体（无需 Value 网络），适合可验证奖励任务
- **On-Policy Distillation**：用教师模型替代奖励模型，泛化到任意任务

## 项目结构

| 模块 | 核心技术 | 本地模型（M4 Pro） | 云端模型（A100） | 数据集 |
|------|---------|-----------------|----------------|--------|
| [01_LoRA/](./01_LoRA/) | 低秩权重分解 W=W₀+BA | Qwen2.5-0.5B 4-bit | Qwen2.5-7B | BELLE + Alpaca |
| [02_SFT/](./02_SFT/) | 指令微调 + label mask | Qwen2.5-0.5B 4-bit | Qwen2.5-7B | BELLE + Alpaca + UltraChat |
| [03_DPO/](./03_DPO/) | log-ratio 对比损失 | Qwen2.5-0.5B 4-bit | Qwen2.5-7B | UltraFeedback |
| [04_PPO/](./04_PPO/) | GAE + 裁剪代理目标 | 0.5B policy + 0.5B RM | 7B + 1.5B RM | Anthropic HH |
| [05_GRPO/](./05_GRPO/) | 组内归一化优势估计 | Qwen2.5-0.5B GSM8K | Qwen2.5-7B HumanEval | GSM8K / HumanEval |
| [06_OnPolicyDistillation/](./06_OnPolicyDistillation/) | KL 蒸馏 + 策略梯度 | 1.5B Teacher + 0.5B Student | 7B + 1.5B | 在线自生成 |
| [LLM_Basics/](./LLM_Basics/) | 跨模块基础知识 | — | — | — |

## 模型选择

| 环境 | 模型 | 参数量 | 精度 | 显存需求 |
|------|------|--------|------|----------|
| Mac M4 Pro（本地） | Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 4-bit 量化 | ~2 GB |
| Mac M4 Pro（本地） | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 4-bit 量化 | ~4 GB |
| 云端 GPU（A100 40G） | Qwen/Qwen2.5-7B-Instruct | 7B | BF16 | ~16 GB |

> 所有 Qwen2.5 模型原生支持中英双语，无需切换模型即可处理中文和英文任务。

## 环境配置

### 本地环境（Mac M4 Pro）

| 库 | 版本 | 说明 |
|----|------|------|
| Python | 3.11 | 推荐版本 |
| torch | ≥2.5.1 | MPS 后端（Apple Silicon） |
| transformers | ≥4.46.0 | Qwen2.5 完整支持 |
| peft | ≥0.13.0 | LoRA / QLoRA |
| trl | ≥0.12.0 | SFTTrainer / DPOTrainer / PPOTrainer / GRPOTrainer |
| bitsandbytes | ≥0.44.0 | 4-bit 量化 |

```bash
# 创建虚拟环境（在项目根目录）
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 安装依赖
pip install -r requirements_local.txt
```

### 云端环境（A100/H100，CUDA 12.1）

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch（CUDA 12.1 版本需单独指定 index）
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements_cloud.txt

# Flash Attention 2（显存优化，强烈推荐，需要 CUDA 编译环境）
pip install flash-attn --no-build-isolation
```

## 运行方式

每个模块都支持本地和云端两种运行模式，通过环境变量 `TRAIN_ENV` 切换：

```bash
# 本地模式（M4 Pro，默认）
cd 01_LoRA
python train_scratch.py

# 云端模式
TRAIN_ENV=cloud python train_scratch.py

# 对比 PEFT 库等价实现
python train_peft.py

# 效果评估
python evaluate.py
```

## 哲学：先问为什么，再问怎么做

每个模块 README 以「上一个方法的局限性」开篇，解释为什么需要这个新算法，然后推导完整数学，最后给出两种实现（从零 vs 库）的对比。

参考 [LLM_Basics/](./LLM_Basics/) 了解跨模块共用的基础技术（量化、tokenizer、显存优化、奖励建模等）。
