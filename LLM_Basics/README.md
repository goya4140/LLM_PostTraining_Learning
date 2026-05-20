# LLM_Basics — 跨模块基础技术速查

后训练学习过程中涉及的基础技术索引。每个条目注明首次出现的模块，方便按需查阅。

## 技术速查表

| 技术 | 分类 | 解决的问题 | 首次出现 | 文档 |
|------|------|-----------|---------|------|
| 4-bit 量化（QLoRA） | 量化 | 7B 模型无法在本地运行（14GB → 3.5GB） | 01_LoRA | [quantization/quantization.md](./quantization/quantization.md) |
| LoRA / PEFT 适配器 | 参数高效 | 全参数微调代价（100% → 0.1% 可训练参数） | 01_LoRA | [peft_adapters/adapter_methods.md](./peft_adapters/adapter_methods.md) |
| Chat Template | 分词 | 不同模型对话格式不兼容 | 02_SFT | [tokenization/tokenization.md](./tokenization/tokenization.md) |
| Label Masking (-100) | 分词 | 对 prompt 计算 loss 引入错误梯度 | 02_SFT | [tokenization/tokenization.md](./tokenization/tokenization.md) |
| Gradient Checkpointing | 显存优化 | 长序列激活值占用大量显存 | 02_SFT | [memory_optimization/memory_tricks.md](./memory_optimization/memory_tricks.md) |
| Flash Attention 2 | 显存优化 | Attention 的 O(L²) 显存占用 | 02_SFT | [memory_optimization/memory_tricks.md](./memory_optimization/memory_tricks.md) |
| Bradley-Terry 模型 | 奖励建模 | 如何将人类偏好转化为标量奖励 | 04_PPO | [reward_modeling/reward_modeling.md](./reward_modeling/reward_modeling.md) |
| KL 散度惩罚 | 对齐训练 | 防止策略在训练中偏离 SFT 基准 | 03_DPO | [rlhf_pipeline/rlhf_overview.md](./rlhf_pipeline/rlhf_overview.md) |
| GAE 优势估计 | RL 基础 | 策略梯度估计方差过高 | 04_PPO | [rlhf_pipeline/rlhf_overview.md](./rlhf_pipeline/rlhf_overview.md) |
| 温度采样 / Top-p | 生成策略 | 确定性输出 vs 多样性探索 | 05_GRPO | [generation_strategies/generation.md](./generation_strategies/generation.md) |
| 序列打包（Packing） | 训练效率 | padding token 浪费计算资源 | 02_SFT | [memory_optimization/memory_tricks.md](./memory_optimization/memory_tricks.md) |
| Verifiable Rewards | 奖励设计 | 训练 RL 需要密集可信的奖励信号 | 05_GRPO | [reward_modeling/reward_modeling.md](./reward_modeling/reward_modeling.md) |

## 子目录说明

```
LLM_Basics/
├── quantization/
│   └── quantization.md        # NF4 量化原理、QLoRA、BitsAndBytes API
├── peft_adapters/
│   └── adapter_methods.md     # LoRA / Prefix Tuning / IA³ 统一对比
├── tokenization/
│   └── tokenization.md        # BPE、chat template、label mask (-100)
├── memory_optimization/
│   └── memory_tricks.md       # 梯度检查点、FlashAttention、ZeRO、梯度累积
├── reward_modeling/
│   └── reward_modeling.md     # Bradley-Terry、比较损失、reward hacking
├── generation_strategies/
│   └── generation.md          # temperature、top-p、top-k、beam search
└── rlhf_pipeline/
    └── rlhf_overview.md       # 完整 RLHF 流程图、各模块连接、论文推荐
```
