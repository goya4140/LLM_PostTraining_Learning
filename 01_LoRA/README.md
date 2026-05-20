# 01_LoRA — LoRA 参数高效微调

本模块分为两个子项目，对应两种学习目的：

| | `learn/`（学习版） | `engineer/`（工程版） |
|--|---|---|
| **目标** | 理解 LoRA 原理，从零实现算法 | 掌握工程工具，完成实际训练任务 |
| **运行环境** | Mac M4 Pro（本地，MPS） | A100/H100（云端 GPU） |
| **实现方式** | 纯 PyTorch，手写 LoRALinear | PEFT + TRL SFTTrainer |
| **主要 README** | 数学推导、梯度流分析 | 超参数选择、工具调用、显存优化 |

## 推荐学习顺序

1. 先读 [learn/README.md](./learn/README.md) — 理解 LoRA 的数学本质
2. 运行 `cd learn && python test.py` — 验证核心属性（无需下载模型，约 2s）
3. 运行 `python train.py` — 在本地 0.5B 模型跑通完整流程
4. 再读 [engineer/README.md](./engineer/README.md) — 了解工程工具与最佳实践
5. 运行 `cd engineer && python train.py` — 在云端 7B 模型完整训练
6. 运行 `python test.py` — 评估训练效果

## 目录结构

```
01_LoRA/
├── learn/                 # 学习版（本地，从零实现）
│   ├── README.md          ← 原理解读（公式驱动）
│   ├── config.py          # 本地配置（M4 Pro）
│   ├── lora_layer.py      # LoRALinear 手工实现
│   ├── model.py           # inject_lora / freeze_non_lora
│   ├── dataset.py         # 数据加载与 label mask
│   ├── train.py           # 手动训练循环
│   ├── test.py            # 单元测试（验证数学属性）
│   └── utils.py           # 日志、可视化工具
└── engineer/              # 工程版（云端，集成框架）
    ├── README.md          ← 实践解读（工具驱动）
    ├── config.py          # 云端配置（A100）
    ├── dataset.py         # HuggingFace datasets 处理
    ├── train.py           # TRL SFTTrainer + PEFT
    ├── test.py            # 评估脚本（PPL + 生成质量）
    └── utils.py           # W&B、显存监控
```
