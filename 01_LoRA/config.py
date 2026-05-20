"""
LoRA 训练配置

通过环境变量 TRAIN_ENV 切换本地和云端配置：
    TRAIN_ENV=local python train_scratch.py    # 默认，Mac M4 Pro
    TRAIN_ENV=cloud python train_scratch.py    # 云端 GPU
"""

import os

ENV = os.environ.get("TRAIN_ENV", "local")


class LocalConfig:
    """Mac M4 Pro 配置：Qwen2.5-0.5B + 4-bit 量化 + MPS"""

    # 模型
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps"
    load_in_4bit = True
    torch_dtype = "float16"           # MPS 上不支持 bfloat16

    # LoRA 超参数
    lora_r = 8                        # 秩 r：权重分解维度
    lora_alpha = 16                   # 缩放因子 α（实际缩放 = α/r = 2.0）
    lora_dropout = 0.05               # LoRA 适配器内部 dropout
    target_modules = ["q_proj", "v_proj"]  # 只加在 QV 上，本地节省计算

    # 训练
    batch_size = 4
    gradient_accumulation_steps = 8   # 等效 batch_size = 32
    max_seq_len = 512
    learning_rate = 2e-4
    num_epochs = 3
    warmup_ratio = 0.03               # 前 3% 的 step 做线性 warmup
    weight_decay = 0.01
    max_grad_norm = 1.0               # 梯度裁剪阈値

    # 数据
    dataset_name = "belle"            # belle | alpaca | belle+alpaca
    max_train_samples = 5000          # 本地用小子集快速验证
    max_eval_samples = 500

    # 保存
    save_dir = "./checkpoints/lora_local"
    log_dir = "./logs/lora_local"
    log_interval = 10                 # 每 10 步记录一次
    save_interval = 500               # 每 500 步保存一次检查点
    seed = 42


class CloudConfig:
    """A100/H100 配置：Qwen2.5-7B + BF16 + CUDA"""

    # 模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda"
    load_in_4bit = False
    torch_dtype = "bfloat16"          # A100 原生支持 bfloat16

    # LoRA 超参数
    lora_r = 16                       # 云端用更大的秩，容量更强
    lora_alpha = 32                   # 缩放因子 α（缩放 = α/r = 2.0，与本地相同）
    lora_dropout = 0.05
    target_modules = [              # 全量加 LoRA，云端资源充足
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # 训练
    batch_size = 8
    gradient_accumulation_steps = 4   # 等效 batch_size = 32
    max_seq_len = 2048
    learning_rate = 1e-4
    num_epochs = 3
    warmup_ratio = 0.03
    weight_decay = 0.01
    max_grad_norm = 1.0

    # 数据
    dataset_name = "belle+alpaca"     # 完整双语数据集
    max_train_samples = None          # 使用完整数据集
    max_eval_samples = 2000

    # 保存
    save_dir = "./checkpoints/lora_cloud"
    log_dir = "./logs/lora_cloud"
    log_interval = 50
    save_interval = 1000
    seed = 42


CONFIG = LocalConfig if ENV == "local" else CloudConfig
