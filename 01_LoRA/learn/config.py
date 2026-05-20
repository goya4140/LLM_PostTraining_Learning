"""
本地训练配置（Mac M4 Pro）

学习版专用——只有一套本地配置，参数较小以便在本地快速迭代。
每个参数都附有注释说明其含义，配合 README.md 理解数学意义。
"""


class Config:
    # 模型（选用最小的 0.5B 方便本地学习）
    model_name  = "Qwen/Qwen2.5-0.5B-Instruct"
    device      = "mps"        # M4 Pro: "mps" | Intel Mac/Linux: "cpu"
    load_in_4bit = True        # 4-bit 量化节省统一内存（见 LLM_Basics/quantization/）
    torch_dtype = "float16"    # MPS 不支持 bfloat16

    # LoRA 超参数（每个参数的含义见 learn/README.md 第 2 节）
    lora_r       = 8           # 秩 r：低秩矩阵维度，控制适配器容量
    lora_alpha   = 16          # 缩放因子 α（建议保持 α = 2r）
    lora_dropout = 0.05        # LoRA 旁路 dropout（正则化）
    target_modules = ["q_proj", "v_proj"]  # 只注入 QV（本地节省计算）

    # 训练超参数
    batch_size                 = 4
    gradient_accumulation_steps = 8    # 等效 effective_batch = 32
    max_seq_len                = 512
    learning_rate              = 2e-4
    num_epochs                 = 3
    warmup_ratio               = 0.03  # 前 3% 的 step 做线性 warmup
    weight_decay               = 0.01
    max_grad_norm              = 1.0   # 梯度裁剪阈値

    # 数据（小子集用于本地快速验证）
    dataset_name        = "belle"      # belle | alpaca | belle+alpaca
    max_train_samples   = 5000
    max_eval_samples    = 500
    seed                = 42

    # 保存路径
    save_dir     = "./checkpoints"
    log_dir      = "./logs"
    log_interval = 10                  # 每 10 步记录一次
