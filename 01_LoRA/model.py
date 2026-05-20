"""
将 LoRA 注入 Qwen2.5 模型

提供以下功能：
  1. 从 HuggingFace 加载 Qwen2.5 基座模型（支持 4-bit 量化）
  2. 将指定线性层替换为 LoRALinear
  3. 冻结非 LoRA 参数
  4. 提供对比 PEFT 库的加载方式
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_layer import LoRALinear


def load_base_model(config):
    """加载基座模型和分词器

    本地配置使用 4-bit 量化降低显存占用，
    云端配置加载完整精度模型。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",   # 训练时右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4 量化
            bnb_4bit_compute_dtype=torch_dtype,  # 计算时使用半精度
            bnb_4bit_use_double_quant=True,      # 嫌套量化，进一步节省显存
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if config.device == "cuda" else "eager",
        )

    return model, tokenizer


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    """将指定模块名中的 nn.Linear 替换为 LoRALinear

    Args:
        model:          原始预训练模型
        r:              LoRA 秩
        alpha:          缩放因子
        dropout:        LoRA 旁路 dropout
        target_modules: 要替换的层名称列表（如 ["q_proj", "v_proj"]）

    Returns:
        注入 LoRA 后的模型（in-place 修改）
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # 检查模块名称是否匹配 target_modules 中任意一个
        module_name = name.split(".")[-1]
        if module_name not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # 获取父模块和属性名
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model

        # 创建 LoRALinear，并将原始权重复制进去
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=module.bias is not None,
        )
        # 复制预训练权重（保持量化状态）
        lora_layer.weight = module.weight
        if module.bias is not None:
            lora_layer.bias = module.bias

        setattr(parent, attr_name, lora_layer)
        replaced += 1

    print(f"  LoRA 注入完成：共替换 {replaced} 个线性层")
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """冻结所有非 LoRA 参数

    LoRA 参数名称含 "lora_A" 或 "lora_B"，其余参数全部冻结。
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_state_dict(model: nn.Module) -> dict:
    """只提取 LoRA 适配器权重，用于轻量级检查点保存

    适配器权重通常只有全量权重的 0.1%，保存和传输成本远低于全量。
    """
    return {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def merge_and_save(model: nn.Module, save_path: str, tokenizer) -> None:
    """将 LoRA 权重内合进 W₀，保存完整模型（推理时无额外延迟）

    内合公式：W_merged = W₀ + (α/r) * B·A
    """
    import copy
    model_copy = copy.deepcopy(model)
    for name, module in list(model_copy.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model_copy.get_submodule(parent_name) if parent_name else model_copy
            setattr(parent, attr_name, module.merge_weights())
    model_copy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  已保存内合 LoRA 的完整模型到: {save_path}")
