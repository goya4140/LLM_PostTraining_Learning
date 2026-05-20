"""
将 LoRA 注入 Qwen2.5 模型
提供: load_base_model / inject_lora / freeze_non_lora / get_lora_state_dict
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_layer import LoRALinear


def load_base_model(config):
    """4-bit 量化加载 Qwen2.5"""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    ) if config.load_in_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if not config.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def inject_lora(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    """将匹配模块名的 nn.Linear 替换为 LoRALinear"""
    replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "LoRALinear":
            continue
        if name.split(".")[-1] not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name, attr = (name.rsplit(".", 1) if "." in name else ("", name))
        parent = model.get_submodule(parent_name) if parent_name else model

        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            r=r, alpha=alpha, dropout=dropout,
            bias=module.bias is not None,
        )
        lora_layer.weight = module.weight
        if module.bias is not None:
            lora_layer.bias = module.bias
        setattr(parent, attr, lora_layer)
        replaced += 1

    print(f"  LoRA 注入完成：替换 {replaced} 个线性层")
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """冻结所有非 LoRA 参数"""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def get_lora_state_dict(model: nn.Module) -> dict:
    """只提取 LoRA 适配器权重（用于轻量级保存）"""
    return {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
