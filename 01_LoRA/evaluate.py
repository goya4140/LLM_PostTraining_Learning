"""
LoRA 效果评估

评估维度：
  1. 参数效率 — 可训练参数比例（LoRA vs 全量）
  2. 模型容量 — 训练前 vs 训练后的 loss 和 PPL
  3. 生成质量 — 同一组提示词的回复对比（基座模型 vs LoRA 微调）

运行：
  python evaluate.py --adapter ./checkpoints/lora_local/best_adapter.pt
"""

import argparse
import math
import torch
from tqdm import tqdm

from config import CONFIG
from model import load_base_model, inject_lora, freeze_non_lora
from dataset import get_dataloaders
from utils import get_device, print_trainable_parameters, set_seed


CHINESE_PROMPTS = [
    "请用中文解释什么是机器学习？",
    "写一首关于秋天的短诗。",
    "如何提高 Python 代码的性能？请给出三个建议。",
]

ENGLISH_PROMPTS = [
    "Explain the concept of gradient descent in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and JavaScript?",
]


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int = 200) -> str:
    """对单个 prompt 生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


@torch.no_grad()
def compute_ppl(model, loader, device) -> float:
    """计算验证集上的 Perplexity"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="  计算 PPL", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def compare_generation(base_model, lora_model, tokenizer, device) -> None:
    """对比基座模型与 LoRA 微调模型的生成质量"""
    all_prompts = CHINESE_PROMPTS + ENGLISH_PROMPTS
    for i, prompt in enumerate(all_prompts):
        lang = "中文" if i < len(CHINESE_PROMPTS) else "英文"
        print(f"\n{'=' * 60}")
        print(f"[提示 {i+1}/{len(all_prompts)}] ({lang})")
        print(f"Q: {prompt}")
        print("-" * 40)

        base_resp = generate_response(base_model, tokenizer, prompt, device)
        print(f"[基座模型] {base_resp[:300]}..." if len(base_resp) > 300 else f"[基座模型] {base_resp}")
        print("-" * 40)

        lora_resp = generate_response(lora_model, tokenizer, prompt, device)
        print(f"[LoRA 微调] {lora_resp[:300]}..." if len(lora_resp) > 300 else f"[LoRA 微调] {lora_resp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="LoRA adapter 路径（.pt 文件）")
    parser.add_argument("--skip_ppl", action="store_true", help="跳过 PPL 计算，仅注重生成对比")
    args = parser.parse_args()

    config = CONFIG()
    set_seed(config.seed)
    device = get_device(config.device)

    print("=" * 60)
    print("LoRA 评估")
    print("=" * 60)

    # 加载基座模型
    print("\n[基座模型]")
    base_model, tokenizer = load_base_model(config)
    trainable, total = sum(p.numel() for p in base_model.parameters() if p.requires_grad), sum(p.numel() for p in base_model.parameters())
    print(f"  参数总量: {total:,}")

    # 加载 LoRA 模型
    print("\n[LoRA 模型]")
    lora_model, _ = load_base_model(config)
    lora_model = inject_lora(lora_model, config.lora_r, config.lora_alpha, config.lora_dropout, config.target_modules)
    freeze_non_lora(lora_model)
    print("  LoRA 注入后的参数情况:")
    print_trainable_parameters(lora_model)

    # 加载 adapter 权重
    adapter_path = args.adapter or f"{config.save_dir}/best_adapter.pt"
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        adapter_state = torch.load(adapter_path, map_location=device)
    else:
        adapter_state = torch.load(adapter_path, map_location="cpu")
    missing, unexpected = lora_model.load_state_dict(adapter_state, strict=False)
    print(f"  Adapter 加载: 缺少 {len(missing)} 个, 多余 {len(unexpected)} 个")

    # PPL 对比
    if not args.skip_ppl:
        print("\n[计算 PPL]")
        _, eval_loader = get_dataloaders(config, tokenizer)
        base_ppl = compute_ppl(base_model, eval_loader, device)
        lora_ppl = compute_ppl(lora_model, eval_loader, device)
        print(f"  基座模型 PPL : {base_ppl:.2f}")
        print(f"  LoRA 微调 PPL: {lora_ppl:.2f}")
        print(f"  PPL 降低幅度  : {(base_ppl - lora_ppl) / base_ppl * 100:.1f}%")

    # 生成对比
    print("\n[生成质量对比]")
    compare_generation(base_model, lora_model, tokenizer, device)


if __name__ == "__main__":
    main()
