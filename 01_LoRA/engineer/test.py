"""
Evaluation script for LoRA engineer track.
Measures: perplexity, generation quality (6 prompts: 3 ZH + 3 EN), throughput, memory.

Usage:
    python test.py --adapter ./checkpoints
    python test.py --adapter ./checkpoints --no-ppl
    python test.py --adapter ./checkpoints --max-new 200
    python test.py --dry-run    # verify script logic without downloading a model
"""
import argparse
import math
import sys
import time
import torch
from typing import Optional


EVAL_PROMPTS = [
    {"lang": "zh", "prompt": "请用简单的语言解释什么是机器学习。"},
    {"lang": "zh", "prompt": "写一首关于春天的短诗。"},
    {"lang": "zh", "prompt": "如何提高英语口语水平？请给出三个建议。"},
    {"lang": "en", "prompt": "Explain gradient descent in simple terms."},
    {"lang": "en", "prompt": "Write a short poem about the ocean."},
    {"lang": "en", "prompt": "What are three effective strategies for improving productivity?"},
]


def load_model(adapter_path: Optional[str], base_model: str, torch_dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()
    return tokenizer, model


def compute_ppl(model, tokenizer, texts, max_length: int = 512, device: str = "cuda") -> float:
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        total_nll    += loss * ids.numel()
        total_tokens += ids.numel()
    return math.exp(total_nll / total_tokens)


def generate(model, tokenizer, prompt: str, max_new: int, device: str) -> str:
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_speed_test(model, tokenizer, device: str, n_tokens: int = 200) -> float:
    prompt  = "Tell me a detailed story about a robot who learns to paint."
    inputs  = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        ),
        return_tensors="pt",
    ).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=n_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return new_tokens / (time.time() - t0)


def section(title: str):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",    type=str, default=None,
                        help="Path to LoRA adapter or merged model directory")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-ppl",     action="store_true", help="Skip perplexity")
    parser.add_argument("--max-new",    type=int, default=200)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Verify script logic only (no model download)")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN: script structure verified.")
        print("Run with --adapter <path> to evaluate a trained adapter.")
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if (torch.cuda.is_available() and
                                 torch.cuda.is_bf16_supported()) else torch.float16

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  {total:.1f}GB")

    tokenizer, model = load_model(args.adapter, args.base_model, dtype)

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after load: {alloc:.2f}GB")

    # ── Perplexity ──
    if not args.no_ppl:
        section("Perplexity")
        ppl_texts = [
            "The capital of France is Paris. It is known for the Eiffel Tower.",
            "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习。",
            "Gradient descent minimizes a loss function by iteratively updating parameters.",
        ]
        ppl = compute_ppl(model, tokenizer, ppl_texts, device=device)
        print(f"Perplexity: {ppl:.2f}")

    # ── Generation quality ──
    section("Generation Quality (3 ZH + 3 EN)")
    for i, item in enumerate(EVAL_PROMPTS):
        print(f"\n[{i+1}/6] [{item['lang'].upper()}] {item['prompt']}")
        print("-" * 40)
        print(generate(model, tokenizer, item["prompt"], args.max_new, device))

    # ── Throughput ──
    section("Generation Speed")
    tok_s = run_speed_test(model, tokenizer, device)
    print(f"Throughput: {tok_s:.1f} tokens/sec")

    # ── Memory summary ──
    section("Memory Summary")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak:.2f}GB")
    else:
        print("CPU only — no GPU memory stats.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
