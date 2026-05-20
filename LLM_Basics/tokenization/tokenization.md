# Tokenization 与数据格式

> 首次出现：01_LoRA（理解输入格式）；02_SFT（chat template + label mask）

---

## 1. BPE 算法原理

BPE（Byte-Pair Encoding）是 LLM 最常用的分词算法，也是 Qwen2.5 的 tokenizer 基础。

**训练过程**：
1. 将文本拆成字符序列（字节级别）
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对 → 生成新 token
4. 重复 N 次直到词表大小达到目标（通常 32K–200K）

```
初始：  h e l l o _ w o r l d
合并1（ll → ll）：h e ll o _ w o r l d  
合并2（lo → lo）：h e ll lo _ w o r l d  ← "hello" 变为 [h, e, ll, lo]
合并3（he → he）：he ll lo _ w o r l d
...
最终 vocab 中可能有 "hello"、"world" 作为整体 token
```

**为什么用 BPE 而不是按空格分词？**
- 处理未登录词（OOV）：拆成子词而非 `[UNK]`
- 跨语言通用：中文、英文、代码共用同一词表
- 控制词表大小与序列长度的平衡

---

## 2. Chat Template

不同模型使用不同的对话格式，格式不对会导致模型无法正确理解对话角色。

### Qwen2.5 ChatML 格式

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{用户输入}<|im_end|>
<|im_start|>assistant
{模型回复}<|im_end|>
```

特殊 token：`<|im_start|>`（151644）、`<|im_end|>`（151645）

### 其他常见格式

```
# Llama-3 格式
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# Gemma 格式
<bos><start_of_turn>user
{user}<end_of_turn>
<start_of_turn>model
{assistant}<end_of_turn>
```

### 用 Transformers 自动应用模板

```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA is..."},
]

# 训练用（包含 assistant 回复）
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)

# 推理用（让模型续写）
text = tokenizer.apply_chat_template(
    messages[:-1], tokenize=False, add_generation_prompt=True
)
# 末尾自动加上 "<|im_start|>assistant\n"
```

---

## 3. Label Masking（-100）

**核心问题**：SFT 时为什么不对 prompt 计算 loss？

如果对整个序列（system + user + assistant）都算 loss，模型会学习预测 prompt 中的每个 token，这会引入两个问题：
1. 浪费梯度信号（prompt 是确定的，不需要学习生成它）
2. 模型可能「聪明地」记忆 prompt 而非学习生成正确回复

**解决方案**：将 prompt token 的 label 设为 `-100`，`CrossEntropyLoss` 会自动跳过这些位置（`ignore_index=-100`）。

```python
# 手动实现（01_LoRA/learn/dataset.py 的逻辑）
labels = input_ids.clone()      # [seq_len]
prompt_len = len(prompt_ids)    # 仅包含 system + user 部分
labels[:prompt_len] = -100      # mask 掉 prompt

loss = F.cross_entropy(logits.view(-1, vocab_size),
                       labels.view(-1),
                       ignore_index=-100)

# TRL 自动处理（01_LoRA/engineer/train.py）
# DataCollatorForCompletionOnlyLM 自动识别 assistant 回复开始位置
```

---

## 4. Padding 策略

```
左 padding（推理推荐）：
  [PAD] [PAD] [PAD] token1 token2 token3
  → 最后一个 token 是序列末尾，用于 next-token prediction 准确

右 padding（训练推荐）：
  token1 token2 token3 [PAD] [PAD] [PAD]
  → 配合 attention_mask=0 忽略 padding，batch 内填充更均匀
```

```python
tokenizer.padding_side = "right"  # SFT 训练
tokenizer.padding_side = "left"   # generate() 推理
```

---

## 5. 特殊 Token 处理

```python
# Qwen2.5 特殊 token
tokenizer.pad_token     # None（需手动设置）
tokenizer.eos_token     # '<|im_end|>'
tokenizer.bos_token     # None（Qwen 无 bos）

# 常见初始化
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # 共用 eos 作为 pad
```

**注意**：`pad_token_id` 对应的 label 需要设为 -100（不对 padding 计算 loss）。
