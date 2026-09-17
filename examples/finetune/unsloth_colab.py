"""免费 Colab（T4 GPU）LoRA 微调示例：基于 Unsloth。

使用步骤：
  1. 打开 Google Colab（colab.research.google.com），新建笔记本；
  2. Runtime 选择 T4 GPU；
  3. 第一个单元格先运行：!pip install unsloth
  4. 把本文件内容粘贴进下一个单元格运行。

本脚本会把“一句话介绍公司”的示例数据微调到 Qwen2.5-1.5B 上。
"""

from __future__ import annotations

import json

import torch

# ---------- 1. 加载模型（4-bit 量化，省显存） ----------
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None  # None 表示自动选择
load_in_4bit = True  # 4-bit 量化，T4 显卡才能装下

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ---------- 2. 配置 LoRA ----------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA 秩
    lora_alpha=16,
    lora_dropout=0,  # Unsloth 建议设 0
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ---------- 3. 准备数据（换成你自己的 JSONL 即可） ----------
builtin_data = [
    {"instruction": "用一句话介绍这家公司", "input": "华为",
     "output": "华为是一家总部位于深圳的全球领先 ICT 基础设施和智能终端提供商。"},
    {"instruction": "用一句话介绍这家公司", "input": "腾讯",
     "output": "腾讯是一家以社交和游戏为核心，业务覆盖金融科技与云服务的互联网公司。"},
    {"instruction": "用一句话介绍这家公司", "input": "宁德时代",
     "output": "宁德时代是全球领先的动力电池和储能电池制造商。"},
    {"instruction": "用一句话介绍这家公司", "input": "贵州茅台",
     "output": "贵州茅台是主营茅台酒生产与销售的白酒龙头企业。"},
    {"instruction": "用一句话介绍这家公司", "input": "字节跳动",
     "output": "字节跳动是一家以推荐算法为核心，产品覆盖短视频与信息分发的科技公司。"},
    {"instruction": "用一句话介绍这家公司", "input": "比亚迪",
     "output": "比亚迪是覆盖新能源汽车、电池与半导体业务的制造业巨头。"},
]

EOS_TOKEN = tokenizer.eos_token


def make_text(item: dict) -> str:
    """按 Qwen 的 chat 模板拼训练文本。"""
    user = item["instruction"] + ("\n" + item["input"] if item.get("input") else "")
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{item['output']}<|im_end|>\n"
    )


dataset = [
    {"text": make_text(item) + EOS_TOKEN}
    for item in builtin_data
]

# ---------- 4. 训练 ----------
from datasets import Dataset  # noqa: E402
from trl import SFTTrainer, TrainingArguments  # noqa: E402

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(dataset),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,          # 数据少，60 步足够演示
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer.train()

# ---------- 5. 微调后验证 ----------
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    ["<|im_start|>user\n用一句话介绍这家公司\n小米<|im_end|>\n<|im_start|>assistant\n"],
    return_tensors="pt",
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ---------- 6.（可选）保存与导出 ----------
# 保存合并后的 16-bit 模型
model.save_pretrained_merged("lora_model", tokenizer, save_method="merged_16bit")

# 导出 GGUF 供 Ollama 使用（需要 HF 账号登录）
# model.save_pretrained_gguf("qwen_gguf", tokenizer, quantization_method="q4_k_m")
