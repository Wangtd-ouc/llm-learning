"""本机（Apple Silicon）LoRA 微调最小示例：基于 mlx-lm 官方训练命令。

说明：
  - 首次运行前：pip install -U mlx-lm
  - 默认使用 Qwen/Qwen2.5-0.5B-Instruct（约 0.6 GB），可换成更大模型；
  - 内置 6 条中文指令数据，也可用 --data 指定自己的 JSONL。

用法：
  python finetune_lora_mlx.py                          # 内置数据，30 步
  python finetune_lora_mlx.py --data my_data.jsonl --steps 60

训练流程：
  1. 把指令数据转成 {train, valid}.jsonl（使用模型的 chat 模板）；
  2. 调用 mlx-lm 官方 LoRA 训练命令；
  3. 加载训练好的 LoRA 适配器，抽查模型在“没见过的公司”上的表现。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 内置示例数据：6 条“指令 → 回答”，让模型学会“一句话介绍公司”
BUILTIN_DATA = [
    {
        "instruction": "用一句话介绍这家公司",
        "input": "华为",
        "output": "华为是一家总部位于深圳的全球领先 ICT 基础设施和智能终端提供商。",
    },
    {
        "instruction": "用一句话介绍这家公司",
        "input": "腾讯",
        "output": "腾讯是一家以社交和游戏为核心，业务覆盖金融科技与云服务的互联网公司。",
    },
    {
        "instruction": "用一句话介绍这家公司",
        "input": "宁德时代",
        "output": "宁德时代是全球领先的动力电池和储能电池制造商。",
    },
    {
        "instruction": "用一句话介绍这家公司",
        "input": "贵州茅台",
        "output": "贵州茅台是主营茅台酒生产与销售的白酒龙头企业。",
    },
    {
        "instruction": "用一句话介绍这家公司",
        "input": "字节跳动",
        "output": "字节跳动是一家以推荐算法为核心，产品覆盖短视频与信息分发的科技公司。",
    },
    {
        "instruction": "用一句话介绍这家公司",
        "input": "比亚迪",
        "output": "比亚迪是覆盖新能源汽车、电池与半导体业务的制造业巨头。",
    },
]


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 数据，字段为 instruction / input / output。"""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def prepare_data(records: list[dict], tokenizer, data_dir: Path) -> None:
    """把指令数据写成 mlx-lm 需要的 {train, valid}.jsonl（一行一条 {"text": ...}）。"""
    data_dir.mkdir(parents=True, exist_ok=True)

    def to_text(record: dict) -> str:
        user_content = record["instruction"]
        if record.get("input"):
            user_content += "\n" + record["input"]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record["output"]},
        ]
        # 用模型自带的 chat 模板拼出训练文本
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    # 前 N-1 条训练，最后 1 条留作验证（演示“留出集”概念）
    train_records, valid_records = records[:-1], records[-1:]
    with (data_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train_records:
            fh.write(json.dumps({"text": to_text(r)}, ensure_ascii=False) + "\n")
    with (data_dir / "valid.jsonl").open("w", encoding="utf-8") as fh:
        for r in valid_records:
            fh.write(json.dumps({"text": to_text(r)}, ensure_ascii=False) + "\n")
    print(f"      训练 {len(train_records)} 条，验证 {len(valid_records)} 条 → {data_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX LoRA 微调最小示例")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=None, help="JSONL 数据文件")
    parser.add_argument("--data-dir", type=Path, default=Path("data_lora"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--merged-path", default="merged_model")
    args = parser.parse_args()

    try:
        from mlx_lm import generate, load
        from mlx_lm.utils import load_tokenizer
    except ImportError:
        sys.exit("缺少 mlx-lm，请先执行：pip install -U mlx-lm")

    print(f"[1/5] 加载分词器：{args.model}（首次会自动下载模型文件）", flush=True)
    tokenizer = load_tokenizer(args.model)

    print("[2/5] 准备数据", flush=True)
    records = load_jsonl(args.data) if args.data else BUILTIN_DATA
    prepare_data(records, tokenizer, args.data_dir)

    print(f"[3/5] 开始 LoRA 训练：{args.steps} 步（模型：{args.model}）", flush=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", str(args.data_dir),
        "--iters", str(args.steps),
        "--adapter-path", args.adapter_path,
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-layers", str(args.lora_layers),
        "--val-batches", "-1",
        "--steps-per-report", "10",
        "--steps-per-eval", "15",
        "--seed", "42",
    ]
    subprocess.run(cmd, check=True)

    print(f"[4/5] 训练完成，LoRA 权重保存在 {args.adapter_path}/", flush=True)

    print("[5/5] 效果对比：微调前 vs 微调后", flush=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍这家公司\n小米"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print("—— 微调前（基础模型，未加载 LoRA）——")
    base_model, _ = load(args.model)
    print(generate(base_model, tokenizer, prompt=prompt, max_tokens=80))

    print("—— 微调后（加载 LoRA 适配器）——")
    model, _ = load(args.model, adapter_path=args.adapter_path)
    print(generate(model, tokenizer, prompt=prompt, max_tokens=80))

    print("—— 泛化抽查：验证集样本（比亚迪，训练时未见过）——")
    valid_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话介绍这家公司\n比亚迪"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(generate(model, tokenizer, prompt=valid_prompt, max_tokens=80))

    print("\n下一步（合并权重并部署，可选）：")
    print(f"  .venv/bin/mlx_lm fuse --model {args.model} "
          f"--adapter-path {args.adapter_path} --save-path {args.merged_path}")
    print(f"  .venv/bin/mlx_lm server --model {args.merged_path} --port 8081")
    print("  （Qwen2 暂不支持直接导出 GGUF，详见 README.md）")


if __name__ == "__main__":
    main()
