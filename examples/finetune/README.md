# 微调示例说明

两个脚本对应两条实操路径，都基于 LoRA（低秩适配），不需要训练全部参数。

## 路径 A：本机 Apple Silicon（MLX）

适合 M 系列 Mac，小模型（0.5B～4B）友好。首次运行需要安装依赖并下载一个约 0.6 GB 的模型。

```bash
cd examples/finetune
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx-lm

# 跑最小实验：内置 6 条中文指令数据，训练 30 步
python finetune_lora_mlx.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 30

# 用自己的数据（JSONL，字段为 instruction / input / output）
python finetune_lora_mlx.py --data my_data.jsonl --steps 60
```

训练产物：

- `adapters/`：LoRA 权重（小文件）；
- `merged_model/`：合并后的完整模型；
- 脚本最后会打印一次微调后的生成结果，用于快速验证。

## 路径 B：免费 Colab（Unsloth）

没有 Mac 或想微调更大模型时使用。在 Google Colab 新建笔记本，粘贴 `unsloth_colab.py` 内容运行（Runtime 选 T4 GPU）。

脚本会：

1. 用 Unsloth 加载 Qwen2.5-1.5B-Instruct（4-bit）；
2. 准备 6 条内置示例数据；
3. LoRA 微调 60 步；
4. 测试微调后的回复；
5. 示范如何导出 GGUF 供 Ollama 使用（可选，需要联网登录 HF）。

## 关键参数速查

| 参数 | 作用 | 建议 |
|---|---|---|
| `--steps` | 训练步数 | 数据少先试 30～60 |
| `--learning-rate` | 学习率 | 默认 1e-4，过拟合就调小 |
| `--lora-rank` | LoRA 秩 | 8～16 起步 |
| `--lora-layers` | 微调多少层 | 默认 8，层越多越贵 |

## 部署：把微调模型用起来

训练产物是 LoRA 适配器（几 MB），不能直接单独使用，需要先合并到基础模型：

```bash
# 合并 LoRA 到基础模型，得到完整的 merged_model/
.venv/bin/mlx_lm fuse --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path adapters --save-path merged_model
```

合并后有两种使用方式：

### 方式 A：本机 OpenAI 兼容服务（已实测）

```bash
.venv/bin/mlx_lm server --model merged_model --port 8081
```

然后用标准 OpenAI 接口调用（任何支持 OpenAI 接口的客户端都可以接）：

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -d '{"model":"merged_model","messages":[{"role":"user","content":"用一句话介绍这家公司\n小米"}]}'
```

实测返回：`小米是一家总部位于深圳的全球领先 ICT 基础设施和智能终端提供商。`（微调后的格式）。

### 方式 B：导入 Ollama（Qwen2 架构当前不支持）

已实测 Ollama 0.32.14 的实验性 safetensors 导入能成功创建模型，但运行时报 `unsupported architecture: Qwen2ForCausalLM`，因此 Qwen2 系列目前需要先用 llama.cpp 的 `convert_hf_to_gguf.py` 转成 GGUF，再通过 `Modelfile` 导入。本目录已附上 `Modelfile` 模板备用。

## 验证状态

## 验证状态

- 2026-08-28：本机（M1 Pro + Python 3.12 venv + mlx-lm）已完整跑通 `finetune_lora_mlx.py`，结果如下：

| 项目 | 实测结果 |
|---|---|
| 可训练参数 | 0.297%（1.466M / 494M），符合 LoRA 特征 |
| 训练损失 | 1.232（第 10 步）→ 0.129（第 30 步） |
| 验证损失 | 5.760 → 1.406（第 15 步）→ 1.906（第 30 步，仅 1 条验证样本，波动正常） |
| 显存峰值 | 1.23 GB（16GB 内存的机器毫无压力） |
| 产物 | `adapters/adapters.safetensors`（5.9 MB）+ 配置 |

效果对比（问题：用一句话介绍小米）：

- 微调前：`小米，一家以创新科技和智能家居产品著称的中国科技公司。`（有知识，但不符合目标格式）
- 微调后：`小米是一家总部位于深圳的全球领先 ICT 基础设施和智能终端提供商。`（格式完全正确，但内容借用了最相似的训练样本“华为”）

这正是微调的精髓：**它学的是行为/格式，不是新知识**。少量数据下内容会“背最像的样本”，所以需要更多高质量数据，或配合 RAG 补知识。

- 注意：mlx 需要 Metal GPU 权限，训练命令必须在非沙箱环境运行（本项目通过授权方式执行）。
