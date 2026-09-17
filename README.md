# LLM 从零到上手：知识框架 + RAG + Agent + 微调

本项目的目标是：让零基础的人建立 LLM 基本知识体系，并能亲手构建三类最常见的 LLM 应用——RAG、Agent、微调。

所有示例都优先跑在本机（macOS + Ollama），不依赖云端 API，随时可复现。

## 学习路线（建议顺序）

| 阶段 | 内容 | 产出 | 对应文件 |
|---|---|---|---|
| 1. 建立框架 | 了解 LLM 是什么：token、Transformer、预训练、指令微调、对齐、推理参数 | 能向别人讲清楚 LLM 的原理链路 | `docs/01-llm基础框架.md` |
| 2. 动手对话 | 用本机 Ollama 跑通第一个对话，理解 system/user/assistant 消息结构 | 跑通 `examples/rag/rag_demo.py` 前的热身 | README 快速开始 |
| 3. RAG | 检索增强生成：文档切块 → 检索 → 拼进提示词 → 生成带出处的回答 | 可运行的本地问答机器人 | `docs/02-RAG.md` + `examples/rag/` |
| 4. Agent | 让模型学会“调用工具”：ReAct 循环、工具定义、多步推理 | 能自己算数、查数据并组合回答的问答程序 | `docs/03-Agent.md` + `examples/agent/` |
| 5. 微调 | 用业务数据调整模型行为：SFT、LoRA/QLoRA、数据格式、评估 | 可运行的本机/Colab 微调脚本 | `docs/04-微调.md` + `examples/finetune/` |
| 6. 综合项目 | 把 RAG + Agent 组合成一个可扩展的问答系统，微调用于稳定格式 | 可运行的组合示例 | `docs/05-综合项目.md` + `examples/assistant/` |
| 7. 自测 | 检验知识框架是否真的建立 | 20 题自测 + 答案解析 | `docs/06-自测题.md` |
| 8. 动手改造 | 给示例加自己的工具或知识文档并验证 | 新增工具 + 评估用例通过 | `docs/07-动手改造指南.md` |
| 9. 审计收尾 | 目标 ↔ 证据对照 + 3 天学习计划 | 完成度审计文档 | `docs/08-完成度审计.md` |
| 10. 第 1 天课程 | LLM 核心框架 + RAG 课程（含答案） | 离线可学的课程文档 | `docs/09-第1天课程.md` |
| 11. 第 2 天课程 | Agent + 微调课程（含答案） | 离线可学的课程文档 | `docs/10-第2天课程.md` |
| 12. 第 3 天课程 | 综合 + 改造 + 收尾课程（含答案） | 离线可学的课程文档 | `docs/11-第3天课程.md` |

## 快速开始

前置条件：

1. 本机已安装并启动 [Ollama](https://ollama.com)（本项目验证时使用 0.32.14）。
2. 已拉取模型（本项目使用 `qwen3.5:4b`，你也可以换成任意 Ollama 支持的中文模型）：

```bash
ollama pull qwen3.5:4b
```

运行示例：

```bash
# RAG：对本地知识库提问，回答带出处
python3 examples/rag/rag_demo.py "LoRA 主要微调的是什么参数？"

# Agent：让模型调用工具做多步计算
python3 examples/agent/agent_demo.py

# 一键验收：自测题库 + RAG 评估 + Agent 评估
python3 examples/practice/verify_all.py
```

示例只使用 Python 标准库，无需 `pip install`。

## 目录结构

```text
llm-learning/
├── README.md                  # 本文件：学习路线 + 快速开始
├── docs/                      # 中文知识框架文档
│   ├── 01-llm基础框架.md
│   ├── 02-RAG.md
│   ├── 03-Agent.md
│   └── 04-微调.md
└── examples/
    ├── common/ollama_client.py     # 极简 Ollama 调用封装（标准库）
    ├── rag/
    │   ├── knowledge_base/         # 示例知识库（3 篇中文文档）
    │   ├── rag_demo.py             # 纯 Python 实现：切块+TF-IDF 检索+生成
    │   ├── rag_embed_demo.py       # 向量检索版：Ollama 嵌入模型 + 语义相似度
    │   └── README.md
    ├── agent/
    │   ├── agent_demo.py           # ReAct 工具调用循环（计算器/股票查询）
    │   ├── agent_tools_demo.py     # 原生函数调用（Function Calling）版
    │   └── README.md
    ├── assistant/
    │   ├── assistant.py            # 综合示例：RAG 检索 + 工具调用
    │   └── README.md
    ├── finetune/
    │   ├── finetune_lora_mlx.py    # 本机 Apple Silicon 用 MLX 跑 LoRA
    │   ├── unsloth_colab.py        # 免费 Colab 用 Unsloth 跑 LoRA
    │   └── README.md
    └── practice/
        ├── quiz.py                 # 交互式自测答题（与 docs/06 同题库）
        ├── verify_all.py           # 一键验收：自测 + RAG 评估 + Agent 评估
        └── track_progress.py       # 学习进度跟踪（跨会话保存到 progress.json）
```

## 验证状态（2026-08-28）

- RAG 示例：已在本机用 `qwen3.5:4b` 跑通，能带出处回答。
- Agent 示例：已在本机用 `qwen3.5:4b` 跑通多步工具调用（先查价、再算钱）。
- 微调示例：已在本机（M1 Pro）完整跑通 30 步 LoRA 训练，训练损失 1.23 → 0.13，显存峰值 1.23 GB，并完成了“微调前/后”效果对比，详见 `examples/finetune/README.md`。
- 综合示例：已在本机用 `qwen3.5:4b` 跑通四类问题（RAG 问答、计算、查价+算钱、时间），详见 `examples/assistant/README.md`。
- 向量检索 RAG：已在本机用 `nomic-embed-text` 嵌入模型验证（见 `examples/rag/README.md`）。
- 原生函数调用 Agent：已在本机用 `qwen3.5:4b` 验证（见 `examples/agent/README.md`）。
- 微调部署：LoRA 合并 → `mlx_lm server` OpenAI 兼容接口已实测；Qwen2 转 GGUF 需用 llama.cpp（Ollama 实验性导入已实测不支持），详见 `examples/finetune/README.md`。
- RAG 评估：10 题测试集实测 TF-IDF Recall@1 = 100%、向量检索 = 80%（小知识库场景 TF-IDF 更优），报告见 `examples/rag/reports/rag_eval_report.md`。
- 全链路实验：微调模型接入综合示例已实测，0.5B 模型只会重复训练格式、不调用工具，结论写入 `docs/05-综合项目.md`。
- Agent 评估：5 个测试用例（工具选择 + 答案正确性）实测 5/5 通过，报告见 `examples/agent/reports/agent_eval_report.md`。
- 一键验收：`examples/practice/verify_all.py` 实测全部通过（自测 20/20、RAG 评估达标、Agent 评估 5/5）。
- 动手改造示范：综合示例新增“汇率查询”工具，Agent 评估用例扩到 6 个并全部通过（6/6），流程见 `docs/07-动手改造指南.md`。
- 完成度审计：目标逐项对照证据已产出，一键验收复跑全绿（2026-08-28），见 `docs/08-完成度审计.md`。

## 完成标准（可核查）

本学习目标是否达成，用下面的清单自检（每项都能用项目里的真实产物验证）：

- [ ] 知识框架：闭卷做完 `docs/06-自测题.md` 的 20 题，答对 ≥16 题；
- [ ] 一键验收：运行 `python3 examples/practice/verify_all.py`，三项验证全部通过；
- [ ] RAG 构建：跑通 `examples/rag/rag_demo.py` 与 `rag_embed_demo.py`，并运行 `evaluate_rag.py` 看懂评估报告；
- [ ] Agent 构建：跑通 `agent_demo.py` 与 `agent_tools_demo.py`，并运行 `evaluate_agent.py` 看懂评估报告；
- [ ] 微调：跑通 `examples/finetune/finetune_lora_mlx.py`，能解释训练损失、验证集和“微调学格式不学知识”；
- [ ] 综合：跑通 `examples/assistant/assistant.py`，能向别人解释 RAG 与 Agent 如何组合；
- [ ] 动手改造：给任一示例新增一个自己的工具或知识文档并验证生效（按 `docs/07-动手改造指南.md` 的 5 步做，参考“汇率工具”示范）。

全部勾选后，即视为本学习目标达成。

## 原则

- 先理解“为什么”，再动手写代码。
- 每个概念尽量配一个可以在本机跑的最小示例。
- 文档中的事实以主流公开资料为准，标注为“概念”而非“建议”。
