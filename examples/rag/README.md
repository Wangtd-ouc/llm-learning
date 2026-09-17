# RAG 示例

一个零依赖的 RAG 问答机器人：本地文档 → 切块 → TF-IDF 检索 → Ollama 生成带出处的回答。

## 运行

```bash
python3 examples/rag/rag_demo.py "LoRA 主要微调的是什么参数？"
python3 examples/rag/rag_demo.py "RAG 的五个环节是什么？"
```

向量检索版（语义检索，需要先 `ollama pull nomic-embed-text`）：

```bash
python3 examples/rag/rag_embed_demo.py "LoRA 主要微调的是什么参数？"
```

可选参数：

- `--model`：换模型，默认 `qwen3.5:4b`；
- `--top-k`：检索返回的文本块数量，默认 3。

## 目录

- `knowledge_base/`：示例知识库（3 篇中文文档）。把你要问的资料放进来即可。
- `rag_demo.py`：主程序，四步：加载切块 → 建索引 → 检索 → 生成。
- `rag_embed_demo.py`：向量检索版，用 Ollama 嵌入模型替代 TF-IDF，结果缓存到 `embeddings_cache.json`。

## 原理与进阶

原理见 [../../docs/02-RAG.md](../../docs/02-RAG.md)。示例用 TF-IDF 是为了零依赖；生产环境把检索换成“嵌入模型 + 向量数据库”即可，其余流程不变。
`rag_embed_demo.py` 已演示“嵌入模型”这一步，向量数据库可用 Milvus / Chroma / pgvector 等替代本地缓存。

## 验证记录（2026-08-28）

- 环境：Ollama 0.32.14 + qwen3.5:4b，本机 M 系列 Mac；
- 测试问题 1：“LoRA 主要微调的是什么参数？”→ 回答正确并标注 `lora-finetune.md`；
- 测试问题 2：见项目根 README 的验证状态（可自行复跑）。

## 检索评估

```bash
python3 examples/rag/evaluate_rag.py
```

输出 TF-IDF 与向量检索的 Recall@1/3/5 对比，报告保存在 `reports/rag_eval_report.md`。实测小知识库上 TF-IDF 命中率更高（100% vs 80% @1），说明检索选型要看场景和数据。
