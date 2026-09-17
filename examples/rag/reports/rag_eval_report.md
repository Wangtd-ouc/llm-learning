# RAG 检索评估报告

- 测试题数：10
- 检索方式：TF-IDF（关键词） vs nomic-embed-text（语义向量）

## Recall@k 汇总（命中率）

| 指标 | TF-IDF | 向量检索 |
|---|---|---|
| Recall@1 | 100% | 80% |
| Recall@3 | 100% | 90% |
| Recall@5 | 100% | 100% |

## 逐题明细

| 问题 | 期望来源 | TF-IDF@1 | 向量@1 |
|---|---|---|---|
| LoRA 主要微调的是什么参数？ | lora-finetune.md | lora-finetune.md | lora-finetune.md |
| 什么是 QLoRA？ | lora-finetune.md | lora-finetune.md | lora-finetune.md |
| 微调的数据格式是什么？ | lora-finetune.md | lora-finetune.md | lora-finetune.md |
| 什么时候该用微调？ | lora-finetune.md | lora-finetune.md | lora-finetune.md |
| RAG 的五个环节是什么？ | rag-qa.md | rag-qa.md | rag-qa.md |
| 为什么需要 RAG？ | rag-qa.md | rag-qa.md | rag-qa.md |
| 检索质量怎么评价？ | rag-qa.md | rag-qa.md | lora-finetune.md |
| Transformer 的自注意力是什么？ | llm-transformer.md | llm-transformer.md | llm-transformer.md |
| 上下文窗口是什么？ | llm-transformer.md | llm-transformer.md | llm-transformer.md |
| 位置编码有什么作用？ | llm-transformer.md | llm-transformer.md | lora-finetune.md |
