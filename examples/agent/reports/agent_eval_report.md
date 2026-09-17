# Agent 自动化评估报告

- 模型：qwen3.5:4b
- 用例数：6，通过：6（100%）
- 判据：预期工具全部被调用，且最终回答包含预期关键词

| 问题 | 工具选择 | 答案 | 结果 |
|---|---|---|---|
| 请计算 23*17 等于多少？ | ✓ (实际 calc) | ✓ | PASS |
| 查询股票 600519 的价格，买 10 手需要多少钱？ | ✓ (实际 get_stock_price,calc) | ✓ | PASS |
| 现在几点？ | ✓ (实际 get_current_time) | ✓ | PASS |
| LoRA 主要微调的是什么参数？ | ✓ (实际 search_knowledge) | ✓ | PASS |
| RAG 的五个环节是什么？ | ✓ (实际 search_knowledge) | ✓ | PASS |
| 查询美元兑人民币汇率，然后计算 100 美元等于多少人民币？ | ✓ (实际 get_exchange_rate,calc) | ✓ | PASS |
