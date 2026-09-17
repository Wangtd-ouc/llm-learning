# Agent 示例

一个手写的 ReAct 工具调用循环：模型“思考 → 调用工具 → 观察结果 → 回答”。

## 运行

```bash
python3 examples/agent/agent_demo.py
```

原生函数调用版（推荐，生产同款）：

```bash
python3 examples/agent/agent_tools_demo.py
```

内置两个工具：

- `calc(表达式)`：安全四则运算；
- `get_stock_price(代码)`：查询示例股票价格（模拟数据，非实时行情）。

脚本会跑两个任务：单步计算（23×17）、多步组合（查股价 × 股数算总价），展示 Agent 的核心循环。

## 关键代码位置

- `TOOLS`：工具注册表；
- `parse_action()`：解析模型输出的“行动”指令；
- `run_agent()`：ReAct 主循环；
- `SYSTEM_PROMPT`：告诉模型工具怎么用、输出格式是什么。

`agent_tools_demo.py` 用 JSON Schema 声明工具，模型直接返回结构化 `tool_calls`，无需解析文本格式。

## 原理与进阶

原理见 [../../docs/03-Agent.md](../../docs/03-Agent.md)。生产环境建议改用模型原生的函数调用（Function Calling）或成熟框架（LangGraph、OpenAI Agents SDK）。
`agent_tools_demo.py` 就是原生函数调用版示例。

## 验证记录（2026-08-28）

- 环境：Ollama 0.32.14 + qwen3.5:4b；
- 任务 1：`calc(23*17)` → 391；
- 任务 2：先查 600519 价格（1500 元），再 `calc(1500*10*100)` → 1,500,000 元，多步工具调用成功。

## 自动化评估

```bash
python3 examples/agent/evaluate_agent.py
```

5 个测试用例检查“工具选择 + 答案正确性”，报告保存在 `reports/agent_eval_report.md`。实测 5/5 通过。
