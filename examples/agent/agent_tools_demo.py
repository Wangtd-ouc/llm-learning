"""原生函数调用版 Agent：让模型直接返回结构化工具调用（Function Calling）。

与 agent_demo.py 的文本协议不同，本示例把工具以 JSON Schema 形式发给模型，
模型返回结构化的 tool_calls（工具名 + 参数 JSON），程序不用解析文本格式，
这是生产环境推荐的做法。

用法：
  python3 examples/agent/agent_tools_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ollama_client import chat, DEFAULT_MODEL  # noqa: E402
from agent.agent_demo import tool_calc, tool_get_stock_price  # noqa: E402


def tool_get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "calc": tool_calc,
    "get_stock_price": tool_get_stock_price,
    "get_current_time": tool_get_current_time,
}

# 工具声明：以 JSON Schema 描述，模型据此生成结构化调用
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "执行四则运算，例如 23*17",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "查询示例股票价格，例如 600519",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间，无参数",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_question(user_question: str, model: str = DEFAULT_MODEL, max_steps: int = 8) -> None:
    messages = [{"role": "user", "content": user_question}]
    print(f"\n问题：{user_question}")

    for step in range(1, max_steps + 1):
        result = chat(
            messages, model=model, temperature=0.0, num_predict=512, tools=TOOL_SCHEMAS
        )
        message = result["message"]
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls") or []

        if content:
            print(f"  [第 {step} 步] 模型：{content[:150]}")

        if not tool_calls:
            print(f"  → 最终回答：{content or '（空）'}")
            return

        # 记录模型这一轮的“调用意图”，供下一轮对话引用
        messages.append({"role": "assistant", "content": content or "", "tool_calls": tool_calls})

        # 执行本轮所有工具调用
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            arguments = fn.get("arguments") or {}
            if name not in TOOLS:
                tool_result = f"未知工具：{name}"
            else:
                try:
                    if arguments:
                        tool_result = TOOLS[name](**arguments)
                    else:
                        tool_result = TOOLS[name]()
                except Exception as exc:  # noqa: BLE001
                    tool_result = f"工具执行出错：{exc}"
            print(f"     工具 {name}({arguments}) → {str(tool_result)[:120]}")
            messages.append({"role": "tool", "tool_name": name, "content": str(tool_result)})

    print("  已达到最大步数。")


if __name__ == "__main__":
    run_question("请计算 23*17 等于多少？")
    run_question("查询股票 600519 的价格，买 10 手需要多少钱？")
    run_question("现在几点？")
