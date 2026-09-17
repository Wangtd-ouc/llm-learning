"""Agent 最小可运行示例：ReAct 工具调用循环（纯 Python 标准库 + 本机 Ollama）。

原理：
  模型本身不会执行计算、不会访问数据，但可以让它“决定调用哪个工具”。
  我们循环做三件事：
    1. 把历史对话发给模型；
    2. 解析模型输出的“行动”指令，例如 行动: calc("23*17")；
    3. 执行工具，把结果放回对话，让模型继续，直到它输出“回答”。
  这种“思考 → 行动 → 观察 → 回答”的循环叫 ReAct，是最基础的 Agent 结构。

用法：
  python3 examples/agent/agent_demo.py
"""

from __future__ import annotations

import ast
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ollama_client import chat, DEFAULT_MODEL  # noqa: E402


# ---------- 工具定义（模拟数据，仅用于演示） ----------

MOCK_STOCKS = {
    "600519": {"name": "贵州茅台", "price": 1500.0},
    "300750": {"name": "宁德时代", "price": 200.0},
    "000001": {"name": "平安银行", "price": 12.5},
}

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def tool_calc(expression: str) -> str:
    """安全计算四则运算表达式（禁止执行任意代码）。"""
    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("不支持的表达式")

    return str(evaluate(tree))


def tool_get_stock_price(code: str) -> str:
    """查询示例股票价格（模拟数据，非实时行情）。"""
    stock = MOCK_STOCKS.get(code)
    if not stock:
        return f"未找到股票代码 {code}"
    return f"{stock['name']}（{code}）现价 {stock['price']} 元"


TOOLS = {
    "calc": tool_calc,
    "get_stock_price": tool_get_stock_price,
}

SYSTEM_PROMPT = """你是工具调用程序。完成任务时，你可以使用以下工具：

- calc(表达式)：执行四则运算，例如 calc("23*17")
- get_stock_price(代码)：查询示例股票价格，例如 get_stock_price("600519")

规则：
1. 需要工具结果时，先输出一行“思考: 你的思路”，再输出一行“行动: 工具名(参数)”。
2. 收到工具结果后，继续思考，直到得到最终答案。
3. 得到答案时，只输出一行“回答: 最终答案”，不要附加其他内容。
4. 不要凭空编造计算结果，必须通过工具获得。
"""


def parse_action(text: str) -> tuple[str, str] | None:
    """从模型输出中解析“行动: 工具名(参数)”。"""
    match = re.search(r"行动[:：]\s*(\w+)\s*\((.+?)\)\s*$", text, re.S)
    if not match:
        return None
    tool_name, raw_args = match.group(1), match.group(2).strip()
    try:
        args = ast.literal_eval(raw_args)
    except (ValueError, SyntaxError):
        args = raw_args  # 模型可能没加引号，按字符串处理
    return tool_name, str(args)


def run_agent(user_question: str, model: str = DEFAULT_MODEL, max_steps: int = 6) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    print(f"问题：{user_question}\n")

    for step in range(1, max_steps + 1):
        result = chat(messages, model=model, temperature=0.0, num_predict=512)
        reply = result["message"]["content"].strip()
        print(f"[第 {step} 步] 模型：{reply[:180]}")

        # 情况 1：模型给出最终回答
        answer_match = re.search(r"回答[:：]\s*(.+)", reply, re.S)
        if answer_match:
            print("\n" + "=" * 50)
            print("最终回答：", answer_match.group(1).strip())
            print("=" * 50)
            return

        # 情况 2：模型要求调用工具
        action = parse_action(reply)
        if action and action[0] in TOOLS:
            tool_name, args = action
            try:
                tool_result = TOOLS[tool_name](args)
            except Exception as exc:  # noqa: BLE001
                tool_result = f"工具执行出错：{exc}"
            print(f"       工具 {tool_name}({args}) → {tool_result}")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"工具结果：{tool_result}"})
        else:
            # 格式不符合预期，提示模型重新输出
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": "请严格按照格式输出：要么“行动: 工具名(参数)”，要么“回答: 最终答案”。",
                }
            )

    print("\n已达到最大步数，未能得到最终答案。")


if __name__ == "__main__":
    # 例 1：单次工具调用（计算）
    run_agent("请计算 23*17 等于多少？")
    print()
    # 例 2：多步工具调用（先查价，再算钱）
    run_agent("查询股票 600519 的价格，然后计算买 10 手（1 手 = 100 股）需要多少钱？")
