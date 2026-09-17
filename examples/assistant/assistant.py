"""综合示例：把 RAG 和 Agent 组合成一个可扩展的问答系统。

能力：
  - search_knowledge(问题)：检索本地知识库并返回相关资料（RAG）；
  - calc(表达式)：安全四则运算；
  - get_stock_price(代码)：查询示例股票价格；
  - get_current_time()：当前时间。

模型在 ReAct 循环里自己决定用哪个工具：
  知识问题 → 调 RAG 工具；计算问题 → 调计算器；组合问题 → 多步调用。

用法：
  python3 examples/assistant/assistant.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# 让脚本能 import 同目录下的 rag/ 和 agent/ 模块
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))

from common.ollama_client import chat, chat_openai, DEFAULT_MODEL  # noqa: E402
from rag.rag_demo import build_index, load_chunks, retrieve  # noqa: E402
from agent.agent_demo import tool_calc, tool_get_stock_price  # noqa: E402


# ---------- 1. 工具注册表 ----------

def tool_search_knowledge(query: str) -> str:
    """RAG 工具：检索知识库，返回最相关的文本块。"""
    hits = retrieve(query, INDEX, IDF, top_k=3)
    if not hits:
        return "知识库中没有检索到相关内容"
    return "\n\n".join(
        f"[来源: {hit['source']}]\n{hit['text']}" for hit in hits
    )


def tool_get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


MOCK_RATES = {
    "USD/CNY": 7.2,
    "HKD/CNY": 0.92,
}


def tool_get_exchange_rate(pair: str) -> str:
    """查询示例汇率（模拟数据，非实时行情）。"""
    rate = MOCK_RATES.get(pair.upper())
    if rate is None:
        return f"未找到汇率：{pair}"
    return f"{pair.upper()} 汇率为 {rate}（模拟数据）"


TOOLS = {
    "search_knowledge": tool_search_knowledge,
    "calc": tool_calc,
    "get_stock_price": tool_get_stock_price,
    "get_current_time": tool_get_current_time,
    "get_exchange_rate": tool_get_exchange_rate,
}

TOOL_DESC = "\n".join(
    [
        "- search_knowledge(问题)：检索本地知识库，例如 search_knowledge(\"LoRA 是什么\")",
        "- calc(表达式)：执行四则运算，例如 calc(\"23*17\")",
        "- get_stock_price(代码)：查询示例股票价格，例如 get_stock_price(\"600519\")",
        "- get_current_time()：获取当前时间",
        "- get_exchange_rate(货币对)：查询示例汇率，例如 get_exchange_rate(\"USD/CNY\")",
    ]
)

SYSTEM_PROMPT = f"""你是综合问答程序，可以调用以下工具：

{TOOL_DESC}

规则：
1. 需要工具结果时，先输出“思考: 你的思路”，再输出“行动: 工具名(参数)”。
2. 收到工具结果后继续思考，直到得出最终答案。
3. 得到答案时，只输出“回答: 最终答案”，回答必须另起一行且以“回答:”开头。
4. 如果使用了 search_knowledge，最终回答末尾必须列出引用的来源文件名。
5. 不要凭空编造数字或资料，必须通过工具获得。
"""

# 全局知识库索引（启动时构建一次）
INDEX, IDF = build_index(load_chunks())


# ---------- 2. ReAct 循环（与 agent 示例相同） ----------

def parse_action(text: str) -> tuple[str, str] | None:
    # 允许空参数（如 get_current_time()），并允许行动行不在文本末尾
    match = re.search(r"行动[:：]\s*(\w+)\s*\(([^)]*)\)", text, re.S)
    if not match:
        return None
    tool_name, raw_args = match.group(1), match.group(2).strip()
    try:
        import ast

        args = ast.literal_eval(raw_args)
    except (ValueError, SyntaxError):
        args = raw_args
    return tool_name, str(args)


def run_question(
    user_question: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = 8,
    api: str = "ollama",
    base_url: str | None = None,
) -> tuple[list[str], str | None]:
    """跑一个问题，返回 (用到的工具列表, 最终回答)。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    print(f"\n问题：{user_question}")
    tools_used: list[str] = []

    for step in range(1, max_steps + 1):
        if api == "openai":
            message = chat_openai(
                messages, base_url=base_url, model=model, num_predict=512
            )
            reply = (message.get("content") or "").strip()
        else:
            result = chat(messages, model=model, temperature=0.0, num_predict=512)
            reply = result["message"]["content"].strip()
        print(f"  [第 {step} 步] 模型：{reply[:150]}")

        # 优先执行行动：模型常在同一轮里先写“行动”又提前写“回答”，
        # 如果先采信回答，就会漏掉工具调用（例如误说“无法获取时间”）。
        action = parse_action(reply)
        if action and action[0] in TOOLS:
            tool_name, args = action
            tools_used.append(tool_name)
            try:
                # 空参数（如 get_current_time()）直接调用，不带参数
                tool_result = (
                    TOOLS[tool_name]() if args == "" else TOOLS[tool_name](args)
                )
            except Exception as exc:  # noqa: BLE001
                tool_result = f"工具执行出错：{exc}"
            print(f"     工具 {tool_name}({args}) → {tool_result[:120]}...")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"工具结果：{tool_result}"})
            continue

        answer_match = re.search(r"回答[:：]\s*(.+)", reply, re.S)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            print(f"  → 最终回答：{final_answer}")
            return tools_used, final_answer

        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": "请严格按照格式输出：要么“行动: 工具名(参数)”，要么“回答: 最终答案”。",
            }
        )

    print("  已达到最大步数，未能得到最终答案。")
    return tools_used, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="综合示例")
    parser.add_argument("--api", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址，如 http://127.0.0.1:8081")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question", default=None, help="只跑一个问题（默认跑全部演示问题）")
    args = parser.parse_args()

    if args.api == "openai" and not args.base_url:
        sys.exit("使用 --api openai 时必须提供 --base-url")

    questions = [
        args.question,
    ] if args.question else [
        "LoRA 主要微调的是什么参数？",                # RAG
        "请计算 23*17 等于多少？",                    # 计算器
        "查询股票 600519 的价格，买 10 手需要多少钱？",  # 查价 + 计算
        "现在几点？",                                  # 时间工具
        "查询美元兑人民币汇率，然后计算 100 美元等于多少人民币？",  # 汇率 + 计算
    ]
    for question in questions:
        run_question(question, model=args.model, api=args.api, base_url=args.base_url)
