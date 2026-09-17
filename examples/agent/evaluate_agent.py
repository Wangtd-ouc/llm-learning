"""Agent 自动化评估：用带预期结果的测试用例，检查“工具选对没、答案对不对”。

每个用例记录两个维度：
  - 工具选择：是否调用了预期工具（该检索时检索、该计算时计算）；
  - 答案正确：最终回答是否包含预期关键词。

用法：
  python3 examples/agent/evaluate_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.assistant import run_question  # noqa: E402
from common.ollama_client import DEFAULT_MODEL  # noqa: E402

# 测试用例：问题 → 预期工具（必须全部被调用）→ 预期答案关键词（至少一个命中）
CASES = [
    {
        "question": "请计算 23*17 等于多少？",
        "expect_tools": ["calc"],
        "expect_keywords": ["391"],
    },
    {
        "question": "查询股票 600519 的价格，买 10 手需要多少钱？",
        "expect_tools": ["get_stock_price", "calc"],
        "expect_keywords": ["1500000", "1,500,000"],
    },
    {
        "question": "现在几点？",
        "expect_tools": ["get_current_time"],
        "expect_keywords": ["2026"],
    },
    {
        "question": "LoRA 主要微调的是什么参数？",
        "expect_tools": ["search_knowledge"],
        "expect_keywords": ["低秩", "lora-finetune.md"],
    },
    {
        "question": "RAG 的五个环节是什么？",
        "expect_tools": ["search_knowledge"],
        "expect_keywords": ["切块", "检索", "生成"],
    },
    {
        "question": "查询美元兑人民币汇率，然后计算 100 美元等于多少人民币？",
        "expect_tools": ["get_exchange_rate", "calc"],
        "expect_keywords": ["720"],
    },
]


def evaluate(model: str = DEFAULT_MODEL) -> list[dict]:
    rows = []
    for case in CASES:
        tools_used, answer = run_question(case["question"], model=model)
        tools_ok = all(t in tools_used for t in case["expect_tools"])
        answer_ok = bool(answer) and any(
            kw in answer for kw in case["expect_keywords"]
        )
        rows.append(
            {
                "question": case["question"],
                "expect_tools": case["expect_tools"],
                "tools_used": tools_used,
                "tools_ok": tools_ok,
                "answer_ok": answer_ok,
                "answer_preview": (answer or "")[:80],
                "pass": tools_ok and answer_ok,
            }
        )
        print(
            f"  {'PASS' if rows[-1]['pass'] else 'FAIL'} | {case['question'][:26]:<28} | "
            f"工具 {'✓' if tools_ok else '✗'} 答案 {'✓' if answer_ok else '✗'}"
        )
    return rows


def save_report(rows: list[dict]) -> None:
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(exist_ok=True)
    passed = sum(r["pass"] for r in rows)
    total = len(rows)

    report = {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total,
        "cases": rows,
        "model": DEFAULT_MODEL,
        "note": "工具维度：预期工具是否都被调用；答案维度：最终回答是否包含预期关键词",
    }
    (report_dir / "agent_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Agent 自动化评估报告",
        "",
        f"- 模型：{DEFAULT_MODEL}",
        f"- 用例数：{total}，通过：{passed}（{passed / total:.0%}）",
        "- 判据：预期工具全部被调用，且最终回答包含预期关键词",
        "",
        "| 问题 | 工具选择 | 答案 | 结果 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | {'✓' if row['tools_ok'] else '✗'} "
            f"(实际 {','.join(row['tools_used']) or '-'}) | "
            f"{'✓' if row['answer_ok'] else '✗'} | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    md_path = report_dir / "agent_eval_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{report_dir / 'agent_eval_report.md'}")


if __name__ == "__main__":
    rows = evaluate()
    save_report(rows)
