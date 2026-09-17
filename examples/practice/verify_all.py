"""一键验收：运行全部自动化验证，检查学习项目是否处于“可通过”状态。

验证项：
  1. 自测题库自检：用标准答案跑 quiz.py，必须 20/20；
  2. RAG 检索评估：TF-IDF Recall@1 ≥ 0.9，向量 Recall@1 ≥ 0.7；
  3. Agent 评估：测试用例通过率 ≥ 80%。

用法（在项目根目录）：
  python3 examples/practice/verify_all.py

任一验证不达标会以非零退出码结束。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUIZ_ANSWER_KEY = "b,b,b,c,b,b,b,b,a,b,b,a,b,b,b,对,错,错,对,错"


def run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def load_report(relative: str) -> dict:
    return json.loads((PROJECT_ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    print("== 1/3 自测题库自检（标准答案应得 20/20）==")
    quiz_ok = (
        run([sys.executable, "examples/practice/quiz.py", "--answers", QUIZ_ANSWER_KEY]) == 0
    )
    results.append(("自测题库自检（20/20）", quiz_ok, "examples/practice/quiz.py"))

    print("\n== 2/3 RAG 检索评估（约 5 秒）==")
    rag_ok_run = run([sys.executable, "examples/rag/evaluate_rag.py"])
    rag_ok = False
    if rag_ok_run == 0:
        report = load_report("examples/rag/reports/rag_eval_report.json")
        summary = report["summary"]
        tfidf_r1 = summary["recall@1"]["tfidf"]
        vector_r1 = summary["recall@1"]["vector"]
        rag_ok = tfidf_r1 >= 0.9 and vector_r1 >= 0.7
        print(f"  TF-IDF Recall@1 = {tfidf_r1:.0%}，向量 Recall@1 = {vector_r1:.0%}")
    results.append(
        ("RAG 检索评估（TF-IDF@1≥90%，向量@1≥70%）", rag_ok,
         "examples/rag/reports/rag_eval_report.md")
    )

    print("\n== 3/3 Agent 评估（约 1-2 分钟，需要 Ollama）==")
    agent_ok_run = run([sys.executable, "examples/agent/evaluate_agent.py"])
    agent_ok = False
    if agent_ok_run == 0:
        report = load_report("examples/agent/reports/agent_eval_report.json")
        rate = report["pass_rate"]
        agent_ok = rate >= 0.8
        print(f"  通过率 = {rate:.0%}（{report['passed']}/{report['total']}）")
    results.append(
        ("Agent 评估（通过率≥80%）", agent_ok,
         "examples/agent/reports/agent_eval_report.md")
    )

    print("\n" + "=" * 50)
    print("验收结果：")
    all_ok = True
    for name, ok, artifact in results:
        all_ok = all_ok and ok
        print(f"  {'✓' if ok else '✗'} {name}  → {artifact}")
    print("=" * 50)
    print("总体：全部通过，可以进入下一步学习。" if all_ok else "总体：有未通过项，请按上面提示检查。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
