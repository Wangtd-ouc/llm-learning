"""学习进度跟踪器：把“完成清单”保存到本地，跨会话持续记录。

交互模式（逐项确认）：
  python3 examples/practice/track_progress.py

常用参数：
  --status            只看当前进度，不提问
  --mark 1,3,5        直接把第 1/3/5 项标记为完成（脚本化验证用）
  --reset             清空进度

进度保存在 examples/practice/progress.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROGRESS_PATH = Path(__file__).resolve().parent / "progress.json"

ITEMS = [
    "闭卷自测 ≥16/20（python3 examples/practice/quiz.py）",
    "一键验收全绿（python3 examples/practice/verify_all.py）",
    "跑通 RAG 两个示例并看懂评估报告（rag_demo / rag_embed_demo / evaluate_rag）",
    "跑通 Agent 两个示例并看懂评估报告（agent_demo / agent_tools_demo / evaluate_agent）",
    "跑通微调并解释训练损失/验证集/效果对比（finetune_lora_mlx.py）",
    "跑通综合示例（assistant.py 四类问题）",
    "亲手改造一个示例并验证生效（按 docs/07-动手改造指南.md）",
]


def load_progress() -> list[bool]:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))["done"]
    return [False] * len(ITEMS)


def save_progress(done: list[bool]) -> None:
    PROGRESS_PATH.write_text(
        json.dumps({"done": done}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def show_status(done: list[bool]) -> None:
    total = len(done)
    passed = sum(done)
    print(f"学习进度：{passed}/{total}（{passed / total:.0%}）\n")
    for i, (item, is_done) in enumerate(zip(ITEMS, done), start=1):
        print(f"  {'✓' if is_done else '○'} [{i}] {item}")
    if passed == total:
        print("\n全部完成！学习目标达成。")
    else:
        print(f"\n剩余 {total - passed} 项，建议按 docs/08-完成度审计.md 的 3 天计划推进。")


def main() -> None:
    parser = argparse.ArgumentParser(description="学习进度跟踪器")
    parser.add_argument("--status", action="store_true", help="只看进度")
    parser.add_argument("--mark", default=None, help="标记完成的项，如 1,3,5")
    parser.add_argument("--reset", action="store_true", help="清空进度")
    args = parser.parse_args()

    if args.reset:
        save_progress([False] * len(ITEMS))
        print("进度已清空。")
        return

    done = load_progress()
    if args.mark:
        for token in args.mark.split(","):
            idx = int(token.strip()) - 1
            if 0 <= idx < len(done):
                done[idx] = True
        save_progress(done)
        print(f"已标记：{args.mark}")
        show_status(done)
        return

    if args.status:
        show_status(done)
        return

    # 交互模式：逐项询问未完成项
    print("逐项确认完成情况（y=完成，n=未完成，回车默认未完成）：\n")
    for i, (item, is_done) in enumerate(zip(ITEMS, done), start=1):
        if is_done:
            print(f"  ✓ [{i}] {item}")
            continue
        answer = input(f"  [{i}] {item}\n     已完成？(y/n) ").strip().lower()
        if answer in ("y", "yes", "是", "对"):
            done[i - 1] = True
    save_progress(done)
    print()
    show_status(done)


if __name__ == "__main__":
    main()
