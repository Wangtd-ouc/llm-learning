"""LLM 知识自测答题程序（与 docs/06-自测题.md 同题库）。

交互模式：
  python3 examples/practice/quiz.py

脚本化验证（自动化测试用）：
  python3 examples/practice/quiz.py --answers "b,b,b,c,b,b,b,b,a,b,b,a,b,b,b,对,错,错,对,错"
"""

from __future__ import annotations

import argparse

# (题目, 选项, 答案, 对应文档与章节提示)
QUESTIONS = [
    ("大语言模型生成文本的本质是：",
     ["从数据库检索答案", "预测下一个最可能出现的 token", "模仿搜索引擎排序", "对输入做数学加密"],
     "b", "docs/01-llm基础框架.md 第 1 节"),
    ("“上下文窗口 4096”指的是：",
     ["最多 4096 个汉字", "最多 4096 个 token", "最多 4096 个句子", "最多 4096 个参数"],
     "b", "docs/01-llm基础框架.md 第 2 节"),
    ("Transformer 中让模型理解“词与词之间关系”的核心机制是：",
     ["卷积核", "自注意力（Self-Attention）", "循环连接", "贝叶斯网络"],
     "b", "docs/01-llm基础框架.md 第 3 节"),
    ("一个对话模型的生命周期顺序通常是：",
     ["对齐 → 预训练 → 指令微调", "指令微调 → 预训练 → 对齐",
      "预训练 → 指令微调 → 对齐", "预训练 → 对齐 → 指令微调"],
     "c", "docs/01-llm基础框架.md 第 4 节"),
    ("温度（temperature）参数主要控制：",
     ["回答长度", "输出的随机性", "上下文窗口大小", "训练轮数"],
     "b", "docs/01-llm基础框架.md 第 5 节"),
    ("“模型一本正经地编造不存在的事实”被称为：",
     ["过拟合", "幻觉（Hallucination）", "梯度消失", "冷启动"],
     "b", "docs/01-llm基础框架.md 第 6 节"),
    ("RAG 的核心思路是：",
     ["重新训练模型来记住新知识", "检索相关资料拼进提示词，让模型基于资料回答",
      "把知识写进模型参数", "用更大模型替代小模型"],
     "b", "docs/02-RAG.md 第 1 节"),
    ("向量检索能工作，是因为：",
     ["向量里包含原始文本", "语义相近的文本向量距离更近", "向量比文本更短", "向量是加密后的文本"],
     "b", "docs/02-RAG.md 第 2 节"),
    ("Agent 的基本组成是：",
     ["模型 + 工具 + 循环", "模型 + 数据库 + 前端", "工具 + 循环 + 网络", "模型 + 提示词 + 缓存"],
     "a", "docs/03-Agent.md 第 1 节"),
    ("ReAct 循环的正确顺序是：",
     ["行动 → 思考 → 观察 → 行动", "思考 → 行动 → 观察 → 继续思考",
      "观察 → 思考 → 行动 → 回答", "回答 → 行动 → 观察 → 思考"],
     "b", "docs/03-Agent.md 第 2 节"),
    ("文本协议工具调用相比原生函数调用的缺点是：",
     ["无法调用工具", "格式可能不稳定，需要解析", "速度一定更慢", "只能在云端运行"],
     "b", "docs/03-Agent.md 第 3 节"),
    ("想让模型“学会固定的输出格式”，最合适的方案是：",
     ["微调（SFT）", "换更大的向量库", "增加上下文窗口", "提高 temperature"],
     "a", "docs/04-微调.md 第 1 节"),
    ("LoRA 为什么省资源？",
     ["它不训练任何参数", "它冻结原模型，只训练少量低秩矩阵",
      "它把模型压缩成文本", "它用更小的模型替代大模型"],
     "b", "docs/04-微调.md 第 4 节"),
    ("QLoRA 相比 LoRA 的进一步优化是：",
     ["训练更多参数", "把原模型权重量化到 4-bit 加载", "使用更大的学习率", "跳过数据准备"],
     "b", "docs/04-微调.md 第 4 节"),
    ("关于微调数据，正确的说法是：",
     ["数据越多一定越好", "几百条高质量数据可能好过几万条脏数据",
      "测试集必须包含在训练集里", "数据格式不重要"],
     "b", "docs/04-微调.md 第 3 节"),
    ("RAG 适合“知识会频繁更新”的场景，因为更新知识库比重训模型便宜。",
     ["对", "错"], "对", "docs/02-RAG.md 第 1 节"),
    ("微调是让模型“记住新事实”的首选方案。",
     ["对", "错"], "错", "docs/04-微调.md 第 1 节"),
    ("Agent 里给模型的工具应该尽可能开放权限，比如允许删除文件。",
     ["对", "错"], "错", "docs/03-Agent.md 第 6 节"),
    ("模型在 Agent 循环里“想当然”直接给出答案而不调用工具，是常见失败模式。",
     ["对", "错"], "对", "docs/03-Agent.md 第 6 节"),
    ("评估微调效果时，只看训练集损失下降就足够。",
     ["对", "错"], "错", "docs/04-微调.md 第 7 节"),
]


def normalize(answer: str) -> str:
    a = answer.strip().lower().replace(" ", "")
    mapping = {
        "a": "a", "b": "b", "c": "c", "d": "d",
        "对": "对", "正确": "对", "t": "对", "true": "对",
        "错": "错", "错误": "错", "f": "错", "false": "错",
    }
    return mapping.get(a, "")


def run(answers: list[str] | None = None) -> int:
    correct = 0
    for i, (question, options, answer, hint) in enumerate(QUESTIONS, start=1):
        print(f"\n第 {i} 题（共 {len(QUESTIONS)} 题）：{question}")
        if len(options) > 2:
            for label, opt in zip("abcd", options):
                print(f"  {label}. {opt}")
        else:
            print("  请输入：对 / 错")

        if answers is not None:
            user_answer = normalize(answers[i - 1])
        else:
            user_answer = normalize(input("你的答案："))

        if not user_answer:
            print("  ⚠ 无法识别该答案，本题记错。")
        elif user_answer == normalize(answer):
            correct += 1
            print(f"  ✓ 正确")
        else:
            print(f"  ✗ 错误（正确答案：{answer}）")
            print(f"    复习提示：{hint}")

    print("\n" + "=" * 40)
    print(f"得分：{correct}/{len(QUESTIONS)}")
    if correct >= 16:
        print("结论：知识框架已建立，可以进入下一个阶段。")
    elif correct >= 12:
        print("结论：基本合格，建议重读错题对应的文档章节后再测一次。")
    else:
        print("结论：建议按 01 → 02/03 → 04/05 的顺序重过一遍，再测一次。")
    return 0 if correct >= 16 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 知识自测")
    parser.add_argument("--answers", default=None,
                        help="脚本化答案（逗号分隔），用于自动化验证")
    args = parser.parse_args()
    supplied = args.answers.split(",") if args.answers else None
    if supplied and len(supplied) != len(QUESTIONS):
        raise SystemExit(f"答案数量应为 {len(QUESTIONS)}，实际 {len(supplied)}")
    raise SystemExit(run(supplied))
