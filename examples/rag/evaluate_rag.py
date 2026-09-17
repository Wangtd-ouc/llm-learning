"""RAG 检索评估：用一组带标准答案的测试题，量化对比两种检索方式。

指标：
  - Recall@k：前 k 个检索结果里是否出现“正确答案所在文档”（命中即 1，否则 0）；
  - 命中率 = 命中题数 / 总题数。

用法：
  python3 examples/rag/evaluate_rag.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ollama_client import embed, DEFAULT_EMBED_MODEL  # noqa: E402
from rag.rag_demo import build_index, load_chunks, retrieve as tfidf_retrieve  # noqa: E402
from rag.rag_embed_demo import CACHE_PATH, get_embedding  # noqa: E402

# 测试集：(问题, 期望命中的来源文件名列表)
EVAL_SET = [
    ("LoRA 主要微调的是什么参数？", ["lora-finetune.md"]),
    ("什么是 QLoRA？", ["lora-finetune.md"]),
    ("微调的数据格式是什么？", ["lora-finetune.md"]),
    ("什么时候该用微调？", ["lora-finetune.md"]),
    ("RAG 的五个环节是什么？", ["rag-qa.md"]),
    ("为什么需要 RAG？", ["rag-qa.md"]),
    ("检索质量怎么评价？", ["rag-qa.md"]),
    ("Transformer 的自注意力是什么？", ["llm-transformer.md"]),
    ("上下文窗口是什么？", ["llm-transformer.md"]),
    ("位置编码有什么作用？", ["llm-transformer.md"]),
]

TOP_KS = [1, 3, 5]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def vector_retrieve(query: str, chunks: list[dict], vectors: list[list[float]], top_k: int) -> list[dict]:
    query_vec = embed(query, model=DEFAULT_EMBED_MODEL)
    scored = sorted(
        ((cosine_similarity(query_vec, v), c) for v, c in zip(vectors, chunks)),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for s, c in scored[:top_k]]


def hit(hits: list[dict], expected: list[str]) -> bool:
    return any(h["source"] in expected for h in hits)


def main() -> None:
    print("[1/3] 加载知识库与索引")
    chunks = load_chunks()
    index, idf = build_index(chunks)

    cache: dict = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    vectors = [get_embedding(c["text"], DEFAULT_EMBED_MODEL, cache) for c in chunks]
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    print("[2/3] 逐题检索（两种方式各取 Top-5）")
    rows = []
    for question, expected in EVAL_SET:
        tfidf_hits = tfidf_retrieve(question, index, idf, top_k=5)
        vector_hits = vector_retrieve(question, chunks, vectors, top_k=5)
        rows.append(
            {
                "question": question,
                "expected": expected[0],
                "tfidf_top1": tfidf_hits[0]["source"] if tfidf_hits else "-",
                "vector_top1": vector_hits[0]["source"] if vector_hits else "-",
                "tfidf_recall3": hit(tfidf_hits[:3], expected),
                "vector_recall3": hit(vector_hits[:3], expected),
            }
        )
        print(f"  {question[:22]:<24} | 期望 {expected[0][:20]:<20} | "
              f"TFIDF@1 {rows[-1]['tfidf_top1'][:18]:<18} | 向量@1 {rows[-1]['vector_top1'][:18]}")

    print("\n[3/3] 汇总（Recall@k，命中率 = 命中题数 / 总题数）")
    n = len(EVAL_SET)
    print(f"  {'指标':<16}{'TF-IDF':>10}{'向量检索':>10}")
    summary = {}
    for k in TOP_KS:
        tfidf_hits = sum(
            hit(tfidf_retrieve(q, index, idf, top_k=k), exp) for q, exp in EVAL_SET
        )
        vector_hits = sum(
            hit(vector_retrieve(q, chunks, vectors, top_k=k), exp) for q, exp in EVAL_SET
        )
        summary[f"recall@{k}"] = {"tfidf": tfidf_hits / n, "vector": vector_hits / n}
        print(f"  {'Recall@' + str(k):<16}{tfidf_hits / n:>10.2f}{vector_hits / n:>10.2f}")

    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(exist_ok=True)
    report = {
        "total": n,
        "eval_set": rows,
        "summary": summary,
        "retriever_note": "TF-IDF 靠关键词重合；向量检索用 nomic-embed-text 语义相似度",
    }
    (report_dir / "rag_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 同时输出人可读的 Markdown 报告
    lines = [
        "# RAG 检索评估报告",
        "",
        f"- 测试题数：{n}",
        "- 检索方式：TF-IDF（关键词） vs nomic-embed-text（语义向量）",
        "",
        "## Recall@k 汇总（命中率）",
        "",
        "| 指标 | TF-IDF | 向量检索 |",
        "|---|---|---|",
    ]
    for k in TOP_KS:
        s = summary[f"recall@{k}"]
        lines.append(f"| Recall@{k} | {s['tfidf']:.0%} | {s['vector']:.0%} |")
    lines += ["", "## 逐题明细", "", "| 问题 | 期望来源 | TF-IDF@1 | 向量@1 |"]
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['question']} | {row['expected']} | "
            f"{row['tfidf_top1']} | {row['vector_top1']} |"
        )
    md_path = report_dir / "rag_eval_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{report_dir / 'rag_eval_report.json'} 和 {md_path}")


if __name__ == "__main__":
    main()
