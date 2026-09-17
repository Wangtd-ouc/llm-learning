"""向量检索版 RAG：用 Ollama 嵌入模型替代 TF-IDF。

与 rag_demo.py 的唯一区别在“检索”这一步：
  - TF-IDF：靠关键词重合；
  - 向量检索：把文本转成语义向量，靠语义相似度。

用法：
  python3 examples/rag/rag_embed_demo.py "LoRA 主要微调的是什么参数？"
  python3 examples/rag/rag_embed_demo.py "RAG 的五个环节是什么？" --top-k 5

首次运行会为知识库生成向量并缓存到 embeddings_cache.json，之后秒开。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ollama_client import (  # noqa: E402
    chat,
    embed,
    DEFAULT_MODEL,
    DEFAULT_EMBED_MODEL,
)
from rag.rag_demo import load_chunks, build_prompt  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "embeddings_cache.json"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def get_embedding(text: str, embed_model: str, cache: dict) -> list[float]:
    """取文本向量；命中缓存则直接返回，避免重复计算。"""
    key = text_hash(text)
    if key in cache:
        return cache[key]
    vector = embed(text, model=embed_model)
    cache[key] = vector
    return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="向量检索版 RAG")
    parser.add_argument("query", help="要问的问题")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="生成回答的模型")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="嵌入模型")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    cache: dict = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    print(f"[1/4] 加载知识库并切块（复用 rag_demo 的逻辑）")
    chunks = load_chunks()
    print(f"      共 {len(chunks)} 个文本块")

    print(f"[2/4] 计算向量索引（嵌入模型：{args.embed_model}）")
    vectors = [get_embedding(c["text"], args.embed_model, cache) for c in chunks]
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[3/4] 检索问题：{args.query}")
    query_vec = get_embedding(args.query, args.embed_model, cache)
    scored = sorted(
        ((cosine_similarity(query_vec, v), c) for v, c in zip(vectors, chunks)),
        key=lambda x: x[0],
        reverse=True,
    )
    hits = [c for s, c in scored[: args.top_k] if s > 0.2]  # 低于阈值的视为不相关
    for c in hits:
        print(f"      ★ {c['source']}（相似度命中）")
    if not hits:
        print("      没有检索到相关内容。")
        return

    print(f"[4/4] 调用 Ollama 生成回答（模型：{args.model}）")
    result = chat(
        [{"role": "user", "content": build_prompt(args.query, hits)}],
        model=args.model,
        temperature=0.2,
    )
    print("\n" + "=" * 50)
    print("回答：")
    print(result["message"]["content"].strip())
    print("=" * 50)
    print("检索到的来源：")
    for c in hits:
        print(f"  - {c['source']}")


if __name__ == "__main__":
    main()
