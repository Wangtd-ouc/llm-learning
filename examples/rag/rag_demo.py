"""RAG 最小可运行示例（纯 Python 标准库 + 本机 Ollama）。

流程：
  1. 读取 knowledge_base/ 下的文档，按段落切块；
  2. 用 TF-IDF 给每个文本块建索引；
  3. 把问题转成向量，检索最相关的 K 个文本块；
  4. 把文本块拼进提示词，让模型只依据资料回答并标注来源。

生产环境通常把第 2、3 步换成“嵌入模型 + 向量数据库”，骨架相同。

用法：
  python3 examples/rag/rag_demo.py "LoRA 主要微调的是什么参数？"
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

# 让脚本可以直接从 examples/ 目录运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ollama_client import chat, DEFAULT_MODEL  # noqa: E402

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge_base"


# ---------- 1. 文档加载与切块 ----------

def load_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict]:
    """读取 .md 文件，按“标题 + 段落”切块；列表项不会被空行切断。"""
    chunks: list[dict] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        heading: str | None = None  # 当前小节标题
        buf: list[str] = []        # 当前段落内容

        def flush() -> None:
            """把缓存中的段落落盘为一个检索块（标题 + 正文）。"""
            text = " ".join(part.strip() for part in buf).strip()
            if len(text) >= 8:  # 过滤太短的碎片
                full = f"{heading}：{text}" if heading else text
                chunks.append({"source": path.name, "text": full})

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#"):
                flush()
                buf = []
                heading = line.lstrip("#").strip()
                i += 1
            elif line == "":
                # 空行时向后看：若下一非空行是列表项，则列表未结束，不切块
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and re.match(
                    r"^(\d+[.、)]|[-*•])\s", lines[j].strip()
                ):
                    i = j
                    continue
                flush()
                buf = []
                i += 1
            else:
                # 去掉行内 Markdown 符号，保留可读文本
                buf.append(re.sub(r"[#*`>|]", "", line))
                i += 1
        flush()
    return chunks


# ---------- 2. 分词与 TF-IDF 索引 ----------

def tokenize(text: str) -> list[str]:
    """简单中文分词：英文/数字按词，中文按相邻双字组合（bigram）。"""
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    cjk = [t for t in tokens if re.fullmatch(r"[\u4e00-\u9fff]", t)]
    bigrams = [a + b for a, b in zip(cjk, cjk[1:])]
    return tokens + bigrams


def build_index(chunks: list[dict]) -> tuple[list[dict], dict]:
    """构建 TF-IDF 索引：返回 (chunks, idf)，idf[term] = log(N/(1+df))。"""
    doc_freq: Counter = Counter()
    chunk_tf: list[Counter] = []
    for chunk in chunks:
        tf = Counter(tokenize(chunk["text"]))
        chunk_tf.append(tf)
        doc_freq.update(tf.keys())
    n = len(chunks)
    idf = {term: math.log(n / (1 + df)) for term, df in doc_freq.items()}
    return list(zip(chunks, chunk_tf)), idf


def query_vector(query: str, idf: dict) -> dict[str, float]:
    q_tf = Counter(tokenize(query))
    return {term: count * idf.get(term, 0) for term, count in q_tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(query: str, index: list, idf: dict, top_k: int = 3) -> list[dict]:
    """返回最相关的 top_k 个文本块。"""
    q_vec = query_vector(query, idf)
    scored = []
    for chunk, tf in index:
        doc_vec = {term: count * idf.get(term, 0) for term, count in tf.items()}
        scored.append((cosine_similarity(q_vec, doc_vec), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k] if score > 0]


# ---------- 3. 生成回答 ----------

def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[来源: {hit['source']}]\n{hit['text']}" for hit in hits
    )
    return (
        "你是知识库问答程序。请只依据下面资料回答，不要编造。\n"
        "如果资料中没有答案，请回答“资料中没有相关信息”。\n"
        "回答末尾用一行列出引用的来源文件名。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{query}\n\n"
        "回答："
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 最小示例")
    parser.add_argument("query", help="要问的问题")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 模型名")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回的文本块数")
    args = parser.parse_args()

    print(f"[1/4] 加载知识库并切块：{KNOWLEDGE_DIR}")
    chunks = load_chunks()
    print(f"      共 {len(chunks)} 个文本块")

    print("[2/4] 建立 TF-IDF 索引")
    index, idf = build_index(chunks)

    print(f"[3/4] 检索问题：{args.query}")
    hits = retrieve(args.query, index, idf, top_k=args.top_k)
    for hit in hits:
        print(f"      ★ {hit['source']}（相似度命中）")
    if not hits:
        print("      没有检索到相关内容，请换一个问题或补充知识库。")
        return

    print(f"[4/4] 调用 Ollama 模型生成回答（模型：{args.model}）")
    messages = [{"role": "user", "content": build_prompt(args.query, hits)}]
    result = chat(messages, model=args.model, temperature=0.2)
    answer = result["message"]["content"].strip()

    print("\n" + "=" * 50)
    print("回答：")
    print(answer)
    print("=" * 50)
    print("检索到的来源：")
    for hit in hits:
        print(f"  - {hit['source']}")


if __name__ == "__main__":
    main()
