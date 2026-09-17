"""极简 Ollama 调用封装（仅用 Python 标准库，无第三方依赖）。

用法：
    from common.ollama_client import chat
    reply = chat("qwen3.5:4b", [{"role": "user", "content": "你好"}])
    print(reply["message"]["content"])
"""

from __future__ import annotations

import json
import urllib.request

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
TIMEOUT = 180  # 生成较长回答时可能需要较久


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    num_predict: int = 1024,
    think: bool = False,
    tools: list | None = None,
) -> dict:
    """调用 Ollama /api/chat，返回完整 JSON 响应。"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # qwen3.5 等推理模型默认会输出“思考”过程；
        # 示例里关闭它以加快速度、让回答更可控。
        "think": think,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(content: str, model: str = DEFAULT_MODEL, **kwargs) -> str:
    """便捷函数：只发一条 user 消息，返回模型文本。"""
    result = chat([{"role": "user", "content": content}], model=model, **kwargs)
    return result["message"]["content"]


def embed(prompt: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    """调用 Ollama /api/embeddings，返回文本向量。"""
    payload = {"model": model, "prompt": prompt}
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))["embedding"]


def chat_openai(
    messages: list[dict],
    base_url: str,
    model: str,
    temperature: float = 0.0,
    num_predict: int = 512,
) -> dict:
    """调用 OpenAI 兼容接口（如 mlx_lm server），返回 message 字典。"""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": num_predict,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]


if __name__ == "__main__":
    # 自检：打印模型回复
    print(ask("用一句话说明什么是 LLM。"))
