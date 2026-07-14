from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass
class ChatResult:
    answer: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    raw: dict
    first_token_latency_ms: int | None = None


@dataclass
class OllamaEmbeddingResult:
    embeddings: list[list[float]]
    latency_ms: int
    model: str


class DoubaoClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        return self.chat_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw, latency_ms = _post_json(
            url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout=60,
            provider="Doubao",
        )
        return _parse_chat_result(raw, latency_ms, provider="Doubao")

    def chat_messages_stream(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], None],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return _stream_chat_result(
            url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout=60,
            provider="Doubao",
            on_delta=on_delta,
        )


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        return self.chat_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        raw, latency_ms = _post_json(
            url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout=120,
            provider="DeepSeek",
        )
        return _parse_chat_result(raw, latency_ms, provider="DeepSeek")

    def chat_messages_stream(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], None],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return _stream_chat_result(
            url=self.base_url,
            api_key=self.api_key,
            payload=payload,
            timeout=120,
            provider="DeepSeek",
            on_delta=on_delta,
        )


def _strip_thinking_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if cleaned:
        return cleaned
    cleaned = re.sub(
        r"^Thinking\.\.\..*?\.\.\.done thinking\.\s*",
        "",
        text,
        flags=re.S,
    ).strip()
    return cleaned or text.strip()


class OllamaChatClient:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户输入：{user_query}"},
            ],
            "stream": False,
            "think": False if self.model.lower().startswith("qwen3") else None,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if payload["think"] is None:
            del payload["think"]

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama 调用失败，请确认本地服务已启动: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        message = data.get("message") or {}
        answer = _strip_thinking_text(str(message.get("content") or ""))
        if not answer:
            raise RuntimeError(f"Ollama 返回为空: {data}")
        return ChatResult(
            answer=answer,
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            raw=data,
        )


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> OllamaEmbeddingResult:
        if not texts:
            return OllamaEmbeddingResult(embeddings=[], latency_ms=0, model=self.model)

        payload = {
            "model": self.model,
            "input": texts,
            "truncate": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama embedding 调用失败，请确认本地服务已启动: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(f"Ollama embedding 返回格式异常: {data}")
        return OllamaEmbeddingResult(
            embeddings=embeddings,
            latency_ms=latency_ms,
            model=str(data.get("model") or self.model),
        )


def _post_json(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    provider: str,
) -> tuple[dict, int]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} API HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider} API 网络请求失败: {exc}") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{provider} API 返回了无效 JSON") from exc
    return raw, latency_ms


def _stream_chat_result(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    provider: str,
    on_delta: Callable[[str], None],
) -> ChatResult:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    start = time.perf_counter()
    first_token_latency_ms: int | None = None
    answer_parts: list[str] = []
    usage: dict = {}
    response_id = ""
    response_model = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if not data_text or data_text == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_text)
                except json.JSONDecodeError:
                    continue

                response_id = str(chunk.get("id") or response_id)
                response_model = str(chunk.get("model") or response_model)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not isinstance(content, str) or not content:
                    continue
                if first_token_latency_ms is None:
                    first_token_latency_ms = int((time.perf_counter() - start) * 1000)
                answer_parts.append(content)
                on_delta(content)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} API HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider} API 网络请求失败: {exc}") from exc

    answer = "".join(answer_parts).strip()
    if not answer:
        raise RuntimeError(f"{provider} API 流式返回了空内容")
    latency_ms = int((time.perf_counter() - start) * 1000)
    raw = {
        "id": response_id,
        "model": response_model,
        "usage": usage,
        "stream": True,
    }
    return ChatResult(
        answer=answer,
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw=raw,
        first_token_latency_ms=first_token_latency_ms,
    )


def _parse_chat_result(raw: dict, latency_ms: int, provider: str) -> ChatResult:
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider} API 返回中没有 choices")

    message = choices[0].get("message") or {}
    answer = (message.get("content") or "").strip()
    if not answer:
        raise RuntimeError(f"{provider} API 返回了空内容")

    usage = raw.get("usage") or {}
    return ChatResult(
        answer=answer,
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        raw=raw,
    )
