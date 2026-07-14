from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaChatResult:
    answer: str
    latency_ms: int
    model: str


def strip_thinking_text(text: str) -> str:
    cleaned = re_sub_think_blocks(text).strip()
    if cleaned:
        return cleaned
    return text.strip()


def re_sub_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^Thinking\.\.\..*?\.\.\.done thinking\.\s*", "", text, flags=re.S)
    return text


class OllamaClient:
    def __init__(
        self,
        *,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _prepare_system_prompt(self, system_prompt: str) -> str:
        if self.model.lower().startswith("qwen3") and not system_prompt.lstrip().startswith("/no_think"):
            return "/no_think\n" + system_prompt
        return system_prompt

    def _prepare_user_query(self, user_query: str) -> str:
        content = f"用户输入：{user_query}"
        if self.model.lower().startswith("qwen3"):
            return "/no_think\n" + content
        return content

    def _chat_generate_raw(
        self,
        *,
        system_prompt: str,
        user_query: str,
        temperature: float,
        max_tokens: int,
    ) -> OllamaChatResult:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n用户输入：{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n/no_think\n"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["<|im_end|>", "<|im_start|>"],
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
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
        answer = strip_thinking_text(str(data.get("response") or ""))
        if not answer:
            if data.get("thinking"):
                raise RuntimeError(
                    "Ollama 返回为空，qwen3 的 thinking 内容可能占满了输出 token。"
                    "请增大 --max-tokens，或换用非 thinking 分类模型。"
                )
            raise RuntimeError(f"Ollama 返回为空: {data}")

        return OllamaChatResult(
            answer=answer,
            latency_ms=latency_ms,
            model=str(data.get("model") or self.model),
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> OllamaChatResult:
        if self.model.lower().startswith("qwen3"):
            return self._chat_generate_raw(
                system_prompt=system_prompt,
                user_query=user_query,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._prepare_system_prompt(system_prompt)},
                {"role": "user", "content": self._prepare_user_query(user_query)},
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
        answer = strip_thinking_text(str(message.get("content") or ""))
        if not answer:
            raise RuntimeError(f"Ollama 返回为空: {data}")

        return OllamaChatResult(
            answer=answer,
            latency_ms=latency_ms,
            model=str(data.get("model") or self.model),
        )
