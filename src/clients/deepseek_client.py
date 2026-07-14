import json
import time
import urllib.error
import urllib.request

from src.clients.doubao_client import ChatResult


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float,
        max_tokens: int,
        thinking_enabled: bool,
        response_format: dict | None = None,
    ) -> ChatResult:
        return self._chat_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            response_format=response_format,
        )

    def _chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        thinking_enabled: bool,
        response_format: dict | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
        }
        if response_format is not None:
            payload["response_format"] = response_format

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API 网络请求失败: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        raw = json.loads(raw_text)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek API 返回中没有 choices: {raw}")

        message = choices[0].get("message") or {}
        answer = (message.get("content") or "").strip()
        if not answer:
            raise RuntimeError("DeepSeek API 返回了空内容")

        usage = raw.get("usage") or {}
        return ChatResult(
            answer=answer,
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=raw,
        )

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        return self._chat(
            system_prompt=system_prompt,
            user_query=user_query,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=False,
        )

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult:
        return self._chat_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=False,
        )

    def chat_json(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResult:
        return self._chat(
            system_prompt=system_prompt,
            user_query=user_query,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_enabled=True,
            response_format={"type": "json_object"},
        )
