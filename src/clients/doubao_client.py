import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatResult:
    answer: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    raw: dict


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

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Doubao API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Doubao API 网络请求失败: {exc}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        raw = json.loads(raw_text)

        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"Doubao API 返回中没有 choices: {raw}")

        message = choices[0].get("message") or {}
        answer = (message.get("content") or "").strip()

        usage = raw.get("usage") or {}
        return ChatResult(
            answer=answer,
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            raw=raw,
        )
