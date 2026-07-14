from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaEmbeddingResult:
    embeddings: list[list[float]]
    latency_ms: int
    model: str


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
