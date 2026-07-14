from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .clients import ChatResult, OllamaChatClient, OllamaEmbeddingClient


SCENE_DESCRIPTIONS = {
    "日常闲聊": (
        "用户没有明确任务，主要是轻松聊天、分享日常、吐槽小事、找话题、放松陪聊。"
        "重点是自然回应和延续聊天。"
    ),
    "情绪陪伴": (
        "用户主要表达难过、焦虑、失落、压力、孤独、烦躁、低落等情绪，"
        "需要被理解、安慰、陪伴或情绪支持。"
    ),
    "知识解释": (
        "用户主要询问概念、原因、区别、原理或事实说明，"
        "希望知道是什么、为什么、有什么不同。"
    ),
    "生活建议": (
        "用户提出具体生活问题，希望获得可直接执行的通用建议，"
        "例如作息、学习、饮食、收纳、效率、放松、日常安排。"
    ),
    "追问澄清": (
        "用户要求推荐、选择、规划、安排、方案或个性化建议，"
        "但缺少目标、预算、时间、地点、偏好、限制条件、当前水平等关键条件，"
        "需要先追问再回答。"
    ),
    "安全敏感": (
        "用户涉及自伤轻生、违法违规、侵犯隐私、危险行为、医疗用药、极端节食、"
        "投资买卖建议、制作危险物品或其他需要谨慎拒绝或安全引导的内容。"
    ),
}

_CENTROID_CACHE: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}


@dataclass
class LocalClassification:
    scene: str
    confidence: str
    reason: str
    result: ChatResult


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("vectors 不能为空")
    size = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(size)]


def load_reference_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            scene = (row.get("scene") or "").strip()
            user_query = (row.get("user_query") or "").strip()
            if scene and user_query:
                rows.append(
                    {
                        "id": (row.get("id") or "").strip(),
                        "scene": scene,
                        "user_query": user_query,
                    }
                )
    if not rows:
        raise RuntimeError(f"参考集为空: {path}")
    return rows


def extract_json_object(text: str, required_key: str | None = None) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text.strip()))):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (required_key is None or required_key in value):
            return value
    raise ValueError(f"未找到有效 JSON 对象: {text}")


def build_rerank_prompt(candidate_scenes: list[str]) -> str:
    descriptions = "\n".join(
        f"- {scene}：{SCENE_DESCRIPTIONS[scene]}"
        for scene in candidate_scenes
    )
    return f"""你是语音助手场景复判器。

只能从候选场景中选择一个，不能输出候选之外的场景。

候选场景：
{descriptions}

判断要求：
1. 如果涉及自伤轻生、违法违规、侵犯隐私、危险物品、医疗用药、极端减肥、借贷风险、投资买卖建议、保证收益、内幕消息、炒币股票基金等风险，优先选择安全敏感。
2. 如果主要是表达难过、焦虑、低落、压力，优先选择情绪陪伴。
3. 如果是在问概念、原因、区别、原理，选择知识解释。
4. 如果是在要推荐、规划、选择、个性化方案，但缺少关键条件，选择追问澄清。
5. 如果可以直接给通用可执行建议，选择生活建议。
6. 如果只是轻松聊天、分享日常、找话题，选择日常闲聊。

只输出一行 JSON：
{{"scene":"场景名","confidence":0.0,"reason":"简短原因"}}"""


def parse_rerank_response(text: str, *, allowed_scenes: set[str]) -> tuple[str, str]:
    data = extract_json_object(text, required_key="scene")
    scene = str(data.get("scene", "")).strip()
    if scene not in allowed_scenes:
        raise ValueError(f"复判输出不在候选范围内: {scene}")
    reason = str(data.get("reason", "")).strip()
    return scene, reason


class LocalSceneClassifier:
    def __init__(
        self,
        *,
        reference_path: str | Path,
        embedding_client: OllamaEmbeddingClient,
        rerank_client: OllamaChatClient,
        rerank_threshold: float = 0.04,
        rerank_candidates: int = 3,
    ) -> None:
        self.reference_path = Path(reference_path)
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.rerank_threshold = rerank_threshold
        self.rerank_candidates = rerank_candidates
        self._centroids: dict[str, list[float]] | None = None
        self._reference_latency_ms = 0

    def _ensure_centroids(self) -> None:
        if self._centroids is not None:
            self._reference_latency_ms = 0
            return

        stat = self.reference_path.stat()
        cache_key = (
            str(self.reference_path.resolve()),
            self.embedding_client.base_url,
            self.embedding_client.model,
            stat.st_mtime_ns,
        )
        cached_centroids = _CENTROID_CACHE.get(cache_key)
        if cached_centroids is not None:
            self._centroids = cached_centroids
            self._reference_latency_ms = 0
            return

        rows = load_reference_rows(self.reference_path)
        result = self.embedding_client.embed([row["user_query"] for row in rows])
        scene_vectors: dict[str, list[list[float]]] = defaultdict(list)
        for row, embedding in zip(rows, result.embeddings):
            scene_vectors[row["scene"]].append(l2_normalize(embedding))

        self._centroids = {
            scene: l2_normalize(mean_vector(vectors))
            for scene, vectors in scene_vectors.items()
        }
        _CENTROID_CACHE[cache_key] = self._centroids
        self._reference_latency_ms = result.latency_ms

    def classify(self, user_query: str) -> LocalClassification:
        self._ensure_centroids()
        if self._centroids is None:
            raise RuntimeError("本地分类中心未初始化")

        query_result = self.embedding_client.embed([user_query])
        query_embedding = l2_normalize(query_result.embeddings[0])
        scores = sorted(
            (
                (scene, cosine(query_embedding, centroid))
                for scene, centroid in self._centroids.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top1_scene, top1_score = scores[0]
        top2_scene, top2_score = scores[1] if len(scores) > 1 else ("", 0.0)
        margin = top1_score - top2_score
        candidate_scenes = [scene for scene, _ in scores[: self.rerank_candidates]]
        predicted_scene = top1_scene
        rerank_used = False
        rerank_reason = ""
        rerank_raw_response = ""
        rerank_latency_ms = 0

        if margin <= self.rerank_threshold:
            rerank_used = True
            system_prompt = build_rerank_prompt(candidate_scenes)
            try:
                rerank_result = self.rerank_client.chat(
                    system_prompt=system_prompt,
                    user_query=user_query,
                    temperature=0.0,
                    max_tokens=128,
                )
                rerank_latency_ms = rerank_result.latency_ms
                rerank_raw_response = rerank_result.answer
                predicted_scene, rerank_reason = parse_rerank_response(
                    rerank_result.answer,
                    allowed_scenes=set(candidate_scenes),
                )
            except Exception as exc:
                predicted_scene = top1_scene
                rerank_reason = f"复判失败，保留 embedding 结果: {exc}"

        raw = {
            "classifier": "local_embedding_centroid_rerank",
            "embedding_model": self.embedding_client.model,
            "rerank_model": self.rerank_client.model,
            "top1_scene": top1_scene,
            "top1_score": round(top1_score, 6),
            "top2_scene": top2_scene,
            "top2_score": round(top2_score, 6),
            "margin": round(margin, 6),
            "candidate_scenes": candidate_scenes,
            "rerank_used": rerank_used,
            "rerank_threshold": self.rerank_threshold,
            "rerank_reason": rerank_reason,
            "rerank_raw_response": rerank_raw_response,
        }
        answer = json.dumps(
            {
                "scene": predicted_scene,
                "confidence": round(max(0.0, min(1.0, margin / self.rerank_threshold)), 4)
                if self.rerank_threshold > 0
                else "",
                "reason": rerank_reason or f"embedding top1={top1_scene}",
            },
            ensure_ascii=False,
        )
        result = ChatResult(
            answer=answer,
            latency_ms=self._reference_latency_ms + query_result.latency_ms + rerank_latency_ms,
            input_tokens=None,
            output_tokens=None,
            raw=raw,
        )
        return LocalClassification(
            scene=predicted_scene,
            confidence="",
            reason=rerank_reason,
            result=result,
        )
