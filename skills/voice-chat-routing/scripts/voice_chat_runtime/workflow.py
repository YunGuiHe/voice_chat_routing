from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .clients import (
    ChatResult,
    DeepSeekClient,
    DoubaoClient,
    OllamaChatClient,
    OllamaEmbeddingClient,
)
from .config import (
    get_classifier_mode,
    get_deepseek_api_key,
    get_deepseek_base_url,
    get_deepseek_model,
    get_doubao_api_key,
    get_doubao_base_url,
    get_doubao_model,
    get_ollama_base_url,
    get_ollama_embedding_model,
    get_ollama_rerank_model,
    get_rerank_candidates,
    get_rerank_threshold,
    get_recent_rounds,
    get_summary_threshold_rounds,
    get_long_term_memory_enabled,
    get_long_term_memory_extract_limit,
    get_long_term_memory_limit,
    get_state_db_path,
    load_skill_environment,
)
from .local_scene_classifier import LocalSceneClassifier
from .memory_store import ConversationMessage, LongTermMemory, LongTermMemoryStore
from .text_utils import read_text, sanitize_tts_text


SKILL_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PROMPT = "assets/prompts/classifier.txt"
BASELINE_PROMPT = "assets/prompts/baseline.txt"
LONG_TERM_MEMORY_PROMPT = "assets/prompts/long_term_memory_extract.txt"
SUMMARY_PROMPT = "assets/prompts/session_summary.txt"
REFERENCE_CASES = "assets/data/scene_reference_centroid_v1.csv"
SCENE_PROMPTS = {
    "日常闲聊": "assets/prompts/scenes/chat.txt",
    "情绪陪伴": "assets/prompts/scenes/emotion.txt",
    "知识解释": "assets/prompts/scenes/knowledge.txt",
    "生活建议": "assets/prompts/scenes/life_advice.txt",
    "追问澄清": "assets/prompts/scenes/clarification.txt",
    "安全敏感": "assets/prompts/scenes/safety.txt",
}
ALLOWED_SCENES = set(SCENE_PROMPTS)
MODEL_ROUTING = {
    "日常闲聊": "doubao",
    "情绪陪伴": "doubao",
    "知识解释": "deepseek",
    "生活建议": "doubao",
    "追问澄清": "doubao",
    "安全敏感": "doubao",
}
ALLOWED_MEMORY_TYPES = {
    "profile",
    "preference",
    "goal",
    "constraint",
    "communication_style",
}


class ChatClient(Protocol):
    model: str

    def chat(
        self,
        system_prompt: str,
        user_query: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult: ...

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult: ...


class StreamGenerationError(RuntimeError):
    def __init__(self, message: str, partial_output: bool) -> None:
        super().__init__(message)
        self.partial_output = partial_output

def parse_classifier_response(raw_response: str) -> tuple[str, str, str]:
    text = raw_response.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"分类结果不是 JSON: {raw_response}")

    data = json.loads(match.group(0))
    predicted_scene = str(data.get("scene", "")).strip()
    if predicted_scene not in ALLOWED_SCENES:
        raise ValueError(f"未知场景: {predicted_scene}")

    confidence = data.get("confidence", "")
    if confidence != "":
        confidence = str(confidence).strip()
    reason = str(data.get("reason", "")).strip()
    return predicted_scene, confidence, reason


def _extract_json_object(raw_response: str) -> dict[str, object]:
    text = raw_response.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"模型返回不是 JSON: {raw_response}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型返回 JSON 不是对象")
    return data


def _long_term_memory_lines(memories: list[LongTermMemory], include_evidence: bool = False) -> str:
    if not memories:
        return "无"
    lines = []
    for memory in memories:
        line = (
            f"[{memory.id}] type={memory.memory_type}, "
            f"priority={memory.priority}, content={memory.content}"
        )
        if include_evidence and memory.evidence:
            line += f", evidence={memory.evidence}"
        lines.append(line)
    return "\n".join(lines)


def _message_lines(messages: list[ConversationMessage]) -> str:
    if not messages:
        return "无"
    role_names = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role_names[item.role]}：{item.content}" for item in messages)


@dataclass
class VoiceChatReply:
    user_query: str
    session_id: str = "default"
    user_id: str = "default"
    scene: str = ""
    answer: str = ""
    model_name: str = ""
    prompt_name: str = ""
    latency_ms: int = 0
    classifier_latency_ms: int = 0
    generation_latency_ms: int = 0
    first_token_latency_ms: int | None = None
    memory_latency_ms: int = 0
    summary_latency_ms: int = 0
    history_mode: str = ""
    history_rounds: int = 0
    recent_messages_used: int = 0
    summary_used: bool = False
    summary_updated: bool = False
    summary_pending: bool = False
    summary_update_mode: str = ""
    long_term_memories_used: int = 0
    long_term_memory_updated: bool = False
    long_term_memory_pending: bool = False
    long_term_memory_update_mode: str = ""
    total_tokens: int = 0
    fallback_used: bool = False
    error: str = ""
    memory_error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VoiceChatSkill:
    def __init__(
        self,
        client: ChatClient,
        skill_root: str | Path = SKILL_ROOT,
        deepseek_client: ChatClient | None = None,
        memory_store: LongTermMemoryStore | None = None,
        classifier_prompt_path: str | Path = CLASSIFIER_PROMPT,
        baseline_prompt_path: str | Path = BASELINE_PROMPT,
        long_term_memory_prompt_path: str | Path = LONG_TERM_MEMORY_PROMPT,
        summary_prompt_path: str | Path = SUMMARY_PROMPT,
        scene_prompt_paths: dict[str, str | Path] | None = None,
        model_routing: dict[str, str] | None = None,
        classifier_mode: str = "local",
        local_classifier: LocalSceneClassifier | None = None,
        classifier_temperature: float = 0.0,
        generation_temperature: float = 0.3,
        summary_temperature: float = 0.0,
        classifier_max_tokens: int = 128,
        generation_max_tokens: int = 512,
        summary_max_tokens: int = 512,
        long_term_memory_max_tokens: int = 512,
        retries: int = 2,
        long_term_memory_enabled: bool = True,
        long_term_memory_limit: int = 5,
        long_term_memory_extract_limit: int = 20,
        summary_threshold_rounds: int = 6,
        recent_rounds: int = 3,
        background_memory_updates: bool = False,
        background_update_delay_seconds: float = 0.0,
        background_update_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.client = client
        self.deepseek_client = deepseek_client
        self.clients: dict[str, ChatClient] = {"doubao": client}
        if deepseek_client is not None:
            self.clients["deepseek"] = deepseek_client
        self.memory_store = memory_store

        self.skill_root = Path(skill_root).resolve()
        self.classifier_temperature = classifier_temperature
        self.generation_temperature = generation_temperature
        self.summary_temperature = summary_temperature
        self.classifier_max_tokens = classifier_max_tokens
        self.generation_max_tokens = generation_max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.long_term_memory_max_tokens = long_term_memory_max_tokens
        self.retries = retries
        self.long_term_memory_enabled = long_term_memory_enabled
        self.long_term_memory_limit = long_term_memory_limit
        self.long_term_memory_extract_limit = long_term_memory_extract_limit
        self.summary_threshold_rounds = summary_threshold_rounds
        self.recent_rounds = recent_rounds
        self.background_memory_updates = background_memory_updates
        self.background_update_delay_seconds = max(background_update_delay_seconds, 0.0)
        self.background_update_callback = background_update_callback
        self._background_update_lock = threading.Lock()
        self._background_update_versions: dict[tuple[str, str], int] = {}
        self.classifier_prompt_path = classifier_prompt_path
        self.baseline_prompt_path = baseline_prompt_path
        self.long_term_memory_prompt_path = long_term_memory_prompt_path
        self.summary_prompt_path = summary_prompt_path
        self.scene_prompt_paths = scene_prompt_paths or SCENE_PROMPTS
        self.model_routing = model_routing or MODEL_ROUTING
        self.classifier_mode = classifier_mode
        self.local_classifier = local_classifier

        self.classifier_prompt = read_text(self._resolve(classifier_prompt_path))
        self.baseline_prompt = read_text(self._resolve(baseline_prompt_path))
        self.long_term_memory_prompt = read_text(self._resolve(long_term_memory_prompt_path))
        self.summary_prompt = read_text(self._resolve(summary_prompt_path))
        self.scene_prompts = {
            scene: read_text(self._resolve(prompt_path))
            for scene, prompt_path in self.scene_prompt_paths.items()
        }

    @classmethod
    def from_env(
        cls,
        skill_root: str | Path = SKILL_ROOT,
        **kwargs,
    ) -> "VoiceChatSkill":
        root = Path(skill_root).resolve()
        load_skill_environment(root)
        client = DoubaoClient(
            api_key=get_doubao_api_key(),
            base_url=get_doubao_base_url(),
            model=get_doubao_model(),
        )
        deepseek_client = None
        try:
            deepseek_client = DeepSeekClient(
                api_key=get_deepseek_api_key(),
                base_url=get_deepseek_base_url(),
                model=get_deepseek_model(),
            )
        except RuntimeError:
            pass
        classifier_mode = get_classifier_mode()
        local_classifier = None
        if classifier_mode == "local":
            ollama_base_url = get_ollama_base_url()
            local_classifier = LocalSceneClassifier(
                reference_path=root / REFERENCE_CASES,
                embedding_client=OllamaEmbeddingClient(
                    model=get_ollama_embedding_model(),
                    base_url=ollama_base_url,
                ),
                rerank_client=OllamaChatClient(
                    model=get_ollama_rerank_model(),
                    base_url=ollama_base_url,
                ),
                rerank_threshold=get_rerank_threshold(),
                rerank_candidates=get_rerank_candidates(),
            )
        kwargs.setdefault("summary_threshold_rounds", get_summary_threshold_rounds())
        kwargs.setdefault("recent_rounds", get_recent_rounds())
        return cls(
            client=client,
            deepseek_client=deepseek_client,
            memory_store=LongTermMemoryStore(get_state_db_path(root)),
            skill_root=root,
            classifier_mode=classifier_mode,
            local_classifier=local_classifier,
            long_term_memory_enabled=get_long_term_memory_enabled(),
            long_term_memory_limit=get_long_term_memory_limit(),
            long_term_memory_extract_limit=get_long_term_memory_extract_limit(),
            **kwargs,
        )

    def _resolve(self, path: str | Path) -> Path:
        prompt_path = Path(path)
        return prompt_path if prompt_path.is_absolute() else self.skill_root / prompt_path

    def _chat_with_retry(
        self,
        client: ChatClient,
        system_prompt: str,
        user_query: str,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                result = client.chat(
                    system_prompt=system_prompt,
                    user_query=user_query,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not result.answer.strip():
                    raise RuntimeError("模型返回了空内容")
                return result
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
        raise RuntimeError(last_error)

    def _chat_messages_with_retry(
        self,
        client: ChatClient,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                result = client.chat_messages(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not result.answer.strip():
                    raise RuntimeError("模型返回了空内容")
                return result
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
        raise RuntimeError(last_error)

    def _load_context(
        self,
        session_id: str,
    ) -> tuple[str, list[ConversationMessage], str, int, bool]:
        if self.memory_store is None:
            return "full", [], "", 0, False

        history_rounds = self.memory_store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return (
                "full",
                self.memory_store.get_messages(session_id),
                "",
                history_rounds,
                False,
            )

        summary = self.memory_store.get_summary(session_id)
        recent_messages = self.memory_store.get_recent_messages(session_id, self.recent_rounds)
        return (
            "summary_window",
            recent_messages,
            summary.summary if summary else "",
            history_rounds,
            bool(summary and summary.summary.strip()),
        )

    def _build_generation_messages(
        self,
        prompt: str,
        user_query: str,
        recent_messages: list[ConversationMessage],
        summary: str,
        long_term_memories: list[LongTermMemory],
    ) -> list[dict[str, str]]:
        system_blocks = [
            prompt,
            "你会收到同一会话的上下文。请结合上下文回答当前用户输入，不要机械复述历史。",
        ]
        if long_term_memories:
            system_blocks.append(
                "长期记忆：\n"
                f"{_long_term_memory_lines(long_term_memories)}\n"
                "这些信息来自用户长期记忆，只在相关时参考；不要主动说明“我记得”。"
            )
        if summary:
            system_blocks.append(f"较早会话摘要：\n{summary}")

        messages = [{"role": "system", "content": "\n\n".join(system_blocks)}]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in recent_messages
        )
        messages.append({"role": "user", "content": user_query})
        return messages

    def _classify_with_llm(self, user_query: str) -> tuple[str, str, str, ChatResult]:
        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                result = self.client.chat(
                    system_prompt=self.classifier_prompt,
                    user_query=user_query,
                    temperature=self.classifier_temperature,
                    max_tokens=self.classifier_max_tokens,
                )
                if not result.answer.strip():
                    raise RuntimeError("模型返回了空内容")
                scene, confidence, reason = parse_classifier_response(result.answer)
                return scene, confidence, reason, result
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.retries:
                    time.sleep(attempt)
        raise RuntimeError(last_error)

    def classify(self, user_query: str) -> tuple[str, str, str, ChatResult]:
        if self.classifier_mode == "local":
            if self.local_classifier is None:
                raise RuntimeError("本地分类器未初始化")
            local_result = self.local_classifier.classify(user_query)
            return (
                local_result.scene,
                local_result.confidence,
                local_result.reason,
                local_result.result,
            )

        return self._classify_with_llm(user_query)

    def _generate(
        self,
        client: ChatClient,
        user_query: str,
        prompt: str,
        recent_messages: list[ConversationMessage],
        summary: str,
        long_term_memories: list[LongTermMemory],
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatResult:
        messages = self._build_generation_messages(
            prompt=prompt,
            user_query=user_query,
            recent_messages=recent_messages,
            summary=summary,
            long_term_memories=long_term_memories,
        )
        stream_method = getattr(client, "chat_messages_stream", None)
        if on_delta is not None and callable(stream_method):
            emitted = False

            def tracked_delta(delta: str) -> None:
                nonlocal emitted
                emitted = True
                on_delta(delta)

            try:
                return stream_method(
                    messages=messages,
                    on_delta=tracked_delta,
                    temperature=self.generation_temperature,
                    max_tokens=self.generation_max_tokens,
                )
            except Exception as exc:
                raise StreamGenerationError(str(exc), partial_output=emitted) from exc

        return self._chat_messages_with_retry(
            client=client,
            messages=messages,
            temperature=self.generation_temperature,
            max_tokens=self.generation_max_tokens,
        )

    def _parse_long_term_memory_update(
        self,
        raw_response: str,
        existing_memory_ids: set[int],
    ) -> tuple[list[dict[str, object]], list[int]]:
        data = _extract_json_object(raw_response)
        raw_memories = data.get("memories", [])
        raw_deactivate_ids = data.get("deactivate_ids", [])
        if not isinstance(raw_memories, list):
            raise ValueError("memories 必须是数组")
        if not isinstance(raw_deactivate_ids, list):
            raise ValueError("deactivate_ids 必须是数组")

        memories: list[dict[str, object]] = []
        for item in raw_memories:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("memory_type", "")).strip()
            content = " ".join(str(item.get("content", "")).split())
            evidence = " ".join(str(item.get("evidence", "")).split())
            if memory_type not in ALLOWED_MEMORY_TYPES or not content:
                continue

            raw_memory_id = item.get("memory_id")
            memory_id = None
            if isinstance(raw_memory_id, int) and raw_memory_id in existing_memory_ids:
                memory_id = raw_memory_id

            try:
                confidence = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            try:
                priority = int(item.get("priority", 0))
            except (TypeError, ValueError):
                priority = 0

            memories.append(
                {
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                    "content": content,
                    "evidence": evidence,
                    "confidence": confidence,
                    "priority": priority,
                }
            )

        deactivate_ids = [
            memory_id
            for memory_id in raw_deactivate_ids
            if isinstance(memory_id, int) and memory_id in existing_memory_ids
        ]
        return memories, deactivate_ids

    def _update_long_term_memories(
        self,
        user_id: str,
        user_query: str,
        assistant_answer: str,
        scene: str,
        conversation_messages: list[ConversationMessage] | None = None,
    ) -> tuple[bool, ChatResult | None]:
        if not self.long_term_memory_enabled or self.memory_store is None:
            return False, None

        existing = self.memory_store.get_active(
            user_id,
            limit=self.long_term_memory_extract_limit,
        )
        existing_ids = {memory.id for memory in existing}
        extraction_query = (
            f"用户ID：{user_id}\n"
            f"已有长期记忆：\n{_long_term_memory_lines(existing, include_evidence=True)}\n\n"
            f"最近对话：\n{_message_lines(conversation_messages or [])}\n\n"
            f"当前场景：{scene}\n"
            f"当前用户输入：{user_query}\n"
            f"当前助手回复：{assistant_answer}"
        )
        result = self._chat_with_retry(
            client=self.client,
            system_prompt=self.long_term_memory_prompt,
            user_query=extraction_query,
            temperature=0.0,
            max_tokens=self.long_term_memory_max_tokens,
        )
        memories, deactivate_ids = self._parse_long_term_memory_update(
            result.answer,
            existing_memory_ids=existing_ids,
        )
        changed = False
        for memory_id in deactivate_ids:
            self.memory_store.deactivate(memory_id, user_id)
            changed = True
        for memory in memories:
            self.memory_store.upsert(
                user_id=user_id,
                memory_type=str(memory["memory_type"]),
                content=str(memory["content"]),
                evidence=str(memory["evidence"]),
                confidence=float(memory["confidence"]),
                priority=int(memory["priority"]),
                memory_id=memory["memory_id"] if isinstance(memory["memory_id"], int) else None,
            )
            changed = True
        return changed, result

    def _update_summary(self, session_id: str) -> tuple[bool, ChatResult | None]:
        if self.memory_store is None:
            return False, None

        history_rounds = self.memory_store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return False, None

        recent_messages = self.memory_store.get_recent_messages(session_id, self.recent_rounds)
        if not recent_messages:
            return False, None
        cutoff_message_id = recent_messages[0].id - 1
        if cutoff_message_id <= 0:
            return False, None

        current_summary = self.memory_store.get_summary(session_id)
        summarized_message_id = current_summary.summarized_message_id if current_summary else 0
        if summarized_message_id >= cutoff_message_id:
            return False, None

        delta_messages = self.memory_store.get_messages_between(
            session_id=session_id,
            after_message_id=summarized_message_id,
            through_message_id=cutoff_message_id,
        )
        if not delta_messages:
            return False, None

        summary_query = (
            f"已有摘要：\n{current_summary.summary if current_summary else '无'}\n\n"
            f"需要合并的新对话：\n{_message_lines(delta_messages)}"
        )
        result = self._chat_with_retry(
            client=self.client,
            system_prompt=self.summary_prompt,
            user_query=summary_query,
            temperature=self.summary_temperature,
            max_tokens=self.summary_max_tokens,
        )
        self.memory_store.upsert_summary(
            session_id=session_id,
            summary=result.answer.strip(),
            summarized_message_id=cutoff_message_id,
        )
        return True, result

    def _summary_update_needed(self, session_id: str) -> bool:
        if self.memory_store is None:
            return False
        history_rounds = self.memory_store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return False

        recent_messages = self.memory_store.get_recent_messages(session_id, self.recent_rounds)
        if not recent_messages:
            return False
        cutoff_message_id = recent_messages[0].id - 1
        if cutoff_message_id <= 0:
            return False

        current_summary = self.memory_store.get_summary(session_id)
        summarized_message_id = current_summary.summarized_message_id if current_summary else 0
        return summarized_message_id < cutoff_message_id

    def _schedule_background_updates(
        self,
        session_id: str,
        user_id: str,
        user_query: str,
        assistant_answer: str,
        scene: str,
    ) -> tuple[bool, bool]:
        summary_needed = self._summary_update_needed(session_id)
        memory_needed = self.long_term_memory_enabled and self.memory_store is not None
        if not summary_needed and not memory_needed:
            return False, False

        key = (session_id, user_id)
        with self._background_update_lock:
            version = self._background_update_versions.get(key, 0) + 1
            self._background_update_versions[key] = version

        def worker() -> None:
            if self.background_update_delay_seconds:
                time.sleep(self.background_update_delay_seconds)
            with self._background_update_lock:
                if self._background_update_versions.get(key) != version:
                    return

            started_at = time.perf_counter()
            summary_changed = False
            memory_changed = False
            summary_result: ChatResult | None = None
            memory_result: ChatResult | None = None
            errors: list[str] = []
            try:
                summary_changed, summary_result = self._update_summary(session_id)
            except Exception as exc:
                errors.append(f"会话摘要更新失败: {exc}")

            try:
                recent_messages = (
                    self.memory_store.get_recent_messages(
                        session_id,
                        max(self.recent_rounds, 5),
                    )
                    if self.memory_store is not None
                    else []
                )
                memory_changed, memory_result = self._update_long_term_memories(
                    user_id=user_id,
                    user_query=user_query,
                    assistant_answer=assistant_answer,
                    scene=scene,
                    conversation_messages=recent_messages,
                )
            except Exception as exc:
                errors.append(f"长期记忆更新失败: {exc}")

            callback = self.background_update_callback
            if callback is not None:
                event: dict[str, object] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "summary_updated": summary_changed,
                    "long_term_memory_updated": memory_changed,
                    "summary_latency_ms": summary_result.latency_ms if summary_result else 0,
                    "memory_latency_ms": memory_result.latency_ms if memory_result else 0,
                    "total_latency_ms": int((time.perf_counter() - started_at) * 1000),
                    "error": "; ".join(errors),
                }
                try:
                    callback(event)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        return summary_needed, bool(memory_needed)

    def _invalidate_background_updates(self, session_id: str, user_id: str) -> None:
        key = (session_id, user_id)
        with self._background_update_lock:
            self._background_update_versions[key] = (
                self._background_update_versions.get(key, 0) + 1
            )

    def reset_session(self, session_id: str) -> None:
        if self.memory_store is not None:
            self.memory_store.reset_session(session_id)

    def reply(
        self,
        user_query: str,
        session_id: str = "default",
        user_id: str = "default",
        on_delta: Callable[[str], None] | None = None,
    ) -> VoiceChatReply:
        query = user_query.strip()
        session = session_id.strip() or "default"
        user = user_id.strip() or "default"
        reply = VoiceChatReply(user_query=query, session_id=session, user_id=user)
        if not query:
            reply.error = "用户输入不能为空"
            return reply

        if self.background_memory_updates:
            self._invalidate_background_updates(session, user)

        history_mode, recent_messages, summary, history_rounds, summary_used = (
            self._load_context(session)
        )
        reply.history_mode = history_mode
        reply.history_rounds = history_rounds
        reply.recent_messages_used = len(recent_messages)
        reply.summary_used = summary_used

        long_term_memories = []
        if self.long_term_memory_enabled and self.memory_store is not None:
            long_term_memories = self.memory_store.get_active(
                user,
                limit=self.long_term_memory_limit,
            )
        reply.long_term_memories_used = len(long_term_memories)

        classifier_result: ChatResult | None = None
        fallback_reason = ""
        try:
            scene, _, _, classifier_result = self.classify(query)
            reply.scene = scene
            generation_prompt = self.scene_prompts[scene]
            generation_prompt_name = Path(self.scene_prompt_paths[scene]).name
            selected_provider = self.model_routing.get(scene, "doubao")
            selected_client = self.clients.get(selected_provider)
        except Exception as exc:
            reply.scene = "通用回退"
            reply.fallback_used = True
            fallback_reason = f"场景分类失败: {exc}"
            generation_prompt = self.baseline_prompt
            generation_prompt_name = Path(self.baseline_prompt_path).name
            selected_provider = "doubao"
            selected_client = self.client

        try:
            if selected_client is None:
                raise RuntimeError(f"未配置模型提供方: {selected_provider}")
            generation_result = self._generate(
                selected_client,
                query,
                generation_prompt,
                recent_messages,
                summary,
                long_term_memories,
                on_delta,
            )
            generation_provider = selected_provider
        except Exception as exc:
            if isinstance(exc, StreamGenerationError) and exc.partial_output:
                reply.error = f"流式回复中断: {exc}"
                return reply
            if reply.scene == "通用回退":
                reply.error = f"{fallback_reason}; 通用回复失败: {exc}"
                return reply

            if selected_provider != "doubao":
                reply.fallback_used = True
                fallback_reason = f"{selected_provider}场景回复失败: {exc}"
                try:
                    generation_result = self._generate(
                        self.client,
                        query,
                        generation_prompt,
                        recent_messages,
                        summary,
                        long_term_memories,
                        on_delta,
                    )
                    generation_provider = "doubao"
                except Exception as doubao_scene_exc:
                    fallback_reason += f"; 豆包场景回复失败: {doubao_scene_exc}"
                    generation_result = None
            else:
                generation_result = None

            if generation_result is None:
                if not reply.fallback_used:
                    reply.fallback_used = True
                    fallback_reason = f"场景回复失败: {exc}"
                generation_prompt_name = Path(self.baseline_prompt_path).name
                try:
                    generation_result = self._generate(
                        self.client,
                        query,
                        self.baseline_prompt,
                        recent_messages,
                        summary,
                        long_term_memories,
                        on_delta,
                    )
                    generation_provider = "doubao"
                except Exception as fallback_exc:
                    reply.error = f"{fallback_reason}; 通用回复失败: {fallback_exc}"
                    return reply

        reply.prompt_name = generation_prompt_name
        reply.model_name = (
            generation_result.raw.get("model")
            or self.clients[generation_provider].model
        )
        reply.answer = sanitize_tts_text(generation_result.answer)
        reply.classifier_latency_ms = classifier_result.latency_ms if classifier_result else 0
        reply.generation_latency_ms = generation_result.latency_ms
        reply.first_token_latency_ms = generation_result.first_token_latency_ms
        if long_term_memories and self.memory_store is not None:
            self.memory_store.mark_used([memory.id for memory in long_term_memories])

        if self.memory_store is not None:
            self.memory_store.append_turn(session, query, reply.answer, reply.scene)

        memory_result: ChatResult | None = None
        summary_result: ChatResult | None = None
        if self.background_memory_updates:
            reply.summary_pending, reply.long_term_memory_pending = (
                self._schedule_background_updates(
                    session_id=session,
                    user_id=user,
                    user_query=query,
                    assistant_answer=reply.answer,
                    scene=reply.scene,
                )
            )
            reply.summary_update_mode = "background" if reply.summary_pending else "none"
            reply.long_term_memory_update_mode = (
                "background" if reply.long_term_memory_pending else "none"
            )
        else:
            memory_errors: list[str] = []
            try:
                reply.summary_updated, summary_result = self._update_summary(session)
                reply.summary_latency_ms = summary_result.latency_ms if summary_result else 0
                reply.summary_update_mode = "synchronous" if reply.summary_updated else "none"
            except Exception as exc:
                memory_errors.append(f"会话摘要更新失败: {exc}")

            try:
                memory_changed, memory_result = self._update_long_term_memories(
                    user_id=user,
                    user_query=query,
                    assistant_answer=reply.answer,
                    scene=reply.scene,
                )
                reply.long_term_memory_updated = memory_changed
                reply.memory_latency_ms = memory_result.latency_ms if memory_result else 0
                reply.long_term_memory_update_mode = (
                    "synchronous" if self.long_term_memory_enabled else "none"
                )
            except Exception as exc:
                memory_errors.append(f"长期记忆更新失败: {exc}")

            reply.memory_error = "; ".join(memory_errors)

        reply.latency_ms = (
            (classifier_result.latency_ms if classifier_result else 0)
            + generation_result.latency_ms
            + reply.summary_latency_ms
            + reply.memory_latency_ms
        )
        reply.total_tokens = sum(
            value or 0
            for value in (
                classifier_result.input_tokens if classifier_result else 0,
                classifier_result.output_tokens if classifier_result else 0,
                generation_result.input_tokens,
                generation_result.output_tokens,
                summary_result.input_tokens if summary_result else 0,
                summary_result.output_tokens if summary_result else 0,
                memory_result.input_tokens if memory_result else 0,
                memory_result.output_tokens if memory_result else 0,
            )
        )
        return reply
