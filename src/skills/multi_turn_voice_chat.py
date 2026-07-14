from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.clients.deepseek_client import DeepSeekClient
from src.clients.doubao_client import ChatResult, DoubaoClient
from src.memory import ConversationMessage, ConversationStore, LongTermMemory
from src.utils.config import (
    get_deepseek_api_key,
    get_deepseek_base_url,
    get_deepseek_model,
    get_doubao_api_key,
    get_doubao_base_url,
    get_doubao_model,
)
from src.utils.io import read_text, sanitize_tts_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_SCRIPTS_DIR = PROJECT_ROOT / "skills" / "voice-chat-routing" / "scripts"
if str(SKILL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS_DIR))

from voice_chat_runtime import LocalSceneClassifier, OllamaChatClient, OllamaEmbeddingClient

CLASSIFIER_PROMPT = "prompts/classifier/scene_classifier.txt"
BASELINE_PROMPT = "prompts/baseline/general_voice_assistant.txt"
SUMMARY_PROMPT = "prompts/memory/session_summary.txt"
LONG_TERM_MEMORY_PROMPT = "prompts/memory/long_term_memory_extract.txt"
LOCAL_REFERENCE_CASES = "skills/voice-chat-routing/assets/data/scene_reference_centroid_v1.csv"
DEFAULT_STATE_DB = ".state/conversations.sqlite3"
SCENE_PROMPTS = {
    "日常闲聊": "prompts/scenes/chat.txt",
    "情绪陪伴": "prompts/scenes/emotion.txt",
    "知识解释": "prompts/scenes/knowledge.txt",
    "生活建议": "prompts/scenes/life_advice.txt",
    "追问澄清": "prompts/scenes/clarification.txt",
    "安全敏感": "prompts/scenes/safety.txt",
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

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> ChatResult: ...


@dataclass
class MultiTurnVoiceChatReply:
    user_query: str
    session_id: str
    scene: str = ""
    answer: str = ""
    model_name: str = ""
    prompt_name: str = ""
    history_mode: str = ""
    history_rounds: int = 0
    recent_messages_used: int = 0
    summary_used: bool = False
    summary_updated: bool = False
    summary_pending: bool = False
    summary_update_mode: str = ""
    user_id: str = "default"
    long_term_memories_used: int = 0
    long_term_memory_updated: bool = False
    long_term_memory_pending: bool = False
    long_term_memory_update_mode: str = ""
    classifier_latency_ms: int = 0
    generation_latency_ms: int = 0
    summary_latency_ms: int = 0
    latency_ms: int = 0
    total_tokens: int = 0
    fallback_used: bool = False
    error: str = ""
    memory_error: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _message_lines(messages: list[ConversationMessage]) -> str:
    if not messages:
        return "无"
    role_names = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role_names[item.role]}：{item.content}" for item in messages)


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


def _token_total(*results: ChatResult | None) -> int:
    total = 0
    for result in results:
        if result is None:
            continue
        total += int(result.input_tokens or 0)
        total += int(result.output_tokens or 0)
    return total


def _extract_json_object(raw_response: str) -> dict[str, object]:
    text = raw_response.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"模型返回不是 JSON: {raw_response}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型返回 JSON 不是对象")
    return data


class MultiTurnVoiceChatSkill:
    def __init__(
        self,
        client: ChatClient,
        store: ConversationStore,
        project_root: str | Path = PROJECT_ROOT,
        deepseek_client: ChatClient | None = None,
        classifier_prompt_path: str | Path = CLASSIFIER_PROMPT,
        baseline_prompt_path: str | Path = BASELINE_PROMPT,
        summary_prompt_path: str | Path = SUMMARY_PROMPT,
        long_term_memory_prompt_path: str | Path = LONG_TERM_MEMORY_PROMPT,
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
        summary_threshold_rounds: int = 6,
        recent_rounds: int = 3,
        long_term_memory_limit: int = 5,
        long_term_memory_extract_limit: int = 20,
        long_term_memory_enabled: bool = True,
        background_memory_updates: bool = True,
    ) -> None:
        self.client = client
        self.clients: dict[str, ChatClient] = {"doubao": client}
        if deepseek_client is not None:
            self.clients["deepseek"] = deepseek_client
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.classifier_temperature = classifier_temperature
        self.generation_temperature = generation_temperature
        self.summary_temperature = summary_temperature
        self.classifier_max_tokens = classifier_max_tokens
        self.generation_max_tokens = generation_max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.long_term_memory_max_tokens = long_term_memory_max_tokens
        self.retries = retries
        self.summary_threshold_rounds = summary_threshold_rounds
        self.recent_rounds = recent_rounds
        self.long_term_memory_limit = long_term_memory_limit
        self.long_term_memory_extract_limit = long_term_memory_extract_limit
        self.long_term_memory_enabled = long_term_memory_enabled
        self.background_memory_updates = background_memory_updates
        self.classifier_prompt_path = classifier_prompt_path
        self.baseline_prompt_path = baseline_prompt_path
        self.summary_prompt_path = summary_prompt_path
        self.long_term_memory_prompt_path = long_term_memory_prompt_path
        self.scene_prompt_paths = scene_prompt_paths or SCENE_PROMPTS
        self.model_routing = model_routing or MODEL_ROUTING
        self.classifier_mode = classifier_mode
        self.local_classifier = local_classifier

        self.classifier_prompt = read_text(self._resolve(classifier_prompt_path))
        self.baseline_prompt = read_text(self._resolve(baseline_prompt_path))
        self.summary_prompt = read_text(self._resolve(summary_prompt_path))
        self.long_term_memory_prompt = read_text(self._resolve(long_term_memory_prompt_path))
        self.scene_prompts = {
            scene: read_text(self._resolve(prompt_path))
            for scene, prompt_path in self.scene_prompt_paths.items()
        }

    @classmethod
    def from_env(
        cls,
        project_root: str | Path = PROJECT_ROOT,
        state_db: str | Path = DEFAULT_STATE_DB,
        **kwargs,
    ) -> "MultiTurnVoiceChatSkill":
        root = Path(project_root).resolve()
        client = DoubaoClient(
            api_key=get_doubao_api_key(root),
            base_url=get_doubao_base_url(root),
            model=get_doubao_model(root),
        )
        deepseek_client = None
        try:
            deepseek_client = DeepSeekClient(
                api_key=get_deepseek_api_key(root),
                base_url=get_deepseek_base_url(root),
                model=get_deepseek_model(root),
            )
        except RuntimeError:
            pass
        classifier_mode = os.getenv("VOICE_CHAT_CLASSIFIER_MODE", "local").strip().lower()
        if classifier_mode not in {"local", "llm"}:
            raise RuntimeError("VOICE_CHAT_CLASSIFIER_MODE 只能是 local 或 llm")

        local_classifier = None
        if classifier_mode == "local":
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            rerank_candidates = int(os.getenv("VOICE_CHAT_RERANK_CANDIDATES", "3"))
            if rerank_candidates < 2:
                raise RuntimeError("VOICE_CHAT_RERANK_CANDIDATES 至少为 2")
            local_classifier = LocalSceneClassifier(
                reference_path=root / LOCAL_REFERENCE_CASES,
                embedding_client=OllamaEmbeddingClient(
                    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip(),
                    base_url=ollama_base_url,
                ),
                rerank_client=OllamaChatClient(
                    model=os.getenv("OLLAMA_RERANK_MODEL", "qwen2.5:3b").strip(),
                    base_url=ollama_base_url,
                ),
                rerank_threshold=float(os.getenv("VOICE_CHAT_RERANK_THRESHOLD", "0.04")),
                rerank_candidates=rerank_candidates,
            )
        state_path = Path(state_db)
        if not state_path.is_absolute():
            state_path = root / state_path
        return cls(
            client=client,
            deepseek_client=deepseek_client,
            store=ConversationStore(state_path),
            project_root=root,
            classifier_mode=classifier_mode,
            local_classifier=local_classifier,
            **kwargs,
        )

    def _resolve(self, path: str | Path) -> Path:
        prompt_path = Path(path)
        return prompt_path if prompt_path.is_absolute() else self.project_root / prompt_path

    def _chat_with_retry(
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

    def _load_context(self, session_id: str) -> tuple[str, list[ConversationMessage], str, int, bool]:
        history_rounds = self.store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return "full", self.store.get_messages(session_id), "", history_rounds, False

        summary = self.store.get_summary(session_id)
        recent_messages = self.store.get_recent_messages(session_id, self.recent_rounds)
        return (
            "summary_window",
            recent_messages,
            summary.summary if summary else "",
            history_rounds,
            bool(summary and summary.summary.strip()),
        )

    def _build_classifier_query(
        self,
        user_query: str,
        history_mode: str,
        recent_messages: list[ConversationMessage],
        summary: str,
    ) -> str:
        return (
            f"会话上下文模式：{history_mode}\n"
            f"较早会话摘要：{summary or '无'}\n"
            f"最近对话：\n{_message_lines(recent_messages)}\n"
            f"当前用户输入：{user_query}\n\n"
            "请只根据当前用户输入在上述上下文中的真实意图进行分类。"
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
        messages.extend({"role": item.role, "content": item.content} for item in recent_messages)
        messages.append({"role": "user", "content": user_query})
        return messages

    def classify(
        self,
        user_query: str,
        history_mode: str,
        recent_messages: list[ConversationMessage],
        summary: str,
    ) -> tuple[str, str, str, ChatResult]:
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

        classifier_query = self._build_classifier_query(
            user_query=user_query,
            history_mode=history_mode,
            recent_messages=recent_messages,
            summary=summary,
        )
        result = self._chat_with_retry(
            client=self.client,
            messages=[
                {"role": "system", "content": self.classifier_prompt},
                {"role": "user", "content": classifier_query},
            ],
            temperature=self.classifier_temperature,
            max_tokens=self.classifier_max_tokens,
        )
        scene, confidence, reason = parse_classifier_response(result.answer)
        return scene, confidence, reason, result

    def _generate(
        self,
        client: ChatClient,
        prompt: str,
        user_query: str,
        recent_messages: list[ConversationMessage],
        summary: str,
        long_term_memories: list[LongTermMemory],
    ) -> ChatResult:
        return self._chat_with_retry(
            client=client,
            messages=self._build_generation_messages(
                prompt=prompt,
                user_query=user_query,
                recent_messages=recent_messages,
                summary=summary,
                long_term_memories=long_term_memories,
            ),
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

    def _maybe_update_long_term_memories(
        self,
        user_id: str,
        user_query: str,
        assistant_answer: str,
        scene: str,
    ) -> tuple[bool, ChatResult | None]:
        if not self.long_term_memory_enabled:
            return False, None

        existing = self.store.get_active_long_term_memories(
            user_id,
            limit=self.long_term_memory_extract_limit,
        )
        existing_ids = {memory.id for memory in existing}
        extraction_query = (
            f"用户ID：{user_id}\n"
            f"已有长期记忆：\n{_long_term_memory_lines(existing, include_evidence=True)}\n\n"
            f"当前场景：{scene}\n"
            f"当前用户输入：{user_query}\n"
            f"当前助手回复：{assistant_answer}"
        )
        result = self._chat_with_retry(
            client=self.client,
            messages=[
                {"role": "system", "content": self.long_term_memory_prompt},
                {"role": "user", "content": extraction_query},
            ],
            temperature=0.0,
            max_tokens=self.long_term_memory_max_tokens,
        )
        memories, deactivate_ids = self._parse_long_term_memory_update(
            result.answer,
            existing_memory_ids=existing_ids,
        )
        changed = False
        for memory_id in deactivate_ids:
            self.store.deactivate_long_term_memory(memory_id, user_id)
            changed = True
        for memory in memories:
            self.store.upsert_long_term_memory(
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

    def _maybe_update_summary(self, session_id: str) -> tuple[bool, ChatResult | None]:
        history_rounds = self.store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return False, None

        recent_messages = self.store.get_recent_messages(session_id, self.recent_rounds)
        if not recent_messages:
            return False, None
        cutoff_message_id = recent_messages[0].id - 1
        if cutoff_message_id <= 0:
            return False, None

        current_summary = self.store.get_summary(session_id)
        summarized_message_id = current_summary.summarized_message_id if current_summary else 0
        if summarized_message_id >= cutoff_message_id:
            return False, None

        delta_messages = self.store.get_messages_between(
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
            messages=[
                {"role": "system", "content": self.summary_prompt},
                {"role": "user", "content": summary_query},
            ],
            temperature=self.summary_temperature,
            max_tokens=self.summary_max_tokens,
        )
        self.store.upsert_summary(
            session_id=session_id,
            summary=result.answer.strip(),
            summarized_message_id=cutoff_message_id,
        )
        return True, result

    def _summary_update_needed(self, session_id: str) -> bool:
        history_rounds = self.store.count_rounds(session_id)
        if history_rounds < self.summary_threshold_rounds:
            return False

        recent_messages = self.store.get_recent_messages(session_id, self.recent_rounds)
        if not recent_messages:
            return False

        cutoff_message_id = recent_messages[0].id - 1
        if cutoff_message_id <= 0:
            return False

        current_summary = self.store.get_summary(session_id)
        summarized_message_id = current_summary.summarized_message_id if current_summary else 0
        return summarized_message_id < cutoff_message_id

    def _start_summary_update(self, session_id: str) -> bool:
        if not self._summary_update_needed(session_id):
            return False

        def worker() -> None:
            try:
                self._maybe_update_summary(session_id)
            except Exception:
                # Background summary failures should not block the user-facing answer.
                pass

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _start_long_term_memory_update(
        self,
        user_id: str,
        user_query: str,
        assistant_answer: str,
        scene: str,
    ) -> bool:
        if not self.long_term_memory_enabled:
            return False

        def worker() -> None:
            try:
                self._maybe_update_long_term_memories(
                    user_id=user_id,
                    user_query=user_query,
                    assistant_answer=assistant_answer,
                    scene=scene,
                )
            except Exception:
                # Background memory extraction failures should not block chatting.
                pass

        threading.Thread(target=worker, daemon=True).start()
        return True

    def reset_session(self, session_id: str) -> None:
        self.store.reset_session(session_id)

    def reply(
        self,
        user_query: str,
        session_id: str = "default",
        user_id: str = "default",
    ) -> MultiTurnVoiceChatReply:
        query = user_query.strip()
        session = session_id.strip() or "default"
        user = user_id.strip() or "default"
        reply = MultiTurnVoiceChatReply(user_query=query, session_id=session)
        reply.user_id = user
        if not query:
            reply.error = "用户输入不能为空"
            return reply

        history_mode, recent_messages, summary, history_rounds, summary_used = self._load_context(session)
        long_term_memories: list[LongTermMemory] = []
        if self.long_term_memory_enabled:
            long_term_memories = self.store.get_active_long_term_memories(
                user,
                limit=self.long_term_memory_limit,
            )
        reply.history_mode = history_mode
        reply.history_rounds = history_rounds
        reply.recent_messages_used = len(recent_messages)
        reply.summary_used = summary_used
        reply.long_term_memories_used = len(long_term_memories)

        classifier_result: ChatResult | None = None
        generation_result: ChatResult | None = None
        fallback_reason = ""
        try:
            scene, _, _, classifier_result = self.classify(
                query,
                history_mode=history_mode,
                recent_messages=recent_messages,
                summary=summary,
            )
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
                client=selected_client,
                prompt=generation_prompt,
                user_query=query,
                recent_messages=recent_messages,
                summary=summary,
                long_term_memories=long_term_memories,
            )
            generation_provider = selected_provider
        except Exception as exc:
            if reply.scene == "通用回退":
                reply.error = f"{fallback_reason}; 通用回复失败: {exc}"
                return reply

            if selected_provider != "doubao":
                reply.fallback_used = True
                generation_result = self._generate(
                    client=self.client,
                    prompt=generation_prompt,
                    user_query=query,
                    recent_messages=recent_messages,
                    summary=summary,
                    long_term_memories=long_term_memories,
                )
                generation_provider = "doubao"
            else:
                reply.fallback_used = True
                generation_prompt = self.baseline_prompt
                generation_prompt_name = Path(self.baseline_prompt_path).name
                generation_result = self._generate(
                    client=self.client,
                    prompt=generation_prompt,
                    user_query=query,
                    recent_messages=recent_messages,
                    summary=summary,
                    long_term_memories=long_term_memories,
                )
                generation_provider = "doubao"

        answer = sanitize_tts_text(generation_result.answer)
        reply.answer = answer
        reply.model_name = self.clients[generation_provider].model
        reply.prompt_name = generation_prompt_name
        reply.classifier_latency_ms = classifier_result.latency_ms if classifier_result else 0
        reply.generation_latency_ms = generation_result.latency_ms
        reply.latency_ms = int(reply.classifier_latency_ms + reply.generation_latency_ms)

        if long_term_memories:
            self.store.mark_long_term_memories_used([memory.id for memory in long_term_memories])
        self.store.append_turn(session, query, answer, reply.scene)
        if self.background_memory_updates:
            try:
                reply.summary_pending = self._start_summary_update(session)
                reply.summary_update_mode = "background" if reply.summary_pending else "none"
                reply.long_term_memory_pending = self._start_long_term_memory_update(
                    user_id=user,
                    user_query=query,
                    assistant_answer=answer,
                    scene=reply.scene,
                )
                reply.long_term_memory_update_mode = (
                    "background" if reply.long_term_memory_pending else "none"
                )
            except Exception as exc:
                reply.memory_error = f"后台记忆启动失败: {exc}"
        else:
            try:
                reply.summary_updated, summary_result = self._maybe_update_summary(session)
                reply.summary_update_mode = "synchronous" if reply.summary_updated else "none"
                if summary_result is not None:
                    reply.summary_latency_ms = summary_result.latency_ms
            except Exception as exc:
                reply.memory_error = f"会话摘要更新失败: {exc}"

            if self.long_term_memory_enabled:
                try:
                    reply.long_term_memory_updated, _ = self._maybe_update_long_term_memories(
                        user_id=user,
                        user_query=query,
                        assistant_answer=answer,
                        scene=reply.scene,
                    )
                    reply.long_term_memory_update_mode = "synchronous"
                except Exception as exc:
                    separator = "; " if reply.memory_error else ""
                    reply.memory_error += f"{separator}长期记忆更新失败: {exc}"

        reply.total_tokens = _token_total(classifier_result, generation_result)
        return reply
