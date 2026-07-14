from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_skill_environment(skill_root: str | Path) -> None:
    configured_path = os.getenv("VOICE_CHAT_ENV_FILE")
    env_path = Path(configured_path) if configured_path else Path(skill_root) / ".env"
    load_dotenv(env_path)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"未设置环境变量 {name}")
    return value


def get_doubao_api_key() -> str:
    return require_env("DOUBAO_API_KEY")


def get_doubao_base_url() -> str:
    return os.getenv(
        "DOUBAO_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    ).strip()


def get_doubao_model() -> str:
    return os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-vision-250815").strip()


def get_deepseek_api_key() -> str:
    return require_env("DEEPSEEK_API_KEY")


def get_deepseek_base_url() -> str:
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def get_deepseek_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()


def get_classifier_mode() -> str:
    mode = os.getenv("VOICE_CHAT_CLASSIFIER_MODE", "local").strip().lower()
    if mode not in {"local", "llm"}:
        raise RuntimeError("VOICE_CHAT_CLASSIFIER_MODE 只能是 local 或 llm")
    return mode


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()


def get_ollama_embedding_model() -> str:
    return os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip()


def get_ollama_rerank_model() -> str:
    return os.getenv("OLLAMA_RERANK_MODEL", "qwen2.5:3b").strip()


def get_rerank_threshold() -> float:
    raw_value = os.getenv("VOICE_CHAT_RERANK_THRESHOLD", "0.04").strip()
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_RERANK_THRESHOLD 必须是数字") from exc


def get_rerank_candidates() -> int:
    raw_value = os.getenv("VOICE_CHAT_RERANK_CANDIDATES", "3").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_RERANK_CANDIDATES 必须是整数") from exc
    if value < 2:
        raise RuntimeError("VOICE_CHAT_RERANK_CANDIDATES 至少为 2")
    return value


def get_long_term_memory_enabled() -> bool:
    raw_value = os.getenv("VOICE_CHAT_LONG_TERM_MEMORY", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def get_long_term_memory_limit() -> int:
    raw_value = os.getenv("VOICE_CHAT_LONG_TERM_MEMORY_LIMIT", "5").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_LONG_TERM_MEMORY_LIMIT 必须是整数") from exc
    return max(value, 0)


def get_long_term_memory_extract_limit() -> int:
    raw_value = os.getenv("VOICE_CHAT_LONG_TERM_MEMORY_EXTRACT_LIMIT", "20").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_LONG_TERM_MEMORY_EXTRACT_LIMIT 必须是整数") from exc
    return max(value, 0)


def get_summary_threshold_rounds() -> int:
    raw_value = os.getenv("VOICE_CHAT_SUMMARY_THRESHOLD_ROUNDS", "6").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_SUMMARY_THRESHOLD_ROUNDS 必须是整数") from exc
    return max(value, 1)


def get_recent_rounds() -> int:
    raw_value = os.getenv("VOICE_CHAT_RECENT_ROUNDS", "3").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("VOICE_CHAT_RECENT_ROUNDS 必须是整数") from exc
    return max(value, 1)


def get_state_db_path(skill_root: str | Path) -> Path:
    raw_value = os.getenv("VOICE_CHAT_STATE_DB", "").strip()
    if raw_value:
        path = Path(raw_value)
        return path if path.is_absolute() else Path(skill_root) / path
    return Path(skill_root) / ".state" / "voice_chat_memory.sqlite3"
