from __future__ import annotations

import sys
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = PROJECT_ROOT / "skills/voice-chat-routing"
SKILL_SCRIPTS = SKILL_ROOT / "scripts"
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))

from voice_chat_runtime import (  # noqa: E402
    ALLOWED_SCENES,
    MODEL_ROUTING,
    VoiceChatReply,
    VoiceChatSkill as RuntimeVoiceChatSkill,
    parse_classifier_response,
)
from src.utils.config import (  # noqa: E402
    get_deepseek_api_key,
    get_deepseek_base_url,
    get_deepseek_model,
    get_doubao_api_key,
    get_doubao_base_url,
    get_doubao_model,
)


# Project-relative paths retained for the known-scene experiment pipeline.
SCENE_PROMPTS = {
    "日常闲聊": "skills/voice-chat-routing/assets/prompts/scenes/chat.txt",
    "情绪陪伴": "skills/voice-chat-routing/assets/prompts/scenes/emotion.txt",
    "知识解释": "skills/voice-chat-routing/assets/prompts/scenes/knowledge.txt",
    "生活建议": "skills/voice-chat-routing/assets/prompts/scenes/life_advice.txt",
    "追问澄清": "skills/voice-chat-routing/assets/prompts/scenes/clarification.txt",
    "安全敏感": "skills/voice-chat-routing/assets/prompts/scenes/safety.txt",
}


class VoiceChatSkill(RuntimeVoiceChatSkill):
    @classmethod
    def from_env(cls, skill_root: str | Path = SKILL_ROOT, **kwargs):
        os.environ.setdefault("DOUBAO_API_KEY", get_doubao_api_key(PROJECT_ROOT))
        os.environ.setdefault("DOUBAO_BASE_URL", get_doubao_base_url(PROJECT_ROOT))
        os.environ.setdefault("DOUBAO_MODEL", get_doubao_model(PROJECT_ROOT))
        try:
            os.environ.setdefault(
                "DEEPSEEK_API_KEY",
                get_deepseek_api_key(PROJECT_ROOT),
            )
            os.environ.setdefault(
                "DEEPSEEK_BASE_URL",
                get_deepseek_base_url(PROJECT_ROOT),
            )
            os.environ.setdefault(
                "DEEPSEEK_MODEL",
                get_deepseek_model(PROJECT_ROOT),
            )
        except RuntimeError:
            pass
        return super().from_env(skill_root=skill_root, **kwargs)


__all__ = [
    "ALLOWED_SCENES",
    "MODEL_ROUTING",
    "SCENE_PROMPTS",
    "VoiceChatReply",
    "VoiceChatSkill",
    "parse_classifier_response",
]
