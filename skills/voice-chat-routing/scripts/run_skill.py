import argparse
import json
import os
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from voice_chat_runtime import VoiceChatSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the voice chat routing skill once.")
    parser.add_argument("query", nargs="?", help="Chinese user query. Reads stdin when omitted.")
    parser.add_argument(
        "--user-id",
        default=os.getenv("VOICE_CHAT_USER_ID", "default"),
        help="Long-term memory user id.",
    )
    parser.add_argument(
        "--session-id",
        default=os.getenv("VOICE_CHAT_SESSION_ID", "default"),
        help="Short-term conversation session id.",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear messages and summary for the selected session before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = args.query if args.query is not None else sys.stdin.read()
    try:
        skill = VoiceChatSkill.from_env(SKILL_ROOT)
        if args.reset_session:
            skill.reset_session(args.session_id)
            if not query.strip():
                print(
                    json.dumps(
                        {
                            "session_id": args.session_id,
                            "reset_session": True,
                            "error": "",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
        result = skill.reply(
            query,
            session_id=args.session_id,
            user_id=args.user_id,
        )
        payload = result.to_dict()
    except Exception as exc:
        payload = {
            "user_query": query.strip(),
            "session_id": args.session_id,
            "user_id": args.user_id,
            "scene": "",
            "answer": "",
            "model_name": "",
            "prompt_name": "",
            "latency_ms": 0,
            "memory_latency_ms": 0,
            "summary_latency_ms": 0,
            "history_mode": "",
            "history_rounds": 0,
            "recent_messages_used": 0,
            "summary_used": False,
            "summary_updated": False,
            "long_term_memories_used": 0,
            "long_term_memory_updated": False,
            "total_tokens": 0,
            "fallback_used": False,
            "error": str(exc),
            "memory_error": "",
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
