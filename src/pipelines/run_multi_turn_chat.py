import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.skills.multi_turn_voice_chat import MultiTurnVoiceChatSkill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project-side multi-turn voice chat workflow.")
    parser.add_argument("query", nargs="?", help="Chinese user query. Reads stdin when omitted.")
    parser.add_argument("--session-id", default="default", help="Conversation session id.")
    parser.add_argument("--user-id", default="default", help="Long-term memory user id.")
    parser.add_argument("--state-db", default=".state/conversations.sqlite3")
    parser.add_argument("--reset-session", action="store_true", help="Clear this session before running.")
    parser.add_argument("--summary-threshold-rounds", type=int, default=6)
    parser.add_argument("--recent-rounds", type=int, default=3)
    parser.add_argument("--classifier-temperature", type=float, default=0.0)
    parser.add_argument("--generation-temperature", type=float, default=0.3)
    parser.add_argument("--summary-temperature", type=float, default=0.0)
    parser.add_argument("--classifier-max-tokens", type=int, default=128)
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--summary-max-tokens", type=int, default=512)
    parser.add_argument("--long-term-memory-max-tokens", type=int, default=512)
    parser.add_argument("--long-term-memory-limit", type=int, default=5)
    parser.add_argument("--disable-long-term-memory", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    query = args.query if args.query is not None else sys.stdin.read()
    skill = MultiTurnVoiceChatSkill.from_env(
        project_root=ROOT_DIR,
        state_db=args.state_db,
        classifier_temperature=args.classifier_temperature,
        generation_temperature=args.generation_temperature,
        summary_temperature=args.summary_temperature,
        classifier_max_tokens=args.classifier_max_tokens,
        generation_max_tokens=args.generation_max_tokens,
        summary_max_tokens=args.summary_max_tokens,
        long_term_memory_max_tokens=args.long_term_memory_max_tokens,
        retries=args.retries,
        summary_threshold_rounds=args.summary_threshold_rounds,
        recent_rounds=args.recent_rounds,
        long_term_memory_limit=args.long_term_memory_limit,
        long_term_memory_enabled=not args.disable_long_term_memory,
        background_memory_updates=False,
    )
    if args.reset_session:
        skill.reset_session(args.session_id)
        if not query.strip():
            print(
                json.dumps(
                    {"session_id": args.session_id, "reset_session": True, "error": ""},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

    result = skill.reply(query, session_id=args.session_id, user_id=args.user_id)
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
