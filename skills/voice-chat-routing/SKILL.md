---
name: voice-chat-routing
description: Use when a Chinese voice-assistant utterance needs scene-aware conversation, multi-turn context, cross-session user memory, model routing, or TTS-ready text.
---

# Voice Chat Routing

Process Chinese utterances through the packaged scene-gated chat workflow.
Scene gating uses local Ollama embedding classification by default.
Short-term context is keyed by `session_id`; long-term memory is keyed by
`user_id`. Both are stored in the Skill-local SQLite database.

## Run The Workflow

1. Start Ollama locally and make sure `bge-m3` and `qwen2.5:3b` are available.
2. Set `DOUBAO_API_KEY`.
3. Set `DEEPSEEK_API_KEY` to enable the knowledge route. If it is absent, use
   Doubao with the knowledge prompt as the fallback.
4. Optionally place these variables in `.env` inside this Skill, or set
   `VOICE_CHAT_ENV_FILE` to another `.env` path.
5. Run the script from any directory:

```bash
python /path/to/voice-chat-routing/scripts/run_skill.py \
  --session-id chat-001 \
  --user-id default \
  "我最近压力很大，什么都不想做"
```

Use the same `session_id` for consecutive turns. Use a new `session_id` to start
a new conversation while retaining long-term memory for the same `user_id`.

## Follow The Routing Logic

Execute these steps in order:

1. Validate that the input text is non-empty.
2. Load the current session history and up to the configured number of active user memories.
3. Load `assets/data/scene_reference_centroid_v1.csv`.
4. Use local `bge-m3` embeddings to compare the input against six scene centroids.
5. If the top scene margin is at or below `VOICE_CHAT_RERANK_THRESHOLD`
   (default `0.04`), ask local `qwen2.5:3b` to rerank the top 3 candidate scenes.
6. Classify the input as one of:
   - 日常闲聊
   - 情绪陪伴
   - 知识解释
   - 生活建议
   - 追问澄清
   - 安全敏感
7. Load the matching prompt from `assets/prompts/scenes/`.
8. Route knowledge explanation to DeepSeek V4 Pro and the other scenes to Doubao.
9. Generate with full history before the summary threshold; afterwards use the
   earlier summary plus the most recent configured rounds.
10. Clean the answer with `sanitize_tts_text` and append the turn to SQLite.
11. Incrementally update the session summary when needed.
12. Use Doubao to extract or update long-term memories.
13. Return the structured result after state updates finish.

Set `VOICE_CHAT_CLASSIFIER_MODE=llm` to use the old LLM classifier prompt instead
of local embedding classification.

## Apply Fallbacks

- LLM classification mode attempts classification at most three times: the initial
  call plus two retries.
- Local classification falls back to the embedding top-1 scene if rerank output is
  invalid or outside the candidate scene list.
- In LLM classification mode, treat an empty classifier response, invalid JSON, or
  an unknown scene name as a failed classification attempt.
- Do not retry a syntactically valid scene merely because an offline evaluator
  might later judge the semantic classification to be wrong.
- Fall back to `assets/prompts/baseline.txt` when classification fails or returns invalid output.
- Fall back from DeepSeek to Doubao with the same scene prompt when DeepSeek is unavailable.
- Fall back to the baseline prompt when scene-specific generation fails.
- Return an explicit error when both the scene flow and baseline fallback fail.
- Never invent a response after all model calls fail.

## Respect The Scope

- Handle Chinese text conversations identified by `session_id` and `user_id`.
- Operate after ASR and before TTS.
- Do not perform ASR, TTS synthesis, tool calls, or RAG.
- Long-term memory is limited to stable user facts, preferences, goals, constraints, and communication style.
- Do not read test labels or evaluation results during normal Skill execution.
- Do not expose API keys in output or logs.

## Use The Core Interface

For Python integration, add the Skill's `scripts/` directory to `sys.path`, then
call:

```python
from voice_chat_runtime import VoiceChatSkill

skill = VoiceChatSkill.from_env("/path/to/voice-chat-routing")
result = skill.reply(user_query, session_id="chat-001", user_id="default")
```

To clear short-term context without deleting long-term user memory:

```python
skill.reset_session("chat-001")
```

Relevant optional environment variables are
`VOICE_CHAT_SUMMARY_THRESHOLD_ROUNDS` (default `6`),
`VOICE_CHAT_RECENT_ROUNDS` (default `3`), and
`VOICE_CHAT_STATE_DB` (default `.state/voice_chat_memory.sqlite3` inside the Skill).

Keep batch experiments and evaluation outside the Skill. Use them only to test whether this workflow improves over the baseline.
