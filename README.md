# 语音助手聊天能力优化实验项目

本项目研究如何通过本地场景门控、多 Prompt、模型路由、上下文记忆和回复后处理，提高中文语音助手的聊天回答质量。

```text
用户文本 -> 读取上下文与长期记忆 -> 本地场景门控 -> Prompt 选择 -> 模型路由 -> 回复生成 -> 文本清洗 -> 更新记忆
```

当前支持六类场景：日常闲聊、情绪陪伴、知识解释、生活建议、追问澄清和安全敏感。

## 目录结构

```text
.agents/skills/                Codex 项目级 Skill 发现入口
data/test_cases/               场景分类参考集与测试集
prompts/
  baseline/                    实验用 baseline Prompt
  classifier/                  低置信场景复判 Prompt
  memory/                      会话摘要与长期记忆提取 Prompt
  scenes/                      场景回复 Prompt
skills/voice-chat-routing/     可独立运行的语音助手路由 Skill
src/
  clients/                     远程模型与本地 Ollama 客户端
  memory/                      会话上下文、摘要与 SQLite 长期记忆
  pipelines/                   多轮调试与本地场景分类入口
  skills/                      实验工程与独立 Skill 的兼容入口
  utils/                       配置、CSV 和文本处理工具
multi_turn_debug_gui.py        多轮对话 GUI 调试入口
```

## 环境配置

Python 优先使用 conda 环境 `voice-agent`。

项目实验脚本可以从根目录 `.env` 或未提交 Git 的 `api.md` 读取密钥。独立 Skill 默认读取：

```text
skills/voice-chat-routing/.env
```

主要环境变量：

```dotenv
DOUBAO_API_KEY=
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_MODEL=doubao-seed-1-6-vision-250815
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-pro
VOICE_CHAT_CLASSIFIER_MODE=local
VOICE_CHAT_RERANK_THRESHOLD=0.04
VOICE_CHAT_RERANK_CANDIDATES=3
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_RERANK_MODEL=qwen2.5:3b
```

`.env` 和 `api.md` 已被 Git 忽略，不要将真实密钥提交到仓库。

## 当前调试脚本

以下命令会调用模型 API 并消耗接口额度：

```bash
# 多轮文本对话调试
conda run -n voice-agent python src/pipelines/run_multi_turn_chat.py "你好，能和我聊聊蛋糕制作吗"
```

实验输出保存在本地 `outputs/`，该目录不提交到 Git。

## Voice Chat Routing Skill

Skill 源码位于：

```text
skills/voice-chat-routing/
```

项目通过以下软链接让 Codex 自动发现 Skill：

```text
.agents/skills/voice-chat-routing
    -> ../../skills/voice-chat-routing
```

在 Codex 中显式调用：

```text
$voice-chat-routing 我最近压力很大，应该怎么办？
```

也可以直接运行：

```bash
conda run -n voice-agent python \
  skills/voice-chat-routing/scripts/run_skill.py \
  "为什么冬天空调房里很干？"
```

当前 Skill 默认使用本地 Ollama 做场景门控：`bge-m3` 负责 embedding 粗分，
低置信样本再由 `qwen2.5:3b` 在候选场景中复判。正常情况下，远程 API 只用于
最终回答生成；失败重试或模型回退会增加 API 调用次数。
