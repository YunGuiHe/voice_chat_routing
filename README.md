# Voice Chat Routing

这是一个中文语音助手的文本对话原型。项目接收用户文本，先在本地判断对话场景，再选择对应的 Prompt 和远程模型生成回答，同时维护当前会话上下文和跨会话长期记忆。

目前支持日常闲聊、情绪陪伴、知识解释、生活建议、追问澄清和安全敏感六类场景。代码以实验和调试为主，不包含 ASR、TTS 模型以及真实手机工具调用。

## 工作流程

```text
用户输入
  -> 读取会话上下文和长期记忆
  -> 本地场景分类
  -> 选择场景 Prompt 和生成模型
  -> 生成并清洗回复
  -> 保存本轮对话，按需更新摘要和长期记忆
```

场景分类分为两步：先用 `bge-m3` 计算输入与各场景参考语句的相似度；当场景之间的分数过于接近时，再调用本地 `qwen2.5:3b` 在候选场景中复判。知识解释场景默认交给 DeepSeek，其余场景默认使用豆包；模型调用失败时会回退到豆包通用 Prompt。

短对话直接携带完整历史。对话超过设定轮数后，系统改用较早内容摘要和最近几轮消息。长期记忆保存在本地 SQLite 中，可在不同会话之间复用用户偏好、身份背景和持续目标等信息。

## 运行环境

- Python 3.10 或更高版本
- [Ollama](https://ollama.com/)
- 豆包 API Key
- DeepSeek API Key，可选；未配置时知识解释场景会回退到豆包

先准备本地分类模型：

```bash
ollama pull bge-m3
ollama pull qwen2.5:3b
```

复制环境变量模板：

```bash
cp .env.example skills/voice-chat-routing/.env
```

打开 `skills/voice-chat-routing/.env`，至少填写：

```dotenv
DOUBAO_API_KEY=
DEEPSEEK_API_KEY=
```

密钥文件、SQLite 数据库和实验输出已通过 `.gitignore` 排除，不应提交到仓库。

## 使用方式

直接运行独立 Skill：

```bash
python skills/voice-chat-routing/scripts/run_skill.py \
  "为什么冬天空调房里很干？"
```

指定用户和会话，可以分别测试长期记忆和当前会话上下文：

```bash
python skills/voice-chat-routing/scripts/run_skill.py \
  --user-id demo-user \
  --session-id demo-session \
  "我最近想换一部手机"
```

启动图形化调试界面：

```bash
python Debug_GUI.py
```

在 Codex 中也可以显式调用项目 Skill：

```text
$voice-chat-routing 我最近压力很大，应该怎么办？
```

## 目录说明

```text
data/test_cases/               场景分类参考集与测试集
prompts/                       实验阶段使用的 Prompt
skills/voice-chat-routing/     可独立运行的 Skill
src/                           项目侧客户端、记忆和调试代码
Debug_GUI.py                   多轮对话调试界面
```

`skills/voice-chat-routing/` 已包含运行所需的 Prompt、场景参考集和 Python 代码，可以脱离 `src/` 独立使用。`src/` 主要保留实验工程和分类测试入口。

## 说明

项目会调用远程模型 API，并产生相应的接口费用。场景分类依赖本地 Ollama 服务；如果 Ollama 未启动或模型未下载，分类流程将无法正常运行。
