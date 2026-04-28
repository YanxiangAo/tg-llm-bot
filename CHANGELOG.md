# 更新日志 Changelog

本文件记录项目的重要变更。  
This file documents notable changes to the project.

## [Unreleased] / 未发布

- 新增会话摘要压缩与输入预算裁剪：支持按消息数/token 阈值触发后台摘要，注入摘要时自动截断，并在请求前做输入 token 兜底裁剪。  
  Added session summarization and prompt-budget trimming: background summarization now triggers by message/token thresholds, summaries are clipped before injection, and requests are hard-capped by estimated input tokens.
- 新增本地 RAG：支持上传 PDF/TXT/MD 入库，基于 embedding + 余弦相似度检索，并将召回片段注入上下文。  
  Added local RAG: upload PDF/TXT/MD files, retrieve by embeddings + cosine similarity, and inject top snippets into context.
- 工具框架重构：将 Tavily 重构为可插拔 `ToolProvider`，由 `ToolManager` 统一管理 schema、开关、超时与配额。  
  Refactored tool framework: Tavily is now a pluggable `ToolProvider` managed by `ToolManager` for schema exposure, enable flags, timeout, and quota.
- 参数面板增强：新增 `prompt_cache`、`summary_model`、`embedding_model` 会话级切换，并支持模型列表刷新。  
  Parameter panel enhanced: added session-scoped toggles/selections for `prompt_cache`, `summary_model`, and `embedding_model`, with model-list refresh.
- 模型回退增强：摘要模型与 embedding 模型支持按优先级自动降级重试。  
  Model fallback improved: summary and embedding models now support ordered fallback retries.
- Token 估算改为优先使用 `tiktoken`，不可用时自动回退到字符粗估。  
  Token estimation now prefers `tiktoken`, with automatic fallback to character-based rough estimation.
- `web_search` 行为调整：移除业务侧关键词拦截，工具是否调用完全由模型决定。  
  `web_search` behavior updated: removed business-side keyword gating; tool invocation is now fully model-driven.
- 会话行为调整：移除 `/reset` 命令；切换/清空系统提示词与切换预设不再清空历史。  
  Session behavior updated: removed `/reset`; changing/clearing system prompts and switching presets no longer clear history.
- 会话级配置：系统提示词与采样参数改为随会话保存，切换回会话时自动恢复。  
  Session-scoped settings: system prompt and sampling parameters are stored per session and restored on session switch.
- 新增 `thinking` 开关：支持 `/params` 与 `/set thinking on|off`，并按模型缓存支持探测结果。  
  Added `thinking` toggle via `/params` and `/set thinking on|off`, with per-model support detection cache.
- `web_search` 升级为外部工具调用：配置 `TAVILY_API_KEY` 后，模型可通过 tool calling 调用 Tavily 搜索；不支持时自动降级回退。  
  `web_search` upgraded to external tool-calling: with `TAVILY_API_KEY`, models can call Tavily via tool calling, with graceful fallback on unsupported paths.
- `/stop` 行为增强：在“思考中动态秒数提示”阶段也可立即终止。  
  `/stop` improved: now also interrupts during the pre-response “thinking seconds” stage.

## [v1.0.0] - 2026-04-26

### 新增 Added

- 基于 OpenAI 兼容接口的 Telegram LLM Bot 初始版本。  
  Initial Telegram LLM bot with OpenAI-compatible API integration.
- 支持从上游 `/v1/models` 动态拉取可用模型。  
  Dynamic model discovery from upstream `/v1/models`.
- 支持按用户配置模型、系统提示词与采样参数。  
  Per-user controls for model, system prompt, and generation parameters.
- 支持流式输出（边生成边编辑消息）。  
  Streaming replies with progressive message editing.
- 支持图片理解（图片 + caption）。  
  Vision input support (image + caption).
- 支持白名单访问控制。  
  Whitelist-based access control.
- 支持 Token 用量统计（prompt/completion/total）。  
  Usage statistics (`prompt/completion/total tokens`).
- 内置预设提示词并支持 `/preset` 快速切换。  
  Built-in prompt presets and `/preset` selection.
- Docker + Compose 一键部署。  
  Dockerized deployment with Compose.
- 完成 GitHub 与 Docker Hub 发布流程。  
  GitHub repository and Docker Hub publishing flow.

### 变更 Changed

- Compose 默认改为直接拉取 `otis49482/tg-llm-bot:latest`。  
  Compose now pulls `otis49482/tg-llm-bot:latest` directly.
- 菜单精简为高频主命令，低频命令保留在 `/help`。  
  Command menu simplified to high-frequency actions; advanced commands remain in `/help`.
- 生成前等待阶段改为动态秒数提示。  
  Waiting UX improved with dynamic elapsed seconds before first token.
- 模型切换页优先实时读取上游模型列表。  
  Model selector now prefers live upstream model list each time.

### 高级会话与参数 Advanced Sessions & Parameters

- 多会话管理：  
  Multi-session conversation management:
  - `/sessions`, `/newchat`, `/use <id>`, `/delsession <id>`
  - 会话持久化到 `data/users.json`  
    Session persistence in `data/users.json`
  - 根据内容自动生成会话标题  
    Auto-generated session titles from content
  - 切换历史会话时自动发送会话摘要  
    Auto summary message when switching to historical sessions
- 新增可调参数：`top_k`、`repeat_penalty`，并在 `/params`、`/set`、默认配置、请求透传中生效。  
  Added `top_k` and `repeat_penalty`, supported in `/params`, `/set`, defaults, and upstream payload.

### 文档 Docs

- 文档双语化：`README.md`（中文）与 `README_EN.md`（英文）。  
  Bilingual docs: `README.md` (ZH) and `README_EN.md` (EN).
- 新增 `LICENSE`（MIT）。  
  Added `LICENSE` (MIT).
- 同步更新文档以覆盖多会话、新参数与运行行为。  
  Updated docs to reflect session flow, new parameters, and runtime behavior.
