# 更新日志 Changelog

本文件记录项目的重要变更。  
This file documents notable changes to the project.

## [Unreleased] / 未发布

- 暂无。  
- No entries yet.

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
