# tg-llm-bot

[English README](./README_EN.md)
[Changelog](./CHANGELOG.md)

一个连接到 OpenAI 兼容 API 的 Telegram Bot，支持：

- 切换模型 (`/model`)
- 查看 / 设置系统提示词 (`/system`) + 预设模板 (`/preset`)
- 微调采样参数 (`/params`：temperature、top_p、repeat_penalty、max_tokens、stream、web_search、thinking、prompt_cache)
- 多轮上下文（保留当前会话历史）
- 多会话管理（查看、切换、删除、自动会话标题）
- 流式输出（边生成边编辑消息）
- 生成前动态等待秒数提示（防止误判卡住）
- Token 累计用量统计 (`/stats`)
- 视觉理解：直接给 Bot 发图片即可（需要 vision 模型）
- 轻量 RAG：直接上传 `PDF/TXT/MD` 到本地知识库（SQLite），聊天自动检索增强
- Telegram user id 白名单
- Docker Compose 一键部署

## 一、准备

1. 在 Telegram 找 [@BotFather](https://t.me/BotFather)，`/newbot` 拿到 token。
2. 拿到 OpenAI 兼容 API 的 `api_key` 和 `base_url`：
   - 直连 OpenAI：`https://api.openai.com/v1`
   - 本机 new-api（你的环境）：`http://127.0.0.1:3001/v1`
3. （可选）查一下你自己的 Telegram user id。如果不知道也没关系，先随便填一个，启动后给 Bot 发任意消息，Bot 会回你说"你的 id 是 xxx，未在白名单中"。

## 二、配置

```bash
cd /root/tg-llm-bot
cp .env.example .env
vim .env   # 填入 token / api_key / base_url / 白名单
```

关键变量：

| 变量 | 说明 |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | BotFather 给的 token |
| `OPENAI_API_KEY` | LLM 网关或厂商 key |
| `OPENAI_BASE_URL` | OpenAI 兼容 endpoint，结尾带 `/v1` |
| `TAVILY_API_KEY` | 外部网页搜索 key；配置后 `web_search` 会通过工具调用 Tavily |
| `ALLOWED_USER_IDS` | 逗号分隔的 Telegram 数字 id；留空 = 不限制（**不推荐**） |
| `DEFAULT_MODEL` | 默认模型名 |
| `AVAILABLE_MODELS` | 可切换的模型列表，逗号分隔；留空则启动时自动 `GET /v1/models` |
| `DEFAULT_REPEAT_PENALTY` | 默认 repeat_penalty（浮点） |
| `WEB_SEARCH_DEFAULT` | 默认是否尝试启用网页搜索（不支持会自动回退） |
| `THINKING_DEFAULT` | 默认是否开启 thinking 模式（不支持会自动回退） |
| `PROMPT_CACHE_DEFAULT` | 默认是否尝试启用 Prompt Cache（OpenAI 兼容网关不支持会自动回退） |
| `EMBEDDING_MODEL` | RAG 向量化模型（OpenAI 兼容 embeddings） |
| `RAG_ENABLED_DEFAULT` | 默认是否开启 RAG 检索增强（可会话内覆盖） |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | RAG 文本分块参数 |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | RAG 检索召回数量与相似度阈值 |
| `TOOLS_ENABLED` | 工具框架启用列表（如 `web_search`） |
| `TOOL_TIMEOUT_SECONDS` / `TOOL_MAX_CALLS_PER_REQUEST` | 工具调用超时与单请求配额 |
| `SUMMARY_TRIGGER_MESSAGES` | 会话历史达到该条数后，后台触发一次摘要压缩 |
| `SUMMARY_TRIGGER_TOKENS` | 会话历史估算 token 达到该值时，也会触发摘要压缩 |
| `SUMMARY_MODEL` | 会话摘要专用模型；留空时跟随当前会话模型 |
| `SUMMARY_KEEP_RECENT_MESSAGES` | 摘要后保留最近多少条原始消息，其余压入会话摘要 |
| `SUMMARY_MAX_TOKENS` | 会话摘要生成时的最大输出 token |
| `SUMMARY_CONTEXT_MAX_TOKENS` | 会话摘要注入上下文时的估算 token 上限（超出会自动截断摘要） |
| `PROMPT_MAX_INPUT_TOKENS` | 单次请求发送给模型的输入估算 token 上限（超出会自动裁剪） |
| `PROMPT_KEEP_RECENT_MESSAGES` | 自动裁剪时最多保留最近多少条原始消息 |

## 三、启动

```bash
cd /root/tg-llm-bot
docker compose pull
docker compose up -d
docker compose logs -f
```

默认会拉取 Docker Hub 镜像 `otis49482/tg-llm-bot:latest`。看到 `Bot ready: @xxx` 即上线。

## 四、常用命令

发给 Bot：

```
/model                    选模型
/system                   查看当前系统提示词
/system 你是一名诗人      设置系统提示词
/system clear             清空系统提示词
/preset                   选预设模板
/params                   按钮调参
/set temperature 0.3      直接设置
/set top_p 0.9
/set repeat_penalty 1.1
/set max_tokens 4096
/set summary_model llama-3.3-70b-versatile
/set embedding_model text-embedding-v3
/set stream off
/set web_search on
/set thinking on
/set prompt_cache on
/set rag on
/sessions                 查看会话列表（可切换/删除）
/newchat                  新建会话
/use <会话ID>             切换到历史会话继续聊
/delsession <会话ID>      删除会话（不带参数删当前）
/ragfiles                 查看已上传知识库文件
/stats                    查看配置和 token 用量
/id                       查询自己的 user id（用于白名单）
```

直接发文字 = 聊天；直接发图片 = 视觉问答（caption 作为问题，没有 caption 会默认问"请描述这张图片"）。  
直接发 `PDF/TXT/MD` 文件 = 入库本地知识库（SQLite + 向量检索），后续聊天会自动检索相关片段。  
切换到历史会话（`/use` 或 `/sessions` 按钮）时，Bot 会自动发送会话摘要，帮助快速衔接上下文。
系统提示词与采样参数也会随会话保存，切回会话后会恢复该会话的原值。
当会话历史过长（按消息条数或估算 token）时，Bot 会在后台自动把较早轮次压缩成“会话摘要”，并只保留最近几条原始消息，以降低长对话超时风险。  
此外，每次请求发给模型前还会做一次输入 token 预算兜底裁剪，避免“消息条数不多但每条很长”导致超时。
thinking 模式支持按模型自动探测：不支持时会自动回退并缓存结果。  
`web_search` 开启且配置了 `TAVILY_API_KEY` 时，会向模型暴露 Tavily 工具；是否实际调用由模型自主决定。
如果开启 `PROMPT_CACHE_DEFAULT=true`，会在请求中附带 `prompt_cache` 与 `prompt_cache_key`；网关/模型不支持时会自动回退到不带缓存参数重试。

## 五、Todo

- [ ] 支持在 Telegram `channel` 中使用机器人（含权限与消息来源识别）。
- [ ] 支持在 Telegram `group/supergroup` 中使用机器人（@提及触发、回复触发、群内权限控制）。

## 六、维护

```bash
docker compose ps                          # 看状态
docker compose logs -f --tail 200          # 看日志
docker compose restart                     # 重启
docker compose pull && docker compose up -d           # 拉取最新版镜像并重启
```

用户配置和历史都保存在 `./data/users.json`，删除该文件相当于重置所有用户状态。

## 七、安全提示

- **强烈建议配置 `ALLOWED_USER_IDS`**，否则任何拿到 Bot 用户名的人都能消耗你的 API 额度。
- `.env` 包含密钥，不要提交到 git。
- 服务使用 `network_mode: host` 是为了直接连本机的 `127.0.0.1:3001` (new-api)。Bot 本身不监听任何端口，只主动外连 Telegram，因此公网攻击面很小。

## 八、目录结构

```
tg-llm-bot/
├── bot/
│   ├── config.py      # 读环境变量
│   ├── llm.py         # OpenAI 异步客户端封装
│   ├── storage.py     # 每用户状态持久化（JSON 文件）
│   ├── presets.py     # 预设系统提示词
│   ├── tools/         # 外部工具（如 Tavily）
│   ├── handlers.py    # 命令 / 按钮 / 消息处理
│   └── main.py        # Application 装配 + 入口
├── data/              # 运行时持久化目录（挂载进容器）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
