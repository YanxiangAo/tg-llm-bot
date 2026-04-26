# tg-llm-bot

一个连接到 OpenAI 兼容 API 的 Telegram Bot，支持：

- 切换模型 (`/model`)
- 查看 / 设置系统提示词 (`/system`) + 预设模板 (`/preset`)
- 微调采样参数 (`/params`：temperature、top_p、max_tokens、stream)
- 多轮上下文 + 一键清空 (`/reset`)
- 流式输出（边生成边编辑消息）
- Token 累计用量统计 (`/stats`)
- 视觉理解：直接给 Bot 发图片即可（需要 vision 模型）
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
| `ALLOWED_USER_IDS` | 逗号分隔的 Telegram 数字 id；留空 = 不限制（**不推荐**） |
| `DEFAULT_MODEL` | 默认模型名 |
| `AVAILABLE_MODELS` | 可切换的模型列表，逗号分隔；留空则启动时自动 `GET /v1/models` |

## 三、启动

```bash
cd /root/tg-llm-bot
docker compose up -d --build
docker compose logs -f
```

第一次会构建镜像（约 1–2 分钟）。看到 `Bot ready: @xxx` 即上线。

## 四、常用命令

发给 Bot：

```
/model                    选模型
/system                   查看当前系统提示词
/system 你是一名诗人      设置系统提示词（清空历史）
/system clear             清空系统提示词
/preset                   选预设模板
/params                   按钮调参
/set temperature 0.3      直接设置
/set max_tokens 4096
/set stream off
/reset                    清空对话历史
/stats                    查看配置和 token 用量
/id                       查询自己的 user id（用于白名单）
```

直接发文字 = 聊天；直接发图片 = 视觉问答（caption 作为问题，没有 caption 会默认问"请描述这张图片"）。

## 五、维护

```bash
docker compose ps                          # 看状态
docker compose logs -f --tail 200          # 看日志
docker compose restart                     # 重启
docker compose down && docker compose up -d --build   # 改完代码重新构建
```

用户配置和历史都保存在 `./data/users.json`，删除该文件相当于重置所有用户状态。

## 六、安全提示

- **强烈建议配置 `ALLOWED_USER_IDS`**，否则任何拿到 Bot 用户名的人都能消耗你的 API 额度。
- `.env` 包含密钥，不要提交到 git。
- 服务使用 `network_mode: host` 是为了直接连本机的 `127.0.0.1:3001` (new-api)。Bot 本身不监听任何端口，只主动外连 Telegram，因此公网攻击面很小。

## 七、目录结构

```
tg-llm-bot/
├── bot/
│   ├── config.py      # 读环境变量
│   ├── llm.py         # OpenAI 异步客户端封装
│   ├── storage.py     # 每用户状态持久化（JSON 文件）
│   ├── presets.py     # 预设系统提示词
│   ├── handlers.py    # 命令 / 按钮 / 消息处理
│   └── main.py        # Application 装配 + 入口
├── data/              # 运行时持久化目录（挂载进容器）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
