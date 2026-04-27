# tg-llm-bot

[中文说明](./README.md)

A Telegram bot connected to any OpenAI-compatible API, with support for:

- Model switching (`/model`)
- System prompt management (`/system`) + prompt presets (`/preset`)
- Parameter tuning (`/params`: temperature, top_p, repeat_penalty, max_tokens, stream, web_search, thinking)
- Multi-turn context (history kept in current session)
- Multi-session management (list, switch, delete, auto titles)
- Streaming replies (edit message while generating)
- Dynamic waiting seconds before first token (to show bot is alive)
- Token usage stats (`/stats`)
- Vision input (send image directly, requires a vision-capable model)
- Telegram user ID whitelist
- One-command Docker Compose deployment

## 1) Prerequisites

1. Create a bot via [@BotFather](https://t.me/BotFather) and get your token.
2. Prepare an OpenAI-compatible API key and base URL:
   - OpenAI: `https://api.openai.com/v1`
   - Local new-api (this VPS): `http://127.0.0.1:3001/v1`
3. (Optional) Get your Telegram numeric user ID. If unknown, start the bot first and it can show your ID via `/id`.

## 2) Configuration

```bash
cd /root/tg-llm-bot
cp .env.example .env
vim .env
```

Key variables:

| Variable | Description |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Token from BotFather |
| `OPENAI_API_KEY` | API key for your LLM provider/gateway |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint, usually ending with `/v1` |
| `TAVILY_API_KEY` | External web-search key; with this set, `web_search` uses Tavily tool-calling |
| `ALLOWED_USER_IDS` | Comma-separated Telegram numeric user IDs; empty means no restriction (**not recommended**) |
| `DEFAULT_MODEL` | Default model ID |
| `AVAILABLE_MODELS` | Keep empty to fetch from `GET /v1/models` dynamically |
| `DEFAULT_REPEAT_PENALTY` | Default repeat penalty float |
| `WEB_SEARCH_DEFAULT` | Try web search by default (auto-fallback if unsupported) |
| `THINKING_DEFAULT` | Enable thinking mode by default (auto-fallback if unsupported) |

## 3) Start

```bash
cd /root/tg-llm-bot
docker compose pull
docker compose up -d
docker compose logs -f
```

By default, it pulls `otis49482/tg-llm-bot:latest` from Docker Hub.  
When logs show `Bot ready: @xxx`, the service is online.

## 4) Common Commands

Send these commands to the bot:

```text
/model                    choose model
/system                   show current system prompt
/system You are a poet    set system prompt
/system clear             clear system prompt
/preset                   choose prompt preset
/params                   tune parameters with inline buttons
/set temperature 0.3      set parameter directly
/set top_p 0.9
/set repeat_penalty 1.1
/set max_tokens 4096
/set stream off
/set web_search on
/set thinking on
/sessions                 list sessions (switch/delete)
/newchat                  create a new session
/use <session_id>         continue an old session
/delsession <session_id>  delete a session (no arg = current)
/stats                    show session config and token usage
/id                       show your Telegram user ID
```

Send text = normal chat.  
Send image = vision chat (caption is used as the question).  
When switching sessions (`/use` or `/sessions` button), the bot also sends a short conversation summary for quick context recovery.
System prompt and sampling parameters are also session-scoped and restored when you switch back.
Thinking-mode support is auto-detected per model, with fallback when unsupported.  
When `web_search` is on and `TAVILY_API_KEY` is configured, the model can call an external Tavily web-search tool before finalizing the answer.

## 5) Todo

- [ ] Support usage in Telegram `channel` chats.
- [ ] Support usage in Telegram `group/supergroup` chats (`@mention`, reply trigger, in-group permissions).
- [ ] Support explicit chain-of-thought display mode (toggleable, off by default).

## 6) Maintenance

```bash
docker compose ps
docker compose logs -f --tail 200
docker compose restart
docker compose pull && docker compose up -d
```

User states and history are persisted in `./data/users.json`.

## 7) Security Notes

- Strongly recommend setting `ALLOWED_USER_IDS`.
- Never commit `.env` to git.
- `network_mode: host` is used to access local `127.0.0.1:3001` directly.  
  The bot does not expose inbound ports; it only makes outbound requests to Telegram and your LLM API.

## 8) Project Structure

```text
tg-llm-bot/
├── bot/
│   ├── config.py
│   ├── llm.py
│   ├── storage.py
│   ├── presets.py
│   ├── tools/
│   ├── handlers.py
│   └── main.py
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
└── README_EN.md
```
