# tg-llm-bot

[中文说明](./README.md)

A Telegram bot connected to any OpenAI-compatible API, with support for:

- Model switching (`/model`)
- System prompt management (`/system`) + prompt presets (`/preset`)
- Parameter tuning (`/params`: temperature, top_p, max_tokens, stream)
- Multi-turn context + one-command reset (`/reset`)
- Streaming replies (edit message while generating)
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
| `ALLOWED_USER_IDS` | Comma-separated Telegram numeric user IDs; empty means no restriction (**not recommended**) |
| `DEFAULT_MODEL` | Default model ID |
| `AVAILABLE_MODELS` | Keep empty to fetch from `GET /v1/models` dynamically |

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
/system You are a poet    set system prompt (resets history)
/system clear             clear system prompt
/preset                   choose prompt preset
/params                   tune parameters with inline buttons
/set temperature 0.3      set parameter directly
/set max_tokens 4096
/set stream off
/reset                    clear conversation history
/stats                    show session config and token usage
/id                       show your Telegram user ID
```

Send text = normal chat.  
Send image = vision chat (caption is used as the question).

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
