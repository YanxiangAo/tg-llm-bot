"""Application entrypoint."""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from telegram import BotCommand
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .config import Config
from .handlers import (
    cb_model,
    cb_param,
    cb_preset,
    cb_session,
    cmd_help,
    cmd_id,
    cmd_model,
    cmd_newchat,
    cmd_params,
    cmd_preset,
    cmd_sessions,
    cmd_delsession,
    cmd_reset,
    cmd_set,
    cmd_start,
    cmd_stats,
    cmd_system,
    cmd_use,
    on_error,
    on_photo,
    on_text,
)
from .llm import LLMClient
from .storage import Storage, UserState


log = logging.getLogger("tg-llm-bot")


async def _post_init(app: Application) -> None:
    cfg: Config = app.bot_data["cfg"]

    # Pre-fill model cache so the first /model is fast
    llm: LLMClient = app.bot_data["llm"]
    try:
        ids = await llm.list_models()
        app.bot_data["models_cache"] = ids or [cfg.default_model]
        log.info("Discovered %d models from upstream", len(ids))
    except Exception as e:
        log.warning("models.list failed: %s", e)
        app.bot_data["models_cache"] = [cfg.default_model]

    # Register the slash-command list shown in the Telegram client
    await app.bot.set_my_commands(
        [
            BotCommand("start", "欢迎"),
            BotCommand("help", "查看所有命令"),
            BotCommand("model", "切换模型"),
            BotCommand("sessions", "查看和切换历史会话"),
            BotCommand("system", "查看/设置系统提示词"),
            BotCommand("preset", "选择预设系统提示词"),
            BotCommand("params", "调节采样参数"),
            BotCommand("set", "/set <key> <value>"),
            BotCommand("reset", "清空对话历史"),
            BotCommand("stats", "查看配置与 token 用量"),
            BotCommand("id", "查看你的 user id"),
        ]
    )

    me = await app.bot.get_me()
    log.info("Bot ready: @%s (%s)", me.username, me.id)
    if cfg.allowed_user_ids:
        log.info("Whitelist: %s", sorted(cfg.allowed_user_ids))
    else:
        log.warning("No whitelist configured – the bot is OPEN to anyone who finds it!")


def build_app(cfg: Config) -> Application:
    storage = Storage(
        cfg.data_dir,
        UserState(
            model=cfg.default_model,
            system_prompt="",
            temperature=cfg.default_temperature,
            top_p=cfg.default_top_p,
            top_k=cfg.default_top_k,
            repeat_penalty=cfg.default_repeat_penalty,
            max_tokens=cfg.default_max_tokens,
            stream=cfg.stream_default,
        ),
    )
    llm = LLMClient(api_key=cfg.api_key, base_url=cfg.base_url)

    app = (
        Application.builder()
        .token(cfg.bot_token)
        .rate_limiter(AIORateLimiter())
        .post_init(_post_init)
        .build()
    )
    app.bot_data["cfg"] = cfg
    app.bot_data["storage"] = storage
    app.bot_data["llm"] = llm

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("sessions", cmd_sessions))
    app.add_handler(CommandHandler("newchat", cmd_newchat))
    app.add_handler(CommandHandler("use", cmd_use))
    app.add_handler(CommandHandler("delsession", cmd_delsession))
    app.add_handler(CommandHandler("system", cmd_system))
    app.add_handler(CommandHandler("preset", cmd_preset))
    app.add_handler(CommandHandler("params", cmd_params))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("settings", cmd_stats))

    app.add_handler(CallbackQueryHandler(cb_model, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(cb_session, pattern=r"^session:"))
    app.add_handler(CallbackQueryHandler(cb_preset, pattern=r"^preset:"))
    app.add_handler(CallbackQueryHandler(cb_param, pattern=r"^param:"))

    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_error_handler(on_error)
    return app


def main() -> None:
    load_dotenv()
    cfg = Config.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # quiet a few noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)

    app = build_app(cfg)
    log.info("Starting Telegram bot, base_url=%s, default_model=%s", cfg.base_url, cfg.default_model)
    app.run_polling(allowed_updates=["message", "callback_query"], drop_pending_updates=True)


if __name__ == "__main__":
    main()
