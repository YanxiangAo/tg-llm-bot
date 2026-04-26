"""Telegram command/message handlers."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from .config import Config
from .llm import LLMClient
from .presets import PRESETS
from .storage import Storage, UserState

log = logging.getLogger(__name__)

TG_TEXT_LIMIT = 4000  # leave a small safety margin under the 4096 hard limit
STREAM_EDIT_INTERVAL = 1.2  # seconds between message edits while streaming


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #


def auth_required(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *a, **kw):
        cfg: Config = context.application.bot_data["cfg"]
        user = update.effective_user
        if user is None:
            return
        if cfg.allowed_user_ids and user.id not in cfg.allowed_user_ids:
            log.warning("Unauthorized access attempt by %s (%s)", user.id, user.username)
            if update.effective_message is not None:
                await update.effective_message.reply_text(
                    f"⛔️ 该 Bot 已启用白名单，你的 user id 是 `{user.id}`，未在白名单中。\n"
                    f"请联系管理员把这个 id 加入 ALLOWED_USER_IDS。",
                    parse_mode="Markdown",
                )
            return
        return await func(update, context, *a, **kw)

    return wrapper


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _split(text: str, n: int = TG_TEXT_LIMIT) -> list[str]:
    if len(text) <= n:
        return [text] if text else [""]
    out: list[str] = []
    cur = text
    while len(cur) > n:
        # try to break on a newline within the last 500 chars to avoid mid-word cuts
        cut = cur.rfind("\n", n - 500, n)
        if cut <= 0:
            cut = n
        out.append(cur[:cut])
        cur = cur[cut:].lstrip("\n")
    if cur:
        out.append(cur)
    return out


async def _safe_edit(msg, text: str) -> None:
    try:
        await msg.edit_text(text, disable_web_page_preview=True)
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return
        # ignore "Message_too_long" etc – caller is expected to have chunked.
        log.debug("edit_text BadRequest: %s", e)
    except Exception as e:
        log.debug("edit_text failed: %s", e)


def _build_messages(state: UserState, user_content: Any) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if state.system_prompt:
        msgs.append({"role": "system", "content": state.system_prompt})
    msgs.extend(state.history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


# --------------------------------------------------------------------------- #
# /start /help /id
# --------------------------------------------------------------------------- #

WELCOME = (
    "👋 你好，我是一个连到 LLM API 的 Telegram Bot。\n\n"
    "直接发文字就能聊天，发图片可以做视觉理解。\n"
    "所有命令见 /help。"
)

HELP_TEXT = (
    "*基础命令*\n"
    "/start 显示欢迎\n"
    "/help 显示本帮助\n"
    "/id 查看你的 Telegram user id（用于白名单）\n"
    "\n"
    "*会话*\n"
    "/reset 清空当前对话历史\n"
    "/stats 查看当前配置 + Token 用量\n"
    "\n"
    "*模型与提示词*\n"
    "/model 切换模型（按钮选择）\n"
    "/system 查看当前系统提示词\n"
    "/system <内容> 设置系统提示词（会清空历史）\n"
    "/system clear 清空系统提示词\n"
    "/preset 选择预置系统提示词\n"
    "\n"
    "*采样参数*\n"
    "/params 用按钮微调 temperature / top\\_p / max\\_tokens / stream\n"
    "/set temperature 0.7\n"
    "/set top\\_p 0.9\n"
    "/set max\\_tokens 4096\n"
    "/set stream on|off\n"
)


@auth_required
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


@auth_required
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # intentionally NOT @auth_required so newcomers can fetch their id
    user = update.effective_user
    if user is None:
        return
    await update.message.reply_text(
        f"你的 Telegram user id: `{user.id}`\n用户名: @{user.username or '(无)'}",
        parse_mode="Markdown",
    )


# --------------------------------------------------------------------------- #
# /model
# --------------------------------------------------------------------------- #


async def _resolve_models(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    cfg: Config = context.application.bot_data["cfg"]
    client: LLMClient = context.application.bot_data["llm"]
    models = await client.list_models()
    if not models:
        cached: list[str] | None = context.application.bot_data.get("models_cache")
        if cached:
            return cached
        models = [cfg.default_model]
    context.application.bot_data["models_cache"] = models
    return models


@auth_required
async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    models = await _resolve_models(context)

    rows: list[list[InlineKeyboardButton]] = []
    for m in models:
        prefix = "✅ " if m == state.model else ""
        rows.append([InlineKeyboardButton(f"{prefix}{m}", callback_data=f"model:{m}")])
    rows.append([InlineKeyboardButton("🔄 刷新模型列表", callback_data="model:_refresh")])

    await update.message.reply_text(
        f"当前模型：`{state.model}`\n点击切换：",
        reply_markup=InlineKeyboardMarkup(rows),
        parse_mode="Markdown",
    )


async def cb_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    cfg: Config = context.application.bot_data["cfg"]
    if cfg.allowed_user_ids and q.from_user.id not in cfg.allowed_user_ids:
        await q.edit_message_text("⛔️ 你不在白名单。")
        return

    payload = q.data.split(":", 1)[1]
    storage: Storage = context.application.bot_data["storage"]

    if payload == "_refresh":
        context.application.bot_data.pop("models_cache", None)
        models = await _resolve_models(context)
        state = await storage.get(q.from_user.id)
        rows = [
            [InlineKeyboardButton(("✅ " if m == state.model else "") + m, callback_data=f"model:{m}")]
            for m in models
        ]
        rows.append([InlineKeyboardButton("🔄 刷新模型列表", callback_data="model:_refresh")])
        await q.edit_message_text(
            f"已刷新。当前模型：`{state.model}`",
            reply_markup=InlineKeyboardMarkup(rows),
            parse_mode="Markdown",
        )
        return

    await storage.update(q.from_user.id, model=payload)
    await q.edit_message_text(f"✅ 已切换模型：`{payload}`", parse_mode="Markdown")


# --------------------------------------------------------------------------- #
# /system /preset
# --------------------------------------------------------------------------- #


@auth_required
async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    args_text = (update.message.text or "").split(maxsplit=1)
    if len(args_text) == 1:
        cur = state.system_prompt or "(空)"
        await update.message.reply_text(
            "当前系统提示词：\n"
            "————————\n"
            f"{cur}\n"
            "————————\n"
            "用法：`/system <内容>` 设置；`/system clear` 清空；`/preset` 选预设。",
            parse_mode="Markdown",
        )
        return

    new_prompt = args_text[1].strip()
    if new_prompt.lower() == "clear":
        await storage.update(update.effective_user.id, system_prompt="", history=[])
        await update.message.reply_text("✅ 系统提示词已清空，对话历史已重置。")
        return

    await storage.update(update.effective_user.id, system_prompt=new_prompt, history=[])
    await update.message.reply_text("✅ 系统提示词已更新，对话历史已重置。")


@auth_required
async def cmd_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = [
        [InlineKeyboardButton(name, callback_data=f"preset:{key}")]
        for key, (name, _) in PRESETS.items()
    ]
    await update.message.reply_text(
        "选择一个预设系统提示词（会清空对话历史）：",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def cb_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    cfg: Config = context.application.bot_data["cfg"]
    if cfg.allowed_user_ids and q.from_user.id not in cfg.allowed_user_ids:
        await q.edit_message_text("⛔️ 你不在白名单。")
        return

    key = q.data.split(":", 1)[1]
    if key not in PRESETS:
        await q.edit_message_text("未知预设。")
        return
    name, prompt = PRESETS[key]
    storage: Storage = context.application.bot_data["storage"]
    await storage.update(q.from_user.id, system_prompt=prompt, history=[])
    await q.edit_message_text(
        f"✅ 已切换到预设「{name}」。\n\n系统提示词：\n{prompt}"
    )


# --------------------------------------------------------------------------- #
# /params /set
# --------------------------------------------------------------------------- #

PARAM_BOUNDS = {
    "temperature": (0.0, 2.0, 0.1),
    "top_p": (0.0, 1.0, 0.05),
    "max_tokens": (256, 32768, 512),
}


def _params_keyboard(state: UserState) -> InlineKeyboardMarkup:
    def row(name: str, value: float | int, fmt: str) -> list[InlineKeyboardButton]:
        return [
            InlineKeyboardButton(f"➖", callback_data=f"param:{name}:dec"),
            InlineKeyboardButton(f"{name}: {value:{fmt}}", callback_data=f"param:{name}:noop"),
            InlineKeyboardButton(f"➕", callback_data=f"param:{name}:inc"),
        ]

    rows = [
        row("temperature", state.temperature, ".2f"),
        row("top_p", state.top_p, ".2f"),
        row("max_tokens", state.max_tokens, "d"),
        [
            InlineKeyboardButton(
                f"stream: {'on ✅' if state.stream else 'off ❌'} (点击切换)",
                callback_data="param:stream:toggle",
            )
        ],
        [InlineKeyboardButton("关闭", callback_data="param:_:close")],
    ]
    return InlineKeyboardMarkup(rows)


def _params_text(state: UserState) -> str:
    return (
        "*采样参数*\n"
        f"temperature = `{state.temperature:.2f}`  范围 0.0–2.0\n"
        f"top\\_p = `{state.top_p:.2f}`  范围 0.0–1.0\n"
        f"max\\_tokens = `{state.max_tokens}`  范围 256–32768\n"
        f"stream = `{state.stream}`\n"
    )


@auth_required
async def cmd_params(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    await update.message.reply_text(
        _params_text(state),
        reply_markup=_params_keyboard(state),
        parse_mode="Markdown",
    )


async def cb_param(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    cfg: Config = context.application.bot_data["cfg"]
    if cfg.allowed_user_ids and q.from_user.id not in cfg.allowed_user_ids:
        await q.edit_message_text("⛔️ 你不在白名单。")
        return

    _, name, action = q.data.split(":", 2)
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(q.from_user.id)

    if action == "close":
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        return
    if action == "noop":
        return

    if name == "stream" and action == "toggle":
        state = await storage.update(q.from_user.id, stream=not state.stream)
    elif name in PARAM_BOUNDS and action in {"inc", "dec"}:
        lo, hi, step = PARAM_BOUNDS[name]
        cur = getattr(state, name)
        nv = cur + step if action == "inc" else cur - step
        nv = max(lo, min(hi, nv))
        if name == "max_tokens":
            nv = int(round(nv))
        else:
            nv = round(nv, 2)
        state = await storage.update(q.from_user.id, **{name: nv})
    else:
        return

    try:
        await q.edit_message_text(
            _params_text(state),
            reply_markup=_params_keyboard(state),
            parse_mode="Markdown",
        )
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            raise


@auth_required
async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    parts = (update.message.text or "").split()
    if len(parts) < 3:
        await update.message.reply_text(
            "用法：`/set <key> <value>`\nkey 支持：`temperature`, `top_p`, `max_tokens`, `stream`",
            parse_mode="Markdown",
        )
        return
    key = parts[1].lower()
    raw = parts[2]
    try:
        if key == "stream":
            val = raw.lower() in {"on", "true", "1", "yes"}
            state = await storage.update(update.effective_user.id, stream=val)
            await update.message.reply_text(f"✅ stream = `{state.stream}`", parse_mode="Markdown")
            return
        if key in {"temperature", "top_p"}:
            v = float(raw)
            lo, hi, _ = PARAM_BOUNDS[key]
            if not (lo <= v <= hi):
                raise ValueError(f"超出范围 [{lo}, {hi}]")
            state = await storage.update(update.effective_user.id, **{key: v})
            await update.message.reply_text(f"✅ {key} = `{v:.3f}`", parse_mode="Markdown")
            return
        if key == "max_tokens":
            v = int(raw)
            lo, hi, _ = PARAM_BOUNDS[key]
            if not (lo <= v <= hi):
                raise ValueError(f"超出范围 [{lo}, {hi}]")
            state = await storage.update(update.effective_user.id, max_tokens=v)
            await update.message.reply_text(f"✅ max_tokens = `{v}`", parse_mode="Markdown")
            return
        await update.message.reply_text(f"未知 key: `{key}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 设置失败：{e}")


# --------------------------------------------------------------------------- #
# /reset /stats
# --------------------------------------------------------------------------- #


@auth_required
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    await storage.reset_history(update.effective_user.id)
    await update.message.reply_text("🧹 对话历史已清空。")


@auth_required
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    sp = state.system_prompt
    sp_short = (sp[:80] + "…") if len(sp) > 80 else sp
    text = (
        "*当前会话*\n"
        f"模型: `{state.model}`\n"
        f"系统提示词: `{sp_short or '(空)'}`\n"
        f"temperature=`{state.temperature:.2f}` top\\_p=`{state.top_p:.2f}` "
        f"max\\_tokens=`{state.max_tokens}` stream=`{state.stream}`\n"
        f"历史消息数: `{len(state.history)}`\n\n"
        "*累计用量*\n"
        f"请求数: `{state.total_requests}`\n"
        f"prompt tokens: `{state.total_prompt_tokens}`\n"
        f"completion tokens: `{state.total_completion_tokens}`\n"
        f"合计 tokens: `{state.total_prompt_tokens + state.total_completion_tokens}`\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# --------------------------------------------------------------------------- #
# Chat (text + photo)
# --------------------------------------------------------------------------- #


async def _download_photo_b64(message, context: ContextTypes.DEFAULT_TYPE) -> str:
    photo = message.photo[-1]
    f = await context.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await f.download_to_memory(out=buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@auth_required
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    await _do_chat(update, context, user_content=text, history_user_msg={"role": "user", "content": text})


@auth_required
async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    caption = msg.caption or "请描述这张图片。"
    try:
        b64 = await _download_photo_b64(msg, context)
    except Exception as e:
        await msg.reply_text(f"❌ 下载图片失败：{e}")
        return
    content = [
        {"type": "text", "text": caption},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]
    history_user_msg = {"role": "user", "content": f"[图片] {caption}"}
    await _do_chat(update, context, user_content=content, history_user_msg=history_user_msg)


async def _do_chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_content: Any,
    history_user_msg: dict[str, Any],
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]
    user_id = update.effective_user.id
    state = await storage.get(user_id)

    messages = _build_messages(state, user_content)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)
    except Exception:
        pass

    try:
        if state.stream:
            await _do_chat_stream(update, context, state, messages, history_user_msg)
        else:
            await _do_chat_once(update, context, state, messages, history_user_msg)
    except Exception as e:
        log.exception("chat failed")
        await update.message.reply_text(f"❌ 调用模型失败：{e}")


async def _do_chat_once(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: UserState,
    messages: list[dict[str, Any]],
    history_user_msg: dict[str, Any],
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]

    text, pt, ct = await llm.chat_once(
        model=state.model,
        messages=messages,
        temperature=state.temperature,
        top_p=state.top_p,
        max_tokens=state.max_tokens,
    )
    if not text:
        text = "(模型返回空内容)"

    for chunk in _split(text):
        await update.message.reply_text(chunk, disable_web_page_preview=True)

    await storage.append_history(
        update.effective_user.id,
        [history_user_msg, {"role": "assistant", "content": text}],
        cap=cfg.max_history_messages,
    )
    await storage.add_usage(update.effective_user.id, pt, ct)


async def _do_chat_stream(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: UserState,
    messages: list[dict[str, Any]],
    history_user_msg: dict[str, Any],
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]

    placeholder = await update.message.reply_text("…", disable_web_page_preview=True)
    full = ""
    cur_msg = placeholder
    cur_text = ""
    last_edit = 0.0
    pt, ct = 0, 0

    try:
        async for delta, p, c in llm.chat_stream(
            model=state.model,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            max_tokens=state.max_tokens,
        ):
            if p or c:
                pt, ct = p or pt, c or ct
            if not delta:
                continue
            full += delta
            cur_text += delta

            # If current chunk grew too large, finalize it and continue in a new message.
            if len(cur_text) >= TG_TEXT_LIMIT:
                # Finalize at the largest paragraph boundary that fits.
                cut = cur_text.rfind("\n", TG_TEXT_LIMIT - 500, TG_TEXT_LIMIT)
                if cut <= 0:
                    cut = TG_TEXT_LIMIT
                final_part = cur_text[:cut]
                rest = cur_text[cut:].lstrip("\n")
                await _safe_edit(cur_msg, final_part)
                cur_msg = await update.message.reply_text(
                    rest or "…", disable_web_page_preview=True
                )
                cur_text = rest
                last_edit = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_edit >= STREAM_EDIT_INTERVAL:
                await _safe_edit(cur_msg, cur_text or "…")
                last_edit = now
    finally:
        # Final flush for the last chunk
        if cur_text:
            await _safe_edit(cur_msg, cur_text)

    if not full:
        await _safe_edit(cur_msg, "(模型返回空内容)")
        full = ""

    await storage.append_history(
        update.effective_user.id,
        [history_user_msg, {"role": "assistant", "content": full}],
        cap=cfg.max_history_messages,
    )
    await storage.add_usage(update.effective_user.id, pt, ct)


# --------------------------------------------------------------------------- #
# Error handler
# --------------------------------------------------------------------------- #


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(f"❌ 出错了：{context.error}")
        except Exception:
            pass
