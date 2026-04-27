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


async def _waiting_indicator_loop(
    msg,
    stop_event: asyncio.Event,
    cancel_event: asyncio.Event | None = None,
) -> None:
    """Update a waiting message every second until stopped or cancelled."""
    started_at = time.monotonic()
    last_elapsed = -1
    while True:
        if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
            break
        elapsed = int(time.monotonic() - started_at)
        if elapsed != last_elapsed:
            await _safe_edit(msg, f"⏳ 正在思考中... {elapsed}s")
            last_elapsed = elapsed
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


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
    "/stop 终止当前正在生成的回复\n"
    "\n"
    "*会话*\n"
    "/sessions 查看历史会话列表\n"
    "/newchat 新建会话\n"
    "/use <会话ID> 切换并继续某个历史会话\n"
    "/delsession <会话ID> 删除会话（不带参数则删当前）\n"
    "/stats 查看当前配置 + Token 用量\n"
    "\n"
    "*模型与提示词*\n"
    "/model 切换模型（按钮选择）\n"
    "/system 查看当前系统提示词\n"
    "/system <内容> 设置系统提示词\n"
    "/system clear 清空系统提示词\n"
    "/preset 选择预置系统提示词\n"
    "\n"
    "*采样参数*\n"
    "/params 用按钮微调 temperature / top\\_p / repeat\\_penalty / max\\_tokens / stream / web\\_search / thinking\n"
    "/set temperature 0.7\n"
    "/set top\\_p 0.9\n"
    "/set repeat\\_penalty 1.1\n"
    "/set max\\_tokens 4096\n"
    "/set stream on|off\n"
    "/set web\\_search on|off\n"
    "/set thinking on|off\n"
)


def _summarize_title(text: str, limit: int = 24) -> str:
    t = (text or "").replace("\n", " ").replace("`", "'").strip()
    if not t:
        return "新会话"
    if t.startswith("[图片]"):
        t = t.replace("[图片]", "", 1).strip()
        t = f"图片: {t}" if t else "图片会话"
    return t[:limit] + ("…" if len(t) > limit else "")


def _session_history_summary(history: list[dict[str, Any]]) -> str:
    """Build a compact plain-text summary from an existing conversation history."""
    if not history:
        return "（该会话暂无历史消息）"

    user_msgs: list[str] = []
    assistant_msgs: list[str] = []
    for m in history:
        role = str(m.get("role", ""))
        content = str(m.get("content", "")).replace("\n", " ").strip()
        if not content:
            continue
        if role == "user":
            user_msgs.append(content)
        elif role == "assistant":
            assistant_msgs.append(content)

    first_user = user_msgs[0][:70] + ("…" if len(user_msgs[0]) > 70 else "") if user_msgs else "（无）"
    last_user = user_msgs[-1][:70] + ("…" if len(user_msgs[-1]) > 70 else "") if user_msgs else "（无）"
    last_assistant = (
        assistant_msgs[-1][:120] + ("…" if len(assistant_msgs[-1]) > 120 else "")
        if assistant_msgs
        else "（无）"
    )

    return (
        f"- 用户轮次：{len(user_msgs)}\n"
        f"- 助手轮次：{len(assistant_msgs)}\n"
        f"- 首个问题：{first_user}\n"
        f"- 最近问题：{last_user}\n"
        f"- 最近回答：{last_assistant}"
    )


async def _render_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(user_id)
    sessions = await storage.list_sessions(user_id)
    rows: list[list[InlineKeyboardButton]] = []
    lines = ["会话列表（最新在前）"]
    for i, s in enumerate(sessions[:12], start=1):
        sid = s.get("id", "")
        title = s.get("title", "新会话")
        hlen = len(s.get("history", []))
        active = " ✅" if sid == state.active_session_id else ""
        lines.append(f"{sid} · {title} ({hlen}条){active}")
        rows.append(
            [
                InlineKeyboardButton(f"切换 #{i}", callback_data=f"session:use:{sid}"),
                InlineKeyboardButton(f"删除 #{i}", callback_data=f"session:del:{sid}"),
            ]
        )
    rows.append([InlineKeyboardButton("➕ 新建会话", callback_data="session:new")])
    await update.effective_message.reply_text(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(rows),
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


@auth_required
async def cmd_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_sessions(update, context, update.effective_user.id)


@auth_required
async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    active = context.application.bot_data.setdefault("active_generations", {})
    run = active.get(update.effective_user.id)
    if not run:
        await update.message.reply_text("当前没有正在生成的回复。")
        return
    cancel_event: asyncio.Event = run["cancel_event"]
    task: asyncio.Task = run["task"]
    cancel_event.set()
    if not task.done():
        task.cancel()
    await update.message.reply_text("⏹️ 已请求停止当前任务（思考/回复）。")


@auth_required
async def cmd_newchat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    sess = await storage.create_session(update.effective_user.id, title="新会话")
    await update.message.reply_text(f"✅ 已新建会话：`{sess['id']}`", parse_mode="Markdown")


@auth_required
async def cmd_use(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    parts = (update.message.text or "").split()
    if len(parts) < 2:
        await update.message.reply_text("用法：`/use <会话ID>`", parse_mode="Markdown")
        return
    sid = parts[1].strip()
    ok = await storage.switch_session(update.effective_user.id, sid)
    if not ok:
        await update.message.reply_text(f"❌ 未找到会话：`{sid}`", parse_mode="Markdown")
        return
    st = await storage.get(update.effective_user.id)
    summary = _session_history_summary(st.history)
    await update.message.reply_text(
        f"✅ 已切换到会话：`{sid}`（{len(st.history)}条历史）\n\n"
        f"该会话摘要：\n{summary}",
        parse_mode="Markdown",
    )


@auth_required
async def cmd_delsession(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    parts = (update.message.text or "").split()
    sid = parts[1].strip() if len(parts) > 1 else state.active_session_id
    ok, active_sid = await storage.delete_session(update.effective_user.id, sid)
    if not ok:
        await update.message.reply_text(f"❌ 未找到会话：`{sid}`", parse_mode="Markdown")
        return
    await update.message.reply_text(
        f"🗑️ 已删除会话：`{sid}`\n当前会话：`{active_sid}`",
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


async def cb_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    cfg: Config = context.application.bot_data["cfg"]
    if cfg.allowed_user_ids and q.from_user.id not in cfg.allowed_user_ids:
        await q.edit_message_text("⛔️ 你不在白名单。")
        return
    storage: Storage = context.application.bot_data["storage"]
    parts = (q.data or "").split(":", 2)
    if len(parts) < 2:
        return
    action = parts[1]
    sid = parts[2] if len(parts) > 2 else ""

    if action == "new":
        sess = await storage.create_session(q.from_user.id, title="新会话")
        await q.edit_message_text(f"✅ 已新建会话：`{sess['id']}`", parse_mode="Markdown")
        return
    if not sid:
        return
    if action == "use":
        ok = await storage.switch_session(q.from_user.id, sid)
        if ok:
            st = await storage.get(q.from_user.id)
            summary = _session_history_summary(st.history)
            await q.edit_message_text(
                f"✅ 已切换会话：`{sid}`（{len(st.history)}条历史）\n\n"
                f"该会话摘要：\n{summary}",
                parse_mode="Markdown",
            )
        else:
            await q.edit_message_text(f"❌ 会话不存在：`{sid}`", parse_mode="Markdown")
        return
    if action == "del":
        ok, active_sid = await storage.delete_session(q.from_user.id, sid)
        if ok:
            await q.edit_message_text(
                f"🗑️ 已删除会话：`{sid}`\n当前会话：`{active_sid}`",
                parse_mode="Markdown",
            )
        else:
            await q.edit_message_text(f"❌ 会话不存在：`{sid}`", parse_mode="Markdown")


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
        await storage.update(update.effective_user.id, system_prompt="")
        await update.message.reply_text("✅ 系统提示词已清空。")
        return

    await storage.update(update.effective_user.id, system_prompt=new_prompt)
    await update.message.reply_text("✅ 系统提示词已更新。")


@auth_required
async def cmd_preset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = [
        [InlineKeyboardButton(name, callback_data=f"preset:{key}")]
        for key, (name, _) in PRESETS.items()
    ]
    await update.message.reply_text(
        "选择一个预设系统提示词：",
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
    await storage.update(q.from_user.id, system_prompt=prompt)
    await q.edit_message_text(
        f"✅ 已切换到预设「{name}」。\n\n系统提示词：\n{prompt}"
    )


# --------------------------------------------------------------------------- #
# /params /set
# --------------------------------------------------------------------------- #

PARAM_BOUNDS = {
    "temperature": (0.0, 2.0, 0.1),
    "top_p": (0.0, 1.0, 0.05),
    "repeat_penalty": (0.8, 2.0, 0.05),
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
        row("repeat_penalty", state.repeat_penalty, ".2f"),
        row("max_tokens", state.max_tokens, "d"),
        [
            InlineKeyboardButton(
                f"stream: {'on ✅' if state.stream else 'off ❌'} (点击切换)",
                callback_data="param:stream:toggle",
            )
        ],
        [
            InlineKeyboardButton(
                f"web_search: {'on ✅' if state.web_search else 'off ❌'} (点击切换)",
                callback_data="param:web_search:toggle",
            )
        ],
        [
            InlineKeyboardButton(
                f"thinking: {'on ✅' if state.thinking else 'off ❌'} (点击切换)",
                callback_data="param:thinking:toggle",
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
        f"repeat\\_penalty = `{state.repeat_penalty:.2f}`  范围 0.8–2.0\n"
        f"max\\_tokens = `{state.max_tokens}`  范围 256–32768\n"
        f"stream = `{state.stream}`\n"
        f"web\\_search = `{state.web_search}`\n"
        f"thinking = `{state.thinking}`\n"
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
    elif name == "web_search" and action == "toggle":
        state = await storage.update(q.from_user.id, web_search=not state.web_search)
    elif name == "thinking" and action == "toggle":
        state = await storage.update(q.from_user.id, thinking=not state.thinking)
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
            "用法：`/set <key> <value>`\n"
            "key 支持：`temperature`, `top_p`, `repeat_penalty`, `max_tokens`, `stream`, `web_search`, `thinking`",
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
        if key == "web_search":
            val = raw.lower() in {"on", "true", "1", "yes"}
            state = await storage.update(update.effective_user.id, web_search=val)
            await update.message.reply_text(f"✅ web_search = `{state.web_search}`", parse_mode="Markdown")
            return
        if key == "thinking":
            val = raw.lower() in {"on", "true", "1", "yes"}
            state = await storage.update(update.effective_user.id, thinking=val)
            await update.message.reply_text(f"✅ thinking = `{state.thinking}`", parse_mode="Markdown")
            return
        if key in {"temperature", "top_p", "repeat_penalty"}:
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
            state = await storage.update(update.effective_user.id, **{key: v})
            await update.message.reply_text(f"✅ {key} = `{v}`", parse_mode="Markdown")
            return
        await update.message.reply_text(f"未知 key: `{key}`", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 设置失败：{e}")


# --------------------------------------------------------------------------- #
# /stats
# --------------------------------------------------------------------------- #

@auth_required
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]
    state = await storage.get(update.effective_user.id)
    support = llm.get_web_search_support(state.model)
    thinking_support = llm.get_thinking_support(state.model)
    support_text = "unknown"
    if support is True:
        support_text = "supported"
    elif support is False:
        support_text = "unsupported"
    thinking_support_text = "unknown"
    if thinking_support is True:
        thinking_support_text = "supported"
    elif thinking_support is False:
        thinking_support_text = "unsupported"
    sp = state.system_prompt
    sp_short = (sp[:80] + "…") if len(sp) > 80 else sp
    session_title = _summarize_title(
        str(state.sessions.get(state.active_session_id, {}).get("title", "新会话")),
        limit=40,
    )
    text = (
        "*当前会话*\n"
        f"会话ID: `{state.active_session_id}`\n"
        f"会话标题: `{session_title}`\n"
        f"模型: `{state.model}`\n"
        f"系统提示词: `{sp_short or '(空)'}`\n"
        f"temperature=`{state.temperature:.2f}` top\\_p=`{state.top_p:.2f}` "
        f"repeat\\_penalty=`{state.repeat_penalty:.2f}` "
        f"max\\_tokens=`{state.max_tokens}` stream=`{state.stream}` web\\_search=`{state.web_search}` thinking=`{state.thinking}`\n"
        f"web search 支持探测: `{support_text}`\n"
        f"thinking 支持探测: `{thinking_support_text}`\n"
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
    active = context.application.bot_data.setdefault("active_generations", {})
    running = active.get(user_id)
    if running and not running["task"].done():
        await update.message.reply_text("你有一条回复还在生成中，可先用 /stop 终止。")
        return
    cancel_event = asyncio.Event()
    active[user_id] = {"task": asyncio.current_task(), "cancel_event": cancel_event}
    state = await storage.get(user_id)

    messages = _build_messages(state, user_content)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)
    except Exception:
        pass

    try:
        if state.stream:
            await _do_chat_stream(update, context, state, messages, history_user_msg, cancel_event)
        else:
            await _do_chat_once(update, context, state, messages, history_user_msg, cancel_event)
    except asyncio.CancelledError:
        log.info("chat cancelled for user %s", user_id)
    except Exception as e:
        log.exception("chat failed")
        await update.message.reply_text(f"❌ 调用模型失败：{e}")
    finally:
        run = active.get(user_id)
        if run and run["task"] is asyncio.current_task():
            active.pop(user_id, None)


async def _do_chat_once(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: UserState,
    messages: list[dict[str, Any]],
    history_user_msg: dict[str, Any],
    cancel_event: asyncio.Event,
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]
    waiting_msg = await update.message.reply_text("⏳ 正在思考中... 0s", disable_web_page_preview=True)
    waiting_stop = asyncio.Event()
    waiting_task = asyncio.create_task(_waiting_indicator_loop(waiting_msg, waiting_stop, cancel_event))

    if cancel_event.is_set():
        waiting_stop.set()
        await waiting_task
        await _safe_edit(waiting_msg, "⏹️ 已停止生成。")
        return

    try:
        text, pt, ct = await llm.chat_once(
            model=state.model,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            repeat_penalty=state.repeat_penalty,
            max_tokens=state.max_tokens,
            web_search=state.web_search,
            thinking=state.thinking,
        )
    finally:
        waiting_stop.set()
        await waiting_task

    if cancel_event.is_set():
        await _safe_edit(waiting_msg, "⏹️ 已停止生成。")
        return

    if not text:
        text = "(模型返回空内容)"

    chunks = _split(text)
    await _safe_edit(waiting_msg, chunks[0])
    for chunk in chunks[1:]:
        await update.message.reply_text(chunk, disable_web_page_preview=True)

    await storage.append_history(
        update.effective_user.id,
        [history_user_msg, {"role": "assistant", "content": text}],
        cap=cfg.max_history_messages,
        title_hint=_summarize_title(str(history_user_msg.get("content", ""))),
    )
    await storage.add_usage(update.effective_user.id, pt, ct)


async def _do_chat_stream(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: UserState,
    messages: list[dict[str, Any]],
    history_user_msg: dict[str, Any],
    cancel_event: asyncio.Event,
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]

    placeholder = await update.message.reply_text("⏳ 正在思考中... 0s", disable_web_page_preview=True)
    waiting_stop = asyncio.Event()
    waiting_task = asyncio.create_task(_waiting_indicator_loop(placeholder, waiting_stop, cancel_event))
    first_token_received = False
    full = ""
    cur_msg = placeholder
    cur_text = ""
    last_edit = 0.0
    pt, ct = 0, 0
    stopped = False

    try:
        async for delta, p, c in llm.chat_stream(
            model=state.model,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            repeat_penalty=state.repeat_penalty,
            max_tokens=state.max_tokens,
            web_search=state.web_search,
            thinking=state.thinking,
        ):
            if cancel_event.is_set():
                stopped = True
                break
            if p or c:
                pt, ct = p or pt, c or ct
            if not delta:
                continue
            if not first_token_received:
                first_token_received = True
                waiting_stop.set()
                await waiting_task
                cur_text = ""
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
        waiting_stop.set()
        await waiting_task
        # Final flush for the last chunk
        if cur_text:
            await _safe_edit(cur_msg, cur_text)

    if not full:
        await _safe_edit(cur_msg, "⏹️ 已停止生成。" if stopped else "(模型返回空内容)")
        full = ""
    elif stopped:
        await _safe_edit(cur_msg, (cur_text or full)[-TG_TEXT_LIMIT:] + "\n\n⏹️ 已停止")

    if stopped:
        return

    await storage.append_history(
        update.effective_user.id,
        [history_user_msg, {"role": "assistant", "content": full}],
        cap=cfg.max_history_messages,
        title_hint=_summarize_title(str(history_user_msg.get("content", ""))),
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
