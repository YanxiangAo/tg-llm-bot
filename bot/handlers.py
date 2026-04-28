"""Telegram command/message handlers."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

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
from .rag import RagStore
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
    sess = state.sessions.get(state.active_session_id, {}) if state.sessions else {}
    sess_summary = str(sess.get("summary", "") or "").strip()
    if sess_summary:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "以下是当前会话的历史摘要（用于衔接上下文，优先级低于更近的原始对话）：\n"
                    f"{sess_summary}"
                ),
            }
        )
    msgs.extend(state.history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _content_for_estimate(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                # Avoid counting base64 payload bytes as text tokens.
                parts.append("[image]")
        return "\n".join(parts)
    return str(content or "")


_ENC_CACHE: dict[str, Any] = {}


def _get_encoder_for_model(model: str | None) -> Any:
    if tiktoken is None:
        return None
    m = (model or "").strip()
    cache_key = m or "__default__"
    enc = _ENC_CACHE.get(cache_key)
    if enc is not None:
        return enc
    try:
        enc = tiktoken.encoding_for_model(m) if m else tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    _ENC_CACHE[cache_key] = enc
    return enc


def _estimate_text_tokens(text: str, *, model: str | None = None) -> int:
    if not text:
        return 0
    enc = _get_encoder_for_model(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: rough multilingual estimate when tokenizer is unavailable.
    return max(1, len(text) // 3)


def _estimate_messages_tokens(messages: list[dict[str, Any]], *, model: str | None = None) -> int:
    total = 0
    for m in messages:
        total += 4
        total += _estimate_text_tokens(_content_for_estimate(m.get("content", "")), model=model)
    return total


def _clip_summary_for_context(summary: str, cfg: Config, *, model: str | None = None) -> str:
    text = str(summary or "").strip()
    if not text:
        return ""
    cap = max(0, int(cfg.summary_context_max_tokens))
    if cap <= 0:
        return text
    # Keep the latest part when summary grows too long.
    while text and _estimate_text_tokens(text, model=model) > cap:
        text = text[len(text) // 5 :].lstrip()
    return text


def _extract_user_text(user_content: Any) -> str:
    return _content_for_estimate(user_content).strip()


def _rag_context_prompt(snippets: list[tuple[str, float, str]]) -> str:
    if not snippets:
        return ""
    lines = [
        "以下是用户私有知识库检索到的片段，仅用于补充事实；若与用户最新问题冲突，以用户最新问题为准。"
    ]
    for i, (fname, score, text) in enumerate(snippets, start=1):
        lines.append(f"[{i}] 来源: {fname} (score={score:.3f})")
        lines.append(text[:1200])
    return "\n".join(lines).strip()


def _unique_nonempty_models(*models: str | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in models:
        v = str(m or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


async def _embed_texts_with_fallback(
    *,
    llm: LLMClient,
    models: list[str],
    inputs: list[str],
    expected_len: int,
    user_id: int,
    scenario: str,
) -> tuple[list[list[float]], str | None]:
    for model in models:
        try:
            vectors = await llm.embed_texts(model=model, inputs=inputs)
            if len(vectors) == expected_len:
                return vectors, model
            log.warning(
                "embedding fallback miss user=%s scenario=%s model=%s reason=invalid_len expected=%s got=%s",
                user_id,
                scenario,
                model,
                expected_len,
                len(vectors),
            )
        except Exception as e:
            log.warning(
                "embedding fallback miss user=%s scenario=%s model=%s err=%s",
                user_id,
                scenario,
                model,
                e,
            )
    return [], None


async def _maybe_build_rag_snippets(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    state: UserState,
    user_content: Any,
) -> list[tuple[str, float, str]]:
    cfg: Config = context.application.bot_data["cfg"]
    rag: RagStore | None = context.application.bot_data.get("rag")
    llm: LLMClient = context.application.bot_data["llm"]
    if rag is None or not state.rag_enabled:
        return []
    query = _extract_user_text(user_content)
    if not query:
        return []
    emb_models = _unique_nonempty_models(
        state.embedding_model,
        cfg.embedding_model,
    )
    emb, used_model = await _embed_texts_with_fallback(
        llm=llm,
        models=emb_models,
        inputs=[query],
        expected_len=1,
        user_id=user_id,
        scenario="rag_query",
    )
    if not emb:
        log.warning("rag query embedding failed user=%s models=%s", user_id, emb_models)
        return []
    if used_model and used_model != (state.embedding_model or "").strip():
        log.info("rag query embedding fallback hit user=%s model=%s", user_id, used_model)
    hits = await rag.search(
        user_id=user_id,
        query_embedding=emb[0],
        top_k=cfg.rag_top_k,
        min_score=cfg.rag_min_score,
    )
    return [(h.file_name, h.score, h.chunk_text) for h in hits]


def _trim_messages_for_budget(
    state: UserState,
    user_content: Any,
    cfg: Config,
    rag_prompt: str = "",
) -> list[dict[str, Any]]:
    # Keep system context, prefer recent raw history, and hard-cap prompt size.
    system_msgs: list[dict[str, Any]] = []
    if state.system_prompt:
        system_msgs.append({"role": "system", "content": state.system_prompt})
    sess = state.sessions.get(state.active_session_id, {}) if state.sessions else {}
    model = (state.model or cfg.default_model or "").strip() or None
    sess_summary = _clip_summary_for_context(
        str(sess.get("summary", "") or "").strip(),
        cfg,
        model=model,
    )
    if sess_summary:
        system_msgs.append(
            {
                "role": "system",
                "content": (
                    "以下是当前会话的历史摘要（用于衔接上下文，优先级低于更近的原始对话）：\n"
                    f"{sess_summary}"
                ),
            }
        )
    if rag_prompt:
        system_msgs.append({"role": "system", "content": rag_prompt})
    user_msg = {"role": "user", "content": user_content}

    keep_recent = max(1, cfg.prompt_keep_recent_messages)
    history_tail = list(state.history[-keep_recent:])
    msgs = [*system_msgs, *history_tail, user_msg]

    budget = max(0, cfg.prompt_max_input_tokens)
    if budget <= 0:
        return msgs
    if _estimate_messages_tokens(msgs, model=model) <= budget:
        return msgs

    trimmed = list(history_tail)
    while trimmed and _estimate_messages_tokens([*system_msgs, *trimmed, user_msg], model=model) > budget:
        trimmed.pop(0)
    return [*system_msgs, *trimmed, user_msg]


def _history_to_summary_text(history: list[dict[str, Any]], max_chars: int = 12000) -> str:
    parts: list[str] = []
    for m in history:
        role = str(m.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        tag = "用户" if role == "user" else "助手"
        parts.append(f"{tag}: {content}")
    merged = "\n".join(parts)
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]


async def _summarize_session_if_needed(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    session_id: str,
) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    storage: Storage = context.application.bot_data["storage"]
    llm: LLMClient = context.application.bot_data["llm"]
    try:
        active_sid, history, existing_summary = await storage.get_active_session_snapshot(user_id)
        if active_sid != session_id:
            log.info(
                "summary skip user=%s session=%s reason=inactive_session active=%s",
                user_id,
                session_id,
                active_sid,
            )
            return
        state = await storage.get(user_id)
        summary_models = _unique_nonempty_models(
            state.summary_model,
            state.model,
            cfg.default_model,
        )
        if not summary_models:
            log.info(
                "summary skip user=%s session=%s reason=empty_summary_model",
                user_id,
                session_id,
            )
            return
        summary_model = summary_models[0]
        est_hist_tokens = _estimate_messages_tokens(history, model=summary_model)
        by_message_count = len(history) >= cfg.summary_trigger_messages
        by_token_size = est_hist_tokens >= cfg.summary_trigger_tokens
        log.info(
            "summary check user=%s session=%s hist_msgs=%s hist_tokens=%s trig_msgs=%s trig_tokens=%s by_count=%s by_tokens=%s",
            user_id,
            session_id,
            len(history),
            est_hist_tokens,
            cfg.summary_trigger_messages,
            cfg.summary_trigger_tokens,
            by_message_count,
            by_token_size,
        )
        if not (by_message_count or by_token_size):
            log.info(
                "summary skip user=%s session=%s reason=below_threshold",
                user_id,
                session_id,
            )
            return
        keep_recent = max(1, cfg.summary_keep_recent_messages)
        # For short-but-very-long histories, reduce keep_recent dynamically so
        # summarization can still happen instead of being blocked by keep_recent.
        if by_token_size and len(history) <= keep_recent + 2:
            old_keep = keep_recent
            keep_recent = max(1, min(keep_recent, len(history) // 2))
            log.info(
                "summary adjust_keep_recent user=%s session=%s old=%s new=%s hist_msgs=%s",
                user_id,
                session_id,
                old_keep,
                keep_recent,
                len(history),
            )
        if len(history) <= keep_recent:
            log.info(
                "summary skip user=%s session=%s reason=no_summarizable_window keep_recent=%s hist_msgs=%s",
                user_id,
                session_id,
                keep_recent,
                len(history),
            )
            return
        to_summarize = history[:-keep_recent]
        summary_input = _history_to_summary_text(to_summarize)
        if not summary_input:
            log.info(
                "summary skip user=%s session=%s reason=empty_summary_input keep_recent=%s",
                user_id,
                session_id,
                keep_recent,
            )
            return
        log.info(
            "summary start user=%s session=%s summarize_msgs=%s keep_recent=%s has_existing=%s model=%s",
            user_id,
            session_id,
            len(to_summarize),
            keep_recent,
            bool(existing_summary),
            summary_model,
        )
        summary_prompt = (
            "你是会话压缩器。请把下面历史对话压缩成可用于后续问答的连续摘要。\n"
            "要求：\n"
            "1) 保留用户目标、约束、已确认结论、未完成事项；\n"
            "2) 保留关键事实/参数/口径，避免冗余细节；\n"
            "3) 使用简明中文，优先条目化；\n"
            "4) 不要臆测，不要新增事实。\n\n"
            f"已有摘要（可为空）：\n{existing_summary or '(无)'}\n\n"
            f"需要压缩的新增历史：\n{summary_input}"
        )
        messages = [
            {"role": "system", "content": "你擅长将长对话压缩为准确、可持续更新的工作摘要。"},
            {"role": "user", "content": summary_prompt},
        ]
        summary_text = ""
        used_summary_model: str | None = None
        for m in summary_models:
            try:
                candidate, _, _ = await llm.chat_once(
                    user_id=user_id,
                    model=m,
                    messages=messages,
                    temperature=0.2,
                    top_p=1.0,
                    repeat_penalty=1.0,
                    max_tokens=cfg.summary_max_tokens,
                    web_search=False,
                    thinking=False,
                    prompt_cache=state.prompt_cache,
                    prompt_cache_key=f"summary:{user_id}:{session_id}:{m}",
                )
                candidate = (candidate or "").strip()
                if candidate:
                    summary_text = candidate
                    used_summary_model = m
                    break
                log.warning(
                    "summary fallback miss user=%s session=%s model=%s reason=empty_output",
                    user_id,
                    session_id,
                    m,
                )
            except Exception as e:
                log.warning(
                    "summary fallback miss user=%s session=%s model=%s err=%s",
                    user_id,
                    session_id,
                    m,
                    e,
                )
        if not summary_text or not used_summary_model:
            log.info(
                "summary skip user=%s session=%s reason=all_fallbacks_failed models=%s",
                user_id,
                session_id,
                summary_models,
            )
            return
        if used_summary_model != summary_model:
            log.info(
                "summary fallback hit user=%s session=%s model=%s",
                user_id,
                session_id,
                used_summary_model,
            )
        new_title = await _generate_summary_title(
            llm,
            model=used_summary_model,
            summary_text=summary_text,
        )
        ok = await storage.set_session_summary_and_trim(
            user_id=user_id,
            session_id=session_id,
            summary=summary_text,
            keep_recent=keep_recent,
            title=new_title,
        )
        if ok:
            log.info(
                "session summarized user=%s session=%s keep_recent=%s hist_msgs=%s hist_tokens=%s title=%s",
                user_id,
                session_id,
                keep_recent,
                len(history),
                est_hist_tokens,
                new_title,
            )
        else:
            log.info(
                "summary skip user=%s session=%s reason=store_rejected",
                user_id,
                session_id,
            )
    except Exception:
        log.exception("session summarization failed user=%s session=%s", user_id, session_id)
    finally:
        tasks = context.application.bot_data.setdefault("summary_tasks", {})
        tasks.pop((user_id, session_id), None)


def _schedule_session_summary(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_id: int,
    session_id: str,
) -> None:
    tasks = context.application.bot_data.setdefault("summary_tasks", {})
    key = (user_id, session_id)
    task: asyncio.Task | None = tasks.get(key)
    if task and not task.done():
        return
    tasks[key] = asyncio.create_task(
        _summarize_session_if_needed(context, user_id=user_id, session_id=session_id)
    )


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
    "/summary 查看当前会话的压缩摘要\n"
    "/ragfiles 查看已上传知识库文件\n"
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
    "/params 用按钮微调 temperature / top\\_p / repeat\\_penalty / max\\_tokens / stream / web\\_search / thinking / prompt\\_cache / rag\n"
    "/set temperature 0.7\n"
    "/set top\\_p 0.9\n"
    "/set repeat\\_penalty 1.1\n"
    "/set max\\_tokens 4096\n"
    "/set summary\\_model llama-3.3-70b-versatile\n"
    "/set stream on|off\n"
    "/set web\\_search on|off\n"
    "/set thinking on|off\n"
    "/set prompt\\_cache on|off\n"
    "/set rag on|off\n"
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


def _stored_session_summary(state: UserState) -> str:
    sess = state.sessions.get(state.active_session_id, {}) if state.sessions else {}
    return str(sess.get("summary", "") or "").strip()


async def _generate_summary_title(
    llm: LLMClient,
    *,
    model: str,
    summary_text: str,
) -> str:
    prompt = (
        "你是会话标题生成器。请基于下面摘要生成一个简短标题。\n"
        "要求：\n"
        "1) 仅输出标题本身；\n"
        "2) 中文优先；\n"
        "3) 不超过18个字；\n"
        "4) 不要标点结尾。\n\n"
        f"摘要：\n{summary_text}"
    )
    try:
        title, _, _ = await llm.chat_once(
            user_id=0,
            model=model,
            messages=[
                {"role": "system", "content": "你擅长将摘要压缩为简洁、可读的会话标题。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=1.0,
            repeat_penalty=1.0,
            max_tokens=32,
            web_search=False,
            thinking=False,
            prompt_cache=False,
        )
        t = str(title or "").strip().strip("\"'`")
        return _summarize_title(t, limit=18)
    except Exception as e:
        log.warning("summary title generation failed: %s", e)
        return _summarize_title(summary_text, limit=18)


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
    summary = _stored_session_summary(st)
    summary_text = summary if summary else "（该会话暂无压缩摘要）"
    await update.message.reply_text(
        f"✅ 已切换到会话：`{sid}`（{len(st.history)}条历史）\n\n"
        f"该会话压缩摘要：\n{summary_text}",
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


@auth_required
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    storage: Storage = context.application.bot_data["storage"]
    state = await storage.get(update.effective_user.id)
    summary = _stored_session_summary(state)
    if not summary:
        await update.message.reply_text(
            f"当前会话：`{state.active_session_id}`\n压缩摘要：`(空)`",
            parse_mode="Markdown",
        )
        return
    await update.message.reply_text(
        f"当前会话：`{state.active_session_id}`\n\n压缩摘要：\n{summary}",
        parse_mode="Markdown",
    )


@auth_required
async def cmd_ragfiles(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rag: RagStore | None = context.application.bot_data.get("rag")
    if rag is None:
        await update.message.reply_text("RAG 存储未初始化。")
        return
    files = await rag.list_files(user_id=update.effective_user.id, limit=20)
    if not files:
        await update.message.reply_text("当前没有已上传的知识库文件。直接发 PDF/TXT/MD 给我即可入库。")
        return
    lines = ["最近知识库文件（最多20条）"]
    for f in files:
        lines.append(f"- {f['file_name']} ({f['file_type']}, chunks={f['chunks']})")
    await update.message.reply_text("\n".join(lines))


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
            summary = _stored_session_summary(st)
            summary_text = summary if summary else "（该会话暂无压缩摘要）"
            await q.edit_message_text(
                f"✅ 已切换会话：`{sid}`（{len(st.history)}条历史）\n\n"
                f"该会话压缩摘要：\n{summary_text}",
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
                f"summary_model: {state.summary_model} (点击切换)",
                callback_data="param:summary_model:open",
            )
        ],
        [
            InlineKeyboardButton(
                f"embedding_model: {state.embedding_model} (点击切换)",
                callback_data="param:embedding_model:open",
            )
        ],
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
        [
            InlineKeyboardButton(
                f"prompt_cache: {'on ✅' if state.prompt_cache else 'off ❌'} (点击切换)",
                callback_data="param:prompt_cache:toggle",
            )
        ],
        [
            InlineKeyboardButton(
                f"rag: {'on ✅' if state.rag_enabled else 'off ❌'} (点击切换)",
                callback_data="param:rag_enabled:toggle",
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
        f"summary\\_model = `{state.summary_model}`\n"
        f"embedding\\_model = `{state.embedding_model}`\n"
        f"stream = `{state.stream}`\n"
        f"web\\_search = `{state.web_search}`\n"
        f"thinking = `{state.thinking}`\n"
        f"prompt\\_cache = `{state.prompt_cache}`\n"
        f"rag = `{state.rag_enabled}`\n"
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
    if name == "summary_model":
        choice_cache: dict[int, list[str]] = context.application.bot_data.setdefault(
            "summary_model_choices", {}
        )
        if action in {"open", "refresh"}:
            if action == "refresh":
                context.application.bot_data.pop("models_cache", None)
            models = await _resolve_models(context)
            choice_cache[q.from_user.id] = models
            state = await storage.get(q.from_user.id)
            rows = [
                [
                    InlineKeyboardButton(
                        ("✅ " if m == state.summary_model else "") + m,
                        callback_data=f"param:summary_model:set:{i}",
                    )
                ]
                for i, m in enumerate(models)
            ]
            rows.append([InlineKeyboardButton("🔄 刷新模型列表", callback_data="param:summary_model:refresh")])
            rows.append([InlineKeyboardButton("⬅️ 返回参数面板", callback_data="param:summary_model:back")])
            await q.edit_message_text(
                f"当前 summary_model：{state.summary_model}\n点击切换：",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return
        if action == "back":
            await q.edit_message_text(
                _params_text(state),
                reply_markup=_params_keyboard(state),
                parse_mode="Markdown",
            )
            return
        if action.startswith("set:"):
            idx_raw = action.split(":", 1)[1].strip()
            try:
                idx = int(idx_raw)
            except ValueError:
                return
            models = choice_cache.get(q.from_user.id) or await _resolve_models(context)
            if idx < 0 or idx >= len(models):
                return
            model = models[idx]
            state = await storage.update(q.from_user.id, summary_model=model)
            choice_cache[q.from_user.id] = models
            rows = [
                [
                    InlineKeyboardButton(
                        ("✅ " if m == state.summary_model else "") + m,
                        callback_data=f"param:summary_model:set:{i}",
                    )
                ]
                for i, m in enumerate(models)
            ]
            rows.append([InlineKeyboardButton("🔄 刷新模型列表", callback_data="param:summary_model:refresh")])
            rows.append([InlineKeyboardButton("⬅️ 返回参数面板", callback_data="param:summary_model:back")])
            await q.edit_message_text(
                f"✅ 已切换 summary_model：{state.summary_model}",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return
    if name == "embedding_model":
        choice_cache: dict[int, list[str]] = context.application.bot_data.setdefault(
            "embedding_model_choices", {}
        )
        if action in {"open", "refresh"}:
            if action == "refresh":
                context.application.bot_data.pop("models_cache", None)
            models = await _resolve_models(context)
            choice_cache[q.from_user.id] = models
            state = await storage.get(q.from_user.id)
            rows = [
                [
                    InlineKeyboardButton(
                        ("✅ " if m == state.embedding_model else "") + m,
                        callback_data=f"param:embedding_model:set:{i}",
                    )
                ]
                for i, m in enumerate(models)
            ]
            rows.append(
                [InlineKeyboardButton("🔄 刷新模型列表", callback_data="param:embedding_model:refresh")]
            )
            rows.append([InlineKeyboardButton("⬅️ 返回参数面板", callback_data="param:embedding_model:back")])
            await q.edit_message_text(
                f"当前 embedding_model：{state.embedding_model}\n点击切换：",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return
        if action == "back":
            await q.edit_message_text(
                _params_text(state),
                reply_markup=_params_keyboard(state),
                parse_mode="Markdown",
            )
            return
        if action.startswith("set:"):
            idx_raw = action.split(":", 1)[1].strip()
            try:
                idx = int(idx_raw)
            except ValueError:
                return
            models = choice_cache.get(q.from_user.id) or await _resolve_models(context)
            if idx < 0 or idx >= len(models):
                return
            model = models[idx]
            state = await storage.update(q.from_user.id, embedding_model=model)
            choice_cache[q.from_user.id] = models
            rows = [
                [
                    InlineKeyboardButton(
                        ("✅ " if m == state.embedding_model else "") + m,
                        callback_data=f"param:embedding_model:set:{i}",
                    )
                ]
                for i, m in enumerate(models)
            ]
            rows.append(
                [InlineKeyboardButton("🔄 刷新模型列表", callback_data="param:embedding_model:refresh")]
            )
            rows.append([InlineKeyboardButton("⬅️ 返回参数面板", callback_data="param:embedding_model:back")])
            await q.edit_message_text(
                f"✅ 已切换 embedding_model：{state.embedding_model}",
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return
    if action == "noop":
        return

    if name == "stream" and action == "toggle":
        state = await storage.update(q.from_user.id, stream=not state.stream)
    elif name == "web_search" and action == "toggle":
        state = await storage.update(q.from_user.id, web_search=not state.web_search)
    elif name == "thinking" and action == "toggle":
        state = await storage.update(q.from_user.id, thinking=not state.thinking)
    elif name == "prompt_cache" and action == "toggle":
        state = await storage.update(q.from_user.id, prompt_cache=not state.prompt_cache)
    elif name == "rag_enabled" and action == "toggle":
        state = await storage.update(q.from_user.id, rag_enabled=not state.rag_enabled)
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
            "key 支持：`temperature`, `top_p`, `repeat_penalty`, `max_tokens`, `summary_model`, `embedding_model`, `stream`, `web_search`, `thinking`, `prompt_cache`, `rag`",
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
        if key == "prompt_cache":
            val = raw.lower() in {"on", "true", "1", "yes"}
            state = await storage.update(update.effective_user.id, prompt_cache=val)
            await update.message.reply_text(
                f"✅ prompt_cache = `{state.prompt_cache}`",
                parse_mode="Markdown",
            )
            return
        if key == "rag":
            val = raw.lower() in {"on", "true", "1", "yes"}
            state = await storage.update(update.effective_user.id, rag_enabled=val)
            await update.message.reply_text(f"✅ rag = `{state.rag_enabled}`", parse_mode="Markdown")
            return
        if key == "summary_model":
            v = raw.strip()
            if not v:
                raise ValueError("summary_model 不能为空")
            state = await storage.update(update.effective_user.id, summary_model=v)
            await update.message.reply_text(
                f"✅ summary_model = {state.summary_model}",
            )
            return
        if key == "embedding_model":
            v = raw.strip()
            if not v:
                raise ValueError("embedding_model 不能为空")
            state = await storage.update(update.effective_user.id, embedding_model=v)
            await update.message.reply_text(
                f"✅ embedding_model = {state.embedding_model}",
            )
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
    prompt_cache_support = llm.get_prompt_cache_support(state.model)
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
    prompt_cache_support_text = "unknown"
    if prompt_cache_support is True:
        prompt_cache_support_text = "supported"
    elif prompt_cache_support is False:
        prompt_cache_support_text = "unsupported"
    sp = state.system_prompt
    sp_short = (sp[:80] + "…") if len(sp) > 80 else sp
    session_title = _summarize_title(
        str(state.sessions.get(state.active_session_id, {}).get("title", "新会话")),
        limit=40,
    )
    session_summary = str(state.sessions.get(state.active_session_id, {}).get("summary", "") or "")
    summary_flag = "yes" if session_summary else "no"
    text = (
        "*当前会话*\n"
        f"会话ID: `{state.active_session_id}`\n"
        f"会话标题: `{session_title}`\n"
        f"模型: `{state.model}`\n"
        f"摘要模型: `{state.summary_model}`\n"
        f"embedding 模型: `{state.embedding_model}`\n"
        f"系统提示词: `{sp_short or '(空)'}`\n"
        f"temperature=`{state.temperature:.2f}` top\\_p=`{state.top_p:.2f}` "
        f"repeat\\_penalty=`{state.repeat_penalty:.2f}` "
        f"max\\_tokens=`{state.max_tokens}` stream=`{state.stream}` web\\_search=`{state.web_search}` thinking=`{state.thinking}` prompt\\_cache=`{state.prompt_cache}` rag=`{state.rag_enabled}`\n"
        f"web search 支持探测: `{support_text}`\n"
        f"thinking 支持探测: `{thinking_support_text}`\n"
        f"prompt cache 支持探测: `{prompt_cache_support_text}`\n"
        f"prompt cache 默认开关: `{cfg.prompt_cache_default}`\n"
        f"会话摘要: `{summary_flag}`\n"
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


@auth_required
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Config = context.application.bot_data["cfg"]
    rag: RagStore | None = context.application.bot_data.get("rag")
    llm: LLMClient = context.application.bot_data["llm"]
    storage: Storage = context.application.bot_data["storage"]
    msg = update.message
    doc = msg.document
    if rag is None or doc is None:
        return
    file_name = doc.file_name or "upload.bin"
    lower = file_name.lower()
    if not (lower.endswith(".pdf") or lower.endswith(".txt") or lower.endswith(".md")):
        await msg.reply_text("暂只支持 PDF/TXT/MD 文件。")
        return
    try:
        state = await storage.get(update.effective_user.id)
        tg_file = await context.bot.get_file(doc.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        raw = buf.getvalue()
        text, file_type = RagStore.parse_upload(file_name, raw)
        chunks = RagStore.chunk_text(
            text=text,
            size=cfg.rag_chunk_size,
            overlap=cfg.rag_chunk_overlap,
        )
        if not chunks:
            await msg.reply_text("文件内容为空或无法提取文本。")
            return
        emb_models = _unique_nonempty_models(
            state.embedding_model,
            cfg.embedding_model,
        )
        embeddings, used_model = await _embed_texts_with_fallback(
            llm=llm,
            models=emb_models,
            inputs=chunks,
            expected_len=len(chunks),
            user_id=update.effective_user.id,
            scenario="rag_ingest",
        )
        if len(embeddings) != len(chunks):
            await msg.reply_text("向量化失败：embedding 结果异常。")
            return
        if used_model and used_model != (state.embedding_model or "").strip():
            log.info("rag ingest embedding fallback hit user=%s model=%s", update.effective_user.id, used_model)
        await rag.ingest(
            user_id=update.effective_user.id,
            file_name=file_name,
            file_type=file_type,
            text=text,
            embeddings=embeddings,
            chunks=chunks,
        )
        await msg.reply_text(
            f"✅ 已入库：`{file_name}`\n"
            f"- 类型：`{file_type}`\n"
            f"- 分块：`{len(chunks)}`\n"
            f"- 检索：`{'on' if state.rag_enabled else 'off'}`",
            parse_mode="Markdown",
        )
    except Exception as e:
        await msg.reply_text(f"❌ 文件入库失败：{e}")


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

    # Do not block user replies on summarization; run it in background.
    _schedule_session_summary(
        context,
        user_id=user_id,
        session_id=state.active_session_id,
    )

    effective_web_search = state.web_search
    effective_thinking = state.thinking

    rag_snippets = await _maybe_build_rag_snippets(
        context=context,
        user_id=user_id,
        state=state,
        user_content=user_content,
    )
    rag_prompt = _rag_context_prompt(rag_snippets)
    messages = _trim_messages_for_budget(state, user_content, cfg, rag_prompt=rag_prompt)

    try:
        await update.message.chat.send_action(ChatAction.TYPING)
    except Exception:
        pass

    try:
        if state.stream:
            await _do_chat_stream(
                update,
                context,
                state,
                messages,
                history_user_msg,
                cancel_event,
                web_search=effective_web_search,
                thinking=effective_thinking,
            )
        else:
            await _do_chat_once(
                update,
                context,
                state,
                messages,
                history_user_msg,
                cancel_event,
                web_search=effective_web_search,
                thinking=effective_thinking,
            )
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
    *,
    web_search: bool,
    thinking: bool,
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
            user_id=update.effective_user.id,
            model=state.model,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            repeat_penalty=state.repeat_penalty,
            max_tokens=state.max_tokens,
            web_search=web_search,
            thinking=thinking,
            prompt_cache=state.prompt_cache,
            prompt_cache_key=(
                f"chat:{update.effective_user.id}:{state.active_session_id}:{state.model}"
            ),
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
    _schedule_session_summary(
        context,
        user_id=update.effective_user.id,
        session_id=state.active_session_id,
    )
    await storage.add_usage(update.effective_user.id, pt, ct)


async def _do_chat_stream(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    state: UserState,
    messages: list[dict[str, Any]],
    history_user_msg: dict[str, Any],
    cancel_event: asyncio.Event,
    *,
    web_search: bool,
    thinking: bool,
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
            user_id=update.effective_user.id,
            model=state.model,
            messages=messages,
            temperature=state.temperature,
            top_p=state.top_p,
            repeat_penalty=state.repeat_penalty,
            max_tokens=state.max_tokens,
            web_search=web_search,
            thinking=thinking,
            prompt_cache=state.prompt_cache,
            prompt_cache_key=(
                f"chat:{update.effective_user.id}:{state.active_session_id}:{state.model}"
            ),
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
    _schedule_session_summary(
        context,
        user_id=update.effective_user.id,
        session_id=state.active_session_id,
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
