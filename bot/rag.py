"""Lightweight local RAG store backed by SQLite."""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    src = (text or "").strip()
    if not src:
        return []
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        end = min(n, i + size)
        # Prefer splitting on newline near window end.
        if end < n:
            cut = src.rfind("\n", i + max(0, size - 300), end)
            if cut > i:
                end = cut
        chunk = src[i:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        i = max(i + 1, end - overlap)
    return out


def _pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


@dataclass
class RagHit:
    file_name: str
    score: float
    chunk_text: str


class RagStore:
    def __init__(self, data_dir: Path):
        self._db_path = data_dir / "rag.sqlite3"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(file_id) REFERENCES rag_files(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_files_user_created ON rag_files(user_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_file ON rag_chunks(file_id, chunk_index)"
            )
            conn.commit()

    @staticmethod
    def parse_upload(file_name: str, payload: bytes) -> tuple[str, str]:
        lower = (file_name or "").lower()
        if lower.endswith(".pdf"):
            text = _pdf_to_text(payload)
            return text, "pdf"
        if lower.endswith(".txt"):
            return _decode_text(payload), "txt"
        if lower.endswith(".md"):
            return _decode_text(payload), "md"
        raise ValueError("仅支持 PDF / TXT / MD")

    async def ingest(
        self,
        *,
        user_id: int,
        file_name: str,
        file_type: str,
        text: str,
        embeddings: list[list[float]],
        chunks: list[str],
    ) -> int:
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        now = time.time()

        def _write() -> int:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO rag_files(user_id, file_name, file_type, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, file_name, file_type, now),
                )
                fid = int(cur.lastrowid)
                conn.executemany(
                    """
                    INSERT INTO rag_chunks(file_id, chunk_index, chunk_text, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (fid, i, chunk, json.dumps(emb, ensure_ascii=False), now)
                        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                    ],
                )
                conn.commit()
                return fid

        async with self._lock:
            return await asyncio.to_thread(_write)

    async def list_files(self, *, user_id: int, limit: int = 20) -> list[dict]:
        lim = max(1, min(int(limit), 100))

        def _read() -> list[dict]:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT f.id, f.file_name, f.file_type, f.created_at, COUNT(c.id) AS chunks
                    FROM rag_files f
                    LEFT JOIN rag_chunks c ON c.file_id = f.id
                    WHERE f.user_id = ?
                    GROUP BY f.id
                    ORDER BY f.created_at DESC
                    LIMIT ?
                    """,
                    (user_id, lim),
                ).fetchall()
                return [dict(r) for r in rows]

        return await asyncio.to_thread(_read)

    async def search(
        self,
        *,
        user_id: int,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[RagHit]:
        k = max(1, min(int(top_k), 20))
        threshold = float(min_score)

        def _query() -> list[RagHit]:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT f.file_name, c.chunk_text, c.embedding_json
                    FROM rag_chunks c
                    JOIN rag_files f ON f.id = c.file_id
                    WHERE f.user_id = ?
                    ORDER BY c.id DESC
                    LIMIT 1500
                    """,
                    (user_id,),
                ).fetchall()
            scored: list[RagHit] = []
            for r in rows:
                try:
                    emb = json.loads(r["embedding_json"])
                except Exception:
                    continue
                score = _cosine(query_embedding, emb)
                if score >= threshold:
                    scored.append(
                        RagHit(
                            file_name=str(r["file_name"]),
                            score=score,
                            chunk_text=str(r["chunk_text"]),
                        )
                    )
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:k]

        return await asyncio.to_thread(_query)

    @staticmethod
    def chunk_text(text: str, size: int, overlap: int) -> list[str]:
        return _chunk_text(text, size=size, overlap=overlap)

