"""대화 저장소 — 사용자가 로그인하면 지난 대화가 그대로 남아 있게 한다.

## 왜 두 가지를 따로 저장하나

`ConversationState` 는 **모델에게 줄 문맥**이다. 요약 압축(`maintain_conversation_summary`)이
최근 6개만 남기고 오래된 메시지를 **지우기 때문에**, 이것만 저장하면 사용자가 다시 로그인했을 때
"지난번에 뭘 물어봤더라" 를 볼 수 없다. 그래서 둘을 나눠 보관한다.

  state       최근 메시지 + 누적 요약 + (선택) 포트폴리오   → 다음 턴의 문맥 복원용
  messages    주고받은 전부, 지워지지 않음                  → 화면에 보여줄 대화 내역

## 개인정보

`portfolio_context` 는 보유 자산·진단 결과라 기본적으로 **저장하지 않는다**(`persist_portfolio=False`).
앱이 로그인 때 자신의 DB 에서 읽어 `ChatService.ask(portfolio=...)` 로 매번 넣어주는 것을 전제로 한다.
개발·시연 목적으로 저장이 필요하면 `persist_portfolio=True` 로 켠다.

## 구현체

  SqliteConversationStore   기본값. 파일 하나, 동시 접근 안전(WAL), 사용자별 조회에 인덱스
  JsonConversationStore     대화 하나당 JSON 파일 하나. 눈으로 열어보기 좋아 개발·테스트용

앱이 자체 DB(Postgres 등)를 쓰면 `ConversationStore` 프로토콜 5개 메서드만 구현해 갈아끼우면 된다.

## 사용자 격리

모든 조회·수정은 `user_id` 를 함께 받고 그것으로 거른다. 남의 `conversation_id` 를 알아도
다른 사용자로는 열리지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol

TITLE_MAX = 40


def now_iso() -> str:
    # 마이크로초까지 — 같은 초에 저장된 대화끼리 목록 정렬이 흔들리지 않게 한다
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def make_title(first_question: str) -> str:
    t = " ".join(first_question.split())
    return t[:TITLE_MAX] + ("…" if len(t) > TITLE_MAX else "")


@dataclass(frozen=True)
class StoredMessage:
    """화면에 보여줄 한 메시지. seq 는 대화 안에서 1부터 증가한다."""

    seq: int
    role: str                      # "user" | "assistant"
    content: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)   # route·status 등 진단 정보 (assistant 만)


@dataclass(frozen=True)
class ConversationMeta:
    """목록 화면용 요약 정보. 본문은 담지 않는다."""

    conversation_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    n_messages: int


@dataclass
class ConversationRecord:
    meta: ConversationMeta
    state: dict[str, Any]          # ConversationState.to_dict()


class ConversationStore(Protocol):
    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationMeta]: ...
    def load(self, user_id: str, conversation_id: str) -> Optional[ConversationRecord]: ...
    def save(self, record: ConversationRecord) -> None: ...
    def append_messages(self, user_id: str, conversation_id: str,
                        messages: list[tuple[str, str, dict[str, Any]]]) -> int: ...
    def messages(self, user_id: str, conversation_id: str, *,
                 limit: Optional[int] = None, before_seq: Optional[int] = None) -> list[StoredMessage]: ...
    def delete(self, user_id: str, conversation_id: str) -> bool: ...


def _check_ids(user_id: str, conversation_id: str = "x") -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 는 비어 있을 수 없습니다.")
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id 는 비어 있을 수 없습니다.")


# ── SQLite ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  conversation_id TEXT PRIMARY KEY,
  user_id         TEXT NOT NULL,
  title           TEXT NOT NULL DEFAULT '',
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  state_json      TEXT NOT NULL,
  n_messages      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  conversation_id TEXT NOT NULL,
  seq             INTEGER NOT NULL,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  created_at      TEXT NOT NULL,
  meta_json       TEXT,
  PRIMARY KEY (conversation_id, seq)
);
"""


class SqliteConversationStore:
    """파일 하나에 담는 기본 저장소. 여러 프로세스가 붙어도 되도록 WAL 을 쓴다."""

    def __init__(self, path: Path | str = "data/chat/conversations.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            yield con
            con.commit()
        finally:
            con.close()

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationMeta]:
        _check_ids(user_id)
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (user_id, limit)).fetchall()
        return [self._meta(r) for r in rows]

    def load(self, user_id: str, conversation_id: str) -> Optional[ConversationRecord]:
        _check_ids(user_id, conversation_id)
        with self._connect() as con:
            r = con.execute("SELECT * FROM conversations WHERE conversation_id=? AND user_id=?",
                            (conversation_id, user_id)).fetchone()
        if r is None:
            return None
        return ConversationRecord(meta=self._meta(r), state=json.loads(r["state_json"]))

    def save(self, record: ConversationRecord) -> None:
        m = record.meta
        _check_ids(m.user_id, m.conversation_id)
        payload = json.dumps(record.state, ensure_ascii=False)
        with self._lock, self._connect() as con:
            con.execute(
                """INSERT INTO conversations
                     (conversation_id, user_id, title, created_at, updated_at, state_json, n_messages)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(conversation_id) DO UPDATE SET
                     title=excluded.title, updated_at=excluded.updated_at,
                     state_json=excluded.state_json, n_messages=excluded.n_messages
                   WHERE conversations.user_id=excluded.user_id""",
                (m.conversation_id, m.user_id, m.title, m.created_at, m.updated_at, payload, m.n_messages))

    def append_messages(self, user_id: str, conversation_id: str,
                        messages: list[tuple[str, str, dict[str, Any]]]) -> int:
        """(role, content, meta) 목록을 이어 붙이고 마지막 seq 를 돌려준다."""
        _check_ids(user_id, conversation_id)
        if not messages:
            return 0
        ts = now_iso()
        with self._lock, self._connect() as con:
            owner = con.execute("SELECT user_id FROM conversations WHERE conversation_id=?",
                                (conversation_id,)).fetchone()
            if owner is not None and owner["user_id"] != user_id:
                raise PermissionError("다른 사용자의 대화에는 쓸 수 없습니다.")
            seq = con.execute("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id=?",
                              (conversation_id,)).fetchone()[0]
            for role, content, meta in messages:
                seq += 1
                con.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content, created_at, meta_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (conversation_id, seq, role, content, ts,
                     json.dumps(meta, ensure_ascii=False) if meta else None))
            con.execute("UPDATE conversations SET n_messages=?, updated_at=? WHERE conversation_id=?",
                        (seq, ts, conversation_id))
        return seq

    def messages(self, user_id: str, conversation_id: str, *,
                 limit: Optional[int] = None, before_seq: Optional[int] = None) -> list[StoredMessage]:
        """오래된 순으로 돌려준다. limit 를 주면 **최근** limit 개를 가져온 뒤 순서를 맞춘다
        (화면에서 '이전 대화 더 보기' 를 하려면 before_seq 와 함께 쓴다)."""
        _check_ids(user_id, conversation_id)
        with self._connect() as con:
            if con.execute("SELECT 1 FROM conversations WHERE conversation_id=? AND user_id=?",
                           (conversation_id, user_id)).fetchone() is None:
                return []
            if limit is None:
                rows = con.execute(
                    "SELECT * FROM messages WHERE conversation_id=? AND (? IS NULL OR seq < ?) ORDER BY seq",
                    (conversation_id, before_seq, before_seq)).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM messages WHERE conversation_id=? AND (? IS NULL OR seq < ?) "
                    "ORDER BY seq DESC LIMIT ?",
                    (conversation_id, before_seq, before_seq, limit)).fetchall()[::-1]
        return [StoredMessage(seq=r["seq"], role=r["role"], content=r["content"], created_at=r["created_at"],
                              meta=json.loads(r["meta_json"]) if r["meta_json"] else {}) for r in rows]

    def delete(self, user_id: str, conversation_id: str) -> bool:
        _check_ids(user_id, conversation_id)
        with self._lock, self._connect() as con:
            n = con.execute("DELETE FROM conversations WHERE conversation_id=? AND user_id=?",
                            (conversation_id, user_id)).rowcount
            if n:
                con.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        return bool(n)

    @staticmethod
    def _meta(r: sqlite3.Row) -> ConversationMeta:
        return ConversationMeta(conversation_id=r["conversation_id"], user_id=r["user_id"], title=r["title"],
                                created_at=r["created_at"], updated_at=r["updated_at"], n_messages=r["n_messages"])


# ── JSON 파일 ─────────────────────────────────────────────────────────────────

class JsonConversationStore:
    """대화 하나 = 파일 하나. 파일명은 conversation_id(16진수)라 경로 조작 위험이 없다."""

    def __init__(self, root: Path | str = "data/chat/conversations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, conversation_id: str) -> Path:
        if not conversation_id.isalnum():
            raise ValueError("conversation_id 는 영숫자여야 합니다.")
        return self.root / f"{conversation_id}.json"

    def _read(self, conversation_id: str) -> Optional[dict]:
        p = self._path(conversation_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        p = self._path(data["conversation_id"])
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)                                   # 쓰다 만 파일이 남지 않게

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationMeta]:
        _check_ids(user_id)
        out = []
        for p in self.root.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if d.get("user_id") == user_id:
                out.append(self._meta(d))
        out.sort(key=lambda m: m.updated_at, reverse=True)
        return out[:limit]

    def load(self, user_id: str, conversation_id: str) -> Optional[ConversationRecord]:
        _check_ids(user_id, conversation_id)
        d = self._read(conversation_id)
        if d is None or d.get("user_id") != user_id:
            return None
        return ConversationRecord(meta=self._meta(d), state=d["state"])

    def save(self, record: ConversationRecord) -> None:
        m = record.meta
        _check_ids(m.user_id, m.conversation_id)
        with self._lock:
            d = self._read(m.conversation_id) or {"messages": []}
            if d.get("user_id") not in (None, m.user_id):
                raise PermissionError("다른 사용자의 대화에는 쓸 수 없습니다.")
            d.update({"conversation_id": m.conversation_id, "user_id": m.user_id, "title": m.title,
                      "created_at": d.get("created_at", m.created_at), "updated_at": m.updated_at,
                      "n_messages": m.n_messages, "state": record.state})
            self._write(d)

    def append_messages(self, user_id: str, conversation_id: str,
                        messages: list[tuple[str, str, dict[str, Any]]]) -> int:
        _check_ids(user_id, conversation_id)
        if not messages:
            return 0
        ts = now_iso()
        with self._lock:
            d = self._read(conversation_id)
            if d is None:
                raise KeyError("저장되지 않은 대화입니다 — save() 를 먼저 호출하세요.")
            if d.get("user_id") != user_id:
                raise PermissionError("다른 사용자의 대화에는 쓸 수 없습니다.")
            seq = d["messages"][-1]["seq"] if d["messages"] else 0
            for role, content, meta in messages:
                seq += 1
                d["messages"].append({"seq": seq, "role": role, "content": content,
                                      "created_at": ts, "meta": meta or {}})
            d["n_messages"] = seq
            d["updated_at"] = ts
            self._write(d)
        return seq

    def messages(self, user_id: str, conversation_id: str, *,
                 limit: Optional[int] = None, before_seq: Optional[int] = None) -> list[StoredMessage]:
        _check_ids(user_id, conversation_id)
        d = self._read(conversation_id)
        if d is None or d.get("user_id") != user_id:
            return []
        rows = [m for m in d["messages"] if before_seq is None or m["seq"] < before_seq]
        if limit is not None:
            rows = rows[-limit:]
        return [StoredMessage(seq=m["seq"], role=m["role"], content=m["content"],
                              created_at=m["created_at"], meta=m.get("meta", {})) for m in rows]

    def delete(self, user_id: str, conversation_id: str) -> bool:
        _check_ids(user_id, conversation_id)
        with self._lock:
            d = self._read(conversation_id)
            if d is None or d.get("user_id") != user_id:
                return False
            self._path(conversation_id).unlink(missing_ok=True)
        return True

    @staticmethod
    def _meta(d: dict) -> ConversationMeta:
        return ConversationMeta(conversation_id=d["conversation_id"], user_id=d["user_id"], title=d.get("title", ""),
                                created_at=d["created_at"], updated_at=d["updated_at"],
                                n_messages=d.get("n_messages", 0))
