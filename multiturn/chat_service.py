"""로그인한 사용자의 대화를 이어가는 층 — 멀티턴 세션을 요청 단위로 돌린다.

두 가지 쓰임새가 있다.

**무상태 (서빙 기본, `ask_stateless`)** — RAG 서버가 DB 를 쓰지 않는다. 앱이 자기 DB 에 대화를
저장하고, 매 요청에 이전 문맥(`state`)을 포트폴리오처럼 실어 보내면 답변과 갱신된 `state` 를
돌려준다. HF Space 같은 호스팅은 바깥 DB 포트(5432)를 막아 RAG 가 DB 에 붙을 수 없기 때문.

    service = ChatService(store=None)
    out = service.ask_stateless("u1", "그럼 종류는?", state=prev_state, portfolio=app_db.portfolio_of("u1"))
    out.answer, out.state, out.messages          # 앱: 메시지 2개 추가 + state 교체

**저장소 포함 (로컬 CLI·테스트, `ask`)** — RAG 가 SQLite/JSON 에 직접 저장한다.

    service = ChatService()                                   # 인덱스·리랭커·재작성기 1회 로드
    convs   = service.list_conversations(user_id)             # 로그인 직후 대화 목록
    msgs    = service.history(user_id, convs[0].conversation_id)   # 지난 대화 그대로 표시
    turn    = service.ask(user_id, "그럼 종류는?",
                          conversation_id=convs[0].conversation_id,
                          portfolio=app_db.portfolio_of(user_id))
    turn.answer, turn.conversation_id

어느 쪽이든 요청마다 state 로 세션을 만들고 답변 뒤 state 를 돌려준다/저장한다 — 요청이 프로세스를
넘나드는 환경을 전제로 해서 서버를 재시작해도 대화가 끊기지 않는다.
무거운 것(FAISS·BM25·리랭커·OpenAI 클라이언트)은 서비스 객체에 한 번만 올린다.

state 는 앱이 해석하지 않는 **불투명 값**이다 — 저장했다가 그대로 돌려보내기만 한다
(`{"version": 1, "messages": [...최근 6개], "conversation_summary": "..."}`, 수 KB).

`conversation_id` 를 안 주면 새 대화를 시작하고 그 id 를 돌려준다. 앱은 그 값을 들고 다니면 된다.
포트폴리오는 기본적으로 저장하지 않는다 — 앱이 매 요청에 넣어주는 것을 전제로 한다
(`ChatService(persist_portfolio=True)` 로 바꿀 수 있다).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from multiturn.session import MultiTurnSession, TurnResponse
from multiturn.state import ConversationState
from multiturn.store import (
    ConversationMeta,
    ConversationRecord,
    ConversationStore,
    StoredMessage,
    make_title,
    new_conversation_id,
    now_iso,
)


@dataclass(frozen=True)
class ChatTurn:
    """앱에 돌려줄 한 턴의 결과."""

    conversation_id: str
    answer: str
    status: str                      # generated | direct | clarify | portfolio_required
    seq: int                         # 이 답변 메시지의 순번
    retrieval_query: str
    route: str
    is_followup: bool
    needs_portfolio: bool
    profile: Optional[str]
    summary_updated: bool
    is_new_conversation: bool


STATE_VERSION = 1


@dataclass(frozen=True)
class StatelessTurn:
    """무상태 호출의 결과. 앱은 messages 를 내역에 추가하고 state 를 통째로 교체한다."""

    answer: str
    status: str
    state: dict[str, Any]            # 갱신된 문맥 — 다음 요청에 그대로 넣음
    messages: list[dict[str, Any]]   # 이번 턴의 user·assistant 메시지 (assistant 는 meta 포함)
    suggested_title: Optional[str]   # 새 대화일 때만 — 첫 질문 앞 40자
    retrieval_query: str
    route: str
    is_followup: bool
    needs_portfolio: bool
    profile: Optional[str]
    summary_updated: bool


def state_to_wire(state: ConversationState) -> dict[str, Any]:
    """앱에 돌려줄 문맥. 포트폴리오는 뺀다 — 앱이 매 요청에 넣어주는 값이고 저장 대상이 아니다."""
    d = state.to_dict()
    return {"version": STATE_VERSION, "messages": d["messages"], "conversation_summary": d["conversation_summary"]}


def state_from_wire(payload: Optional[dict[str, Any]], portfolio: Optional[dict[str, Any]]) -> ConversationState:
    """앱이 보낸 문맥을 검증해 복원한다. 형식이 틀리면 ValueError (API 에서 422)."""
    if payload is None or payload == {}:
        return ConversationState(portfolio_context=portfolio)
    if not isinstance(payload, dict):
        raise ValueError("state 는 객체여야 합니다.")
    ver = payload.get("version", STATE_VERSION)
    if ver != STATE_VERSION:
        raise ValueError(f"state.version {ver!r} 은 지원하지 않습니다 (현재 {STATE_VERSION}).")
    try:
        state = ConversationState.from_dict({
            "messages": payload.get("messages", []),
            "conversation_summary": payload.get("conversation_summary", ""),
            "portfolio_context": None,
        })
    except (TypeError, ValueError) as e:
        raise ValueError(f"state 형식 오류: {e}") from e
    state.set_portfolio_context(portfolio)
    return state


class ChatService:
    def __init__(
        self,
        *,
        store: Optional[ConversationStore] = None,
        pipeline=None,
        orchestrator=None,
        summarizer=None,
        summarize: bool = True,
        recent_message_limit: int = 6,
        replies: Optional[dict[str, str]] = None,
        general_prompt_preset: Optional[str] = None,
        persist_portfolio: bool = False,
        verbose: bool = False,
    ) -> None:
        # store=None 이면 무상태 전용 — ask_stateless 만 쓸 수 있다
        self.store: Optional[ConversationStore] = store
        self.recent_message_limit = recent_message_limit
        self.replies = replies
        self.persist_portfolio = persist_portfolio

        if orchestrator is None:
            from core.pipeline import FinalPipeline
            from multiturn.generation_adapter import MultiTurnGeneratorAdapter
            from multiturn.orchestrator import MultiTurnOrchestrator
            from multiturn.query_rewriter import QueryRewriter
            from multiturn.rag_adapter import MultiTurnRAGAdapter

            if pipeline is None:
                pipeline = FinalPipeline(verbose=verbose)
            gen_kwargs = {"general_prompt_preset": general_prompt_preset} if general_prompt_preset else {}
            orchestrator = MultiTurnOrchestrator(
                QueryRewriter(recent_message_limit=recent_message_limit),
                MultiTurnRAGAdapter(pipeline),
                MultiTurnGeneratorAdapter(**gen_kwargs),
            )
        self.orchestrator = orchestrator

        if summarizer is None and summarize:
            from multiturn.conversation_summarizer import ConversationSummarizer
            summarizer = ConversationSummarizer()
        self.summarizer = summarizer

    # ── 조회 ──────────────────────────────────────────────────────────────

    def _require_store(self) -> ConversationStore:
        if self.store is None:
            raise RuntimeError("저장소가 없습니다 — 무상태 서비스는 ask_stateless() 만 지원합니다.")
        return self.store

    # ── 공통 — 한 턴 실행 ─────────────────────────────────────────────────

    def run_turn(self, state: ConversationState, question: str, *, user_id: Optional[str] = None,
                 thread_id: Optional[str] = None) -> tuple[TurnResponse, dict[str, Any]]:
        """state 위에서 한 턴을 돌리고 (응답, assistant 메시지 meta) 를 돌려준다. state 는 제자리 갱신."""
        session = MultiTurnSession(
            self.orchestrator, summarizer=self.summarizer, state=state,
            recent_message_limit=self.recent_message_limit, replies=self.replies,
            thread_id=thread_id, user_id=user_id)
        resp = session.ask(question)
        meta = {
            "status": resp.status, "route": resp.rewrite.route,
            "is_followup": resp.rewrite.is_followup, "needs_portfolio": resp.rewrite.needs_portfolio,
            "retrieval_query": resp.rewrite.retrieval_query, "profile": resp.profile,
            "context_chunk_ids": list(resp.rag.reranked_ids[:3]) if resp.rag else [],
            # 검색 진단(한 문서 독식·미국 청크 수·리랭커 실패 등) — 운영 로그로 품질을 본다
            **{k: v for k, v in session.last_diagnostics.items()
               if k in ("top3_distinct_docs", "us_in_top3", "us_in_candidates", "rerank_failed")},
        }
        return resp, meta

    def ask_stateless(self, user_id: str, question: str, *, state: Optional[dict[str, Any]] = None,
                      portfolio: Optional[dict[str, Any]] = None,
                      conversation_id: Optional[str] = None) -> StatelessTurn:
        """DB 없이 한 턴. conversation_id 는 저장에 안 쓰고 추적(LangSmith thread) 묶음에만 쓴다."""
        if not question.strip():
            raise ValueError("question 은 비어 있을 수 없습니다.")
        st = state_from_wire(state, portfolio)          # 형식 검증 먼저 — 틀리면 ValueError
        before = state_to_wire(st)
        is_new = not before["messages"] and not before["conversation_summary"]
        resp, meta = self.run_turn(st, question, user_id=user_id, thread_id=conversation_id)
        return StatelessTurn(
            answer=resp.answer, status=resp.status, state=state_to_wire(st),
            messages=[{"role": "user", "content": question},
                      {"role": "assistant", "content": resp.answer, "meta": meta}],
            suggested_title=make_title(question) if is_new else None,
            retrieval_query=resp.rewrite.retrieval_query, route=resp.rewrite.route,
            is_followup=resp.rewrite.is_followup, needs_portfolio=resp.rewrite.needs_portfolio,
            profile=resp.profile, summary_updated=resp.summary_updated)

    # ── 저장소 포함 (로컬 CLI·테스트) ────────────────────────────────────

    def list_conversations(self, user_id: str, *, limit: int = 50) -> list[ConversationMeta]:
        return self._require_store().list_conversations(user_id, limit=limit)

    def latest_conversation_id(self, user_id: str) -> Optional[str]:
        convs = self._require_store().list_conversations(user_id, limit=1)
        return convs[0].conversation_id if convs else None

    def history(self, user_id: str, conversation_id: str, *,
                limit: Optional[int] = None, before_seq: Optional[int] = None) -> list[StoredMessage]:
        """화면에 뿌릴 전체 대화 내역. 요약 압축과 무관하게 주고받은 전부가 남아 있다."""
        return self._require_store().messages(user_id, conversation_id, limit=limit, before_seq=before_seq)

    # ── 대화 ──────────────────────────────────────────────────────────────

    def ask(self, user_id: str, question: str, *, conversation_id: Optional[str] = None,
            portfolio: Optional[dict[str, Any]] = None) -> ChatTurn:
        if not question.strip():
            raise ValueError("question 은 비어 있을 수 없습니다.")

        store = self._require_store()
        record = None
        if conversation_id:
            record = store.load(user_id, conversation_id)
            if record is None:
                raise KeyError(f"대화를 찾을 수 없습니다: {conversation_id}")

        is_new = record is None
        if is_new:
            conversation_id = new_conversation_id()
            created_at = now_iso()
            title = make_title(question)
            state = ConversationState(portfolio_context=portfolio)
            n_messages = 0
        else:
            conversation_id = record.meta.conversation_id
            created_at = record.meta.created_at
            title = record.meta.title or make_title(question)
            state = ConversationState.from_dict(record.state)
            n_messages = record.meta.n_messages
            if portfolio is not None:                    # 앱이 매 요청에 넣어주는 최신 포트폴리오
                state.set_portfolio_context(portfolio)

        resp, msg_meta = self.run_turn(state, question, user_id=user_id, thread_id=conversation_id)

        meta = ConversationMeta(conversation_id=conversation_id, user_id=user_id, title=title,
                                created_at=created_at, updated_at=now_iso(), n_messages=n_messages)
        store.save(ConversationRecord(meta=meta, state=self._state_payload(state)))
        seq = store.append_messages(user_id, conversation_id, [
            ("user", question, {}),
            ("assistant", resp.answer, msg_meta),
        ])

        return ChatTurn(
            conversation_id=conversation_id, answer=resp.answer, status=resp.status, seq=seq,
            retrieval_query=resp.rewrite.retrieval_query, route=resp.rewrite.route,
            is_followup=resp.rewrite.is_followup, needs_portfolio=resp.rewrite.needs_portfolio,
            profile=resp.profile, summary_updated=resp.summary_updated, is_new_conversation=is_new)

    # ── 관리 ──────────────────────────────────────────────────────────────

    def rename(self, user_id: str, conversation_id: str, title: str) -> bool:
        store = self._require_store()
        record = store.load(user_id, conversation_id)
        if record is None:
            return False
        m = record.meta
        store.save(ConversationRecord(
            meta=ConversationMeta(conversation_id=m.conversation_id, user_id=m.user_id, title=title.strip(),
                                  created_at=m.created_at, updated_at=now_iso(), n_messages=m.n_messages),
            state=record.state))
        return True

    def delete(self, user_id: str, conversation_id: str) -> bool:
        return self._require_store().delete(user_id, conversation_id)

    def _state_payload(self, state: ConversationState) -> dict[str, Any]:
        payload = state.to_dict()
        if not self.persist_portfolio:
            payload["portfolio_context"] = None          # 개인 자산 정보는 저장하지 않는다
        return payload
