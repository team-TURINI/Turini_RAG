"""RAG 서비스 HTTP API — 앱 백엔드가 호출한다. **무상태**.

배포를 쪼갠 구조에서 이 프로세스만 무겁다(코퍼스·FAISS·BM25·리랭커). 앱과 포트폴리오 생성은
API 호출만 하므로 가볍고, 그래서 메모리 넉넉한 곳(HF Space)에 이 서비스만 올린다.

HF Space 는 바깥 DB 포트(5432)로 나가는 연결을 막는다. 그래서 RAG 는 DB 를 쓰지 않고,
**대화 저장은 앱이 한다.** 앱은 매 요청에 이전 문맥(`state`)을 포트폴리오처럼 실어 보내고,
응답으로 답변·이번 턴 메시지 2개·갱신된 `state` 를 받아 자기 DB 에 저장한다.

  앱 백엔드 ──HTTPS──> 이 서비스(RAG) ──> OpenAI / Cohere
      │      {question, state, portfolio}         │ 기동 시 내려받음
      └──> 앱 DB (대화·포트폴리오)             └──< HF Dataset (코퍼스·인덱스)

엔드포인트
  GET  /health   기동 완료 여부 — 기동 중이면 503 (깨우기·헬스체크용, 인증 없음)
  POST /chat     질문 → 답변 + 갱신된 state

인증
  RAG_API_KEY 환경변수를 두면 `/chat` 에 `X-API-Key` 헤더를 요구한다.
  (OpenAI·Cohere 비용이 나가는 서비스라 공개 URL 로 열어두면 안 된다. 배포 시 반드시 설정할 것.)

실행
  RAG_API_KEY=... uvicorn serve.api:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

API_KEY = os.getenv("RAG_API_KEY") or ""
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
QUESTION_MAX = 2000      # 앱의 500자 제한과 별도 안전장치

_state: dict[str, Any] = {"service": None, "ready": False, "error": None, "boot_s": None}


def build_default_service():
    """검색 데이터가 없으면 HF Dataset 에서 내려받고, 파이프라인을 서빙 모드로 올린다."""
    from serve.bootstrap import ensure_data
    ensure_data()

    from core.pipeline import FinalPipeline
    from multiturn.chat_service import ChatService
    pipeline = FinalPipeline(verbose=True, serve_mode=True)     # 429 에 죽지 않는 모드
    return ChatService(store=None, pipeline=pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 기동에 40초 안팎 걸린다. 그동안 /health 는 503 을 돌려주므로 앱이 기다릴 수 있다.
    t0 = time.time()
    try:
        _state["service"] = app.state.service_factory()
        _state["ready"] = True
    except Exception as exc:                                    # 기동 실패해도 /health 는 답해야 한다
        _state["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("서비스 기동 실패")
    _state["boot_s"] = round(time.time() - t0, 1)
    yield


app = FastAPI(title="Turini RAG API", lifespan=lifespan)
app.state.service_factory = build_default_service
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def auth(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key 가 없거나 올바르지 않습니다.")


def service():
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "기동 중입니다. 잠시 뒤 다시 시도하세요.",
                            headers={"Retry-After": "10"})
    return _state["service"]


# ── 스키마 ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, description="앱 사용자 id — 저장엔 안 쓰고 추적 메타데이터로만")
    question: str = Field(min_length=1, max_length=QUESTION_MAX)
    conversation_id: Optional[str] = Field(default=None, description="앱이 발급한 대화 id — 추적 묶음용")
    state: Optional[dict[str, Any]] = Field(
        default=None, description="직전 응답의 state 를 그대로. 새 대화면 생략")
    portfolio: Optional[dict[str, Any]] = Field(default=None, description="앱이 매 요청에 넣어준다(저장 안 함)")


class ChatResponse(BaseModel):
    answer: str
    status: str = Field(description="generated | direct | clarify | portfolio_required")
    state: dict[str, Any] = Field(description="갱신된 문맥 — 앱이 통째로 저장했다가 다음 요청에 그대로 넣음")
    messages: list[dict[str, Any]] = Field(description="이번 턴의 user·assistant 메시지. 앱 내역 테이블에 추가")
    suggested_title: Optional[str] = Field(description="새 대화일 때만 — 첫 질문 앞 40자")
    needs_portfolio: bool
    is_followup: bool
    retrieval_query: str
    route: str
    profile: Optional[str]
    degraded: bool = Field(description="리랭킹 없이 답한 경우 true — 품질이 평소보다 낮다")
    latency_s: float


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """인증 없이 열어 둔다 — 깨우기 ping·플랫폼 헬스체크가 쓴다. 기동 중·실패면 503."""
    out: dict[str, Any] = {"ready": _state["ready"], "boot_s": _state["boot_s"], "error": _state["error"]}
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=out, headers={"Retry-After": "10"})
    svc = _state["service"]
    pipe = getattr(svc.orchestrator.rag_adapter, "pipeline", None)
    out["corpus_chunks"] = len(getattr(pipe, "corpus", {}) or {})
    return out


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
def chat(req: ChatRequest, svc=Depends(service)) -> ChatResponse:
    t0 = time.time()
    try:
        turn = svc.ask_stateless(req.user_id, req.question, state=req.state,
                                 portfolio=req.portfolio, conversation_id=req.conversation_id)
    except ValueError as exc:                                   # state 형식 오류·빈 질문
        raise HTTPException(status_code=422, detail=str(exc))
    pipe = getattr(svc.orchestrator.rag_adapter, "pipeline", None)
    degraded = bool(getattr(getattr(pipe, "reranker", None), "last_failed", False))
    return ChatResponse(
        answer=turn.answer, status=turn.status, state=turn.state, messages=turn.messages,
        suggested_title=turn.suggested_title, needs_portfolio=turn.needs_portfolio,
        is_followup=turn.is_followup, retrieval_query=turn.retrieval_query, route=turn.route,
        profile=turn.profile, degraded=degraded, latency_s=round(time.time() - t0, 2))
