"""관찰(LangSmith) 공통 유틸 — 파이프라인 동작은 바꾸지 않는다.

## 구조

한 대화 = LangSmith Thread(`thread_id` = 우리 `conversation_id`),
한 질문/답변 = 하나의 trace, 그 안에 단계가 child run 으로 들어간다.

    Turn (parent)            route·status·진단 지표를 metadata 로
    ├─ query_rewrite         (multiturn/query_rewriter.py — 이미 있음, 자동 중첩)
    ├─ retrieval             단계별 순위 이동 + 국가 필터 결과
    ├─ rerank                점수 · 실패 여부
    ├─ generation            프롬프트(포트폴리오 마스킹) · 토큰
    └─ summary_update        (multiturn/conversation_summarizer.py — 이미 있음)

## 꺼져 있을 때

`LANGSMITH_TRACING` 이 false 거나 `LANGSMITH_API_KEY` 가 없으면 **아무것도 하지 않는다**
(no-op 객체를 돌려주므로 호출부 코드는 그대로 두면 된다). 배치 실험(157문항 등)은 기본이 꺼짐이라
무료 한도를 쓰지 않는다 — 켜려면 환경변수로 켠다.

## 개인정보

포트폴리오(보유 자산·금액·진단)는 기본적으로 **값을 가리고 필드 구조만** 기록한다.
디버깅 때문에 원본이 필요하면 `TRACE_PORTFOLIO=raw` 로 켠다(실사용자 데이터에는 쓰지 말 것).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

import config as _cfg

log = logging.getLogger(__name__)

MASK = "***"
# 값을 가리지 않아도 되는 키 — 구조 파악에 필요하고 개인 식별 정보가 아니다
PORTFOLIO_KEEP_KEYS = {"input_mode", "level_type", "risk_type"}


def enabled() -> bool:
    return bool(getattr(_cfg, "LANGSMITH_TRACING", False) and getattr(_cfg, "LANGSMITH_API_KEY", None))


def portfolio_raw() -> bool:
    return os.getenv("TRACE_PORTFOLIO", "").strip().lower() == "raw"


def mask_portfolio(value: Any) -> Any:
    """키 구조는 남기고 값만 가린다 — '어떤 필드가 들어왔나'는 보되 자산 내역은 안 보이게."""
    if portfolio_raw():
        return value
    if isinstance(value, dict):
        return {k: (v if k in PORTFOLIO_KEEP_KEYS else mask_portfolio(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_portfolio(v) for v in value]
    if isinstance(value, str) and not value.strip():
        return value
    return MASK


def mask_prompt(text: str, portfolio: Optional[dict]) -> str:
    """생성 프롬프트에서 포트폴리오 블록만 마스킹본으로 바꾼다 (자료·질문·지시는 그대로 본다)."""
    if portfolio is None or portfolio_raw():
        return text
    import json

    marker = "# 사용자 포트폴리오"
    i = text.find(marker)
    if i < 0:
        return text
    j = text.find("# 사용자 질문", i)
    masked = json.dumps(mask_portfolio(portfolio), ensure_ascii=False, indent=2, sort_keys=True)
    tail = text[j:] if j > 0 else ""
    return f"{text[:i]}{marker}\n{masked}\n\n{tail}"


# 턴 안에서 깊은 단계(검색 등)가 남긴 진단 지표를 턴 단위로 모은다.
# 추적이 꺼져 있어도 동작한다 — 같은 값을 대화 DB 에도 저장하기 때문.
_turn_diag: ContextVar[Optional[dict]] = ContextVar("turn_diag", default=None)


@contextmanager
def collect() -> Iterator[dict]:
    """한 턴 동안의 진단 지표를 모으는 그릇. 중첩되면 바깥 것을 그대로 쓴다."""
    existing = _turn_diag.get()
    if existing is not None:
        yield existing
        return
    bucket: dict = {}
    token = _turn_diag.set(bucket)
    try:
        yield bucket
    finally:
        _turn_diag.reset(token)


def annotate(md: dict) -> None:
    """현재 턴에 진단 지표를 남긴다 (턴 밖에서 호출하면 아무 일도 하지 않는다)."""
    bucket = _turn_diag.get()
    if bucket is not None:
        bucket.update({k: v for k, v in md.items() if v is not None})


def mask_messages(messages: list) -> list:
    """프롬프트 메시지에서 포트폴리오 블록만 마스킹본으로 교체한다."""
    if portfolio_raw():
        return messages
    return [{**m, "content": _mask_portfolio_block(m.get("content", ""))}
            if isinstance(m.get("content"), str) else m for m in messages]


def _mask_portfolio_block(text: str) -> str:
    """`# 사용자 포트폴리오` **한 줄 제목 다음의 JSON 블록**만 바꾼다.

    접두사로 찾으면 시스템 프롬프트의 `# 사용자 포트폴리오 추가 규칙` 절까지 지워져
    정작 디버깅에 필요한 규칙이 추적에서 사라진다 (테스트로 잡은 실제 버그).
    """
    import json
    import re

    marker, after = "# 사용자 포트폴리오", "# 사용자 질문"
    m = re.search(rf"(?m)^{re.escape(marker)}[ \t]*$", text)
    if m is None:
        return text
    i = m.start()
    j = text.find(after, m.end())
    block = text[m.end():j if j > 0 else len(text)].strip()
    try:
        masked = json.dumps(mask_portfolio(json.loads(block)), ensure_ascii=False, indent=2, sort_keys=True)
    except (ValueError, TypeError):
        masked = MASK                       # 형식이 예상과 다르면 통째로 가린다
    return f"{text[:i]}{marker}\n{masked}\n\n{text[j:] if j > 0 else ''}"


class _NoopSpan:
    """추적이 꺼졌을 때 쓰는 빈 객체 — 호출부가 분기하지 않아도 되게."""

    def end(self, **_: Any) -> None: ...
    def add_metadata(self, _: dict) -> None: ...
    def add_tags(self, _: list) -> None: ...


class _Span:
    def __init__(self, run_tree) -> None:
        self._rt = run_tree

    def end(self, outputs: Optional[dict] = None, **kw: Any) -> None:
        try:
            if outputs is not None:
                self._rt.add_outputs(outputs)
            if kw:
                self._rt.add_metadata(kw)
        except Exception:                      # 추적 실패가 답변을 막으면 안 된다
            log.debug("trace end 실패", exc_info=True)

    def add_metadata(self, md: dict) -> None:
        try:
            self._rt.add_metadata(md)
        except Exception:
            log.debug("trace metadata 실패", exc_info=True)

    def add_tags(self, tags: list) -> None:
        try:
            self._rt.add_tags(tags)
        except Exception:
            log.debug("trace tags 실패", exc_info=True)


@contextmanager
def span(name: str, *, run_type: str = "chain", inputs: Optional[dict] = None,
         metadata: Optional[dict] = None, tags: Optional[list] = None) -> Iterator[Any]:
    """child run 하나. 추적이 꺼져 있거나 LangSmith 가 없으면 no-op 을 돌려준다.

    본문에서 난 예외는 그대로 올린다 — LangSmith 에도 실패로 기록되고 호출부도 알아야 한다.
    """
    cm = None
    if enabled():
        try:
            from langsmith import trace as ls_trace
            cm = ls_trace(name=name, run_type=run_type, inputs=inputs or {},
                          project_name=getattr(_cfg, "LANGSMITH_PROJECT", None),
                          metadata=metadata, tags=tags)
        except Exception:                      # 추적을 못 켜도 작업은 계속한다
            log.debug("trace 시작 실패", exc_info=True)
            cm = None
    if cm is None:
        yield _NoopSpan()
        return
    with cm as rt:
        yield _Span(rt)
