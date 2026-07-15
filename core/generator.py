"""제너레이터 — 검색된 contexts + 질문 → 답변.

RAG 베이스라인의 생성 구성요소. 리트리버가 가져온 문서를 근거로 LLM 이 답변을
만든다. 의도분류·쿼리재작성·멀티턴요약 같은 부가 단계는 베이스라인에서 제외.

------------------------------------------------------------
실험 포인트 (config.py 값 변경만으로 가능):
  - MODEL        : 답변 LLM
  - TEMPERATURE  : 0.0~0.3 (사실성↑) vs 0.7 (다양성↑)
  - MAX_TOKENS   : 답변 길이 상한
  - TOP_P        : 확률 컷오프
프롬프트를 실험하려면 SYSTEM_PROMPT 를 버전 분리 후 토글.
------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI

from config import MAX_TOKENS, MODEL, TEMPERATURE, TOP_P


# =============================================================================
# SYSTEM 프롬프트
#   - "[자료]에 명시된 내용만" 강제 → hallucination 차단
#   - 금융 도메인 안전 가드 (투자 권유·확정 수익 약속 금지)
# =============================================================================
SYSTEM_PROMPT = """당신은 사용자의 금융상품·포트폴리오를 안내하는 정보 도우미입니다.
아래 [금융상품 자료]에 명시된 내용만을 근거로 사용자의 질문에 답변하세요.

# 규칙
- [금융상품 자료]에 있는 내용만 사용해 답변할 것.
- 관련 정보가 자료에 없으면 정확히 "제공된 자료에서는 해당 내용을 확인할 수 없습니다." 라고만 답변할 것.
- 숫자(수익률·비용·기간·한도)는 자료에 적힌 그대로 인용. 단위 변환·반올림·추측 금지.
- 자료에 없는 상품명·수치·운용사를 지어내지 말 것.
- 확정 수익 약속·단정적 투자 권유 금지. 답변은 정보 제공이며 투자 권유가 아님.
- 정중하고 명료한 존댓말 사용.
"""


# USER 메시지 템플릿
USER_TEMPLATE = """# 금융상품 자료
{retrieved_context}

# 사용자 질문
{user_question}"""


# =============================================================================
# 응답 구조체
# =============================================================================
@dataclass
class GenerationResult:
    answer: str
    latency: float
    input_tokens: int
    output_tokens: int


# =============================================================================
# OpenAI 클라이언트 지연 초기화
# =============================================================================
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 필요합니다. .env 파일을 확인하세요."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# =============================================================================
# Public API
# =============================================================================
def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[문서 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    """OpenAI Chat API messages 리스트 생성."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                retrieved_context=format_contexts(contexts).strip(),
                user_question=question.strip(),
            ),
        },
    ]


def generate(question: str, contexts: list[str]) -> GenerationResult:
    """답변 + 메타데이터(latency, tokens) 반환."""
    messages = build_messages(question, contexts)
    client = _get_client()

    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    latency = time.time() - start

    return GenerationResult(
        answer=resp.choices[0].message.content,
        latency=latency,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )


def generate_answer(question: str, contexts: list[str]) -> str:
    """답변 문자열만 반환하는 간편 함수."""
    return generate(question, contexts).answer
