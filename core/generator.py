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
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from config import MAX_TOKENS, MODEL, TEMPERATURE, TOP_P


# =============================================================================
# SYSTEM 프롬프트 — 버전별로 관리한다
#
#   프롬프트는 이 실험의 최대 변수다. 상수 하나를 직접 고쳐가며 실험하면 결과 파일에
#   "어떤 프롬프트로 낸 점수인지"가 남지 않아, 며칠 뒤 results 를 열었을 때 설정과
#   점수의 대응이 끊긴다. 버전을 키로 두고 실행 인자로 고르게 한다.
#
#   v1  현행 기준선 — "[자료]에 명시된 내용만" 강제 + 금융 도메인 안전 가드
# =============================================================================
PROMPTS: dict[str, str] = {
    "v1": """당신은 사용자의 금융상품·포트폴리오를 안내하는 정보 도우미입니다.
아래 [금융상품 자료]에 명시된 내용만을 근거로 사용자의 질문에 답변하세요.

# 규칙
- [금융상품 자료]에 있는 내용만 사용해 답변할 것.
- 관련 정보가 자료에 없으면 정확히 "제공된 자료에서는 해당 내용을 확인할 수 없습니다." 라고만 답변할 것.
- 숫자(수익률·비용·기간·한도)는 자료에 적힌 그대로 인용. 단위 변환·반올림·추측 금지.
- 자료에 없는 상품명·수치·운용사를 지어내지 말 것.
- 확정 수익 약속·단정적 투자 권유 금지. 답변은 정보 제공이며 투자 권유가 아님.
- 정중하고 명료한 존댓말 사용.
""",

    # v2 — 답변 스펙(scripts/answer_spec.py)을 지시문으로 옮긴 것.
    #
    #   튜닝 결과물이 아니라 **실험 조건 정비**다. v1 에는 길이·형식·판정 우선·위험 고지·
    #   권유 아님 명시 지시가 아예 없어서, 그 상태로 모델을 비교하면 "시키지 않아도 알아서
    #   우리 스펙대로 쓰는 모델"을 고르게 된다. 실서비스에서 필요한 것은 "지시하면 지키는
    #   모델"이므로 먼저 지시를 넣는다.
    #
    #   실측 준수율(v1, dev 107) — 권유 아님 0.009 / 위험 고지 0.120 / 길이 0.477
    #
    #   길이 제약과 필수 문구가 충돌하지 않도록 두 가지를 했다.
    #     · 위험 고지·권유 아님을 **한 문장으로 합치도록** 예시를 제시
    #     · 목록 금지를 명시해 불필요한 줄바꿈·번호로 길이가 늘어나는 것을 막음
    "v2": """당신은 사용자의 금융상품·포트폴리오를 안내하는 정보 도우미입니다.
아래 [금융상품 자료]에 명시된 내용만을 근거로 답변하세요.

# 내용 규칙
- [금융상품 자료]에 있는 내용만 사용할 것. 자료에 없는 상품명·수치·운용사를 지어내지 말 것.
- 숫자는 자료에 적힌 그대로 쓰고, **단위와 적용 기준을 함께** 밝힐 것.
  (예: "20%" 가 아니라 "연 20%(지방소득세 포함)", "1,800만원" 이 아니라 "연 1,800만원")
- 관련 정보가 자료에 없으면, **무엇을 확인할 수 없는지 밝히며** 답변할 수 없다고 안내할 것.

# 형식 규칙
- **150~250자**로 작성할 것. 짧으면 불친절하고 길면 읽지 않는다.
- **줄바꿈·번호·불릿을 쓰지 말고** 이어지는 문장으로 설명할 것.
- 예/아니오로 답할 수 있는 질문이면 **첫 문장을 "네" 또는 "아니요"로 시작**할 것.
- 둘 이상을 비교하는 질문이면 **양쪽을 모두** 설명할 것.
- 정중하고 명료한 존댓말을 사용할 것.

# 반드시 지킬 것
- 원금손실이 가능한 상품(펀드·ETF·주식·채권 등)을 언급할 때는 **손실 가능성을 함께** 안내할 것.
- 답변 말미에 **투자 권유가 아니라 정보 제공임을 한 문장으로** 밝힐 것.
  위 두 가지는 가능하면 한 문장으로 합쳐 분량을 아낄 것.
  (예: "원금 손실이 발생할 수 있으므로 유의하시기 바라며, 본 안내는 투자 권유가 아닌 정보 제공입니다.")
- 확정 수익 약속·단정적 투자 권유는 금지.

# 절대 하지 말 것
- "제공된 자료에 따르면", "자료에는" 처럼 **내부 사정을 드러내는 표현**
- "[문서 2]" 처럼 **자료 번호를 인용**하는 것
""",

    # v3 — v2 의 실측 약점 3가지를 겨냥해 다듬음. 여기서부터가 진짜 프롬프트 튜닝임.
    #
    #   1) 길이 — 모델마다 **반대 방향으로** 틀렸다.
    #        gemini-3.5-flash-lite  150자 미만 30건 (너무 짧음)
    #        gpt-4.1-nano           250자 초과 25건 (너무 김)
    #      한쪽만 강조하면 다른 쪽이 나빠지므로 상한·하한을 각각 처방과 함께 적었다.
    #   2) 위험 고지 — gpt-4.1-nano 0.789. 누락 15건의 평균 길이가 229자로,
    #      **길게 쓰면서도 빠뜨린다.** 분량 문제가 아니라 잊는 문제이므로,
    #      마지막에 붙일 **마무리 문장 형태를 고정**해 누락 자체를 막는다.
    #   3) 숫자 기준 — 전 모델 0.78~0.89. 무엇을 '기준'으로 봐야 하는지 예시를 늘렸다.
    #
    #   마무리 문장 고정은 세 가지를 동시에 해결한다 — 위험 고지 누락 방지,
    #   면책 문구 보장, 그리고 짧은 답변의 분량 보충(40~50자).
    "v3": """당신은 사용자의 금융상품·포트폴리오를 안내하는 정보 도우미입니다.
아래 [금융상품 자료]에 명시된 내용만을 근거로 답변하세요.

# 답변 작성 순서
1) 질문에 대한 답을 [금융상품 자료]에서 찾아 설명합니다.
2) 마지막에 위험 고지와 면책을 한 문장으로 덧붙입니다.
3) 전체 길이가 150~250자인지 확인합니다.

# 내용 규칙
- [금융상품 자료]에 있는 내용만 사용할 것. 자료에 없는 상품명·수치·운용사를 지어내지 말 것.
- 숫자는 자료에 적힌 그대로 쓰고, **적용 기준을 함께** 밝힐 것.
  기준이란 기간·대상·범위를 말합니다.
  (예: "20%" → "연 20%(지방소득세 포함)", "1,800만원" → "연 1,800만원", "5년" → "가입 후 5년")
- 관련 정보가 자료에 없으면, **무엇을 확인할 수 없는지 밝히며** 답변할 수 없다고 안내할 것.

# 형식 규칙
- **150자 이상 250자 이하.** 이 범위를 지키는 것이 중요합니다.
  · 150자에 못 미치면 → 근거·조건·예외를 한 문장 더 덧붙여 채울 것
  · 250자를 넘으면 → 질문과 직접 관련 없는 배경 설명을 덜어낼 것
- **줄바꿈·번호·불릿을 쓰지 말고** 이어지는 문장으로 설명할 것.
- 예/아니오로 답할 수 있는 질문이면 **첫 문장을 "네" 또는 "아니요"로 시작**할 것.
- 둘 이상을 비교하는 질문이면 **양쪽을 모두** 설명할 것.
- 정중하고 명료한 존댓말을 사용할 것.

# 마무리 문장 (빠뜨리지 말 것)
원금손실이 가능한 상품(펀드·ETF·ETN·주식·채권·연금저축펀드 등)을 언급했다면
아래 형태로 마무리합니다.

  "원금 손실이 발생할 수 있으므로 유의하시기 바라며, 본 안내는 투자 권유가 아닌 정보 제공입니다."

원금이 보장되는 상품만 다룬 답변이라면 위험 고지 없이 면책만 붙입니다.

  "본 안내는 투자 권유가 아닌 정보 제공입니다."

# 절대 하지 말 것
- "제공된 자료에 따르면", "자료에는" 처럼 **내부 사정을 드러내는 표현**
- "[문서 2]" 처럼 **자료 번호를 인용**하는 것
- 확정 수익 약속·단정적 투자 권유
""",
}

DEFAULT_PROMPT_VERSION = "v1"

# 기존 코드 호환용 (core/pipeline.py, scripts/evaluate.py 가 참조)
SYSTEM_PROMPT = PROMPTS[DEFAULT_PROMPT_VERSION]


def get_prompt(version: str) -> str:
    if version not in PROMPTS:
        raise KeyError(f"없는 프롬프트 버전: {version} (가능: {sorted(PROMPTS)})")
    return PROMPTS[version]


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
    # finish_reason 이 "length" 면 MAX_TOKENS 에서 잘린 것이다. 잘린 답변은 채점에서
    # 불리해지므로 점수를 해석하기 전에 이 값부터 확인해야 한다.
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


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
# Anthropic — 프로바이더 간 모델 비교용
#
#   결과 형식(GenerationResult)은 OpenAI 와 **동일하게 맞춘다.** 그래야 채점기·분석
#   코드를 그대로 쓸 수 있다. 특히 stop_reason 을 finish_reason 규격으로 변환하지
#   않으면 "답변이 잘렸는지" 감지가 조용히 깨진다.
# =============================================================================
_anthropic_client = None


def _provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    return "openai"


# =============================================================================
# Google Gemini
#
#   ⚠ 실측 주의 — gemini-3.5-flash 는 **내부 추론(thinking) 토큰이 출력 예산을
#     잠식한다.** 사소한 질문 하나에 thinking 312토큰을 쓰고, max_output_tokens=300
#     에서는 본문을 10토큰만 내고 MAX_TOKENS 로 잘렸다.
#     그래서 이 계열만 상한을 넉넉히 준다. 다른 모델은 130토큰 안팎에서 끝나 상한이
#     구속력이 없으므로, 상한을 올려도 비교가 흔들리지 않는다.
#   ⚠ gemini-3.5-flash-lite 는 thinking_config 자체를 거부한다(400). 추론을 강제로
#     끄는 대신 **기본 설정 그대로** 두고 상한만 올리는 쪽을 택했다 — 배포하면 쓰게 될
#     동작이 곧 기본 설정이므로 그게 대표성 있는 비교다.
# =============================================================================
_google_client = None
# thinking 토큰까지 감안한 상한. 실측 — gemini-3.5-flash 는 2,500자 컨텍스트 한 건에
# thinking 1,740토큰을 쓰고 본문은 109토큰만 낸다. 1500 으로는 본문이 잘려 채점이
# 불공정해지므로 4000 으로 잡았다. 잘림 없이 재는 것이 우선이고, 느린 것과 비싼 것은
# latency·output_tokens 지표로 드러나게 둔다.
GOOGLE_MAX_TOKENS = 4000


def _get_google():
    global _google_client
    if _google_client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다.")
        _google_client = genai.Client(api_key=api_key)
    return _google_client


def _generate_google(question: str, contexts: list[str], prompt_version: str,
                     model: str, temperature: float, max_tokens: int,
                     top_p: float) -> GenerationResult:
    from google.genai import types

    client = _get_google()
    user = USER_TEMPLATE.format(
        retrieved_context=format_contexts(contexts).strip(),
        user_question=question.strip(),
    )
    start = time.time()
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=get_prompt(prompt_version),
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max(max_tokens, GOOGLE_MAX_TOKENS),
        ),
    )
    latency = time.time() - start
    um = resp.usage_metadata
    thoughts = getattr(um, "thoughts_token_count", None) or 0
    fr = str(getattr(resp.candidates[0], "finish_reason", "")).upper()
    return GenerationResult(
        answer=resp.text or "",          # thinking 중 잘리면 text 가 None 이 된다
        latency=latency,
        input_tokens=um.prompt_token_count,
        # thinking 도 과금되는 출력 토큰이므로 비용 비교를 위해 합산한다
        output_tokens=(um.candidates_token_count or 0) + thoughts,
        finish_reason="length" if "MAX_TOKENS" in fr else "stop",
    )


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY 환경변수가 필요합니다.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _generate_anthropic(question: str, contexts: list[str], prompt_version: str,
                        model: str, temperature: float, max_tokens: int,
                        top_p: float) -> GenerationResult:
    client = _get_anthropic()
    user = USER_TEMPLATE.format(
        retrieved_context=format_contexts(contexts).strip(),
        user_question=question.strip(),
    )
    # ⚠ Anthropic 은 temperature 와 top_p 를 **동시에 지정하면 400** 을 낸다 (실측).
    #   우리 파라미터 튜닝 축은 temperature 이므로 그쪽만 보낸다. 이 차이는 프로바이더
    #   간 비교 시 "완전히 동일한 설정"이 불가능하다는 뜻이므로 결과 해석에 명시할 것.
    start = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,          # Anthropic 은 필수 파라미터
        temperature=temperature,
        system=get_prompt(prompt_version),   # system 은 messages 가 아니라 별도 인자
        messages=[{"role": "user", "content": user}],
    )
    latency = time.time() - start
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return GenerationResult(
        answer=text,
        latency=latency,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        # "max_tokens" → "length" 로 맞춰 truncated 판정을 공용화
        finish_reason="length" if resp.stop_reason == "max_tokens" else "stop",
    )


# =============================================================================
# Public API
# =============================================================================
def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[문서 {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))


def _token_limit_param(model: str) -> str:
    """출력 길이 제한 파라미터 이름은 모델 계열마다 다르다.

    실측 — gpt-5.4-mini/nano 는 `max_tokens` 를 400 으로 거부하고
    `max_completion_tokens` 만 받는다. gpt-4.1 계열은 `max_tokens` 를 받는다.
    모델을 바꿔가며 비교하는 실험이라 호출부에서 자동으로 갈라줘야 한다.
    """
    return "max_completion_tokens" if re.match(r"^(gpt-5|o[1-9])", model) else "max_tokens"


def build_messages(question: str, contexts: list[str],
                   prompt_version: str = DEFAULT_PROMPT_VERSION) -> list[dict]:
    """OpenAI Chat API messages 리스트 생성."""
    return [
        {"role": "system", "content": get_prompt(prompt_version)},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                retrieved_context=format_contexts(contexts).strip(),
                user_question=question.strip(),
            ),
        },
    ]


MAX_RETRY = 6


def _with_retry(fn, *args):
    """레이트 리밋(429)에 지수 백오프로 재시도.

    프로바이더마다 제한이 크게 다르다 (실측):
      OpenAI gpt-4.1      TPM 30,000
      Gemini flash-lite   분당 15회      ← 재시도 없이는 107문항을 못 돌린다
      Gemini flash        **하루 20회**   ← 무료 등급에서는 실험 자체가 불가
    한 문항이 실패해 전체가 죽는 것을 막는 것이 목적이다.
    """
    for attempt in range(MAX_RETRY):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if attempt == MAX_RETRY - 1 or not any(
                    k in msg for k in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "overloaded")):
                raise
            # 서버가 알려준 대기 시간이 있으면 존중한다
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay': '(\d+)s", msg)
            wait = float(m.group(1)) + 1 if m else min(60, 5 * (2 ** attempt))
            time.sleep(wait)
    raise RuntimeError("unreachable")


def generate(question: str, contexts: list[str], *,
             prompt_version: str = DEFAULT_PROMPT_VERSION,
             model: str | None = None,
             temperature: float | None = None,
             max_tokens: int | None = None,
             top_p: float | None = None) -> GenerationResult:
    """답변 + 메타데이터(latency, tokens) 반환.

    실험 파라미터는 인자로 받는다. config.py 를 고쳐가며 스윕하면 결과 파일과 설정의
    대응이 끊기므로, 스윕 축은 전부 호출 인자로 넘기고 결과에 함께 기록한다.
    인자를 생략하면 config.py 기본값을 쓴다.
    """
    m = model or MODEL
    temp = TEMPERATURE if temperature is None else temperature
    tp = TOP_P if top_p is None else top_p
    mt = max_tokens or MAX_TOKENS

    prov = _provider(m)
    if prov == "anthropic":
        return _with_retry(_generate_anthropic, question, contexts, prompt_version, m, temp, mt, tp)
    if prov == "google":
        return _with_retry(_generate_google, question, contexts, prompt_version, m, temp, mt, tp)

    messages = build_messages(question, contexts, prompt_version)
    client = _get_client()

    kwargs = {"temperature": temp, "top_p": tp, _token_limit_param(m): mt}

    start = time.time()
    resp = _with_retry(lambda: client.chat.completions.create(model=m, messages=messages, **kwargs))
    latency = time.time() - start

    choice = resp.choices[0]
    return GenerationResult(
        answer=choice.message.content,
        latency=latency,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        finish_reason=choice.finish_reason or "stop",
    )


def generate_answer(question: str, contexts: list[str]) -> str:
    """답변 문자열만 반환하는 간편 함수."""
    return generate(question, contexts).answer
