"""답변 스펙 — 우리 도메인이 원하는 답변의 정의와 그 준수 여부 검사.

왜 필요한가 — correctness/groundedness 는 **"틀렸나"** 만 본다. 실측 결과 조건 A·A2
모두 전 문항 만점이라 그 지표로는 프롬프트를 비교할 수 없다. 그런데 같은 만점 안에
63자 한 문장과 599자 번호 목록이 섞여 있고, 위험 고지는 7% 만 포함하고 있다.

**"좋은 답변인가"를 재려면 좋은 답변이 뭔지 먼저 정해야 한다.** 이 파일이 그 정의다.
여기 기준이 바뀌면 프롬프트도 같이 바뀌어야 하므로, 스펙과 검사를 한 파일에 둔다.

------------------------------------------------------------------
스펙 (2026-08-11 확정)

  1. 길이      150~250자 (200자 내외) 서술형
  2. 형식      목록·번호 매기기 금지. 문장으로 설명
  3. 위험 고지  원금손실 가능 상품을 언급하면 위험 문구 포함
  4. 권유 아님  투자 권유가 아니라 정보 제공임을 명시
  5. 수치 표기  숫자에 단위·기준을 붙임 ("20%" 가 아니라 "연 20%(지방소득세 포함)")
  6. 출처      표시하지 않음. 내부 사정("제공된 자료에 따르면", "[문서 2]") 노출 금지
  7. 거부      무엇을 확인할 수 없는지 밝히며 거부

------------------------------------------------------------------
규칙으로 재는 것과 못 재는 것

  규칙으로 판정 — 길이 / 형식 / 권유아님 / 메타노출 / 거부구체성 / 단위누락
  judge 가 필요 — "위험 고지가 **필요한 맥락인가**" 는 문맥 판단이라 규칙으로 한계가 있음.
                 여기서는 위험 상품 언급 여부로 근사하고, 정밀 판정은 judge 에 맡긴다.
"""

from __future__ import annotations

import re
from typing import Dict

# ---- 1. 길이 --------------------------------------------------------------
LEN_MIN, LEN_MAX = 150, 250

# ---- 2. 형식 — 목록 마커 --------------------------------------------------
LIST_MARKER = re.compile(r"(?m)^\s*(?:[-·•*]|\d+\s*[.)])\s")

# ---- 3. 위험 고지 ---------------------------------------------------------
# 원금손실이 가능한 상품이 언급되면 위험 문구가 있어야 한다.
RISKY_PRODUCT = re.compile(
    r"(펀드|ETF|ETN|주식|채권|레버리지|파생|投資|투자상품|실적배당|연금저축펀드)")
RISK_NOTICE = re.compile(
    r"(원금\s*손실|원금이?\s*보장되지|손실이?\s*발생|투자\s*위험|손실\s*위험|"
    r"가격\s*변동|원금\s*보장이?\s*되지)")

# ---- 4. 투자 권유 아님 ----------------------------------------------------
NO_SOLICIT = re.compile(
    r"(투자\s*권유가?\s*아니|권유가\s*아닌|정보\s*제공\s*(목적|이며|입니다)|"
    r"투자\s*판단.{0,12}(책임|참고)|참고용)")

# ---- 5. 수치 표기 — 숫자 뒤에 단위가 붙었나 -------------------------------
# "20%" "1,800만원" "3년" 은 통과, 문장 속에 홀로 뜬 "20" 은 미준수.
NUMBER = re.compile(r"\d[\d,.]*")
UNIT_AFTER = re.compile(
    r"^\s*(%p|%|퍼센트포인트|퍼센트|원|만원|억원|조원|백만원|천만원|달러|조|억|만|천|"
    r"년|개월|월|일|주일|주|분기|배|회|세|명|건|위|개|종목|가지|곳|자|"
    r"영업일|거래일|시간|시|분|초|"
    r"포인트|bp|단계|등급|호|차|번)")

# 범위 표기 — "연 1~2%", "1~5년" 처럼 단위가 범위 **뒤에** 한 번만 붙는다.
# 앞 숫자만 보고 "단위 없음"으로 판정하면 오탐이 된다. (실측 오탐 원인 2위)
RANGE_AFTER = re.compile(r"^\s*[~∼〜\-–]\s*\d")

# 숫자가 이름의 일부인 경우 — KODEX200, KOSPI 200, 코스피200 등.
# 수량이 아니라 고유명사이므로 단위를 요구하면 안 된다. (실측 오탐 원인 1위)
IDENT_PREFIX = re.compile(r"(?:[A-Z]{2,}|코스피|코스닥|나스닥|다우|S&P|MSCI|KODEX|TIGER)\s?$")

# "[문서 2]" 같은 내부 컨텍스트 라벨 인용 — 사용자에게 노출되면 안 되는 내부 사정
LABEL_CITE = re.compile(r"\[?\s*문서\s*\d+\s*\]?")

# ---- 6. 내부 사정 노출 ----------------------------------------------------
META_REF = re.compile(r"(제공된\s*자료|자료에\s*따르|자료에\s*의하|주어진\s*자료|"
                      r"본\s*자료|자료에는|자료상)")

# ---- 7. 거부 --------------------------------------------------------------
REFUSAL = re.compile(r"(확인할\s*수\s*없|찾을\s*수\s*없|나와\s*있지\s*않|"
                     r"포함되어\s*있지\s*않|안내해\s*드리기\s*어렵)")
# 거부하면서 '무엇이' 없는지 밝혔는가 — 정형 문구만 쓰면 미준수
REFUSAL_GENERIC = re.compile(r"^제공된\s*자료에서는\s*해당\s*내용을\s*확인할\s*수\s*없습니다\.?$")


def _norm(t: str) -> str:
    return "".join((t or "").split())


def numbers_without_unit(answer: str) -> list[str]:
    """단위가 안 붙은 숫자. 목록 번호는 제외한다."""
    text = LABEL_CITE.sub("", LIST_MARKER.sub("", answer or ""))
    out = []
    for m in NUMBER.finditer(text):
        head = text[:m.start()]
        if head and (head[-1].isalnum() or IDENT_PREFIX.search(head)):
            continue                     # KODEX200 / KOSPI 200 — 이름의 일부
        tail = text[m.end():m.end() + 8]
        if UNIT_AFTER.match(tail) or RANGE_AFTER.match(tail):
            continue
        out.append(m.group())
    return out


def check(answer: str) -> Dict[str, object]:
    """스펙 준수 여부. 해당 없는 항목은 None (준수율 계산에서 제외)."""
    a = answer or ""
    n = len(a)
    is_refusal = bool(REFUSAL.search(a))

    r: Dict[str, object] = {}

    if is_refusal:
        # 거부 답변에는 길이·위험고지·권유아님을 요구하지 않는다
        r["len_ok"] = None
        r["prose_ok"] = None
        r["risk_notice_ok"] = None
        r["no_solicit_ok"] = None
        r["refusal_specific_ok"] = not REFUSAL_GENERIC.match(a.strip())
    else:
        r["len_ok"] = LEN_MIN <= n <= LEN_MAX
        r["prose_ok"] = not LIST_MARKER.search(a)
        r["risk_notice_ok"] = (bool(RISK_NOTICE.search(a))
                               if RISKY_PRODUCT.search(a) else None)
        r["no_solicit_ok"] = bool(NO_SOLICIT.search(a))
        r["refusal_specific_ok"] = None

    bad_nums = numbers_without_unit(a)
    r["unit_ok"] = (not bad_nums) if NUMBER.search(a) else None
    r["bad_numbers"] = bad_nums
    r["no_meta_ok"] = not META_REF.search(a)
    r["no_label_ok"] = not LABEL_CITE.search(a)
    r["is_refusal"] = is_refusal
    r["chars"] = n

    checks = [v for k, v in r.items() if k.endswith("_ok") and v is not None]
    r["compliance"] = round(sum(checks) / len(checks), 4) if checks else None
    return r


RULE_LABELS = {
    "len_ok": "길이 150~250자",
    "prose_ok": "서술형(목록 금지)",
    "risk_notice_ok": "위험 고지",
    "no_solicit_ok": "권유 아님 명시",
    "unit_ok": "숫자 단위·기준",
    "no_meta_ok": "내부 사정 미노출",
    "no_label_ok": "내부 라벨 미인용",
    "refusal_specific_ok": "거부 시 사유 명시",
}
