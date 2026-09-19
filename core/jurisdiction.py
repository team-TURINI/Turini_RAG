"""관할(국가) 판별 + 검색 필터 — jurisdiction-aware retrieval.

문제: 코퍼스에 미국 자료(SEC·FRED·yfinance)가 섞여 있어 "채권 종류가 뭐야?" 에 미국 채권
3유형(회사채·지방채 munis·국채)으로 답하는 일이 생겼다(2026-09-19 정성평가 1번 문항).
프롬프트에 "한국 기준" 을 적어도 검색이 미국 문서를 주면 소용없다 → **리랭커 앞에서** 거른다.

  질문 → detect() → allowed 집합 → Dense/BM25 가 allowed 청크만 반환 → RRF → Cohere → 생성

정책 (2026-09-19 팀 확정)
  - 국가 언급 없음  → KR (서비스 기본값). US 문서 **완전 제외**, GLOBAL 은 허용
  - 미국 언급       → 필터 없음. 한국 문서에도 해외주식 세금·해외 ETF 설명이 있어 그게 정답인 경우가 많음
  - GLOBAL (원자재·환율·일반 경제원리) → 항상 허용

라벨은 data/jurisdiction.json (scripts/label_jurisdiction.py 산출, doc_id 단위 사이드카).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Set

import config as _cfg

LABELS = ("KR", "US", "GLOBAL")
DEFAULT = "KR"

# 규칙 기반 — LLM 안 씀. 나중에 분류기로 바꿔도 반환 형식은 유지.
# 영문 약어는 한글이 바로 붙는 경우("SEC는")가 많아 \b 대신 영문자 경계로 잡는다.
# 뮤추얼펀드·지방채·달러는 한국에서도 쓰는 말이라 US 신호로 안 본다 (뮤추얼펀드 질문은 KR 기본 → 자료 없으면 회피).
def _en(w: str) -> str:
    return rf"(?<![A-Za-z]){w}(?![A-Za-z])"


US_SIGNALS = [
    r"미국", r"미\s?국채", r"연준", _en("Fed"), _en("FOMC"), _en("SEC"), r"나스닥", _en("NASDAQ"), r"S&P", r"다우",
    r"해외\s?주식", r"해외\s?ETF", r"해외\s?자산", r"해외\s?투자", r"월가", r"뉴욕",
    r"테슬라", r"애플", r"엔비디아", r"401\(?k\)?", _en("IRA"), _en("munis?"),
]
KR_SIGNALS = [r"한국", r"국내", r"우리나라", r"코스피", r"KOSPI", r"코스닥", r"KOSDAQ", r"금융위", r"금감원",
              r"한국은행", r"원화", r"KRX", r"국고채", r"KODEX", r"TIGER"]
_US_RE = re.compile("|".join(US_SIGNALS), re.IGNORECASE)
_KR_RE = re.compile("|".join(KR_SIGNALS), re.IGNORECASE)


def detect(query: str) -> Dict[str, object]:
    """질문의 관할. 미국 신호가 있으면 US, 아니면 KR(기본값). 한국 신호는 기록만 한다.

    US 신호와 KR 신호가 같이 있으면(예: "미국 주식 팔면 한국에서 세금?") US 로 본다 —
    US 는 '필터 없음' 이라 KR 문서도 그대로 검색되므로 손해가 없다.
    """
    us = sorted({m.group(0) for m in _US_RE.finditer(query)})
    kr = sorted({m.group(0) for m in _KR_RE.finditer(query)})
    j = "US" if us else DEFAULT
    return {"jurisdiction": j, "allowed": allowed_for(j), "signals_us": us, "signals_kr": kr}


def allowed_for(jurisdiction: str) -> Optional[FrozenSet[str]]:
    """허용 라벨 집합. None = 필터 없음."""
    if jurisdiction == "KR":
        return frozenset({"KR", "GLOBAL"})
    return None                                    # US → 필터 없음 (팀 확정)


class JurisdictionMap:
    """doc_id → 라벨. chunk_id 는 'doc_id__…' 꼴이라 앞부분만 잘라 찾는다."""

    def __init__(self, path: Optional[Path] = None) -> None:
        path = path or getattr(_cfg, "JURISDICTION_PATH", _cfg.PROJECT_ROOT / "data" / "jurisdiction.json")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version = raw.get("version")
        self._doc: Dict[str, str] = {d: v["jurisdiction"] for d, v in raw["docs"].items()}
        self.counts = raw.get("chunk_counts", {})

    def of_doc(self, doc_id: str) -> str:
        return self._doc.get(doc_id, DEFAULT)

    def of_chunk(self, chunk_id: str) -> str:
        return self.of_doc(chunk_id.split("__", 1)[0])

    def chunk_ok(self, allowed: Optional[FrozenSet[str]]) -> Callable[[str], bool]:
        """chunk_id → 허용 여부. allowed=None 이면 항상 True."""
        if allowed is None:
            return lambda cid: True
        return lambda cid: self.of_chunk(cid) in allowed

    def metadata_filter(self, allowed: Optional[FrozenSet[str]]):
        """LangChain FAISS `filter=` 콜백 (metadata dict → bool). allowed=None 이면 None."""
        if allowed is None:
            return None
        return lambda meta: self.of_doc(meta.get("doc_id", "")) in allowed

    def filter_ids(self, ids: List[str], allowed: Optional[FrozenSet[str]]) -> List[str]:
        ok = self.chunk_ok(allowed)
        return [c for c in ids if ok(c)]
