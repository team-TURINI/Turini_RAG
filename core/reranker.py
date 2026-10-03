"""리랭커 — 검색 후보를 질문과의 관련도로 다시 줄 세운다.

리트리버는 질문과 문서를 **따로** 벡터로 만들어 비교하지만(bi-encoder), 리랭커는
질문+문서를 **한 쌍으로 붙여** 통째로 읽는다(cross-encoder). 훨씬 정확하지만 느려서
전체 코퍼스가 아니라 검색이 추려온 상위 후보(candidate_k)에만 적용한다.

실측 — 리랭커는 "새로 찾는" 장치가 아니라 "위로 올리는" 장치다.
  USR@50  변화 0         (후보 집합은 그대로)
  USR@1   +21.7%p        (1등 자리가 달라짐)
그래서 LLM 에 적게 넘길수록(Top1·Top3) 이득이 크다.

CohereReranker 는 scripts/run_cohere_rerank.py 의 호출·정렬 로직을 그대로 옮긴 것이다.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence

from config import COHERE_API_KEY

# Cohere Trial 키는 분당 10회다. 호출 간격을 강제해 429 를 미리 피하고, 그래도 걸리면
# 물러났다가 다시 시도한다. Production 키면 min_interval=0 으로 두면 된다.
COHERE_TRIAL_MIN_INTERVAL_S = 6.5
COHERE_MAX_RETRY = 6

# 실패했을 때 — 배치 실험과 서빙의 요구가 반대다.
#   raise        지표를 내는 실험용. 리랭킹 없는 순위가 결과에 섞이면 안 되므로 멈춘다(이어서 재실행 가능).
#   passthrough  서빙용. 사용자를 기다리게 하거나 에러를 내느니 하이브리드 순위 그대로 답한다(품질은 하락).
ON_FAILURE = ("raise", "passthrough")
SERVE_MIN_INTERVAL_S = 0.0        # 서빙에선 간격을 강제하지 않는다 — 대기가 곧 응답 지연이다
SERVE_MAX_RETRY = 2               # 짧게 두 번만 물러난다


class CohereReranker:
    """Cohere rerank API. 상위 candidate_k 만 재정렬하고 나머지는 원순서로 뒤에 붙인다.

    동점 처리: relevance_score 내림차순, 같으면 chunk_id 오름차순.
    """

    def __init__(self, model: str, *, candidate_k: int,
                 text_field: str = "embedding_text",
                 min_interval_s: float = COHERE_TRIAL_MIN_INTERVAL_S,
                 on_failure: str = "raise",
                 max_retry: int = COHERE_MAX_RETRY,
                 backoff_s: float = 10.0) -> None:
        if on_failure not in ON_FAILURE:
            raise ValueError(f"on_failure 는 {ON_FAILURE} 중 하나여야 합니다.")
        import cohere
        if not COHERE_API_KEY:
            raise RuntimeError("COHERE_API_KEY 가 설정돼 있지 않습니다 (.env 또는 환경변수).")
        self._client = cohere.ClientV2(api_key=COHERE_API_KEY)
        self.model = model
        self.candidate_k = candidate_k
        self.text_field = text_field
        self.min_interval_s = min_interval_s
        self.on_failure = on_failure
        self.max_retry = max_retry
        self.backoff_s = backoff_s
        self.last_failed = False          # 직전 호출이 리랭킹 없이 통과했는가 (서빙 품질 표시용)
        self._last_call = 0.0

    def _call(self, query: str, docs: List[str]):
        import cohere
        for attempt in range(self.max_retry):
            wait = self.min_interval_s - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self._client.rerank(model=self.model, query=query,
                                           documents=docs, top_n=len(docs))
                self._last_call = time.time()
                return resp
            except cohere.errors.TooManyRequestsError:
                self._last_call = time.time()
                time.sleep(self.backoff_s * (attempt + 1))      # 10s, 20s, 30s …
        if self.on_failure == "passthrough":
            return None
        raise RuntimeError(f"Cohere rerank 이 {self.max_retry}회 연속 429 — 잠시 뒤 다시 실행하면 이어서 진행됨")

    def rerank(self, query: str, ranked_ids: Sequence[str],
               corpus: Dict[str, dict]) -> List[str]:
        """ranked_ids 의 앞 candidate_k 개를 재정렬해 돌려준다. 길이는 입력과 같다."""
        cand = list(ranked_ids[: self.candidate_k])
        rest = list(ranked_ids[self.candidate_k:])
        if not cand:
            return rest
        docs = [corpus[c][self.text_field] for c in cand]
        resp = self._call(query, docs)
        if resp is None:                   # passthrough — 하이브리드 순위를 그대로 쓴다
            self.last_failed = True
            return cand + rest
        self.last_failed = False
        scored = [(cand[r.index], r.relevance_score) for r in resp.results]
        reranked = [c for c, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
        return reranked + rest
