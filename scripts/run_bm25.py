"""BM25 검색 실행 → run 파일 생성.

scripts/run_dense_baseline.py 의 BM25 판이다. 출력 형식과 규칙은 동일하다.

⚠ 팀 전체가 동일한 토크나이저·파라미터를 써야 한다. 다르면 "BM25 비교"가 아니라
   "토크나이저 비교"가 된다. 아래 기본값을 합의값으로 쓰고, 바꿀 때는 팀에 공유할 것.

⚠ 검색 대상은 text 가 아니라 embedding_text (= "제목\\n\\n본문") 이다.
   Dense 인덱스가 이 필드로 만들어졌으므로 맞춰야 공정한 비교가 된다.

실행:
  python scripts/run_bm25.py
  python scripts/run_bm25.py --k1 1.5 --b 0.6 --run-name bm25_k1.5_b0.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "chunks.jsonl"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve-k", "--fetch-k", dest="retrieve_k", type=int, default=50,
                    help="저장할 후보 수. 50 으로 한 번 저장해두면 @10/@20/@30/@50 을 재검색 없이 비교 가능")
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--run-name", default="bm25_kiwi_k1.2_b0.75")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from kiwipiepy import Kiwi
    from rank_bm25 import BM25Okapi

    kiwi = Kiwi()

    def tokenize(t: str) -> List[str]:
        return [tok.form for tok in kiwi.tokenize(t)]

    chunks = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
    ids = [c["chunk_id"] for c in chunks]
    print(f"[load] 코퍼스 {len(chunks)}청크 — 토크나이즈 중...")
    corpus_tokens = [tokenize(c["embedding_text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens, k1=a.k1, b=a.b)

    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    print(f"[load] 평가셋 {len(testset)}문항")

    items, lat = [], []
    for i, r in enumerate(testset, 1):
        t0 = time.perf_counter()
        scores = bm25.get_scores(tokenize(r["question"]))  # 쿼리는 question 원문만
        lat.append(time.perf_counter() - t0)
        # 동점 처리: 점수 내림차순, 같으면 chunk_id 오름차순
        ranked = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))[: a.retrieve_k]
        items.append({"id": r["id"], "retrieved": [cid for cid, _ in ranked]})
        if i % 50 == 0:
            print(f"  {i}/{len(testset)}")

    import kiwipiepy
    import rank_bm25

    run = {
        "run_name": a.run_name,
        "retriever": "bm25",
        "config": {
            "tokenizer": "kiwi",
            "kiwipiepy_version": kiwipiepy.__version__,
            "rank_bm25_version": getattr(rank_bm25, "__version__", "0.2.2"),
            "k1": a.k1,
            "b": a.b,
            "retrieve_k": a.retrieve_k,
            "search_text_field": "embedding_text",
            "tie_break": "score desc, chunk_id asc",
        },
        "corpus_sha256": sha256(CORPUS),
        "avg_search_latency_ms": round(1000 * sum(lat) / len(lat), 1),
        "items": items,
    }
    out = Path(a.out) if a.out else PROJECT_ROOT / "runs" / f"{a.run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved → {out}  ({len(items)}문항, 평균 {run['avg_search_latency_ms']}ms)")
    print(f"\n채점:\n  python scripts/eval_retriever.py --run {out}")


if __name__ == "__main__":
    main()
