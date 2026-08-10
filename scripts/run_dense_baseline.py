"""Dense 기준선 검색 실행 — run 파일 생성 예시 겸 팀 공용 baseline.

공유된 fixed_450_70 FAISS 인덱스를 그대로 로드해 157문항을 검색하고,
채점기(scripts/eval_retriever.py)가 읽는 run 파일을 만든다.

각자 BM25/Hybrid/Reranker 를 구현할 때 **이 파일의 출력 형식과 규칙을 그대로 따를 것.**
특히 아래 세 가지가 어긋나면 성능 비교가 아니라 설정 비교가 된다.

  1. 쿼리는 question 원문만 사용 (answer/gold_* 는 채점 전용, 넣으면 데이터 누수)
  2. 검색 대상 텍스트는 embedding_text (= "제목\n\n본문"). BM25 도 동일하게 쓸 것
  3. 동점 처리 규칙 통일 (점수순, 같으면 chunk_id 오름차순)

fetch_k 는 고정값이 아니라 튜닝 대상이다. 단독 리트리버에서는 20 이든 100 이든 top-10 이
같아 점수가 안 변하지만, Hybrid/Reranker 에서는 후보 풀 크기라 결과가 바뀐다.
값은 자유롭게 바꾸되 run_name·config 에 반드시 기록할 것.

실행:
  python scripts/run_dense_baseline.py
  python scripts/run_dense_baseline.py --fetch-k 20 --out runs/dense_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBEDDING_MODEL, OPENAI_API_KEY

TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70.jsonl"
INDEX_DIR = PROJECT_ROOT / "vectorstores" / "fixed_450_70"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve-k", "--fetch-k", dest="retrieve_k", type=int, default=50,
                    help="저장할 후보 수. 50 으로 한 번 저장해두면 @10/@20/@30/@50 을 재검색 없이 비교 가능")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "dense_baseline.json"))
    ap.add_argument("--run-name", default="dense_baseline")
    a = ap.parse_args()

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    if not (INDEX_DIR / "index.faiss").exists():
        raise SystemExit(f"{INDEX_DIR} 에 FAISS 인덱스가 없습니다.")

    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    testset = json.loads(TESTSET.read_text(encoding="utf-8"))
    print(f"[load] 평가셋 {len(testset)}문항")

    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    print(f"[load] FAISS {vs.index.ntotal} 벡터")

    items, lat = [], []
    for i, r in enumerate(testset, 1):
        t0 = time.perf_counter()
        # 쿼리는 question 원문만. answer/gold_* 는 절대 사용 금지.
        pairs = vs.similarity_search_with_score(r["question"], k=a.retrieve_k)
        lat.append(time.perf_counter() - t0)
        # 동점 처리 규칙 — 점수 동일 시 chunk_id 오름차순 (구현체별 순서 차이 제거)
        ranked = sorted(
            ((d.metadata.get("chunk_id"), float(s)) for d, s in pairs),
            key=lambda x: (x[1], x[0]),
        )
        items.append({"id": r["id"], "retrieved": [cid for cid, _ in ranked]})
        if i % 50 == 0:
            print(f"  {i}/{len(testset)}")

    run = {
        "run_name": a.run_name,
        "retriever": "dense",
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "retrieve_k": a.retrieve_k,
            "search_text_field": "embedding_text",
            "tie_break": "score asc, chunk_id asc",
        },
        "corpus_sha256": sha256(CORPUS),
        "index_sha256": sha256(INDEX_DIR / "index.faiss"),
        "avg_search_latency_ms": round(1000 * sum(lat) / len(lat), 1),
        "items": items,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved → {out}  ({len(items)}문항, 평균 {run['avg_search_latency_ms']}ms)")
    print("\n채점:")
    print(f"  python scripts/eval_retriever.py --run {out}")


if __name__ == "__main__":
    main()
