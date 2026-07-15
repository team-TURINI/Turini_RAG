"""RAG 베이스라인 배치 평가 — Retrieval 메트릭 (+ 선택적 Generation 로그).

config.py 값만 바꾸고 이 스크립트를 재실행하면 새 설정으로 평가된다.
실험 워크플로:
  1) 베이스라인 측정          python scripts/evaluate.py --tag baseline
  2) 변수 하나만 변경 후 재측정 (예: config.RETRIEVE_K 조정)
     python scripts/evaluate.py --tag k10
  3) results/ 의 결과 비교

평가 데이터 (goldset) 형식 — data/goldset.json :
  [
    {
      "query_id": "Q001",
      "question": "보수적 투자자에게 맞는 채권 ETF 알려줘",
      "relevant_ids": ["chunk_...", "chunk_..."]   // 정답 문서 id (doc_id 와 매칭)
    },
    ...
  ]
  (호환: relevant_ids 대신 relevant_portfolio_ids 도 허용)

산출:
  results/eval_<tag>_<timestamp>.json  — 메트릭 + 항목별 로그
  콘솔에 요약 메트릭 출력

메트릭:
  - Hit@K  : top-K 안에 정답이 하나라도 있나
  - MRR@K  : 1 / (정답이 처음 등장한 rank)
  - nDCG@K : 다중 정답 위치 가중합 / 이상적 DCG
  - (--generate 시) 평균 생성 latency, input/output tokens
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EMBEDDING_MODEL,
    MODEL,
    PROJECT_ROOT,
    RETRIEVE_K,
    TEMPERATURE,
    TOP_K_GEN,
)
from core.generator import generate
from core.retriever import DenseRetriever


# ----------------------------------------------------------------------
# 평가 헬퍼
# ----------------------------------------------------------------------
def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    topk = retrieved_ids[:k]
    return 1.0 if any(g in topk for g in gold_ids) else 0.0


def mrr_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Binary relevance nDCG."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def _gold_ids(item: dict) -> list[str]:
    return item.get("relevant_ids") or item.get("relevant_portfolio_ids", [])


# ----------------------------------------------------------------------
# 평가 루프
# ----------------------------------------------------------------------
def run_eval(
    goldset_path: Path,
    output_dir: Path,
    tag: str,
    k_values: list[int],
    do_generate: bool,
) -> dict:
    with open(goldset_path, "r", encoding="utf-8") as f:
        goldset = json.load(f)

    retriever = DenseRetriever()

    per_item: list[dict] = []
    gen_latencies: list[float] = []
    gen_in_tokens: list[int] = []
    gen_out_tokens: list[int] = []

    # 검색은 최대 k 이상 충분히 가져와서 상위 K 별로 메트릭 계산
    fetch_k = max(max(k_values), RETRIEVE_K)

    for item in goldset:
        question = item["question"]
        gold_ids = _gold_ids(item)

        candidates = retriever.search(question, k=fetch_k)
        retrieval_ids = [c["doc_id"] for c in candidates]

        record = {
            "query_id": item.get("query_id"),
            "question": question,
            "gold_ids": gold_ids,
            "retrieval_top_ids": retrieval_ids[: max(k_values)],
        }

        if do_generate:
            contexts = [c["page_content"] for c in candidates[:TOP_K_GEN]]
            result = generate(question, contexts)
            gen_latencies.append(result.latency)
            gen_in_tokens.append(result.input_tokens)
            gen_out_tokens.append(result.output_tokens)
            record["answer"] = result.answer
            record["gen_latency_s"] = round(result.latency, 3)
            record["gen_input_tokens"] = result.input_tokens
            record["gen_output_tokens"] = result.output_tokens

        per_item.append(record)

    # 집계
    def agg(name_fn, k):
        return mean(name_fn(r["retrieval_top_ids"], r["gold_ids"], k) for r in per_item)

    metrics: dict = {}
    for k in k_values:
        metrics[f"hit@{k}"] = round(agg(hit_at_k, k), 4)
        metrics[f"mrr@{k}"] = round(agg(mrr_at_k, k), 4)
        metrics[f"ndcg@{k}"] = round(agg(ndcg_at_k, k), 4)

    if do_generate and gen_latencies:
        metrics["avg_gen_latency_s"] = round(mean(gen_latencies), 3)
        metrics["avg_input_tokens"] = round(mean(gen_in_tokens), 1)
        metrics["avg_output_tokens"] = round(mean(gen_out_tokens), 1)

    # 결과 직렬화
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_{tag}_{ts}.json"
    payload = {
        "tag": tag,
        "timestamp": ts,
        "config_snapshot": {
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "RETRIEVE_K": RETRIEVE_K,
            "TOP_K_GEN": TOP_K_GEN,
            "MODEL": MODEL,
            "TEMPERATURE": TEMPERATURE,
        },
        "metrics": metrics,
        "items": per_item,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n== eval: {tag} ==")
    for k, v in metrics.items():
        print(f"  {k:22s}  {v}")
    print(f"\nsaved → {out_file}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", default=str(PROJECT_ROOT / "data" / "goldset.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--tag", default="baseline", help="실험 식별자 (결과 파일명에 사용)")
    parser.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5, 10],
        help="retrieval 메트릭 계산할 K 값들",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="답변 생성까지 수행 (LLM 비용 발생, latency/token 로그 포함)",
    )
    args = parser.parse_args()

    run_eval(
        goldset_path=Path(args.goldset),
        output_dir=Path(args.output),
        tag=args.tag,
        k_values=args.k,
        do_generate=args.generate,
    )


if __name__ == "__main__":
    main()
