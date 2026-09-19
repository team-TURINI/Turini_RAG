"""리트리버 run 파일 → 답변 생성.

리트리버 실험에서 만든 run 파일의 검색 결과를 컨텍스트로 넣어 답변을 생성한다.
**검색을 다시 하지 않으므로** 어떤 리트리버 구성이든 run 파일만 있으면 비교 가능하다.

  run 파일 (chunk_id 순위)  →  청크 텍스트로 변환  →  LLM  →  답변

컨텍스트 구성 — 두 방식 중 선택
  · --top-k N     상위 N개 청크 (개수 기준)
  · --budget N    상위부터 N자까지 (문자 기준, 실서비스에 가까움)
  둘 다 주면 budget 이 우선. 기본은 budget=2500 (리트리버 평가 지표와 동일 기준).

⚠ 데이터 누수 금지 — 컨텍스트는 검색된 청크만. answer/gold_* 는 절대 넣지 말 것.

실행:
  python scripts/run_generation.py --run runs/rr_bge_p10_hybrid.json
  python scripts/run_generation.py --run runs/dense_k50.json --budget 2500 --limit 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MAX_TOKENS, MODEL, TEMPERATURE, TOP_P
from core.generator import generate

CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70.jsonl"
TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CTX_FIELD = "embedding_text"   # 리트리버 평가의 예산 계산과 동일 기준


def build_context(retrieved: List[str], corpus: Dict[str, dict],
                  top_k: int | None, budget: int | None) -> List[str]:
    """검색 순위대로 컨텍스트를 채운다."""
    out, total = [], 0
    for cid in retrieved:
        text = corpus.get(cid, {}).get(CTX_FIELD, "")
        if not text:
            continue
        if budget:
            if out and total + len(text) > budget:
                break
        elif top_k and len(out) >= top_k:
            break
        out.append(text)
        total += len(text)
    return out


def main() -> None:
    global CORPUS, TESTSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="리트리버 run 파일")
    ap.add_argument("--context", help="확정된 컨텍스트 파일 (data/gen_contexts/*.json). "
                                      "자르지 않고 저장된 청크를 그대로 사용")
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="all",
                    help="--context 사용 시 분할 필터")
    ap.add_argument("--budget", type=int, default=2500, help="컨텍스트 문자 예산 (기본 2500)")
    ap.add_argument("--top-k", type=int, default=None, help="개수 기준으로 자를 때 (budget 대신)")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만 (파일럿용)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--instruction-placement", choices=["front", "sandwich"], default="front",
                    help="sandwich = 컨텍스트 뒤에 핵심 규칙을 재기재 "
                         "(GPT-4.1 가이드: 장문맥에서는 지시를 앞뒤 양쪽에 두는 편이 낫다)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--corpus", default=str(CORPUS), help="코퍼스 경로 override")
    ap.add_argument("--testset", default=str(TESTSET), help="평가셋 경로 override")
    a = ap.parse_args()

    CORPUS, TESTSET = Path(a.corpus), Path(a.testset)

    if not a.run and not a.context:
        raise SystemExit("--run 또는 --context 중 하나가 필요합니다")

    budget = None if a.top_k else a.budget
    corpus = {c["chunk_id"]: c for c in
              (json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip())}
    testset = {r["id"]: r for r in json.loads(TESTSET.read_text(encoding="utf-8"))}

    # --context : 이미 확정된 컨텍스트를 그대로 쓴다.
    #   조건 A / A2 / X 는 검색 결과가 아니라 규칙으로 만든 컨텍스트라 run 파일이 없다.
    #   여기서 다시 자르면 컨텍스트를 확정해 둔 의미가 없으므로 budget/top_k 를 무시한다.
    if a.context:
        run = json.loads(Path(a.context).read_text(encoding="utf-8"))
        run["run_name"] = run["context_name"]
        items = run["items"]
        if a.split != "all":
            items = [it for it in items if it.get("split") == a.split]
        for it in items:                       # run 파일과 같은 키로 맞춰준다
            it["retrieved"] = it["context_chunk_ids"]
        budget, a.top_k = None, None
        mode = "context 고정"
    else:
        run = json.loads(Path(a.run).read_text(encoding="utf-8"))
        items = run["items"]
        # 검색 run 파일에는 dev/holdout 표시가 없다. 분할 맵을 붙여 --split 이 듣게 한다.
        # (스윕은 dev 에서만 하고 holdout 은 최종 1회만 보기 위한 장치)
        smap_path = PROJECT_ROOT / "data" / "gen_contexts" / "split.json"
        if smap_path.exists():
            smap = json.loads(smap_path.read_text(encoding="utf-8"))["split"]
            items = [dict(it, split=smap.get(it["id"])) for it in items]
        elif a.split != "all":
            raise SystemExit(f"[fatal] --split 을 쓰려면 {smap_path} 가 필요합니다")
        if a.split != "all":
            items = [it for it in items if it.get("split") == a.split]
        mode = f"budget={budget}자" if budget else f"top_k={a.top_k}"

    items = items[: a.limit] if a.limit else items
    model = a.model or MODEL
    print(f"[gen] {run.get('run_name')} / {len(items)}문항 / {mode} / {model}")

    out_items, lat, tin, tout, trunc = [], [], [], [], 0
    t0 = time.time()
    for n, it in enumerate(items, start=1):
        q = testset[it["id"]]["question"]
        ctx = build_context(it["retrieved"], corpus, a.top_k, budget)
        res = generate(q, ctx, prompt_version=a.prompt_version,
                       placement=a.instruction_placement, model=model)
        lat.append(res.latency); tin.append(res.input_tokens); tout.append(res.output_tokens)
        trunc += int(res.truncated)
        out_items.append({
            "id": it["id"],
            "question": q,
            "reference": testset[it["id"]]["answer"],   # 채점 전용 — 프롬프트에 안 들어감
            "answer": res.answer,
            # 채점기가 컨텍스트 원문을 복원하려면 chunk_id 가 필요하다.
            # (Faithfulness·숫자 정확도는 "자료에 있었나"를 봐야 하므로)
            "context_chunk_ids": [c for c in it["retrieved"]][:len(ctx)],
            "split": it.get("split"),
            "n_context_chunks": len(ctx),
            "context_chars": sum(len(c) for c in ctx),
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "latency_s": round(res.latency, 3),
            "truncated": res.truncated,
        })
        if n % 20 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(items)}  경과 {el/60:.1f}분 / 남은 예상 "
                  f"{el/n*(len(items)-n)/60:.1f}분", flush=True)

    payload = {
        "run_name": run.get("run_name"),
        "retriever_config": run.get("config"),
        "generation_config": {
            "model": model, "prompt_version": a.prompt_version,
            "instruction_placement": a.instruction_placement,
            # 어떤 데이터로 돌렸는지 결과 파일만 보고 알 수 있어야 한다.
            # (기본값이 v1 평가셋이라, v2 를 쓰고도 기록이 없으면 나중에 구분이 안 된다)
            "corpus": str(CORPUS.name), "testset": str(TESTSET.name),
            "split": a.split,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_p": TOP_P,
            "context_mode": mode, "context_text_field": CTX_FIELD,
        },
        "summary": {
            "n": len(out_items),
            "avg_latency_s": round(statistics.mean(lat), 3),
            "avg_input_tokens": round(statistics.mean(tin), 1),
            "avg_output_tokens": round(statistics.mean(tout), 1),
            "avg_context_chunks": round(statistics.mean(x["n_context_chunks"] for x in out_items), 2),
            "truncated": trunc,
        },
        "items": out_items,
    }
    out = Path(a.out) if a.out else PROJECT_ROOT / "gen_runs" / f"gen_{run.get('run_name')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    s = payload["summary"]
    print(f"\nsaved → {out}")
    print(f"  평균 {s['avg_context_chunks']}청크 / input {s['avg_input_tokens']} tok "
          f"/ output {s['avg_output_tokens']} tok / {s['avg_latency_s']}s")
    if trunc:
        print(f"  ⚠ MAX_TOKENS 에서 잘린 답변 {trunc}건 — 점수 해석 전 확인할 것")


if __name__ == "__main__":
    main()
