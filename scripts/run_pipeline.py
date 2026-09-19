"""최종 파이프라인 E2E 실행 — 테스트셋 전체를 FinalPipeline 에 통과시켜 저장한다.

한 번의 실행으로 검색·리랭킹·생성 결과가 **같은 폴더**에 남는다. 그래서 정량 평가와
정성 평가가 같은 답변을 본다.

  results_e2e/<tag>/
    config.json          실행 사양 + 코퍼스·인덱스 sha256
    ret_dense.json       ┐
    ret_bm25.json        │ 단계별 검색 순위 — 기존 run 파일과 같은 형식.
    ret_hybrid.json      │ eval_retriever_coverageaware.py 로 바로 채점되고,
    ret_reranked.json    ┘ runs/*.json 과 순위 단위로 대조할 수 있다(회귀 검증)
    generation.json      생성 결과 — run_judge.py · eval_generation.py · answer_spec 호환
    _partial.json        중간 저장 (중단돼도 이어서)

실행:
  python scripts/run_pipeline.py --tag final_v1                  # 157문항 전부
  python scripts/run_pipeline.py --tag final_v1 --split holdout  # holdout 50 만
  python scripts/run_pipeline.py --tag chk --skip-generation     # 검색만 (회귀 대조용, 무료)
  python scripts/run_pipeline.py --tag smoke --limit 5
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

import config as cfg  # noqa: E402
from core.pipeline import BaselinePipeline, FinalPipeline  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "results_e2e"


def run_file(name: str, retriever: str, items: List[dict], key: str, snap: dict) -> dict:
    """runs/*.json 과 같은 형식. eval_retriever_coverageaware.py 가 그대로 읽는다."""
    return {
        "run_name": name,
        "retriever": retriever,
        "config": snap,
        "corpus_sha256": snap["corpus_sha256"],
        "items": [{"id": it["id"], "retrieved": it[key]} for it in items],
    }


def check_local_files() -> None:
    """git 에 없는 로컬 파일이 다 있는지 먼저 확인한다 — 없으면 뭘 받아야 하는지 알려준다."""
    need = [
        (cfg.FINAL_CORPUS, "정리본 코퍼스 v2 (드라이브 공유)"),
        (cfg.FINAL_INDEX_DIR / "index.faiss", "FAISS 인덱스 v2 (드라이브 공유, 폴더째)"),
        (cfg.FINAL_TESTSET, "평가셋 v2 (드라이브 공유)"),
        (cfg.FINAL_SPLIT, "dev/holdout 분할 (드라이브 공유)"),
    ]
    missing = [(p, why) for p, why in need if not p.exists()]
    if missing:
        lines = "\n".join(f"  - {p.relative_to(PROJECT_ROOT)}   ← {why}" for p, why in missing)
        raise SystemExit(f"[fatal] 로컬 파일이 없습니다 (git 에는 안 올라감):\n{lines}\n"
                         f"  README '최종 파이프라인 실행' 절을 보세요.")
    if not cfg.OPENAI_API_KEY or not cfg.COHERE_API_KEY:
        raise SystemExit("[fatal] .env 에 OPENAI_API_KEY 와 COHERE_API_KEY 가 모두 필요합니다 (.env.example 참고)")


def main() -> None:
    check_local_files()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="results_e2e/<tag>/ 에 저장")
    ap.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-preset", default="v2_4")
    ap.add_argument("--mode", choices=["final", "baseline"], default="final",
                    help="baseline = 같은 코퍼스에서 Dense 단독 Top5 + 프롬프트 v1")
    ap.add_argument("--skip-generation", action="store_true",
                    help="검색·리랭킹까지만 (무료·회귀 대조용)")
    a = ap.parse_args()

    testset = json.loads(cfg.FINAL_TESTSET.read_text(encoding="utf-8"))
    split = json.loads(cfg.FINAL_SPLIT.read_text(encoding="utf-8"))["split"]
    if a.split != "all":
        testset = [r for r in testset if split.get(r["id"]) == a.split]
    if a.limit:
        testset = testset[: a.limit]

    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "_partial.json"
    done: Dict[str, dict] = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    if done:
        print(f"[resume] 중간 저장 {len(done)}문항 — 이어서 진행")

    pipe = BaselinePipeline() if a.mode == "baseline" else FinalPipeline(prompt_preset=a.prompt_preset)
    snap = pipe.config_snapshot()
    snap.update({"testset": cfg.FINAL_TESTSET.name, "split": a.split, "n": len(testset),
                 "skip_generation": a.skip_generation})
    (out_dir / "config.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = "검색만" if a.skip_generation else "검색+생성"
    print(f"\n[run] {a.tag} / {len(testset)}문항 / split={a.split} / {mode}")
    t0, n_new = time.time(), 0
    items: List[dict] = []
    for n, r in enumerate(testset, start=1):
        if r["id"] in done and (a.skip_generation or "answer" in done[r["id"]]):
            items.append(done[r["id"]])
            continue
        o = pipe.run(r["question"], skip_generation=a.skip_generation)
        o.update({"id": r["id"], "reference": r["answer"], "split": split.get(r["id"])})
        items.append(o)
        done[r["id"]] = o
        n_new += 1
        if n_new % 10 == 0 or n == len(testset):
            el = time.time() - t0
            print(f"  {n}/{len(testset)}  경과 {el/60:.1f}분 / 남은 예상 {el/n_new*(len(testset)-n)/60:.1f}분",
                  flush=True)
            ckpt.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")

    # 단계별 검색 run 파일
    for key, rname in [("dense", "dense"), ("bm25", "bm25"), ("hybrid", "hybrid"), ("reranked", "reranker")]:
        if key not in items[0]:
            continue
        (out_dir / f"ret_{key}.json").write_text(
            json.dumps(run_file(f"{a.tag}_{key}", rname, items, key, snap), ensure_ascii=False),
            encoding="utf-8")

    # 생성 결과 — 기존 gen_runs/*.json 과 같은 형식 (판정기·answer_spec 호환)
    if not a.skip_generation:
        gen_items = [{
            "id": it["id"], "question": it["question"], "reference": it["reference"],
            "answer": it["answer"], "context_chunk_ids": it["context_chunk_ids"],
            "split": it["split"], "n_context_chunks": len(it["context_chunk_ids"]),
            "context_chars": it["context_chars"], "input_tokens": it["input_tokens"],
            "output_tokens": it["output_tokens"], "latency_s": it["generation_latency_s"],
            "truncated": it["finish_reason"] == "length",
        } for it in items]
        gen = {
            "run_name": f"{a.tag}",
            "retriever_config": snap["reranker"] | {"base": a.mode},
            "generation_config": {
                "model": cfg.FINAL_MODEL, "prompt_version": pipe.prompt_version,
                "instruction_placement": pipe.placement, "prompt_preset": pipe.prompt_preset,
                "corpus": cfg.FINAL_CORPUS.name, "testset": cfg.FINAL_TESTSET.name, "split": a.split,
                "temperature": cfg.FINAL_TEMPERATURE, "max_tokens": cfg.FINAL_MAX_TOKENS,
                "top_p": cfg.FINAL_TOP_P, "context_mode": f"top_k={cfg.FINAL_TOP_K_GEN}",
                "context_text_field": cfg.FINAL_CTX_FIELD,
            },
            "summary": {
                "n": len(gen_items),
                "avg_latency_s": round(statistics.mean(g["latency_s"] for g in gen_items), 3),
                "avg_input_tokens": round(statistics.mean(g["input_tokens"] for g in gen_items), 1),
                "avg_output_tokens": round(statistics.mean(g["output_tokens"] for g in gen_items), 1),
                "avg_context_chunks": round(statistics.mean(g["n_context_chunks"] for g in gen_items), 2),
                "truncated": sum(g["truncated"] for g in gen_items),
            },
            "items": gen_items,
        }
        (out_dir / "generation.json").write_text(json.dumps(gen, ensure_ascii=False, indent=1), encoding="utf-8")
        s = gen["summary"]
        print(f"\n[gen] 평균 {s['avg_context_chunks']}청크 / input {s['avg_input_tokens']} tok / "
              f"output {s['avg_output_tokens']} tok / {s['avg_latency_s']}s / 절단 {s['truncated']}")

    ckpt.unlink(missing_ok=True)
    print(f"\nsaved → {out_dir}")
    print("검색 채점:  python scripts/eval_retriever_coverageaware.py --run "
          f"{out_dir / 'ret_reranked.json'} --corpus {cfg.FINAL_CORPUS}")
    if not a.skip_generation:
        print(f"생성 판정:  python scripts/run_judge.py --run {out_dir / 'generation.json'} "
              f"--corpus {cfg.FINAL_CORPUS} --testset {cfg.FINAL_TESTSET}")


if __name__ == "__main__":
    main()
