"""테스트셋 v3 생성 — 관할 라벨 + SEC gold 13문항 재라벨/회피 정답.

입력
  data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v2.json   (v2, 157문항)
  data/testset/jurisdiction_relabel.json                                (13문항 처리 방침 — 사람이 정함)
  data/docs_metadata_cleaned.jsonl · 정리본 코퍼스 v2                    (앵커 → offset → gold_chunks)
출력
  data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v3.json

v3 에서 달라지는 것
  - 전 문항에 expected_jurisdiction (core.jurisdiction.detect — KR 기본, 미국 언급 시 US)
  - relabel 11문항: doc_id·answer·앵커·gold_chunks 를 한국 자료로 교체. 원래 값은 `v2_orig` 에 보존
  - refusal 2문항: gold 없음, retrieval_eval=false. 생성 평가에서는 회피했는지 본다
  - 나머지 144문항: 그대로 (13 = 11 + 2)

실행: python scripts/apply_jurisdiction_relabel.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config as cfg  # noqa: E402
from core.jurisdiction import detect  # noqa: E402
from remap_testset import load_chunks, load_docs, locate, rebuild_gold_chunks  # noqa: E402

V2 = cfg.FINAL_TESTSET_V2
V3 = V2.with_name(V2.name.replace("_v2.json", "_v3.json"))
PLAN = PROJECT_ROOT / "data" / "testset" / "jurisdiction_relabel.json"
META = PROJECT_ROOT / "data" / "docs_metadata_cleaned.jsonl"


def main() -> None:
    ts = json.loads(V2.read_text(encoding="utf-8"))
    plan = {p["id"]: p for p in json.loads(PLAN.read_text(encoding="utf-8"))["items"]}
    docs = load_docs(META)
    chunks_by_doc = defaultdict(list)
    for c in load_chunks(cfg.FINAL_CORPUS):
        chunks_by_doc[c["doc_id"]].append(c)

    keep = {"doc_id", "answer", "span_start", "span_end", "gold_start", "gold_end", "gold_length",
            "gold_chunks", "union_gold_coverage", "eval_source"}
    n_re, n_ref = 0, 0
    for r in ts:
        r["expected_jurisdiction"] = detect(r["question"])["jurisdiction"]
        r["retrieval_eval"] = True
        p = plan.get(r["id"])
        if not p:
            continue
        r["v2_orig"] = {k: r[k] for k in keep if k in r}
        r["relabel_why"] = p["why"]
        if p["action"] == "refusal":
            r.update({"doc_id": None, "answer": p["answer"], "span_start": None, "span_end": None,
                      "gold_start": None, "gold_end": None, "gold_length": 0, "gold_chunks": [],
                      "union_gold_coverage": 0.0, "eval_source": "v3_refusal_kr", "retrieval_eval": False,
                      "expected_behavior": "refusal"})
            n_ref += 1
            continue
        text = docs[p["doc_id"]]
        gs, ge, note = locate(text, p["span_start"], p["span_end"], None)
        if gs is None:
            raise SystemExit(f"[fatal] {r['id']} 앵커 못 찾음: {note}")
        golds = rebuild_gold_chunks(p["doc_id"], gs, ge, chunks_by_doc)
        if not golds:
            raise SystemExit(f"[fatal] {r['id']} 구간과 겹치는 청크 없음")
        cov = sum(g["overlap_chars"] for g in golds) / (ge - gs)
        r.update({"doc_id": p["doc_id"], "answer": p["answer"], "span_start": p["span_start"],
                  "span_end": p["span_end"], "gold_start": gs, "gold_end": ge, "gold_length": ge - gs,
                  "gold_chunks": golds, "union_gold_coverage": round(min(1.0, cov), 6),
                  "eval_source": "v3_relabel_kr"})
        n_re += 1
        print(f"  {r['id']:12s} → {p['doc_id']:22s} {ge-gs:4d}자  청크 {[g['chunk_id'].split('__')[-1] for g in golds]}  {note}")

    V3.write_text(json.dumps(ts, ensure_ascii=False, indent=1), encoding="utf-8")
    # 검색 채점기(eval_retriever_coverageaware.py)는 gold 없는 문항을 fail-closed 로 거부하므로 회피 문항을 뺀 판을 따로 둔다
    V3R = V3.with_name(V3.name.replace("_v3.json", "_v3_retrieval.json"))
    V3R.write_text(json.dumps([r for r in ts if r["retrieval_eval"]], ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"\n[v3] {len(ts)}문항  재라벨 {n_re} · 회피 {n_ref}  검색평가 대상 {sum(r['retrieval_eval'] for r in ts)}"
          f"  관할 {dict(Counter(r['expected_jurisdiction'] for r in ts))}")
    print(f"[저장] {V3}")
    print(f"[저장] {V3R}  (검색 채점용, 회피 문항 제외)")


if __name__ == "__main__":
    main()
