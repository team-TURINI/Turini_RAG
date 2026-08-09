"""리트리버 공용 채점기 — 팀 전원이 이 스크립트로만 채점할 것.

각자 채점 코드를 구현하면 반올림·동점·클립 처리 차이만으로 1~2%p 가 흔들려서
실제 성능 차이와 구분이 안 된다. 검색은 각자, 채점은 이 파일 하나로 통일한다.

------------------------------------------------------------------
입력 — run 파일 (각자 검색 결과를 이 형식으로 저장)

  {
    "run_name": "bm25_kiwi_k1.2_b0.75",
    "retriever": "bm25",
    "config": {"tokenizer": "kiwi", "k1": 1.2, "b": 0.75, "fetch_k": 20},
    "corpus_sha256": "...",          # 선택, 있으면 코퍼스 일치 검증
    "index_sha256": "...",           # 선택
    "items": [
      {"id": "rag_fund_01", "retrieved": ["chunk_id_1", "chunk_id_2", ...]},
      ...
    ]
  }

  * retrieved 는 순위 오름차순(1등이 앞)
  * answer / gold_* 필드를 검색 입력에 쓰면 데이터 누수. 채점 전용이다.

------------------------------------------------------------------
지표

  [정답이 들어왔나]
  Coverage@K  주 지표. top-K 가 정답 구간을 덮은 '문자 비율'.
              gold 청크끼리 오버랩이 있으므로 단순 합이 아니라 구간 합집합으로 계산한다.
  Hit@K       top-K 에 gold 청크가 하나라도 있으면 1. 보조 지표(관대함).
  MRR@K       첫 gold 청크의 순위 역수.
  nDCG@K      청크별 gold_coverage 를 등급 관련도로 사용.

  [쓰레기가 얼마나 딸려왔나]
  Precision@K top-K 중 gold 청크의 비율.
  DocHit@K    top-K 중 정답 문서에서 온 청크의 비율.

  ⚠ Hit@K 만으로 판단하지 말 것. 157문항 중 93문항이 gold 청크 2개 이상이고,
    그중 50문항은 단일 청크로 100% 커버가 불가능하다. Hit 로만 채점하면 정답의
    일부만 가져와도 만점 처리되어 리트리버 간 차이가 뭉개진다.

------------------------------------------------------------------
실행

  # 단일 run 채점
  python scripts/eval_retriever.py --run runs/dense.json

  # 여러 run 비교 (첫 번째가 기준선)
  python scripts/eval_retriever.py --compare runs/dense.json runs/bm25.json runs/hybrid.json

  # run 파일 형식·정합성만 검사 (채점 없이)
  python scripts/eval_retriever.py --run runs/dense.json --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "chunks.jsonl"
INDEX_FAISS = PROJECT_ROOT / "vectorstores" / "fixed_450_70" / "index.faiss"

DEFAULT_K = [1, 3, 5, 10]
# 157문항 기준 1문항 = 0.64%p. 이보다 작은 차이는 노이즈로 본다.
NOISE_PP = 2.0


# =============================================================================
# 로드
# =============================================================================
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_testset() -> List[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


def load_corpus() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


# =============================================================================
# run 파일 검증 — 채점 전에 반드시 통과해야 한다
# =============================================================================
def validate_run(run: dict, testset: List[dict], corpus: Dict[str, dict],
                 kmax: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    if not isinstance(run.get("items"), list):
        return ["items 가 없거나 리스트가 아님"], warns

    want = {r["id"] for r in testset}
    got_list = [it.get("id") for it in run["items"]]
    got = set(got_list)

    if len(got_list) != len(got):
        dups = [i for i, n in Counter(got_list).items() if n > 1]
        errors.append(f"run 안에 중복 id {len(dups)}건: {dups[:5]}")
    missing = want - got
    extra = got - want
    if missing:
        errors.append(f"평가셋 문항 {len(missing)}건 누락: {sorted(missing)[:5]}")
    if extra:
        errors.append(f"평가셋에 없는 id {len(extra)}건: {sorted(extra)[:5]}")

    unknown, short, dup_in = 0, 0, 0
    for it in run["items"]:
        rid = it.get("id")
        ret = it.get("retrieved")
        if not isinstance(ret, list):
            errors.append(f"{rid}: retrieved 가 리스트가 아님")
            continue
        if len(ret) != len(set(ret)):
            dup_in += 1
        if len(ret) < kmax:
            short += 1
        unknown += sum(1 for c in ret if c not in corpus)
    if unknown:
        errors.append(f"코퍼스에 없는 chunk_id {unknown}건 — 다른 코퍼스로 검색했을 가능성")
    if dup_in:
        errors.append(f"retrieved 안에 중복 chunk_id 가 있는 문항 {dup_in}건")
    if short:
        warns.append(f"retrieved 길이가 K={kmax} 보다 짧은 문항 {short}건 — 해당 K 지표가 불리해짐")

    # 코퍼스/인덱스 해시 대조
    if run.get("corpus_sha256"):
        actual = sha256(CORPUS)
        if run["corpus_sha256"] != actual:
            errors.append(f"corpus_sha256 불일치 — 다른 코퍼스로 검색함\n"
                          f"      run: {run['corpus_sha256']}\n      실제: {actual}")
    else:
        warns.append("corpus_sha256 미기록 — 같은 코퍼스인지 확인 불가")
    if run.get("index_sha256") and INDEX_FAISS.exists():
        actual = sha256(INDEX_FAISS)
        if run["index_sha256"] != actual:
            warns.append("index_sha256 불일치 — Dense 계열이면 다른 인덱스일 수 있음")

    return errors, warns


# =============================================================================
# 채점
# =============================================================================
def _union_len(spans: Sequence[Tuple[int, int]]) -> int:
    """구간들의 합집합 길이. gold 청크끼리 오버랩이 있어 단순 합은 중복 계산된다."""
    if not spans:
        return 0
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def coverage_at(item: dict, retrieved: Sequence[str], k: int) -> float:
    """top-K 가 정답 구간을 덮은 문자 비율 (구간 합집합 기준)."""
    gs, ge = item["gold_start"], item["gold_end"]
    total = ge - gs
    if total <= 0:
        return 0.0
    gmap = {g["chunk_id"]: g for g in item["gold_chunks"]}
    spans = []
    for cid in retrieved[:k]:
        g = gmap.get(cid)
        if g is None:
            continue
        s, e = max(g["start_char"], gs), min(g["end_char"], ge)
        if e > s:
            spans.append((s, e))
    return min(1.0, _union_len(spans) / total)


def score_item(item: dict, retrieved: Sequence[str], corpus: Dict[str, dict],
               k_values: Sequence[int]) -> dict:
    gmap = {g["chunk_id"]: g for g in item["gold_chunks"]}
    gold_ids = set(gmap)
    is_gold = [cid in gold_ids for cid in retrieved]
    same_doc = [corpus.get(cid, {}).get("doc_id") == item["doc_id"] for cid in retrieved]

    rec: Dict[str, Any] = {
        "id": item["id"],
        "eval_source": item["eval_source"],
        "topic": item["topic"],
        "gold_type": Counter(corpus[g]["chunk_type"] for g in gold_ids
                             if g in corpus).most_common(1)[0][0] if gold_ids else "?",
        "n_gold": len(gold_ids),
    }
    for k in k_values:
        top = list(retrieved[:k])
        n = max(1, min(k, len(retrieved)))
        rec[f"coverage@{k}"] = round(coverage_at(item, retrieved, k), 6)
        rec[f"hit@{k}"] = 1.0 if any(is_gold[:k]) else 0.0
        rec[f"precision@{k}"] = round(sum(is_gold[:k]) / n, 6)
        rec[f"dochit@{k}"] = round(sum(same_doc[:k]) / n, 6)
        # MRR — 첫 gold 청크 순위의 역수
        rr = 0.0
        for i, cid in enumerate(top, start=1):
            if cid in gold_ids:
                rr = 1.0 / i
                break
        rec[f"mrr@{k}"] = round(rr, 6)
        # nDCG — gold_coverage 를 등급 관련도로
        rels = [gmap[cid]["gold_coverage"] if cid in gmap else 0.0 for cid in top]
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
        ideal = sorted((g["gold_coverage"] for g in item["gold_chunks"]), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
        rec[f"ndcg@{k}"] = round(dcg / idcg, 6) if idcg > 0 else 0.0
    return rec


def aggregate(rows: List[dict], k_values: Sequence[int]) -> Dict[str, Any]:
    m: Dict[str, Any] = {"n": len(rows)}
    if not rows:
        return m
    for k in k_values:
        for name in ("coverage", "hit", "mrr", "ndcg", "precision", "dochit"):
            key = f"{name}@{k}"
            m[key] = round(statistics.mean(r[key] for r in rows), 4)
    return m


def score_run(run: dict, testset: List[dict], corpus: Dict[str, dict],
              k_values: Sequence[int]) -> dict:
    by_id = {it["id"]: it.get("retrieved", []) for it in run["items"]}
    rows = [score_item(t, by_id.get(t["id"], []), corpus, k_values) for t in testset]

    segs: Dict[str, Dict[str, Any]] = {"전체": aggregate(rows, k_values)}
    for src in ("v1", "v2_prose"):
        sub = [r for r in rows if r["eval_source"] == src]
        if sub:
            segs[f"src:{src}"] = aggregate(sub, k_values)
    for gt in ("size", "section", "faq", "whole"):
        sub = [r for r in rows if r["gold_type"] == gt]
        if sub:
            segs[f"type:{gt}"] = aggregate(sub, k_values)
    for tp, cnt in Counter(r["topic"] for r in rows).most_common():
        if cnt >= 10:  # 10문항 미만은 노이즈라 분해하지 않음
            segs[f"topic:{tp}"] = aggregate([r for r in rows if r["topic"] == tp], k_values)

    return {
        "run_name": run.get("run_name", "(unnamed)"),
        "retriever": run.get("retriever"),
        "config": run.get("config", {}),
        "corpus_sha256": run.get("corpus_sha256"),
        "k_values": list(k_values),
        "metrics": segs["전체"],
        "metrics_by_segment": segs,
        "items": rows,
    }


# =============================================================================
# 출력
# =============================================================================
def print_single(res: dict, k_values: Sequence[int]) -> None:
    kmain = 5 if 5 in k_values else max(k_values)
    print(f"\n== {res['run_name']} ==  retriever={res.get('retriever')}")
    if res.get("config"):
        print(f"   config: {json.dumps(res['config'], ensure_ascii=False)}")
    cols = [f"Cov@{k}" for k in k_values] + [f"Hit@{kmain}", f"MRR@{kmain}",
                                             f"nDCG@{kmain}", f"P@{kmain}", f"DocHit@{kmain}"]
    hdr = f"{'세그먼트':14s} {'n':>4s} " + " ".join(f"{c:>9s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, m in res["metrics_by_segment"].items():
        vals = [m[f"coverage@{k}"] for k in k_values] + [
            m[f"hit@{kmain}"], m[f"mrr@{kmain}"], m[f"ndcg@{kmain}"],
            m[f"precision@{kmain}"], m[f"dochit@{kmain}"]]
        print(f"{name:14s} {m['n']:>4d} " + " ".join(f"{v:>9.3f}" for v in vals))
    print(f"\n※ 주 지표는 Coverage@K. Hit@K 는 정답 일부만 가져와도 1이라 관대함.")


def print_compare(results: List[dict], k_values: Sequence[int]) -> None:
    kmain = 5 if 5 in k_values else max(k_values)
    base = results[0]
    print(f"\n== 비교 (기준선: {base['run_name']}) ==")
    print(f"※ 157문항 기준 1문항 = 0.64%p. {NOISE_PP}%p 미만 차이는 노이즈로 볼 것.\n")

    for seg in base["metrics_by_segment"]:
        if seg != "전체" and not seg.startswith("src:"):
            continue
        n = base["metrics_by_segment"][seg]["n"]
        print(f"[{seg}]  n={n}")
        hdr = (f"  {'run':26s} " + " ".join(f"{'Cov@'+str(k):>9s}" for k in k_values)
               + f" {'Hit@'+str(kmain):>9s} {'MRR@'+str(kmain):>9s} {'P@'+str(kmain):>9s}"
               + f" {'ΔCov@'+str(kmain):>10s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        b = base["metrics_by_segment"][seg][f"coverage@{kmain}"]
        for r in results:
            m = r["metrics_by_segment"].get(seg)
            if not m:
                continue
            d = (m[f"coverage@{kmain}"] - b) * 100
            mark = "" if abs(d) < NOISE_PP else ("  ▲" if d > 0 else "  ▼")
            row = (f"  {r['run_name'][:26]:26s} "
                   + " ".join(f"{m[f'coverage@{k}']:>9.3f}" for k in k_values)
                   + f" {m[f'hit@{kmain}']:>9.3f} {m[f'mrr@{kmain}']:>9.3f} {m[f'precision@{kmain}']:>9.3f}"
                   + f" {d:>+9.1f}p{mark}")
            print(row)
        print()


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", help="채점할 run 파일")
    ap.add_argument("--compare", nargs="+", help="비교할 run 파일들 (첫 번째가 기준선)")
    ap.add_argument("--k", nargs="+", type=int, default=DEFAULT_K)
    ap.add_argument("--out", default=None, help="채점 결과 저장 경로 (기본: results_retriever/)")
    ap.add_argument("--validate-only", action="store_true", help="형식 검사만 하고 종료")
    a = ap.parse_args()

    paths = [Path(p) for p in (a.compare or a.run or [])]
    if not paths:
        ap.error("--run 또는 --compare 중 하나는 필요합니다")

    k_values = sorted(set(a.k))
    testset, corpus = load_testset(), load_corpus()
    print(f"[load] 평가셋 {len(testset)}문항 / 코퍼스 {len(corpus)}청크")

    results = []
    for p in paths:
        run = json.loads(p.read_text(encoding="utf-8"))
        errors, warns = validate_run(run, testset, corpus, max(k_values))
        label = run.get("run_name", p.name)
        for w in warns:
            print(f"  [warn] {label}: {w}")
        if errors:
            for e in errors:
                print(f"  [ERROR] {label}: {e}")
            raise SystemExit(f"\n[fatal] {p} 검증 실패 — 채점을 중단합니다.")
        print(f"  [ok] {label} 검증 통과")
        if not a.validate_only:
            results.append(score_run(run, testset, corpus, k_values))

    if a.validate_only:
        print("\n검증만 수행했습니다.")
        return

    out_dir = Path(a.out) if a.out else PROJECT_ROOT / "results_retriever"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        f = out_dir / f"score_{res['run_name']}.json"
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print_single(res, k_values)
        print(f"saved → {f}")

    if len(results) > 1:
        print_compare(results, k_values)


if __name__ == "__main__":
    main()
