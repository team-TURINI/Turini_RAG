"""리트리버 공용 채점기 — 팀 전원이 이 스크립트로만 채점할 것.

각자 채점 코드를 구현하면 반올림·경계 처리 차이만으로 수치가 흔들려 실제 성능 차이와
구분이 안 된다. 검색은 각자, 채점은 이 파일 하나로 통일한다.

------------------------------------------------------------------
정답의 기준

  원본 기준은 **원문 Gold span** (`doc_id` + `gold_start`~`gold_end`) 이다.
  `gold_chunks` 는 그 span 이 fixed_450 청크 중 어디에 걸리는지 **미리 계산해 둔 캐시**다.
  157문항 전부 아래가 검증되어 있으므로 채점에 그대로 사용한다.

    gold 청크 offset == 코퍼스 청크 offset      261/261
    overlap_chars   == 실제 교집합 길이          261/261
    gold_coverage   == overlap / gold_length     261/261
    union_gold_coverage == 합집합 / gold_length  157/157

------------------------------------------------------------------
지표

  [근거 확보 능력]
  UnionSpanRecall@K  주 지표. top-K 의 gold 청크들이 정답 구간을 덮은 문자 비율.
                     청크끼리 70자 overlap 이 있으므로 gold_coverage 를 단순 합산하지
                     않고 **문자 구간의 합집합**으로 계산한다.
                     (실제로 157문항 중 89건은 단순 합산 시 1.0 을 초과함)
  CoverageHit@K-τ    UnionSpanRecall@K >= τ 인 문항 비율. τ = 0.5 / 0.8
  EvidenceHit@K      top-K 에 gold 청크가 하나라도 있으면 1

  [순위 품질]
  MRR@K              첫 gold 청크 순위의 역수
  nDCG@K             청크별 gold_coverage 를 등급 관련도로 사용

  [진단]
  DocHit@K           top-K 에 정답 문서 청크가 **하나라도** 있으면 1 (binary)
  DocPrecision@K     top-K 중 정답 문서 청크의 **비율**
  Precision@K        top-K 중 gold 청크의 비율

  [실서비스 관점 — 문자 예산 고정]
  UnionSpanRecall@<budget>   순위대로 예산(기본 2,500자)까지 채웠을 때의 커버리지
  CoverageHit@<budget>-0.8   그 값이 0.8 이상인 문항 비율
  ChunksUsed@<budget>        예산 안에 들어간 평균 청크 수

  관계: DocHit >= EvidenceHit >= CoverageHit@K-0.5 >= CoverageHit@K-0.8

------------------------------------------------------------------
실행

  python scripts/eval_retriever.py --run runs/dense.json
  python scripts/eval_retriever.py --compare runs/dense.json runs/bm25.json
  python scripts/eval_retriever.py --run runs/dense_k50.json --k 10 20 30 50   # 후보수 saturation
  python scripts/eval_retriever.py --run runs/x.json --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70.jsonl"
INDEX_FAISS = PROJECT_ROOT / "vectorstores" / "fixed_450_70" / "index.faiss"

DEFAULT_K = [1, 3, 5, 10]
OFFICIAL_K = 10                 # 공식 보고값 기준 (MRR@10, nDCG@10)
TAUS = (0.5, 0.8)               # CoverageHit 임계값
DEFAULT_BUDGET = 2500           # 실서비스 context 문자 예산
# 예산 계산에 쓰는 텍스트 필드. LLM 에 실제로 들어가는 텍스트와 맞춰야 한다.
# embedding_text = "제목\n\n본문" 이며 Dense 인덱스도 이 필드로 만들어졌다.
BUDGET_TEXT_FIELD = "embedding_text"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260805


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
# run 파일 검증
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
    if want - got:
        errors.append(f"평가셋 문항 {len(want - got)}건 누락: {sorted(want - got)[:5]}")
    if got - want:
        errors.append(f"평가셋에 없는 id {len(got - want)}건: {sorted(got - want)[:5]}")

    unknown, short, dup_in = 0, 0, 0
    for it in run["items"]:
        ret = it.get("retrieved")
        if not isinstance(ret, list):
            errors.append(f"{it.get('id')}: retrieved 가 리스트가 아님")
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

    if run.get("corpus_sha256"):
        actual = sha256(CORPUS)
        if run["corpus_sha256"] != actual:
            errors.append(f"corpus_sha256 불일치 — 다른 코퍼스로 검색함\n"
                          f"      run: {run['corpus_sha256']}\n      실제: {actual}")
    else:
        warns.append("corpus_sha256 미기록 — 같은 코퍼스인지 확인 불가")
    if run.get("index_sha256") and INDEX_FAISS.exists():
        if run["index_sha256"] != sha256(INDEX_FAISS):
            warns.append("index_sha256 불일치 — 인덱스를 새로 만들었다면 config 에 기록할 것")

    return errors, warns


# =============================================================================
# 채점
# =============================================================================
def _union_len(spans: Sequence[Tuple[int, int]]) -> int:
    if not spans:
        return 0
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def union_span_recall(item: dict, retrieved: Sequence[str]) -> float:
    """검색된 gold 청크들이 정답 구간을 덮은 문자 비율 (합집합)."""
    gs, ge = item["gold_start"], item["gold_end"]
    total = ge - gs
    if total <= 0:
        return 0.0
    gmap = {g["chunk_id"]: g for g in item["gold_chunks"]}
    spans = []
    for cid in retrieved:
        g = gmap.get(cid)
        if g is None:
            continue
        s, e = max(g["start_char"], gs), min(g["end_char"], ge)
        if e > s:
            spans.append((s, e))
    return min(1.0, _union_len(spans) / total)


def budget_cut(retrieved: Sequence[str], corpus: Dict[str, dict], budget: int) -> List[str]:
    """순위대로 문자 예산까지 채운다 (최소 1개는 포함)."""
    out: List[str] = []
    total = 0
    for cid in retrieved:
        n = len(corpus.get(cid, {}).get(BUDGET_TEXT_FIELD, ""))
        if out and total + n > budget:
            break
        out.append(cid)
        total += n
    return out


def score_item(item: dict, retrieved: Sequence[str], corpus: Dict[str, dict],
               k_values: Sequence[int], budget: int) -> dict:
    gmap = {g["chunk_id"]: g for g in item["gold_chunks"]}
    gold_ids = set(gmap)
    is_gold = [cid in gold_ids for cid in retrieved]
    same_doc = [corpus.get(cid, {}).get("doc_id") == item["doc_id"] for cid in retrieved]

    gold_types = [corpus[g]["chunk_type"] for g in gold_ids if g in corpus]
    rec: Dict[str, Any] = {
        "id": item["id"],
        "eval_source": item["eval_source"],
        "topic": item["topic"],
        "gold_type": Counter(gold_types).most_common(1)[0][0] if gold_types else "?",
        "n_gold": len(gold_ids),
    }

    for k in k_values:
        top = list(retrieved[:k])
        n = max(1, min(k, len(retrieved)))
        usr = union_span_recall(item, top)
        rec[f"usr@{k}"] = round(usr, 6)
        for tau in TAUS:
            rec[f"covhit@{k}-{tau}"] = 1.0 if usr >= tau else 0.0
        rec[f"evhit@{k}"] = 1.0 if any(is_gold[:k]) else 0.0
        rec[f"dochit@{k}"] = 1.0 if any(same_doc[:k]) else 0.0          # binary
        rec[f"docprec@{k}"] = round(sum(same_doc[:k]) / n, 6)           # 비율
        rec[f"precision@{k}"] = round(sum(is_gold[:k]) / n, 6)
        rr = 0.0
        for i, cid in enumerate(top, start=1):
            if cid in gold_ids:
                rr = 1.0 / i
                break
        rec[f"mrr@{k}"] = round(rr, 6)
        rels = [gmap[cid]["gold_coverage"] if cid in gmap else 0.0 for cid in top]
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
        ideal = sorted((g["gold_coverage"] for g in item["gold_chunks"]), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
        rec[f"ndcg@{k}"] = round(dcg / idcg, 6) if idcg > 0 else 0.0

    sel = budget_cut(retrieved, corpus, budget)
    busr = union_span_recall(item, sel)
    rec[f"usr@{budget}c"] = round(busr, 6)
    rec[f"covhit@{budget}c-0.8"] = 1.0 if busr >= 0.8 else 0.0
    rec[f"chunks@{budget}c"] = len(sel)
    rec[f"chars@{budget}c"] = sum(len(corpus.get(c, {}).get(BUDGET_TEXT_FIELD, "")) for c in sel)
    return rec


METRIC_KEYS_FOR_BOOTSTRAP = "usr"  # 부트스트랩 대상 (연속값)


def aggregate(rows: List[dict], k_values: Sequence[int], budget: int) -> Dict[str, Any]:
    m: Dict[str, Any] = {"n": len(rows)}
    if not rows:
        return m
    names = ["usr", "evhit", "dochit", "docprec", "precision", "mrr", "ndcg"]
    for k in k_values:
        for name in names:
            m[f"{name}@{k}"] = round(statistics.mean(r[f"{name}@{k}"] for r in rows), 4)
        for tau in TAUS:
            key = f"covhit@{k}-{tau}"
            m[key] = round(statistics.mean(r[key] for r in rows), 4)
    for key in (f"usr@{budget}c", f"covhit@{budget}c-0.8", f"chunks@{budget}c", f"chars@{budget}c"):
        m[key] = round(statistics.mean(r[key] for r in rows), 4)
    return m


def score_run(run: dict, testset: List[dict], corpus: Dict[str, dict],
              k_values: Sequence[int], budget: int) -> dict:
    by_id = {it["id"]: it.get("retrieved", []) for it in run["items"]}
    rows = [score_item(t, by_id.get(t["id"], []), corpus, k_values, budget) for t in testset]

    # 정합성: DocHit >= EvidenceHit 이어야 함 (gold 청크는 정의상 정답 문서 소속)
    for k in k_values:
        bad = [r["id"] for r in rows if r[f"dochit@{k}"] < r[f"evhit@{k}"]]
        assert not bad, f"DocHit < EvidenceHit @{k}: {bad[:3]}"

    segs: Dict[str, Dict[str, Any]] = {"전체": aggregate(rows, k_values, budget)}
    for src in ("v1", "v2_prose"):
        sub = [r for r in rows if r["eval_source"] == src]
        if sub:
            segs[f"src:{src}"] = aggregate(sub, k_values, budget)
    for gt, _ in Counter(r["gold_type"] for r in rows).most_common():
        segs[f"type:{gt}"] = aggregate([r for r in rows if r["gold_type"] == gt], k_values, budget)
    for tp, _ in Counter(r["topic"] for r in rows).most_common():
        segs[f"topic:{tp}"] = aggregate([r for r in rows if r["topic"] == tp], k_values, budget)

    return {
        "run_name": run.get("run_name", "(unnamed)"),
        "retriever": run.get("retriever"),
        "config": run.get("config", {}),
        "corpus_sha256": run.get("corpus_sha256"),
        "k_values": list(k_values),
        "budget": budget,
        "budget_text_field": BUDGET_TEXT_FIELD,
        "metrics": segs["전체"],
        "metrics_by_segment": segs,
        "items": rows,
    }


# =============================================================================
# paired bootstrap — "2%p 미만은 노이즈" 같은 임의 규칙 대신 신뢰구간으로 판단
# =============================================================================
def paired_bootstrap(base_rows: List[dict], other_rows: List[dict], key: str,
                     n_iter: int = BOOTSTRAP_N) -> Dict[str, float]:
    a = {r["id"]: r[key] for r in base_rows}
    b = {r["id"]: r[key] for r in other_rows}
    ids = [i for i in a if i in b]
    diffs = [b[i] - a[i] for i in ids]
    obs = statistics.mean(diffs)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(diffs)
    means = []
    for _ in range(n_iter):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter) - 1]
    p_better = sum(1 for m in means if m > 0) / n_iter
    return {"delta": obs, "ci_low": lo, "ci_high": hi, "p_better": p_better}


# =============================================================================
# 출력
# =============================================================================
def print_single(res: dict, k_values: Sequence[int], budget: int) -> None:
    ko = OFFICIAL_K if OFFICIAL_K in k_values else max(k_values)
    print(f"\n== {res['run_name']} ==  retriever={res.get('retriever')}")
    if res.get("config"):
        print(f"   config: {json.dumps(res['config'], ensure_ascii=False)}")

    print(f"\n[근거 확보]  USR = UnionSpanRecall,  CH = CoverageHit")
    cols = [f"USR@{k}" for k in k_values] + [f"CH@{ko}-0.5", f"CH@{ko}-0.8", f"EvHit@{ko}"]
    hdr = f"{'세그먼트':16s} {'n':>4s} " + " ".join(f"{c:>10s}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for name, m in res["metrics_by_segment"].items():
        vals = [m[f"usr@{k}"] for k in k_values] + [
            m[f"covhit@{ko}-0.5"], m[f"covhit@{ko}-0.8"], m[f"evhit@{ko}"]]
        tag = "  ※참고" if m["n"] < 10 and name != "전체" else ""
        print(f"{name:16s} {m['n']:>4d} " + " ".join(f"{v:>10.3f}" for v in vals) + tag)

    print(f"\n[순위 품질 · 진단]")
    cols2 = [f"MRR@{ko}", f"nDCG@{ko}", f"DocHit@{ko}", f"DocPrec@{ko}", f"P@{ko}"]
    hdr2 = f"{'세그먼트':16s} {'n':>4s} " + " ".join(f"{c:>10s}" for c in cols2)
    print(hdr2); print("-" * len(hdr2))
    for name, m in res["metrics_by_segment"].items():
        vals = [m[f"mrr@{ko}"], m[f"ndcg@{ko}"], m[f"dochit@{ko}"],
                m[f"docprec@{ko}"], m[f"precision@{ko}"]]
        tag = "  ※참고" if m["n"] < 10 and name != "전체" else ""
        print(f"{name:16s} {m['n']:>4d} " + " ".join(f"{v:>10.3f}" for v in vals) + tag)

    m = res["metrics"]
    print(f"\n[실서비스 — {budget}자 예산, {BUDGET_TEXT_FIELD} 기준]")
    print(f"  UnionSpanRecall@{budget}c = {m[f'usr@{budget}c']:.3f}"
          f"   CoverageHit@{budget}c-0.8 = {m[f'covhit@{budget}c-0.8']:.3f}"
          f"   ChunksUsed = {m[f'chunks@{budget}c']:.2f}"
          f"   (평균 {m[f'chars@{budget}c']:.0f}자)")
    print(f"\n※ 주 지표는 UnionSpanRecall@K. EvidenceHit 은 근거 일부만 찾아도 1이라 관대함.")
    print(f"※ n<10 세그먼트는 참고용. 결론 근거로 쓰지 말 것.")


def print_compare(results: List[dict], k_values: Sequence[int], budget: int) -> None:
    ko = OFFICIAL_K if OFFICIAL_K in k_values else max(k_values)
    base = results[0]
    print(f"\n== 비교 (기준선: {base['run_name']}) ==")
    print(f"   Δ 는 paired bootstrap {BOOTSTRAP_N}회 · 95% 신뢰구간. CI 가 0 을 포함하면 유의하지 않음\n")

    for seg in ["전체", "src:v1", "src:v2_prose"]:
        if seg not in base["metrics_by_segment"]:
            continue
        bm = base["metrics_by_segment"][seg]
        brows = [r for r in base["items"] if _in_seg(r, seg)]
        print(f"[{seg}]  n={bm['n']}")
        hdr = (f"  {'run':26s} " + " ".join(f"{'USR@'+str(k):>8s}" for k in k_values)
               + f" {'CH@'+str(ko)+'-.8':>9s} {'MRR@'+str(ko):>8s} {'USR@bud':>8s}"
               + f" | {'ΔUSR@'+str(ko):>10s} {'95% CI':>18s} {'P(>0)':>7s}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in results:
            m = r["metrics_by_segment"].get(seg)
            if not m:
                continue
            row = (f"  {r['run_name'][:26]:26s} "
                   + " ".join(f"{m[f'usr@{k}']:>8.3f}" for k in k_values)
                   + f" {m[f'covhit@{ko}-0.8']:>9.3f} {m[f'mrr@{ko}']:>8.3f}"
                   + f" {m[f'usr@{budget}c']:>8.3f}")
            if r is base:
                row += f" | {'(기준선)':>12s}"
            else:
                orows = [x for x in r["items"] if _in_seg(x, seg)]
                bs = paired_bootstrap(brows, orows, f"usr@{ko}")
                sig = "" if bs["ci_low"] <= 0 <= bs["ci_high"] else "  *"
                row += (f" | {bs['delta']*100:>+9.1f}p"
                        f" [{bs['ci_low']*100:>+6.1f}, {bs['ci_high']*100:>+6.1f}]"
                        f" {bs['p_better']:>7.3f}{sig}")
            print(row)
        print()
    print("  * = 95% 신뢰구간이 0 을 포함하지 않음 (차이가 유의)")


def _in_seg(row: dict, seg: str) -> bool:
    if seg == "전체":
        return True
    kind, val = seg.split(":", 1)
    return {"src": row["eval_source"], "type": row["gold_type"], "topic": row["topic"]}[kind] == val


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", help="채점할 run 파일")
    ap.add_argument("--compare", nargs="+", help="비교할 run 파일들 (첫 번째가 기준선)")
    ap.add_argument("--k", nargs="+", type=int, default=DEFAULT_K)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="실서비스 문자 예산")
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate-only", action="store_true")
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
            results.append(score_run(run, testset, corpus, k_values, a.budget))

    if a.validate_only:
        print("\n검증만 수행했습니다.")
        return

    out_dir = Path(a.out) if a.out else PROJECT_ROOT / "results_retriever"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        f = out_dir / f"score_{res['run_name']}.json"
        f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print_single(res, k_values, a.budget)
        print(f"saved → {f}")

    if len(results) > 1:
        print_compare(results, k_values, a.budget)


if __name__ == "__main__":
    main()
