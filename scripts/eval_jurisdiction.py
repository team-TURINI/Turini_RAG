"""관할(국가) 지표 — 검색이 다른 나라 문서를 얼마나 섞는지, 생성이 그걸 얼마나 따라가는지.

검색 (ret_*.json)
  JP@k  Jurisdiction Precision@k = 상위 k 중 허용 관할 청크 비율 (KR 문항 → KR·GLOBAL 허용, US 문항 → 전부 허용)
  leak  KR 문항인데 상위 k 에 US 청크가 1개라도 있는 문항 수
  필터를 켠 run 은 당연히 1.0 — 이 지표의 용도는 **필터 전 run(final_v1) 이 얼마나 새고 있었는지** 와 라우터 오판 검출

생성 (generation.json)
  foreign   KR 문항 답변에 미국 제도 용어(munis·12b-1·FDIC·SEC·401(k)·IRA·뮤추얼펀드…) 등장 문항 수
  kr_mark   답변에 "한국 기준" 류 표현이 있는 문항 수 (v2j 가드레일이 지시한 것)
  refusal   회피 정답 문항(v3 expected_behavior=refusal)에서 실제로 회피한 수

run 필터 (--filter-run)
  회피 문항을 뺀 run 파일을 만든다 → eval_retriever_coverageaware.py --testset *_v3_retrieval.json 에 넣는다

실행
  python scripts/eval_jurisdiction.py --run results_e2e/final_v1/ret_reranked.json
  python scripts/eval_jurisdiction.py --gen results_e2e/final_v2/generation.json
  python scripts/eval_jurisdiction.py --filter-run results_e2e/final_v1/ret_reranked.json   → *_r<문항수>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import config as cfg  # noqa: E402
from core.jurisdiction import JurisdictionMap, allowed_for  # noqa: E402
from answer_spec import REFUSAL  # noqa: E402

FOREIGN = re.compile(r"munis?|무니스|12b-1|FDIC|(?<![A-Za-z])SEC(?![A-Za-z])|401\(?k\)?|(?<![A-Za-z])IRA(?![A-Za-z])|"
                     r"뮤추얼\s?펀드|미국|연방|달러", re.IGNORECASE)
KR_MARK = re.compile(r"한국\s?기준|국내\s?기준|한국에서는|우리나라")


def load_testset(path: Path):
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def eval_run(run_path: Path, ts, jm: JurisdictionMap, ks=(3, 10, 20)):
    run = json.loads(run_path.read_text(encoding="utf-8"))
    jp = {k: [] for k in ks}; leak = {k: 0 for k in ks}; n_kr = 0
    for it in run["items"]:
        r = ts.get(it["id"])
        if r is None:
            continue
        allowed = allowed_for(r.get("expected_jurisdiction", "KR"))
        if allowed is None:                       # US 문항 — 전부 허용
            for k in ks: jp[k].append(1.0)
            continue
        n_kr += 1
        for k in ks:
            top = it["retrieved"][:k]
            ok = [jm.of_chunk(c) in allowed for c in top]
            jp[k].append(sum(ok) / len(top) if top else 0.0)
            leak[k] += any(jm.of_chunk(c) == "US" for c in top)
    print(f"[검색] {run_path}  문항 {len(jp[ks[0]])} (KR 기대 {n_kr})")
    for k in ks:
        print(f"  JP@{k:<2d} {sum(jp[k])/len(jp[k]):.4f}   US 누출 문항 {leak[k]}/{n_kr}")
    return {f"JP@{k}": round(sum(jp[k]) / len(jp[k]), 4) for k in ks} | {f"leak@{k}": leak[k] for k in ks}


def eval_gen(gen_path: Path, ts):
    g = json.loads(gen_path.read_text(encoding="utf-8"))
    foreign, kr_mark, n_kr = [], 0, 0
    ref_total, ref_hit = 0, 0
    for it in g["items"]:
        r = ts.get(it["id"])
        if r is None:
            continue
        a = it["answer"]
        if r.get("expected_behavior") == "refusal":
            ref_total += 1; ref_hit += bool(REFUSAL.search(a))
        if r.get("expected_jurisdiction", "KR") == "KR":
            n_kr += 1
            m = FOREIGN.findall(a)
            if m: foreign.append((it["id"], sorted(set(m))))
            kr_mark += bool(KR_MARK.search(a))
    print(f"[생성] {gen_path}  KR 문항 {n_kr}")
    print(f"  미국 제도 용어 등장  {len(foreign)}/{n_kr}   " + (", ".join(f"{i}{t}" for i, t in foreign[:8]) if foreign else ""))
    print(f"  '한국 기준' 명시     {kr_mark}/{n_kr}")
    if ref_total:
        print(f"  회피 정답 문항 회피  {ref_hit}/{ref_total}")
    return {"foreign": len(foreign), "foreign_items": foreign, "kr_mark": kr_mark, "n_kr": n_kr,
            "refusal_hit": ref_hit, "refusal_total": ref_total}


def filter_run(run_path: Path, ts) -> Path:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    keep = {i for i, r in ts.items() if r.get("retrieval_eval", True)}
    run["items"] = [it for it in run["items"] if it["id"] in keep]
    n = len(run["items"])
    run["run_name"] = run.get("run_name", run_path.stem) + f"_r{n}"
    out = run_path.with_name(run_path.stem + f"_r{n}.json")
    out.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    print(f"[필터] {len(run['items'])}문항 → {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--gen", nargs="*", default=[])
    ap.add_argument("--filter-run", nargs="*", default=[])
    ap.add_argument("--testset", default=str(cfg.FINAL_TESTSET))
    a = ap.parse_args()
    ts = load_testset(Path(a.testset))
    jm = JurisdictionMap()
    for p in a.run:
        eval_run(Path(p), ts, jm)
    for p in a.gen:
        eval_gen(Path(p), ts)
    for p in a.filter_run:
        filter_run(Path(p), ts)


if __name__ == "__main__":
    main()
