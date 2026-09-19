"""생성 실험 결과를 한 표로 모아 출력한다.

지표가 세 군데에 흩어져 있어 매번 손으로 합치게 된다. 그걸 없애는 스크립트다.

  gen_runs/<name>.json          생성 결과      토큰 · 지연 · 컨텍스트 크기
  gen_runs/score_<name>.json    eval_generation  correctness · 회피율
  results_triad/triad_<name>.json  eval_rag_triad  faithfulness · AR · CR
  (answer_spec 준수율은 생성 결과의 answer 로 즉석 계산 — 무료)

없는 파일은 "—" 로 비워 두고 있는 것만 채운다. 그래서 채점을 아직 안 돌린
구성도 섞어서 볼 수 있다.

실행:
  python scripts/report_gen_metrics.py base_v1_dev base_v2_dev guide_v5r2_dev
  python scripts/report_gen_metrics.py --all
  python scripts/report_gen_metrics.py --all --csv out.csv
  python scripts/report_gen_metrics.py base_v1_dev base_v2_dev --diff   # 첫 번째 대비 증감
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

GEN_DIR = PROJECT_ROOT / "gen_runs"
TRIAD_DIR = PROJECT_ROOT / "results_triad"

from answer_spec import check  # noqa: E402

# (표시명, 출처, 키)  출처: gen / score / triad / spec
ROWS = [
    ("correctness",        "score", "correctness"),
    ("  만점 비율",         "score", "correct_rate"),
    ("faithfulness(micro)", "triad", "faithfulness_micro"),
    ("  미근거 주장률",      "triad", "unsupported_claim_rate"),
    ("  주장 수/문항",       "triad", "avg_claims"),
    ("answer_relevance",   "triad", "answer_relevance"),
    ("  회피 제외",         "triad", "answer_relevance_committal"),
    ("context_relevance",  "triad", "context_relevance"),
    ("회피율",              "score", "refusal_rate"),
    ("준수율(전체)",         "spec",  "compliance"),
    ("  길이 150~250",      "spec",  "len_ok"),
    ("  권유 아님 명시",     "spec",  "no_solicit_ok"),
    ("  위험 고지",         "spec",  "risk_notice_ok"),
    ("  숫자 단위·기준",     "spec",  "unit_ok"),
    ("  서술형",            "spec",  "prose_ok"),
    ("  내부사정 미노출",    "spec",  "no_meta_ok"),
    ("예시문구 반복",        "spec",  "sample_verbatim"),
    ("평균 글자수",          "spec",  "chars"),
    ("입력 토큰",           "gen",   "avg_input_tokens"),
    ("출력 토큰",           "gen",   "avg_output_tokens"),
]
INT_KEYS = {"chars", "avg_input_tokens", "avg_output_tokens", "avg_claims"}


def load(name: str) -> Dict[str, Optional[dict]]:
    def _read(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return {
        "gen":   _read(GEN_DIR / f"{name}.json"),
        "score": _read(GEN_DIR / f"score_{name}.json"),
        "triad": _read(TRIAD_DIR / f"triad_{name}.json"),
    }


def spec_metrics(gen: Optional[dict]) -> Dict[str, float]:
    if not gen:
        return {}
    rows = [check(x["answer"]) for x in gen["items"]]
    out = {}
    for k in ("compliance", "len_ok", "no_solicit_ok", "risk_notice_ok",
              "unit_ok", "prose_ok", "no_meta_ok", "sample_verbatim", "chars"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = statistics.mean(float(v) for v in vals)
    return out


def value(src: Dict, spec: Dict, where: str, key: str):
    if where == "spec":
        return spec.get(key)
    d = src.get(where)
    if not d:
        return None
    if where == "gen":
        return (d.get("summary") or {}).get(key)
    return (d.get("metrics") or {}).get(key)


def fmt(v, key: str) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}" if key in INT_KEYS else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="gen_runs/<name>.json 의 <name>")
    ap.add_argument("--all", action="store_true", help="gen_runs 의 모든 결과")
    ap.add_argument("--diff", action="store_true", help="첫 번째 구성 대비 증감도 출력")
    ap.add_argument("--csv", default=None, help="CSV 로도 저장")
    a = ap.parse_args()

    names: List[str] = a.names
    if a.all:
        names = sorted(p.stem for p in GEN_DIR.glob("*.json")
                       if not p.stem.startswith(("score_", "_")))
    if not names:
        ap.error("구성 이름을 주거나 --all 을 쓰세요")

    src = {n: load(n) for n in names}
    spec = {n: spec_metrics(src[n]["gen"]) for n in names}
    missing = [n for n in names if src[n]["gen"] is None]
    if missing:
        print(f"[warn] 생성 결과 없음: {', '.join(missing)}\n")

    w = max(13, max(len(n) for n in names) + 1)
    print(f"{'지표':22s}" + "".join(f"{n[:w-1]:>{w}s}" for n in names))
    print("-" * (22 + w * len(names)))
    table = []
    for lab, where, key in ROWS:
        vals = [value(src[n], spec[n], where, key) for n in names]
        if all(v is None for v in vals):
            continue
        print(f"{lab:22s}" + "".join(f"{fmt(v, key):>{w}s}" for v in vals))
        table.append((lab, key, vals))

    if a.diff and len(names) > 1:
        base = names[0]
        print(f"\n[{base} 대비 증감]")
        print(f"{'지표':22s}" + "".join(f"{n[:w-1]:>{w}s}" for n in names[1:]))
        print("-" * (22 + w * (len(names) - 1)))
        for lab, key, vals in table:
            if vals[0] is None:
                continue
            cells = []
            for v in vals[1:]:
                cells.append("—" if v is None else
                             (f"{v - vals[0]:+.1f}" if key in INT_KEYS else f"{v - vals[0]:+.4f}"))
            print(f"{lab:22s}" + "".join(f"{c:>{w}s}" for c in cells))

    print("\n출처  correctness·회피율 = eval_generation.py / faithfulness·relevance = eval_rag_triad.py")
    print("      준수율·글자수 = answer_spec.py (무료, 즉석 계산) / 토큰 = 생성 결과")
    print("'—' 는 해당 채점을 아직 안 돌린 것입니다.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["지표"] + names)
            for lab, key, vals in table:
                wr.writerow([lab.strip()] + [("" if v is None else round(v, 4)) for v in vals])
        print(f"\n[csv] {a.csv}")


if __name__ == "__main__":
    main()
