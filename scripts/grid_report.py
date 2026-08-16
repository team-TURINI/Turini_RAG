"""모델 × 프롬프트 격자 리포트 — 스펙 준수율·비용·지연을 한 표로.

1단계(모델 선정)의 주 산출물이다. correctness/groundedness 는 조건 A·A2 에서 이미
전 모델 만점이 나와 변별이 안 되므로, **답변 스펙 준수율**이 모델을 가르는 지표다.

------------------------------------------------------------------
읽는 법

  준수율      answer_spec.py 의 규칙 준수 비율. **주 지표**
  v2-v1       프롬프트로 지시했을 때 얼마나 따라오나 = **지시 이행 능력**
              실서비스에서 필요한 것은 "시키지 않아도 알아서" 가 아니라 "시키면 지킴" 이다
  출력토큰     비용. Gemini 는 thinking 토큰이 합산되어 있음
  지연        실시간 응답 가능성

실행
  python scripts/grid_report.py                       # 규칙 기반만 (무료)
  python scripts/grid_report.py --detail              # 규칙별 준수율까지
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_spec import RULE_LABELS, check  # noqa: E402

GEN_DIR = PROJECT_ROOT / "gen_runs"

# 표시 이름 — 파일 태그를 사람이 읽는 이름으로
MODEL_LABEL = {
    "41nano": "gpt-4.1-nano", "41mini": "gpt-4.1-mini",
    "54nano": "gpt-5.4-nano", "54mini": "gpt-5.4-mini",
    "haiku45": "claude-haiku-4.5", "sonnet45": "claude-sonnet-4.5",
    "g35lite": "gemini-3.5-flash-lite", "g35flash": "gemini-3.5-flash",
}
PROVIDER = {"41": "OpenAI", "54": "OpenAI", "ha": "Anthropic",
            "so": "Anthropic", "g3": "Google"}


def load_runs() -> dict:
    """{(모델태그, 프롬프트): 측정치}"""
    out = {}
    for p in sorted(GEN_DIR.glob("grid_*.json")):
        m = re.match(r"grid_(.+)_(v\d+)\.json$", p.name)
        if not m:
            continue
        tag, pv = m.groups()
        gen = json.loads(p.read_text(encoding="utf-8"))
        items = gen["items"]
        rows = [check(it["answer"]) for it in items]
        comp = [r["compliance"] for r in rows if r["compliance"] is not None]
        rec = {
            "n": len(items),
            "compliance": statistics.mean(comp) if comp else float("nan"),
            "chars": statistics.mean(r["chars"] for r in rows),
            "in_tok": statistics.mean(it["input_tokens"] for it in items),
            "out_tok": statistics.mean(it["output_tokens"] for it in items),
            "latency": statistics.mean(it["latency_s"] for it in items),
            "truncated": sum(1 for it in items if it.get("truncated")),
            "empty": sum(1 for it in items if not (it["answer"] or "").strip()),
            "rules": {k: [r[k] for r in rows if r[k] is not None] for k in RULE_LABELS},
        }
        out[(tag, pv)] = rec
    return out


def show_failures(run_tag: str, rule: str | None, n: int) -> None:
    """미준수 답변을 읽는다 — 프롬프트를 고치려면 무엇이 왜 틀렸는지 봐야 한다.

    점수는 "졌다"만 알려주고 "왜 졌는지"는 안 알려준다. 다음 프롬프트 아이디어는
    실제 답변에서 나온다. 이 함수가 그 출발점이다. API 호출 없이 무료로 돈다.
    """
    p = GEN_DIR / (run_tag if run_tag.endswith(".json") else f"grid_{run_tag}.json")
    if not p.exists():
        avail = sorted(x.stem.replace("grid_", "") for x in GEN_DIR.glob("grid_*.json"))
        raise SystemExit(f"없는 런: {run_tag}\n  가능: {avail}")

    items = json.loads(p.read_text(encoding="utf-8"))["items"]
    rows = [(it, check(it["answer"])) for it in items]

    if rule:
        if rule not in RULE_LABELS:
            raise SystemExit(f"없는 규칙: {rule}\n  가능: {list(RULE_LABELS)}")
        bad = [(it, r) for it, r in rows if r.get(rule) is False]
        print(f"[{p.stem}] '{RULE_LABELS[rule]}' 미준수 {len(bad)}/{len(rows)}건\n")
    else:
        bad = [(it, r) for it, r in rows
               if any(r.get(k) is False for k in RULE_LABELS)]
        print(f"[{p.stem}] 하나 이상 미준수 {len(bad)}/{len(rows)}건\n")

    for it, r in bad[:n]:
        miss = [RULE_LABELS[k] for k in RULE_LABELS if r.get(k) is False]
        print(f"--- {it['id']} | {r['chars']}자 | 미준수: {', '.join(miss)}")
        print(f"Q : {it['question']}")
        print(f"A : {it['answer']}")
        if r["bad_numbers"]:
            print(f"    ⚠ 근거 없는 숫자: {r['bad_numbers']}")
        print()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="규칙별 준수율까지 출력")
    ap.add_argument("--show", metavar="RUN",
                    help="특정 런의 미준수 답변을 읽는다 (예: 41mini_v2). 프롬프트 튜닝의 출발점")
    ap.add_argument("--rule", default=None,
                    help=f"--show 에서 볼 규칙. 미지정 시 전체. 가능: {list(RULE_LABELS)}")
    ap.add_argument("--n", type=int, default=5, help="--show 로 출력할 건수")
    a = ap.parse_args()

    if a.show:
        show_failures(a.show, a.rule, a.n)
        return

    runs = load_runs()
    if not runs:
        raise SystemExit("gen_runs/grid_*.json 이 없습니다")
    tags = sorted({t for t, _ in runs}, key=lambda t: list(MODEL_LABEL).index(t)
                  if t in MODEL_LABEL else 99)

    print(f"모델 × 프롬프트 격자 — 스펙 준수율 (dev {max(r['n'] for r in runs.values())}문항)\n")
    hdr = (f"{'모델':22s} {'제공':10s} {'v1':>7s} {'v2':>7s} {'v2-v1':>7s} "
           f"{'글자':>5s} {'출력tok':>7s} {'지연':>6s} {'잘림':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for t in tags:
        r1, r2 = runs.get((t, "v1")), runs.get((t, "v2"))
        if not r1 and not r2:
            continue
        prov = PROVIDER.get(t[:2], "?")
        # 미완료 런을 다른 프롬프트 값으로 채우면 오독하게 되므로 '-' 로 비워 둔다
        c1 = f"{r1['compliance']:>7.3f}" if r1 else f"{'-':>7s}"
        c2 = f"{r2['compliance']:>7.3f}" if r2 else f"{'-':>7s}"
        dv = (f"{r2['compliance'] - r1['compliance']:>+7.3f}" if r1 and r2 else f"{'-':>7s}")
        r = r2 or r1
        print(f"{MODEL_LABEL.get(t, t):22s} {prov:10s} {c1} {c2} {dv} "
              f"{r['chars']:>5.0f} {r['out_tok']:>7.0f} {r['latency']:>5.1f}s "
              f"{r['truncated']:>5d}")

    if a.detail:
        print("\n\n[v2 기준 규칙별 준수율]")
        hdr2 = f"{'모델':22s} " + " ".join(f"{lab[:8]:>9s}" for lab in RULE_LABELS.values())
        print(hdr2)
        print("-" * len(hdr2))
        for t in tags:
            r = runs.get((t, "v2"))
            if not r:
                continue
            cells = []
            for k in RULE_LABELS:
                v = r["rules"][k]
                cells.append(f"{sum(v) / len(v):>9.3f}" if v else f"{'-':>9s}")
            print(f"{MODEL_LABEL.get(t, t):22s} " + " ".join(cells))

    miss = [f"{MODEL_LABEL.get(t, t)}/{pv}" for t in tags for pv in ("v1", "v2")
            if (t, pv) not in runs]
    if miss:
        print(f"\n⚠ 아직 없는 런: {miss}")
    empty = [(t, pv) for (t, pv), r in runs.items() if r["empty"]]
    if empty:
        print(f"⚠ 빈 답변이 있는 런: {empty}")


if __name__ == "__main__":
    main()
