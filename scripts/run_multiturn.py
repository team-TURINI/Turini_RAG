"""멀티턴 시나리오 실행 — 대화 묶음을 세션에 통과시켜 턴별 판단·검색·답변을 저장한다.

run_pipeline.py 의 멀티턴판. 시나리오마다 새 세션(새 state)으로 시작하고 FinalPipeline 은 공유한다.
실제 OpenAI(재작성·요약·생성)·Cohere(리랭커)를 호출한다.

  results_multiturn/<tag>/
    turns.json       시나리오·턴별: 재작성 질문, route, is_followup, needs_portfolio, 국가 판정,
                     생성 자료 출처, 답변, 요약 갱신 여부, 그 시점의 요약문
    transcript.md    사람이 읽는 대화록

실행:
  python scripts/run_multiturn.py --tag smoke                                  # scripts/scenarios/multiturn_smoke.json
  python scripts/run_multiturn.py --tag smoke --scenarios my.json --only followup_bond_kr_us
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from core.jurisdiction import detect  # noqa: E402
from core.pipeline import FinalPipeline  # noqa: E402
from multiturn.session import build_session  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "results_multiturn"
DEFAULT_SCENARIOS = PROJECT_ROOT / "scripts" / "scenarios" / "multiturn_smoke.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--only", nargs="*", help="시나리오 이름으로 골라 실행")
    ap.add_argument("--no-summary", action="store_true", help="요약 압축 끄기")
    a = ap.parse_args()

    if not cfg.OPENAI_API_KEY or not cfg.COHERE_API_KEY:
        raise SystemExit("[fatal] .env 에 OPENAI_API_KEY 와 COHERE_API_KEY 가 필요합니다")
    spec = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))
    scenarios = [s for s in spec["scenarios"] if not a.only or s["name"] in a.only]
    if not scenarios:
        raise SystemExit("[fatal] 실행할 시나리오가 없습니다")

    pipeline = FinalPipeline(verbose=True)
    corpus = pipeline.corpus
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results, md = [], [f"# 멀티턴 실행 — {a.tag}", ""]
    t0 = time.time()
    for sc in scenarios:
        sess = build_session(pipeline=pipeline, portfolio=sc.get("portfolio"), summarize=not a.no_summary)
        print(f"\n=== {sc['name']}  ({sc.get('purpose', '')})")
        md += [f"## {sc['name']}", f"_{sc.get('purpose', '')}_", ""]
        turns = []
        for q in sc["turns"]:
            r = sess.ask(q, scenario_name=sc["name"])
            j = detect(r.rewrite.retrieval_query) if r.rewrite.route == "rag" else None
            srcs = []
            if r.rag is not None:
                for cid in r.rag.reranked_ids[: cfg.FINAL_TOP_K_GEN]:
                    c = corpus.get(cid, {})
                    srcs.append(f"{c.get('source_name', '?')}/{c.get('title', '?')[:24]}")
            turns.append({
                "turn": r.turn_index, "question": q,
                "retrieval_query": r.rewrite.retrieval_query, "route": r.rewrite.route,
                "is_followup": r.rewrite.is_followup, "needs_portfolio": r.rewrite.needs_portfolio,
                "jurisdiction": j["jurisdiction"] if j else None, "signals_us": j["signals_us"] if j else [],
                "status": r.status, "profile": r.profile,
                "context_chunk_ids": list(r.rag.reranked_ids[: cfg.FINAL_TOP_K_GEN]) if r.rag else [],
                "context_sources": srcs, "answer": r.answer,
                "input_tokens": r.generation.input_tokens if r.generation else None,
                "summary_updated": r.summary_updated, "summary_after": sess.state.conversation_summary,
            })
            tag = f"[{r.rewrite.route}{'/followup' if r.rewrite.is_followup else ''}"
            tag += f"{'/portfolio' if r.rewrite.needs_portfolio else ''}{'/' + j['jurisdiction'] if j else ''}] {r.status}"
            print(f"  U: {q}\n     → 재작성: {r.rewrite.retrieval_query or '-'}   {tag}")
            print(f"  A: {r.answer[:140]}{'…' if len(r.answer) > 140 else ''}")
            if r.summary_updated:
                print(f"     (요약 갱신) {sess.state.conversation_summary[:100]}…")
            md += [f"**U{r.turn_index}:** {q}", f"- 재작성: `{r.rewrite.retrieval_query or '-'}` {tag}",
                   f"- 자료: {', '.join(srcs) or '-'}", f"", f"**A{r.turn_index}:** {r.answer}", ""]
            if r.summary_updated:
                md += [f"> 요약 갱신: {sess.state.conversation_summary}", ""]
        results.append({"name": sc["name"], "purpose": sc.get("purpose"), "has_portfolio": sc.get("portfolio") is not None,
                        "turns": turns, "final_summary": sess.state.conversation_summary,
                        "final_messages": sess.transcript()})

    (out_dir / "turns.json").write_text(json.dumps(
        {"tag": a.tag, "scenarios_file": str(a.scenarios), "n_scenarios": len(results),
         "pipeline": pipeline.config_snapshot(), "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "transcript.md").write_text("\n".join(md), encoding="utf-8")
    n_turns = sum(len(r["turns"]) for r in results)
    print(f"\n[done] {len(results)}시나리오 {n_turns}턴 / {(time.time() - t0) / 60:.1f}분  → {out_dir}")


if __name__ == "__main__":
    main()
