"""생성 답변 채점 — 리트리버 구성별 최종 답변 품질 비교.

검색 지표(UnionSpanRecall 등)는 **대리 지표**다. 정작 알고 싶은 것은
"검색이 좋아지면 답변도 좋아지는가" 이며, 그건 답변을 직접 채점해야 알 수 있다.

------------------------------------------------------------------
지표 두 가지

  [규칙 기반 — 무료·결정적]
  RefOverlap    참조답변(testset 의 answer)의 문장 단위를 생성답변이 얼마나 담았나
  Refusal       "제공된 자료에서는 확인할 수 없습니다" 로 회피한 비율
                → 검색이 나쁠수록 올라간다. 검색 품질의 간접 신호

  [LLM 판정 — 유료·비결정적]
  Correctness   참조답변 대비 사실이 맞는가 (0~2)
  Groundedness  주어진 자료에 근거하는가, 지어내지 않았는가 (0~2)
                → 판정 시 **자료 원문을 함께 보여준다.** 자료 없이 물으면 "지어냈나"가
                  아니라 "그럴듯한가"를 재게 된다
                → RAG 에서 가장 중요. 검색이 나쁘면 모델이 지어내기 시작한다

  판정 모델은 생성 모델과 분리하는 것이 원칙이나, 이번 실험은 **구성 간 상대 비교**가
  목적이라 동일 모델을 써도 편향이 양쪽에 동일하게 걸린다.

실행:
  python scripts/eval_generation.py --gen gen_runs/gen_xxx.json                 # 규칙 기반만 (무료)
  python scripts/eval_generation.py --gen gen_runs/gen_xxx.json --judge         # LLM 판정 포함
  python scripts/eval_generation.py --compare gen_runs/a.json gen_runs/b.json --judge
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# config 를 import 하면 .env 가 로드된다 (판정용 OPENAI_API_KEY 가 여기 있음)
from config import OPENAI_API_KEY
# 레이트 리밋 재시도는 생성기 것을 재사용한다. 판정 모델(gpt-4.1)은 TPM 30,000 이라
# 107문항 × 여러 런을 연속으로 돌리면 반드시 429 가 난다 (실측).
from core.generator import _with_retry

CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70.jsonl"
CTX_FIELD = "embedding_text"

REFUSAL_PAT = re.compile(r"제공된\s*자료에서는?\s*해당\s*내용을\s*확인할\s*수\s*없")
# 판정 모델은 **실험 후보에 없는 모델**로 고정한다.
# 후보(gpt-4.1-nano/mini, gpt-5.4-nano/mini)와 같은 모델로 채점하면 자기 답변을
# 자기가 채점하게 되어 자기선호 편향이 든다. gpt-4.1(표준)은 후보에서 빠졌으므로 여기 쓴다.
JUDGE_MODEL = "gpt-4.1-2025-04-14"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260805

JUDGE_PROMPT = """당신은 RAG 시스템의 답변을 채점하는 평가자입니다.
아래 [질문], [참조답변], [생성답변] 을 보고 두 항목을 채점하세요.

# 채점 기준
correctness — 생성답변이 참조답변과 비교해 사실적으로 맞는가
  2 = 참조답변의 핵심 내용을 정확히 담음
  1 = 부분적으로 맞으나 핵심 일부가 빠지거나 부정확
  0 = 틀렸거나, 답을 못 함(회피 포함)

groundedness — 생성답변이 [자료] 에 근거하는가, 지어낸 내용을 포함하는가
  2 = 전부 자료로 뒷받침됨
  1 = 일부 불확실하거나 자료를 넘어선 서술 있음
  0 = 자료에 없는 수치·사실이 명백히 있음
  ※ "확인할 수 없습니다" 라고 회피한 답변은 지어내지 않았으므로 groundedness=2
  ※ 참조답변에 있어도 [자료] 에 없으면 근거 없는 것으로 본다.
     반대로 참조답변에 없어도 [자료] 에 있으면 근거 있는 것이다.
  ※ 아래는 **회사 정책상 반드시 넣도록 지시된 정형 문구**이며 상품에 대한 새로운 사실
     주장이 아니다. [자료]에 없더라도 **감점하지 말 것.**
       - 투자 위험 고지 ("원금 손실이 발생할 수 있습니다" 등)
       - 면책 문구 ("투자 권유가 아닌 정보 제공입니다" 등)
     단, 자료와 **모순되는** 경우(원금이 보장된다고 적힌 상품에 손실 위험을 붙이는 등)
     는 감점한다.

# 출력 형식 (JSON 만, 다른 말 금지)
{"correctness": 0|1|2, "groundedness": 0|1|2, "reason": "한 문장"}"""


def norm(t: str) -> str:
    return "".join((t or "").split())


def ref_overlap(reference: str, answer: str) -> float:
    """참조답변을 문장 단위로 쪼개, 생성답변이 담은 비율."""
    parts = re.split(r"[.\n]|다\.", reference)
    units = [norm(p) for p in parts if len(norm(p)) >= 10]
    if not units:
        units = [norm(reference)]
    a = norm(answer)
    return sum(1 for u in units if u in a) / len(units)


def load_corpus() -> Dict[str, dict]:
    return {c["chunk_id"]: c for c in
            (json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip())}


def judge_one(client, question: str, reference: str, answer: str, context: str = "") -> dict:
    r = _with_retry(lambda: client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content":
                (f"[자료]\n{context}\n\n" if context else "")
                + f"[질문]\n{question}\n\n[참조답변]\n{reference}\n\n[생성답변]\n{answer}"},
        ],
    ))
    try:
        d = json.loads(r.choices[0].message.content)
        return {"correctness": int(d.get("correctness", 0)),
                "groundedness": int(d.get("groundedness", 0)),
                "reason": d.get("reason", "")}
    except Exception as e:
        return {"correctness": 0, "groundedness": 0, "reason": f"(판정 파싱 실패: {e})"}


def score(gen: dict, use_judge: bool) -> dict:
    rows = []
    client = None
    if use_judge:
        from openai import OpenAI
        if not OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
        client = OpenAI(api_key=OPENAI_API_KEY)

    # groundedness 는 "자료에 있었나"를 보는 지표라 자료 원문 없이는 판정할 수 없다.
    # 자료 없이 물으면 '그럴듯한가'를 재게 된다. 생성 결과에 chunk_id 가 있으면 복원한다.
    corpus = load_corpus() if (use_judge and gen["items"]
                               and gen["items"][0].get("context_chunk_ids")) else None
    if use_judge and corpus is None:
        print("  ⚠ context_chunk_ids 가 없어 자료 없이 판정함 — groundedness 는 참고용")

    t0 = time.time()
    for n, it in enumerate(gen["items"], start=1):
        rec = {
            "id": it["id"],
            "ref_overlap": round(ref_overlap(it["reference"], it["answer"]), 4),
            "refusal": 1.0 if REFUSAL_PAT.search(it["answer"]) else 0.0,
            "output_tokens": it["output_tokens"],
        }
        if use_judge:
            ctx = ("\n\n".join(corpus[c][CTX_FIELD] for c in it["context_chunk_ids"]
                               if c in corpus) if corpus else "")
            j = judge_one(client, it["question"], it["reference"], it["answer"], ctx)
            rec.update({"correctness": j["correctness"], "groundedness": j["groundedness"],
                        "judge_reason": j["reason"]})
            if n % 20 == 0:
                el = time.time() - t0
                print(f"  판정 {n}/{len(gen['items'])} ({el/60:.1f}분)", flush=True)
        rows.append(rec)

    m = {"n": len(rows),
         "ref_overlap": round(statistics.mean(r["ref_overlap"] for r in rows), 4),
         "refusal_rate": round(statistics.mean(r["refusal"] for r in rows), 4),
         "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows), 1)}
    if use_judge:
        m["correctness"] = round(statistics.mean(r["correctness"] for r in rows), 4)
        m["groundedness"] = round(statistics.mean(r["groundedness"] for r in rows), 4)
        m["correct_rate"] = round(sum(1 for r in rows if r["correctness"] == 2) / len(rows), 4)
    return {"run_name": gen.get("run_name"), "metrics": m,
            "generation_config": gen.get("generation_config"), "items": rows}


def boot(a: List[dict], b: List[dict], key: str) -> tuple:
    A = {r["id"]: r[key] for r in a}
    B = {r["id"]: r[key] for r in b}
    ids = [i for i in A if i in B]
    d = [B[i] - A[i] for i in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    N = len(d)
    means = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(BOOTSTRAP_N))
    return statistics.mean(d), means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", help="채점할 생성 결과 파일")
    ap.add_argument("--compare", nargs="+", help="비교 (첫 번째가 기준선)")
    ap.add_argument("--judge", action="store_true", help="LLM 판정 포함 (유료)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL,
                    help="판정 모델. 실험 후보에 없는 모델을 쓸 것 (자기선호 편향 방지)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = [Path(p) for p in (a.compare or a.gen or [])]
    if not paths:
        ap.error("--gen 또는 --compare 필요")

    globals()["JUDGE_MODEL"] = a.judge_model
    out_dir = Path(a.out) if a.out else PROJECT_ROOT / "gen_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in paths:
        out_file = out_dir / f"score_{p.stem}.json"
        # 이미 채점된 런은 건너뛴다 — 중간에 끊겨도 이어서 돌릴 수 있게
        if out_file.exists():
            prev = json.loads(out_file.read_text(encoding="utf-8"))
            if bool(prev["items"] and "correctness" in prev["items"][0]) == a.judge:
                print(f"[skip]  {p.stem} — 이미 채점됨")
                results.append(prev)
                continue

        gen = json.loads(p.read_text(encoding="utf-8"))
        print(f"[score] {p.stem} ({len(gen['items'])}문항)"
              f"{' + LLM 판정' if a.judge else ''}")
        r = score(gen, a.judge)
        # ⚠ run_name 은 리트리버 이름이라 컨텍스트 구성이 다르면 서로 덮어씀.
        #    파일명·표시명은 gen 파일 이름을 기준으로 한다.
        r["run_name"] = p.stem
        r["retriever_run"] = gen.get("run_name")
        # ⚠ 런 하나가 끝날 때마다 바로 저장한다. 마지막에 몰아서 쓰면 도중에 죽었을 때
        #    앞서 판정한 것까지 전부 날아간다 (실측 — 429 로 8런이 통째로 소실됐다).
        out_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(r)
    cols = ["ref_overlap", "refusal_rate", "avg_output_tokens"]
    if a.judge:
        cols = ["correctness", "groundedness", "correct_rate"] + cols
    hdr = f"{'run':32s} {'n':>4s} " + " ".join(f"{c:>16s}" for c in cols)
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for r in results:
        m = r["metrics"]
        print(f"{r['run_name'][:32]:32s} {m['n']:>4d} " +
              " ".join(f"{m[c]:>16.4f}" for c in cols))

    if len(results) > 1:
        base = results[0]
        keys = ["ref_overlap", "refusal"] + (["correctness", "groundedness"] if a.judge else [])
        print(f"\n기준선 대비 (paired bootstrap 95% CI)")
        print(f"{'run':32s} " + " ".join(f"{k:>26s}" for k in keys))
        print("-" * (32 + 27 * len(keys)))
        for r in results[1:]:
            cells = []
            for k in keys:
                d, lo, hi = boot(base["items"], r["items"], k)
                sig = "*" if not (lo <= 0 <= hi) else " "
                cells.append(f"{d:>+8.3f} [{lo:>+6.3f},{hi:>+6.3f}]{sig}")
            print(f"{r['run_name'][:32]:32s} " + " ".join(cells))
        print("\n* = 95% 신뢰구간이 0 을 포함하지 않음 (유의)")


if __name__ == "__main__":
    main()
