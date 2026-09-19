"""LLM judge 실행기 — 생성된 답변을 채점해 judge_runs/ 에 저장한다.

정답 answer 가 원문의 **패러프레이즈**(gold 청크와 문자 2-gram 포함률 중앙 0.76)라서
EM·ROUGE 같은 문자열 매칭으로는 "맞게 말을 바꾼 답"과 "틀린 답"을 구분할 수 없다.
그래서 judge 가 선택이 아니라 필수다.

------------------------------------------------------------------
judge 를 둘로 나누는 기준은 '무엇을 보여주느냐'다

  faithfulness   [자료] + 답변            — 모범답안을 **가린다**
  correctness    질문 + 모범답안 + 답변   — [자료]를 **가린다**

  한 프롬프트에서 둘을 같이 물으면 오염된다. 모범답안을 본 judge 는 "모범답안과
  비슷하니 자료에도 있겠지" 라며 faithfulness 를 후하게 준다. 판정의 독립성이
  정보 차단에서 나온다.

  completeness 는 correctness 와 **같은 정보**(질문·모범답안·답변)만 있으면 되므로
  같은 호출에서 함께 받는다. 굳이 나누면 비용만 2배가 된다.

------------------------------------------------------------------
설계에 반영한 실측 사항

  ① 답변이 모범답안보다 넓다 — 조건 A 의 컨텍스트는 gold **청크 전체**라 정답 구간
     보다 넓다. 실제로 모델이 모범답안에 없는 내용을 정확히 덧붙인 사례가 있다
     (v2_086 공매도 처벌 강화, rag_bank_05 보호대상 목록).
     → correctness 프롬프트에 "추가 정보는 감점하지 않음. 단 모순이면 감점" 을 명시.

  ② 답변이 모범답안의 1.45배 길다 — 장황한 답에 후한 judge 는 "자세히 답하라" 프롬프트를
     무조건 승자로 만든다.
     → 문장 판정에 `no_claim` 범주를 둬서 "유의하시기 바랍니다" 같은 빈 문장이 분모에서
       빠지게 했다. 부연을 붙여도 점수가 오르지도 내리지도 않는다.
     → correctness 프롬프트에 "길이는 판정 근거가 아니다" 를 명시.

  두 장치가 실제로 작동하는지는 scripts/make_judge_probe.py 로 검증한다.

  ③ 규정 고지 문구 — 2026-09-19 결정. "원금 손실 가능" · "투자 권유 아님" 은
     answer_spec 이 **요구하는** 문구인데 자료에는 없으므로, 그대로 두면 규정을 지킬수록
     faithfulness 가 깎이는 구조가 된다(실측: v1 0.9845 → v2 0.9428, 유의).
     → no_claim 으로 분류한다. 판정 설계의 결함을 고친 것이지 기준을 낮춘 것이 아니다.
     이 변경 전 판정 결과(judge_runs/judge_base_*_dev.json 등)와는 직접 비교하지 말 것.

실행
  python scripts/run_judge.py --run gen_runs/A_gold_v1_mini.json
  python scripts/run_judge.py --run gen_runs/probe_calibration.json --judge-model gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OPENAI_API_KEY  # noqa: E402

TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval.json"
CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70.jsonl"
CTX_DIR = PROJECT_ROOT / "data" / "gen_contexts"
OUT_DIR = PROJECT_ROOT / "judge_runs"
TEXT_FIELD = "embedding_text"

# 생성 모델보다 상위 모델을 쓴다. 같은 모델로 자기 답을 채점하면 자기 편향이 든다.
DEFAULT_JUDGE_MODEL = "gpt-4.1"

# gpt-4.1 은 이 계정에서 TPM 30,000 이다. faithfulness 호출 하나가 1,000토큰 안팎이라
# 동시 실행을 높이면 곧바로 429 가 난다. 대기를 길게 잡고 동시 실행을 낮춘다.
MAX_RETRY = 8
DEFAULT_WORKERS = 3


# =============================================================================
# 프롬프트
# =============================================================================
FAITH_SYSTEM = """당신은 RAG 답변이 주어진 자료에 근거하는지 검증하는 엄격한 평가자입니다.

번호가 매겨진 답변 문장 각각에 대해, 그 문장이 [자료]로 뒷받침되는지 판정하세요.

# 판정 범주
- supported   : 자료에 명시되어 있거나 자료에서 직접 도출됨
- partial     : 일부만 자료에 있고 나머지는 자료에 없음
- unsupported : 자료에 없는 내용 (세상에서 참이어도 자료에 없으면 unsupported)
- no_claim    : 사실 주장이 아님 — 인사, 안내 문구("유의하시기 바랍니다"), 질문 되받기,
                앞 문장의 요약 반복 등
                **규정상 필수 고지 문구도 여기에 속한다**: "원금 손실이 발생할 수 있다",
                "본 안내는 투자 권유가 아닌 정보 제공이다" 처럼 상품 설명이 아니라
                금융 안내 규정에 따라 붙이는 상투적 고지는 자료 근거를 따지지 않는다.
                단, 고지 문구 안에 **구체적 수치·상품명·조건**이 들어 있으면 그 부분은
                사실 주장이므로 supported/unsupported 로 판정한다.

# 규칙
- 판정 전에 근거를 먼저 찾을 것. supported/partial 이면 자료에서 근거 문구를 **그대로 인용**한다.
- 사실 여부가 아니라 '자료에 있는지'만 본다.
- 표현이 달라도 내용이 같으면 supported 다. 자료의 재서술을 틀렸다고 하지 말 것.
- 길이는 판정과 무관하다.

# 출력 (JSON만)
{"sentences":[{"i":1,"verdict":"supported","evidence":"자료에서 그대로 인용"}]}
evidence 는 unsupported/no_claim 이면 빈 문자열."""

CORRECT_SYSTEM = """당신은 질문에 대한 [모범답안]과 [평가대상 답변]을 비교해 정확성을 판정하는 평가자입니다.
자료 원문은 제공되지 않습니다. 모범답안 대비 내용 일치만 판단하세요.

# 판정
- 2 : 질문에 대한 답이 모범답안과 실질적으로 일치
- 1 : 부분적으로 맞음 — 핵심 일부가 빠졌거나 일부가 틀림
- 0 : 틀렸거나 질문에 답하지 않음 (거부 답변 포함)

# 반드시 지킬 것
- 표현이 달라도 내용이 같으면 일치다. 모범답안 자체가 원문의 재서술이다.
- **모범답안에 없는 내용이 추가로 들어 있어도 감점하지 않는다.** 답변은 모범답안보다
  넓은 자료를 근거로 하므로 추가 정보가 있는 것이 정상이다.
  단, 추가 내용이 모범답안과 **모순**되면 감점한다.
- **길이는 판정 근거가 아니다.** 길다고 좋은 답이 아니고 짧다고 나쁜 답이 아니다.
  부연이 많아도 질문에 답하지 못했으면 낮은 점수다.
- 숫자가 모범답안과 다르면 명확한 감점 사유다.

# 출력 (JSON만)
{"key_points":[{"point":"모범답안의 핵심 요소","covered":true}],
 "verdict":2,"reason":"한 문장"}"""


# =============================================================================
# 유틸
# =============================================================================
def split_sentences(text: str) -> List[str]:
    """답변을 문장 단위로 나눈다. judge 에게 맡기지 않고 여기서 고정한다 —
    분해가 매번 달라지면 분모가 흔들려 점수를 비교할 수 없다."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and len(p) < 8:      # 너무 짧은 조각은 앞 문장에 붙인다
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def load_corpus() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


def call_json(client, model: str, system: str, user: str) -> dict:
    """JSON 강제 + 파싱 실패 시 재시도."""
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,          # 판정은 결정적이어야 한다
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRY - 1:
                raise
            # 429 는 분 단위로 풀리므로 짧은 백오프로는 못 빠져나온다
            wait = min(60, 4 * (2 ** attempt)) if "429" in str(e) else 2 ** attempt
            time.sleep(wait)
    raise RuntimeError("unreachable")


# =============================================================================
# 채점
# =============================================================================
def judge_faithfulness(client, model: str, context: str, answer: str) -> dict:
    sents = split_sentences(answer)
    if not sents:
        return {"n_sentences": 0, "score": None, "sentences": []}
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sents))
    user = f"# 자료\n{context}\n\n# 답변 (문장별)\n{numbered}"
    res = call_json(client, model, FAITH_SYSTEM, user)

    verdicts = {int(s["i"]): s for s in res.get("sentences", []) if str(s.get("i", "")).isdigit()}
    rows = []
    for i, s in enumerate(sents, start=1):
        v = verdicts.get(i, {})
        rows.append({"i": i, "text": s,
                     "verdict": v.get("verdict", "unsupported"),
                     "evidence": v.get("evidence", "")})
    # no_claim 은 분모에서 뺀다 — 빈 문장을 붙여도 점수가 안 오르게
    claims = [r for r in rows if r["verdict"] != "no_claim"]
    score = None
    if claims:
        score = sum(1.0 if r["verdict"] == "supported" else 0.5 if r["verdict"] == "partial" else 0.0
                    for r in claims) / len(claims)
    return {"n_sentences": len(sents), "n_claims": len(claims),
            "score": None if score is None else round(score, 4), "sentences": rows}


def judge_correctness(client, model: str, question: str, gold: str, answer: str) -> dict:
    user = (f"# 질문\n{question}\n\n# 모범답안\n{gold}\n\n# 평가대상 답변\n{answer}")
    res = call_json(client, model, CORRECT_SYSTEM, user)
    kps = res.get("key_points") or []
    covered = sum(1 for k in kps if k.get("covered"))
    try:
        verdict = int(res.get("verdict", 0))
    except (TypeError, ValueError):
        verdict = 0
    return {
        "verdict": max(0, min(2, verdict)),
        "completeness": round(covered / len(kps), 4) if kps else None,
        "n_key_points": len(kps),
        "key_points": kps,
        "reason": res.get("reason", ""),
    }


# =============================================================================
# main
# =============================================================================
def main() -> None:
    global CORPUS
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--corpus", default=str(CORPUS), help="코퍼스 경로 override")
    ap.add_argument("--testset", default=str(TESTSET), help="평가셋 경로 override")
    a = ap.parse_args()

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    # ⚠ run["run_name"] 은 --run 방식일 때 **검색 run 이름**이라, 프롬프트만 다른
    #   구성들이 전부 같은 파일로 덮어써진다(eval_generation.py 에도 같은 결함이 있었다).
    #   파일명·표시명은 **생성 결과 파일 이름**을 기준으로 한다.
    label = Path(a.run).stem
    testset = {r["id"]: r for r in
               json.loads(Path(a.testset).read_text(encoding="utf-8"))}
    CORPUS = Path(a.corpus)
    corpus = load_corpus()

    # 컨텍스트 출처는 두 가지다.
    #   ① --context 로 만든 결과 : run["context_file"] 이 있고 컨텍스트가 확정돼 있다
    #   ② --run  으로 만든 결과 : 각 item 이 context_chunk_ids 를 직접 들고 있다
    # 후자를 못 읽어서 판정기가 둘로 갈라져 있었다. 여기서 합친다.
    if run.get("context_file"):
        ctx = json.loads((CTX_DIR / run["context_file"]).read_text(encoding="utf-8"))
        ctx_by_id = {r["id"]: r for r in ctx["items"]}
        ctx_src = f'context_file={run["context_file"]}'
    elif all("context_chunk_ids" in it for it in run["items"]):
        ctx_by_id = {it["id"]: {"context_chunk_ids": it["context_chunk_ids"]}
                     for it in run["items"]}
        ctx_src = "item.context_chunk_ids"
    else:
        raise SystemExit("[fatal] 컨텍스트를 찾을 수 없습니다 — "
                         "run 파일에 context_file 도 context_chunk_ids 도 없습니다")
    print(f"[ctx] {ctx_src}")

    items = run["items"][:a.limit] if a.limit else run["items"]
    print(f"[judge] {label} — {len(items)}문항 × 2호출, 모델 {a.judge_model}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"judge_{label}.json"

    # resume
    done: Dict[str, dict] = {}
    if out_file.exists():
        prev = json.loads(out_file.read_text(encoding="utf-8"))
        if prev.get("judge_model") == a.judge_model:
            done = {r["id"]: r for r in prev["items"]}
            print(f"[resume] 기존 판정 {len(done)}건 재사용")

    todo = [it for it in items if it["id"] not in done]
    t0 = time.time()
    n = 0

    def work(it: dict) -> dict | None:
        try:
            return _work(it)
        except Exception as e:  # noqa: BLE001
            print(f"    [실패] {it['id']}: {type(e).__name__} — 재실행하면 이어서 채웁니다")
            return None

    def _work(it: dict) -> dict:
        # 변형 문항(probe)은 base_id 로 원본 정보를 찾는다
        base = it.get("base_id", it["id"])
        c = ctx_by_id[base]
        context = "\n\n".join(corpus[cid][TEXT_FIELD] for cid in c["context_chunk_ids"])
        t = testset[base]
        f = judge_faithfulness(client, a.judge_model, context, it["answer"])
        cc = judge_correctness(client, a.judge_model, t["question"], t["answer"], it["answer"])
        return {"id": it["id"], "base_id": base, "label": it.get("label"),
                "split": it.get("split"), "stratum": it.get("stratum"),
                "faithfulness": f, "correctness": cc}

    results = dict(done)
    if todo:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(work, todo):
                n += 1
                if r is None:
                    continue
                results[r["id"]] = r
                if n % 20 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}  ({time.time() - t0:.0f}s)")

    rows = [results[it["id"]] for it in items if it["id"] in results]
    if len(rows) < len(items):
        print(f"\n⚠ {len(items) - len(rows)}건 실패 — 같은 명령을 다시 실행하면 실패분만 채웁니다")
    Path(out_file).write_text(json.dumps({
        "run_name": label,
        "source_run": run.get("run_name"),
        "judge_model": a.judge_model,
        "gen_config": run.get("config") or run.get("generation_config"),
        "corpus": Path(a.corpus).name,
        "testset": Path(a.testset).name,
        "context_source": ctx_src,
        "condition": run.get("condition"),
        "n_items": len(rows),
        "items": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    fs = [r["faithfulness"]["score"] for r in rows if r["faithfulness"]["score"] is not None]
    cs = [r["correctness"]["verdict"] for r in rows]
    cp = [r["correctness"]["completeness"] for r in rows
          if r["correctness"]["completeness"] is not None]
    print("\n" + "=" * 60)
    print(f"Faithfulness  {statistics.mean(fs):.3f}   (n={len(fs)})")
    print(f"Correctness   {statistics.mean(cs):.3f} / 2  "
          f"(2점 {cs.count(2)} / 1점 {cs.count(1)} / 0점 {cs.count(0)})")
    print(f"Completeness  {statistics.mean(cp):.3f}   (n={len(cp)})")
    print(f"\nsaved → {out_file}")


if __name__ == "__main__":
    main()
