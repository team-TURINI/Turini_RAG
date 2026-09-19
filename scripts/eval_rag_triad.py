"""RAG triad 채점 — Faithfulness · Answer Relevance · Context Relevance.

기존 eval_generation.py 의 groundedness 는 **포인트와이즈(0~2 총점)** 라 판정 모델이
후하게 몰려 1.98~2.00 으로 포화됐다. 검색 상한이 21.7%p 차이 나는 두 조건조차
구분하지 못한다. 그래서 표준 방식으로 다시 만든다.

------------------------------------------------------------------
세 지표

  Faithfulness        답변이 **주어진 자료로 뒷받침되나**
                      답변을 원자적 주장으로 쪼갠 뒤 주장마다 판정한다.
                      점수 = 뒷받침되는 주장 / 전체 주장  →  연속값이라 포화되지 않는다.

  Answer Relevance    답변이 **질문에 실제로 답하나**
                      답변만 보고 "이 답변이 답하는 질문"을 3개 역생성해
                      원 질문과의 임베딩 유사도를 잰다.
                      ※ 회피 답변은 구조적으로 낮게 나오므로 **따로 집계**한다.

  Context Relevance   가져온 자료가 **질문에 쓸모 있나**
                      컨텍스트를 문장으로 쪼개 질문과 관련된 문장 비율을 잰다.

------------------------------------------------------------------
비용 최적화

  Faithfulness / Answer Relevance   답변에 의존  →  조건마다 계산
  Context Relevance                 답변과 무관  →  **한 번 계산해 캐시 재사용**

  Context Relevance 는 (질문, 컨텍스트) 만 쓰므로 프롬프트를 바꿔도 변하지 않는다.
  같은 검색 구성이면 캐시가 그대로 맞는다.

------------------------------------------------------------------
gold 라벨과의 대조

  coverage_precision(라벨 기반) 이 낮은데 Context Relevance(라벨 무관) 가 높으면
  **라벨 누락**이다. 둘 다 낮으면 진짜 검색 실패다. 다중 정답 문제의 독립 증거가 된다.

실행:
  python scripts/eval_rag_triad.py --gen gen_runs/ctx2_rr_top3.json
  python scripts/eval_rag_triad.py --compare gen_runs/ctx2_*.json --workers 6
  python scripts/eval_rag_triad.py --gen ... --metrics faithfulness   # 일부만
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config import EMBEDDING_MODEL, OPENAI_API_KEY

CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70_v2.jsonl"
CTX_FIELD = "embedding_text"

# 판정 모델은 생성 후보(gpt-4.1-mini 등)와 분리한다. 같은 모델로 채점하면 자기선호
# 편향이 든다. eval_generation.py 와 같은 모델을 써서 두 채점 결과를 나란히 볼 수 있게 한다.
JUDGE_MODEL = "gpt-4.1-2025-04-14"
N_GEN_QUESTIONS = 3
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260805
ALL_METRICS = ("faithfulness", "answer_relevance", "context_relevance")

# 회피 판정 — answer_spec.py 의 것을 그대로 쓴다.
# eval_generation.py 의 정규식은 "제공된 자료에서는 **해당 내용을** 확인할 수 없습니다"
# 라는 정확한 문구만 잡아 모델이 문장을 바꿔 쓰면 놓친다(실측 회피율이 0 으로 나왔다).
from answer_spec import REFUSAL as REFUSAL_PAT


CLAIM_PROMPT = """당신은 문장 분석기입니다. 주어진 [답변]을 **원자적 주장**으로 쪼개세요.

# 규칙
- 하나의 주장은 **하나의 사실만** 담아야 합니다. "A이고 B이다" 는 두 개로 쪼갭니다.
- 대명사를 풀어 **그 문장만 읽어도 뜻이 통하게** 씁니다.
- 인사말·권유 아님 고지·일반적 당부("유의하시기 바랍니다")처럼 **사실 주장이 아닌 문장은 제외**합니다.
- 답변이 "확인할 수 없다"는 취지뿐이면 주장은 빈 목록입니다.

# 출력 (JSON 만)
{"claims": ["주장1", "주장2"]}"""

VERDICT_PROMPT = """당신은 사실 검증기입니다. [자료]만 근거로 각 [주장]을 판정하세요.

# 판정 기준
supported = true   자료에서 직접 확인되거나 자료로부터 곧바로 추론됨
supported = false  자료에 없음. **일반 상식으로는 맞더라도 자료에 없으면 false**

# 주의
- 표현이 달라도 내용이 같으면 true 입니다.
- 자료보다 **더 구체적인 수치·날짜·고유명사**가 답변에 있으면 false 입니다.

# 출력 (JSON 만, 입력 주장과 같은 순서·같은 개수)
{"verdicts": [{"supported": true, "reason": "한 문장"}]}"""

QGEN_PROMPT = """주어진 [답변]만 보고, **이 답변이 답하고 있는 질문**을 정확히 {n}개 만드세요.

# 규칙
- 답변에 실제로 담긴 내용만 근거로 삼습니다. 답변에 없는 내용을 묻는 질문을 지어내지 마세요.
- 자연스러운 한국어 질문으로 씁니다.
- 답변이 "확인할 수 없다"는 취지뿐이라 무엇을 묻는 질문인지 특정할 수 없으면
  noncommittal 을 true 로 두고 질문은 최선의 추정으로 채웁니다.

# 출력 (JSON 만)
{{"questions": ["질문1", ...], "noncommittal": false}}"""

CTXREL_PROMPT = """[질문]에 답하는 데 **실제로 쓸모 있는 문장**의 번호를 고르세요.

# 기준
- 질문에 대한 답·근거·필수 배경을 담은 문장만 고릅니다.
- 같은 주제라는 이유만으로 고르지 마세요. **그 문장이 없으면 답을 못 하는가**를 기준으로 합니다.
- 쓸모 있는 문장이 하나도 없으면 빈 목록입니다.

# 출력 (JSON 만)
{"relevant": [1, 4, 5]}"""


# =============================================================================
# 유틸
# =============================================================================
_print_lock = threading.Lock()


def load_corpus() -> Dict[str, dict]:
    return {c["chunk_id"]: c for c in
            (json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip())}


def split_sentences(text: str) -> List[str]:
    """한국어 문장 분리 — 종결어미·물음표·줄바꿈 기준의 근사."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 5:            # 제목·머리표 같은 짧은 조각은 버린다
            out.append(p)
    return out


def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def ctx_key(question: str, chunk_ids: List[str]) -> str:
    raw = question + "||" + "|".join(chunk_ids)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _chat(client, system: str, user: str, model: str) -> dict:
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if attempt == 4:
                return {"__error__": str(e)}
            time.sleep(2 ** attempt)
    return {"__error__": "unreachable"}


# =============================================================================
# 지표 세 가지
# =============================================================================
def faithfulness(client, answer: str, context: str, model: str) -> dict:
    """답변을 주장으로 쪼갠 뒤 주장마다 자료 뒷받침 여부를 판정한다."""
    d = _chat(client, CLAIM_PROMPT, f"[답변]\n{answer}", model)
    claims = [c for c in (d.get("claims") or []) if isinstance(c, str) and c.strip()]
    if not claims:
        # 회피 답변 등 사실 주장이 없는 경우. 지어낸 것이 없으므로 감점 대상이 아니지만
        # 평균에 1.0 으로 섞으면 회피가 많은 조건이 유리해진다. None 으로 두고 제외한다.
        return {"faithfulness": None, "n_claims": 0, "n_supported": 0, "claims": []}

    listed = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    v = _chat(client, VERDICT_PROMPT,
              f"[자료]\n{context}\n\n[주장]\n{listed}", model)
    verdicts = v.get("verdicts") or []
    rows = []
    for i, c in enumerate(claims):
        sup = bool(verdicts[i].get("supported")) if i < len(verdicts) and isinstance(verdicts[i], dict) else False
        reason = verdicts[i].get("reason", "") if i < len(verdicts) and isinstance(verdicts[i], dict) else "(판정 누락)"
        rows.append({"claim": c, "supported": sup, "reason": reason})
    n_sup = sum(1 for r in rows if r["supported"])
    return {"faithfulness": round(n_sup / len(rows), 4),
            "n_claims": len(rows), "n_supported": n_sup, "claims": rows}


def answer_relevance(client, embed, question: str, answer: str, model: str) -> dict:
    """답변에서 질문을 역생성해 원 질문과의 임베딩 유사도를 잰다."""
    d = _chat(client, QGEN_PROMPT.format(n=N_GEN_QUESTIONS), f"[답변]\n{answer}", model)
    qs = [q for q in (d.get("questions") or []) if isinstance(q, str) and q.strip()][:N_GEN_QUESTIONS]
    noncommittal = bool(d.get("noncommittal")) or bool(REFUSAL_PAT.search(answer))
    if not qs:
        return {"answer_relevance": None, "noncommittal": noncommittal, "gen_questions": []}
    vecs = embed([question] + qs)
    sims = [cosine(vecs[0], v) for v in vecs[1:]]
    return {"answer_relevance": round(statistics.mean(sims), 4),
            "noncommittal": noncommittal, "gen_questions": qs}


def context_relevance(client, question: str, context: str, model: str) -> dict:
    """컨텍스트 문장 중 질문에 실제로 쓸모 있는 문장의 비율."""
    sents = split_sentences(context)
    if not sents:
        return {"context_relevance": None, "n_sentences": 0, "n_relevant": 0}
    listed = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))
    d = _chat(client, CTXREL_PROMPT, f"[질문]\n{question}\n\n[문장]\n{listed}", model)
    idx = {i for i in (d.get("relevant") or []) if isinstance(i, int) and 1 <= i <= len(sents)}
    return {"context_relevance": round(len(idx) / len(sents), 4),
            "n_sentences": len(sents), "n_relevant": len(idx)}


# =============================================================================
# 실행
# =============================================================================
def score_gen(gen: dict, corpus: Dict[str, dict], metrics: set, model: str,
              workers: int, ctx_cache: dict, cache_path: Optional[Path]) -> dict:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(texts: List[str]) -> List[List[float]]:
        r = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in r.data]

    items = gen["items"]
    done = [0]
    t0 = time.time()

    def one(it: dict) -> dict:
        ctx = "\n\n".join(corpus[c][CTX_FIELD] for c in it.get("context_chunk_ids", [])
                          if c in corpus)
        rec = {"id": it["id"], "n_context_chunks": len(it.get("context_chunk_ids", []))}

        if "faithfulness" in metrics:
            rec.update(faithfulness(client, it["answer"], ctx, model))
        if "answer_relevance" in metrics:
            rec.update(answer_relevance(client, embed, it["question"], it["answer"], model))
        if "context_relevance" in metrics:
            k = ctx_key(it["question"], it.get("context_chunk_ids", []))
            if k in ctx_cache:
                rec.update(ctx_cache[k])
                rec["ctx_cached"] = True
            else:
                r = context_relevance(client, it["question"], ctx, model)
                ctx_cache[k] = r
                rec.update(r)
                rec["ctx_cached"] = False

        with _print_lock:
            done[0] += 1
            n = done[0]
            if n % 20 == 0 or n == len(items):
                el = time.time() - t0
                eta = el / n * (len(items) - n)
                print(f"    {n}/{len(items)}  경과 {el/60:.1f}분 / 남은 예상 {eta/60:.1f}분", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, items))

    if cache_path is not None and "context_relevance" in metrics:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(ctx_cache, ensure_ascii=False), encoding="utf-8")

    by_id = {r["id"]: r for r in rows}
    rows = [by_id[it["id"]] for it in items]      # 입력 순서 복원

    def mean_of(key, rs=None):
        vals = [r[key] for r in (rs or rows) if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    committal = [r for r in rows if not r.get("noncommittal")]
    m = {"n": len(rows)}
    if "faithfulness" in metrics:
        # macro — 문항별 비율의 평균. 문항당 주장이 4개 안팎이라 0/.25/.5/.75/1 로
        #         양자화돼 분산이 크다. 해석은 쉬우나 검정력이 낮다.
        m["faithfulness"] = mean_of("faithfulness")
        m["faithfulness_n_scored"] = sum(1 for r in rows if r.get("faithfulness") is not None)
        m["avg_claims"] = round(statistics.mean(r.get("n_claims", 0) for r in rows), 2)
        tot = sum(r.get("n_claims", 0) for r in rows)
        # micro — 전체 주장 중 뒷받침 비율. 양자화가 없어 구간이 훨씬 좁다.
        #         실측에서 macro 는 검색 품질 차(21.7%p)를 못 잡았고 micro 는 잡았다.
        #         **주 집계는 micro 를 쓴다.**
        m["faithfulness_micro"] = round(sum(r.get("n_supported", 0) for r in rows) / tot, 4) if tot else None
        m["unsupported_claim_rate"] = round(1 - m["faithfulness_micro"], 4) if tot else None
        m["n_claims_total"] = tot
    if "answer_relevance" in metrics:
        m["answer_relevance"] = mean_of("answer_relevance")
        m["answer_relevance_committal"] = mean_of("answer_relevance", committal)
        m["noncommittal_rate"] = round(
            sum(1 for r in rows if r.get("noncommittal")) / len(rows), 4)
    if "context_relevance" in metrics:
        m["context_relevance"] = mean_of("context_relevance")
        m["avg_context_sentences"] = round(
            statistics.mean(r.get("n_sentences", 0) for r in rows), 1)

    return {"run_name": gen.get("run_name"), "judge_model": model,
            "metrics": m, "items": rows}


def boot_micro(a: List[dict], b: List[dict]):
    """micro-faithfulness 의 짝지은 부트스트랩.

    문항을 복원추출하되 **재표본마다 주장 총합으로 micro 를 다시 계산**한다.
    주장 단위로 무작정 뽑으면 짝이 깨지고, 문항 비율을 평균내면 양자화 분산이 남는다.
    """
    A = {r["id"]: (r.get("n_supported", 0), r.get("n_claims", 0)) for r in a}
    B = {r["id"]: (r.get("n_supported", 0), r.get("n_claims", 0)) for r in b}
    ids = [i for i in A if i in B and (A[i][1] or B[i][1])]
    if len(ids) < 5:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(ids)
    diffs = []
    for _ in range(BOOTSTRAP_N):
        sa = ca = sb = cb = 0
        for _ in range(n):
            i = ids[rng.randrange(n)]
            sa += A[i][0]; ca += A[i][1]
            sb += B[i][0]; cb += B[i][1]
        if ca and cb:
            diffs.append(sb / cb - sa / ca)
    if not diffs:
        return None
    diffs.sort()
    tot_a = sum(A[i][1] for i in ids); tot_b = sum(B[i][1] for i in ids)
    obs = sum(B[i][0] for i in ids) / tot_b - sum(A[i][0] for i in ids) / tot_a
    return obs, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1], len(ids)


def boot(a: List[dict], b: List[dict], key: str):
    A = {r["id"]: r[key] for r in a if r.get(key) is not None}
    B = {r["id"]: r[key] for r in b if r.get(key) is not None}
    ids = [i for i in A if i in B]
    if len(ids) < 5:
        return None
    d = [B[i] - A[i] for i in ids]
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_N))
    return statistics.mean(d), means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N) - 1], len(ids)


def main() -> None:
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", help="채점할 생성 결과 파일")
    ap.add_argument("--compare", nargs="+", help="비교 (첫 번째가 기준선)")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--metrics", default=",".join(ALL_METRICS),
                    help="쉼표 구분 (faithfulness,answer_relevance,context_relevance)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=4, help="동시 호출 수")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만 (파일럿용)")
    ap.add_argument("--ctx-cache", default=str(PROJECT_ROOT / "results_triad" / "ctx_relevance_cache.json"),
                    help="Context Relevance 캐시 (답변과 무관하므로 조건 간 재사용)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results_triad"))
    a = ap.parse_args()

    CORPUS = Path(a.corpus)
    metrics = {m.strip() for m in a.metrics.split(",") if m.strip()}
    bad = metrics - set(ALL_METRICS)
    if bad:
        raise SystemExit(f"없는 지표: {sorted(bad)} (가능: {list(ALL_METRICS)})")

    paths = [Path(p) for p in (a.compare or a.gen or [])]
    if not paths:
        ap.error("--gen 또는 --compare 필요")

    corpus = load_corpus()
    cache_path = Path(a.ctx_cache)
    ctx_cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if ctx_cache:
        print(f"[cache] Context Relevance 캐시 {len(ctx_cache)}건 로드")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in paths:
        gen = json.loads(p.read_text(encoding="utf-8"))
        if a.limit:
            gen["items"] = gen["items"][: a.limit]
        print(f"\n[triad] {p.stem} ({len(gen['items'])}문항) / {sorted(metrics)} / {a.judge_model}")
        r = score_gen(gen, corpus, metrics, a.judge_model, a.workers, ctx_cache, cache_path)
        # results_e2e/<tag>/generation.json 처럼 파일명이 generic 하면 부모 폴더명으로 구분한다.
        # 안 그러면 여러 tag 의 결과가 triad_generation.json 하나로 덮어써진다.
        label = p.stem if p.stem != "generation" else f"{p.parent.name}"
        r["run_name"] = label
        r["retriever_run"] = gen.get("run_name")
        (out_dir / f"triad_{label}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(r)

    cols = [c for c in ["faithfulness_micro", "faithfulness", "unsupported_claim_rate",
                        "answer_relevance", "answer_relevance_committal", "noncommittal_rate",
                        "context_relevance", "avg_claims"]
            if any(c in r["metrics"] for r in results)]
    hdr = f"{'run':30s} {'n':>4s} " + " ".join(f"{c[:18]:>19s}" for c in cols)
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for r in results:
        m = r["metrics"]
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append(f"{v:>19.4f}" if isinstance(v, (int, float)) else f"{'—':>19s}")
        print(f"{r['run_name'][:30]:30s} {m['n']:>4d} " + " ".join(cells))

    if len(results) > 1:
        base = results[0]
        keys = [k for k in ["faithfulness_micro", "faithfulness",
                            "answer_relevance", "context_relevance"]
                if k.split("_micro")[0] in metrics or k in metrics]
        print(f"\n기준선({base['run_name']}) 대비 — paired bootstrap {BOOTSTRAP_N}회 95% CI")
        print(f"{'run':30s} " + " ".join(f"{k[:22]:>30s}" for k in keys))
        print("-" * (30 + 31 * len(keys)))
        for r in results[1:]:
            cells = []
            for k in keys:
                res = (boot_micro(base["items"], r["items"]) if k == "faithfulness_micro"
                       else boot(base["items"], r["items"], k))
                if res is None:
                    cells.append(f"{'(표본 부족)':>30s}")
                    continue
                d, lo, hi, n = res
                sig = "*" if not (lo <= 0 <= hi) else " "
                cells.append(f"{d:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}]{sig}")
            print(f"{r['run_name'][:30]:30s} " + " ".join(cells))
        print("\n* = 95% 신뢰구간이 0 을 포함하지 않음 (유의)")
        print("※ faithfulness_micro = 주장 단위 집계(주 지표). 재표본마다 주장 총합으로 다시 계산함")
        print("※ faithfulness(macro) 는 주장이 없는 문항(회피 등)을 제외하고 짝지어 비교함")


if __name__ == "__main__":
    main()
