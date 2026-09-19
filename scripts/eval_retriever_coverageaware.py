r"""Retriever Coverage-Aware 평가기

수정 이유
---------
testset v2에서는 동일한 정답 근거가 문서 내 재수록 또는 중복 문서에 존재하는 경우
해당 청크들도 valid Gold로 인정하기 위해 duplicate Gold가 추가되어 있다.

기존 UnionSpanRecall은 canonical Gold span의 문자 합집합을 기준으로 계산하므로
중복 Gold가 여러 개 있어도 같은 근거를 반복 보상하지 않는다. 반면 기존 nDCG는
각 gold_chunk의 gold_coverage를 독립 relevance로 사용하기 때문에 동일 근거의
원본/중복 청크가 각각 별도 정답처럼 평가될 수 있다. 이 파일은 이러한 중복 relevance
문제를 줄이기 위해 검색 순서상 Gold span에서 새롭게 확보한 문자 coverage를 순위별
gain으로 사용하는 coverage-aware nDCG를 공식 ranking metric으로 추가한다.

수정 범위
---------
- USR@K, CoverageHit@K, EvidenceHit@K, MRR@K는 기존 방식을 유지한다.
- DocHit@K, DocPrecision@K와 문자 예산 기반 USR도 기존 방식을 유지한다.
- coverage_ndcg@K를 공식 ranking metric으로 추가한다.
- 기존 nDCG는 legacy_ndcg@K로 보존하여 과거 결과와 비교할 수 있게 한다.
- legacy Precision@K는 duplicate Gold를 각각 relevant hit로 세므로 동일한 근거의 반복
  검색을 높게 평가할 수 있지만, 과거 결과 호환을 위해 기존 계산과 key를 그대로 유지한다.
- coverage_precision@K는 이전 rank까지 확보하지 못한 canonical Gold 문자를 실제로
  1자 이상 추가한 청크만 세며, 중복 retrieval과 context 효율 진단용으로 추가한다.
- coverage_precision@K는 모델 선택의 primary metric이나 공식 ranking metric이 아니다.
- Gold 판정과 검색 입력은 계속 chunk_id와 gold_chunks를 사용한다.
- actual_start_char/actual_end_char는 provenance이며 canonical USR 좌표로 사용하지 않는다.

기존 coverage_ndcg와 기존 retrieval metric의 의미와 계산식은 변경하지 않는다.

coverage-aware IDCG는 동일 canonical interval을 먼저 중복 제거한 뒤 subset dynamic
programming으로 정확하게 계산한다. unique interval이 18개를 초과하면 근사 fallback을
사용하지 않고 fail closed한다. 현재 v2 testset의 최대 unique interval 수는 6개다.

기존 평가기
-----------
scripts/eval_retriever.py는 과거 실험 재현용으로 수정하지 않고 유지한다. 이 파일은
기존 평가기의 변경되지 않은 metric을 매 실행마다 in-memory regression 비교한다.
새 결과는 score_coverageaware_<run_name>.json으로 저장하여 기존 score 파일을 덮어쓰지
않는다.

testset 선택
------------
기본값은 기존 testset_v2이다. canonical testset(v1) --testset으로 명시적으로 선택한다. 상대경로는
PROJECT_ROOT 기준으로 해석한다.

사용법
------
python scripts/eval_retriever_coverageaware.py --run runs/dense_baseline.json

python scripts/eval_retriever_coverageaware.py \
  --testset data/testset/rag_testset_retriever_v1v2_fixed450_team_eval.json \
  --run runs/dense_baseline.json

python scripts/eval_retriever_coverageaware.py \
  --testset data/testset/rag_testset_retriever_v1v2_fixed450_team_eval.json \
  --compare runs/dense_baseline.json runs/hybrid_rrf_k20_d50_b50.json

python scripts/eval_retriever_coverageaware.py \
  --testset data/testset/rag_testset_retriever_v1v2_fixed450_team_eval.json \
  --run runs/dense_baseline.json \
  --validate-only

PowerShell 실행 예시
--------------------
python .\scripts\eval_retriever_coverageaware.py `
  --testset data\testset\rag_testset_retriever_v1v2_fixed450_team_eval.json `
  --run runs\dense_baseline.json

python .\scripts\eval_retriever_coverageaware.py `
  --testset data\testset\rag_testset_retriever_v1v2_fixed450_team_eval.json `
  --compare runs\dense_baseline.json runs\hybrid_rrf_k20_d50_b50.json

python .\scripts\eval_retriever_coverageaware.py `
  --testset data\testset\rag_testset_retriever_v1v2_fixed450_team_eval.json `
  --run runs\dense_baseline.json `
  --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import eval_retriever as legacy_evaluator
except ModuleNotFoundError:  # package-style import during local tests
    from scripts import eval_retriever as legacy_evaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TESTSET = (
    PROJECT_ROOT
    / "data"
    / "testset"
    / "rag_testset_retriever_v1v2_fixed450_team_eval_v2.json"
)
CORPUS = (
    PROJECT_ROOT
    / "data"
    / "chunking_data"
    / "fixed_450_70"
    / "clean_chunks_450_70_v2.jsonl"
)
INDEX_FAISS = PROJECT_ROOT / "vectorstores" / "fixed_450_70" / "index.faiss"

DEFAULT_K = [1, 3, 5, 10]
OFFICIAL_K = 10
TAUS = (0.5, 0.8)
DEFAULT_BUDGET = 2500
BUDGET_TEXT_FIELD = "embedding_text"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260805
MAX_EXACT_IDCG_INTERVALS = 18
EVALUATOR_NAME = "eval_retriever_coverageaware"
EVALUATOR_VERSION = "1.1"
FLOAT_TOLERANCE = 1e-12


# =============================================================================
# 로드
# =============================================================================
def sha256(path: Path) -> str:
    """파일의 SHA-256 digest를 반환한다."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    """상대경로를 PROJECT_ROOT 기준으로 해석한다."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def provenance_path(path: Path) -> str:
    """프로젝트 내부 경로는 읽기 쉬운 상대경로로 기록한다."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_testset(path: Path) -> List[dict]:
    """명시적으로 선택한 UTF-8 testset을 읽는다."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"testset은 JSON list여야 함: {path}")
    return value


def load_corpus() -> Dict[str, dict]:
    """고정 corpus를 chunk_id 기준으로 읽는다."""

    out: Dict[str, dict] = {}
    with CORPUS.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"corpus {line_number}행의 chunk_id가 유효하지 않음")
            if chunk_id in out:
                raise ValueError(f"corpus 중복 chunk_id: {chunk_id}")
            out[chunk_id] = chunk
    return out


# =============================================================================
# 입력 검증
# =============================================================================
def validate_run(
    run: dict,
    testset: List[dict],
    corpus: Dict[str, dict],
    kmax: int,
) -> Tuple[List[str], List[str]]:
    """기존 평가기와 동일한 run 검증을 수행한다."""

    errors: List[str] = []
    warns: List[str] = []

    if not isinstance(run, dict):
        return ["run top-level이 object가 아님"], warns
    if not isinstance(run.get("items"), list):
        return ["items가 없거나 리스트가 아님"], warns

    want = {row["id"] for row in testset}
    got_list = [item.get("id") for item in run["items"]]
    got = set(got_list)

    if len(got_list) != len(got):
        duplicates = [item_id for item_id, count in Counter(got_list).items() if count > 1]
        errors.append(f"run 안에 중복 id {len(duplicates)}건: {duplicates[:5]}")
    if want - got:
        errors.append(f"평가셋 문항 {len(want - got)}건 누락: {sorted(want - got)[:5]}")
    if got - want:
        errors.append(f"평가셋에 없는 id {len(got - want)}건: {sorted(got - want)[:5]}")

    unknown = 0
    short = 0
    duplicate_retrieved = 0
    for item in run["items"]:
        retrieved = item.get("retrieved")
        if not isinstance(retrieved, list):
            errors.append(f"{item.get('id')}: retrieved가 리스트가 아님")
            continue
        if len(retrieved) != len(set(retrieved)):
            duplicate_retrieved += 1
        if len(retrieved) < kmax:
            short += 1
        unknown += sum(1 for chunk_id in retrieved if chunk_id not in corpus)
    if unknown:
        errors.append(f"코퍼스에 없는 chunk_id {unknown}건 - 다른 코퍼스일 가능성")
    if duplicate_retrieved:
        errors.append(f"retrieved 안에 중복 chunk_id가 있는 문항 {duplicate_retrieved}건")
    if short:
        warns.append(f"retrieved 길이가 K={kmax}보다 짧은 문항 {short}건")

    actual_corpus_sha = sha256(CORPUS)
    if run.get("corpus_sha256"):
        if run["corpus_sha256"] != actual_corpus_sha:
            errors.append(
                "corpus_sha256 불일치 - 다른 코퍼스로 검색함\n"
                f"      run: {run['corpus_sha256']}\n      실제: {actual_corpus_sha}"
            )
    else:
        warns.append("corpus_sha256 미기록 - 같은 코퍼스인지 확인 불가")
    if run.get("index_sha256") and INDEX_FAISS.exists():
        if run["index_sha256"] != sha256(INDEX_FAISS):
            warns.append("index_sha256 불일치 - run 생성 당시 인덱스와 현재 로컬 인덱스가 다름 "
                        "(저장된 retrieved 결과 재평가에는 영향 없음)")

    return errors, warns


def validate_gold_annotations(testset: Sequence[Mapping[str, Any]]) -> None:
    """canonical interval과 duplicate provenance의 구조를 fail closed 검증한다."""

    seen_ids: set[str] = set()
    for position, item in enumerate(testset):
        if not isinstance(item, Mapping):
            raise ValueError(f"testset {position}번 항목이 object가 아님")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"testset {position}번 항목의 id가 유효하지 않음")
        if item_id in seen_ids:
            raise ValueError(f"testset 중복 id: {item_id}")
        seen_ids.add(item_id)

        gold_start = item.get("gold_start")
        gold_end = item.get("gold_end")
        if (
            type(gold_start) is not int
            or type(gold_end) is not int
            or gold_start >= gold_end
        ):
            raise ValueError(f"{item_id}: gold_start < gold_end 조건 위반")
        if item.get("gold_length") != gold_end - gold_start:
            raise ValueError(f"{item_id}: gold_length 불일치")
        gold_chunks = item.get("gold_chunks")
        if not isinstance(gold_chunks, list) or not gold_chunks:
            raise ValueError(f"{item_id}: gold_chunks가 비어 있거나 list가 아님")

        seen_chunks: set[str] = set()
        for gold_position, gold in enumerate(gold_chunks):
            if not isinstance(gold, Mapping):
                raise ValueError(f"{item_id}: Gold {gold_position}가 object가 아님")
            chunk_id = gold.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"{item_id}: Gold {gold_position}의 chunk_id가 유효하지 않음")
            if chunk_id in seen_chunks:
                raise ValueError(f"{item_id}: 중복 Gold chunk_id {chunk_id}")
            seen_chunks.add(chunk_id)

            start = gold.get("start_char")
            end = gold.get("end_char")
            if type(start) is not int or type(end) is not int or start >= end:
                raise ValueError(f"{item_id}/{chunk_id}: canonical interval이 유효하지 않음")
            overlap_start = max(start, gold_start)
            overlap_end = min(end, gold_end)
            if overlap_start >= overlap_end:
                raise ValueError(
                    f"{item_id}/{chunk_id}: canonical interval이 Gold span과 교차하지 않음"
                )

            actual_present = "actual_start_char" in gold or "actual_end_char" in gold
            if actual_present:
                actual_start = gold.get("actual_start_char")
                actual_end = gold.get("actual_end_char")
                if (
                    type(actual_start) is not int
                    or type(actual_end) is not int
                    or actual_start >= actual_end
                ):
                    raise ValueError(
                        f"{item_id}/{chunk_id}: actual_start_char < actual_end_char 조건 위반"
                    )
            if "dup_of" in gold and not isinstance(gold["dup_of"], str):
                raise ValueError(f"{item_id}/{chunk_id}: dup_of는 문자열이어야 함")

        intervals = unique_gold_intervals(item)
        if len(intervals) > MAX_EXACT_IDCG_INTERVALS:
            raise ValueError(
                f"{item_id}: unique Gold interval {len(intervals)}개가 exact DP 한계 "
                f"{MAX_EXACT_IDCG_INTERVALS}개를 초과함"
            )


# =============================================================================
# 기존 지표와 coverage-aware ranking 지표
# =============================================================================
def _union_len(spans: Sequence[Tuple[int, int]]) -> int:
    """겹치는 반개구간의 합집합 길이를 계산한다."""

    if not spans:
        return 0
    merged: List[List[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def canonical_gold_interval(
    item: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> Tuple[int, int] | None:
    """Gold chunk를 canonical Gold span에 clip한 반개구간으로 반환한다."""

    start = max(gold["start_char"], item["gold_start"])
    end = min(gold["end_char"], item["gold_end"])
    return (start, end) if end > start else None


def unique_gold_intervals(item: Mapping[str, Any]) -> Tuple[Tuple[int, int], ...]:
    """동일 canonical coverage interval을 제거해 deterministic하게 반환한다."""

    intervals = {
        interval
        for gold in item["gold_chunks"]
        if (interval := canonical_gold_interval(item, gold)) is not None
    }
    return tuple(sorted(intervals))


def union_span_recall(item: dict, retrieved: Sequence[str]) -> float:
    """검색된 Gold chunk가 canonical Gold span을 덮은 문자 합집합 비율이다."""

    gold_start = item["gold_start"]
    gold_end = item["gold_end"]
    total = gold_end - gold_start
    if total <= 0:
        return 0.0
    gold_map = {gold["chunk_id"]: gold for gold in item["gold_chunks"]}
    spans = []
    for chunk_id in retrieved:
        gold = gold_map.get(chunk_id)
        if gold is None:
            continue
        start = max(gold["start_char"], gold_start)
        end = min(gold["end_char"], gold_end)
        if end > start:
            spans.append((start, end))
    return min(1.0, _union_len(spans) / total)


def marginal_coverage_gains(
    item: Mapping[str, Any],
    retrieved: Sequence[str],
) -> List[float]:
    """rank별로 아직 확보하지 못했던 canonical Gold coverage gain을 계산한다."""

    total = item["gold_end"] - item["gold_start"]
    if total <= 0:
        return [0.0 for _ in retrieved]
    gold_map = {gold["chunk_id"]: gold for gold in item["gold_chunks"]}
    covered: List[Tuple[int, int]] = []
    covered_length = 0
    gains: List[float] = []
    for chunk_id in retrieved:
        gold = gold_map.get(chunk_id)
        interval = canonical_gold_interval(item, gold) if gold is not None else None
        if interval is None:
            gains.append(0.0)
            continue
        new_covered = _union_len([*covered, interval])
        new_chars = new_covered - covered_length
        gains.append(new_chars / total)
        covered.append(interval)
        covered_length = new_covered
    return gains


def coverage_dcg(item: Mapping[str, Any], retrieved: Sequence[str]) -> float:
    """marginal new coverage에 log discount를 적용한 DCG를 계산한다."""

    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(marginal_coverage_gains(item, retrieved), start=1)
    )


@lru_cache(maxsize=None)
def _coverage_idcg_exact(
    intervals: Tuple[Tuple[int, int], ...],
    gold_start: int,
    gold_length: int,
    k: int,
) -> float:
    """unique interval의 최적 순서를 subset DP로 정확히 계산한다."""

    interval_count = len(intervals)
    if not intervals or gold_length <= 0 or k <= 0:
        return 0.0
    if interval_count > MAX_EXACT_IDCG_INTERVALS:
        raise ValueError(
            f"unique Gold interval {interval_count}개가 exact DP 한계 "
            f"{MAX_EXACT_IDCG_INTERVALS}개를 초과함"
        )

    interval_masks: List[int] = []
    for start, end in intervals:
        length = end - start
        interval_masks.append(((1 << length) - 1) << (start - gold_start))

    state_count = 1 << interval_count
    union_masks = [0] * state_count
    for mask in range(1, state_count):
        lowest = mask & -mask
        interval_index = lowest.bit_length() - 1
        union_masks[mask] = union_masks[mask ^ lowest] | interval_masks[interval_index]

    negative_infinity = float("-inf")
    dp = [negative_infinity] * state_count
    dp[0] = 0.0
    best = 0.0
    rank_limit = min(k, interval_count)
    for mask in range(state_count):
        if dp[mask] == negative_infinity:
            continue
        selected = mask.bit_count()
        best = max(best, dp[mask])
        if selected >= rank_limit:
            continue
        old_union = union_masks[mask]
        old_chars = old_union.bit_count()
        rank = selected + 1
        for interval_index, interval_mask in enumerate(interval_masks):
            bit = 1 << interval_index
            if mask & bit:
                continue
            next_mask = mask | bit
            new_chars = (old_union | interval_mask).bit_count() - old_chars
            gain = (new_chars / gold_length) / math.log2(rank + 1)
            dp[next_mask] = max(dp[next_mask], dp[mask] + gain)
    return best


def coverage_idcg(item: Mapping[str, Any], k: int) -> float:
    """canonical Gold interval로 exact coverage-aware IDCG를 계산한다."""

    intervals = unique_gold_intervals(item)
    return _coverage_idcg_exact(
        intervals,
        item["gold_start"],
        item["gold_end"] - item["gold_start"],
        k,
    )


def coverage_ndcg(item: Mapping[str, Any], retrieved: Sequence[str], k: int) -> float:
    """중복 보상 없는 marginal coverage DCG를 exact IDCG로 정규화한다."""

    top = list(retrieved[:k])
    idcg = coverage_idcg(item, k)
    if idcg <= 0.0:
        return 0.0
    score = coverage_dcg(item, top) / idcg
    if score < -FLOAT_TOLERANCE or score > 1.0 + FLOAT_TOLERANCE:
        raise ValueError(
            f"{item.get('id')}: coverage_ndcg@{k} 범위 위반: {score}"
        )
    return min(1.0, max(0.0, score))


def legacy_ndcg(item: Mapping[str, Any], retrieved: Sequence[str], k: int) -> float:
    """기존 gold_coverage 독립 relevance 방식의 nDCG를 보존한다."""

    gold_map = {gold["chunk_id"]: gold for gold in item["gold_chunks"]}
    top = list(retrieved[:k])
    relevances = [
        gold_map[chunk_id]["gold_coverage"] if chunk_id in gold_map else 0.0
        for chunk_id in top
    ]
    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )
    ideal = sorted(
        (gold["gold_coverage"] for gold in item["gold_chunks"]),
        reverse=True,
    )[:k]
    idcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def budget_cut(
    retrieved: Sequence[str],
    corpus: Dict[str, dict],
    budget: int,
) -> List[str]:
    """기존 방식대로 순위순 문자 예산까지 complete chunk를 선택한다."""

    out: List[str] = []
    total = 0
    for chunk_id in retrieved:
        length = len(corpus.get(chunk_id, {}).get(BUDGET_TEXT_FIELD, ""))
        if out and total + length > budget:
            break
        out.append(chunk_id)
        total += length
    return out


def score_item(
    item: dict,
    retrieved: Sequence[str],
    corpus: Dict[str, dict],
    k_values: Sequence[int],
    budget: int,
) -> dict:
    """기존 지표와 legacy/coverage-aware nDCG를 문항별로 계산한다."""

    gold_map = {gold["chunk_id"]: gold for gold in item["gold_chunks"]}
    gold_ids = set(gold_map)
    is_gold = [chunk_id in gold_ids for chunk_id in retrieved]
    same_doc = [
        corpus.get(chunk_id, {}).get("doc_id") == item["doc_id"]
        for chunk_id in retrieved
    ]

    gold_types = [corpus[chunk_id]["chunk_type"] for chunk_id in gold_ids if chunk_id in corpus]
    record: Dict[str, Any] = {
        "id": item["id"],
        "eval_source": item["eval_source"],
        "topic": item["topic"],
        "gold_type": Counter(gold_types).most_common(1)[0][0] if gold_types else "?",
        "n_gold": len(gold_ids),
    }

    for k in k_values:
        top = list(retrieved[:k])
        denominator = max(1, min(k, len(retrieved)))
        usr = union_span_recall(item, top)
        record[f"usr@{k}"] = round(usr, 6)
        for tau in TAUS:
            record[f"covhit@{k}-{tau}"] = 1.0 if usr >= tau else 0.0
        record[f"evhit@{k}"] = 1.0 if any(is_gold[:k]) else 0.0
        record[f"dochit@{k}"] = 1.0 if any(same_doc[:k]) else 0.0
        record[f"docprec@{k}"] = round(sum(same_doc[:k]) / denominator, 6)
        # Precision@K는 duplicate Gold를 각각 세므로 diagnostic only다.
        record[f"precision@{k}"] = round(sum(is_gold[:k]) / denominator, 6)
        gains = marginal_coverage_gains(item, top)
        contributing = sum(1 for gain in gains if gain > FLOAT_TOLERANCE)
        record[f"coverage_precision@{k}"] = round(
            contributing / denominator,
            6,
        )
        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(top, start=1):
            if chunk_id in gold_ids:
                reciprocal_rank = 1.0 / rank
                break
        record[f"mrr@{k}"] = round(reciprocal_rank, 6)
        record[f"legacy_ndcg@{k}"] = round(legacy_ndcg(item, top, k), 6)
        record[f"coverage_ndcg@{k}"] = round(coverage_ndcg(item, top, k), 6)

    selected = budget_cut(retrieved, corpus, budget)
    budget_usr = union_span_recall(item, selected)
    record[f"usr@{budget}c"] = round(budget_usr, 6)
    record[f"covhit@{budget}c-0.8"] = 1.0 if budget_usr >= 0.8 else 0.0
    record[f"chunks@{budget}c"] = len(selected)
    record[f"chars@{budget}c"] = sum(
        len(corpus.get(chunk_id, {}).get(BUDGET_TEXT_FIELD, ""))
        for chunk_id in selected
    )
    return record


METRIC_KEYS_FOR_BOOTSTRAP = "usr"


def aggregate(
    rows: List[dict],
    k_values: Sequence[int],
    budget: int,
) -> Dict[str, Any]:
    """문항별 결과를 기존 precision으로 집계한다."""

    metrics: Dict[str, Any] = {"n": len(rows)}
    if not rows:
        return metrics
    names = [
        "usr",
        "evhit",
        "dochit",
        "docprec",
        "precision",
        "coverage_precision",
        "mrr",
        "legacy_ndcg",
        "coverage_ndcg",
    ]
    for k in k_values:
        for name in names:
            metrics[f"{name}@{k}"] = round(
                statistics.mean(row[f"{name}@{k}"] for row in rows),
                4,
            )
        for tau in TAUS:
            key = f"covhit@{k}-{tau}"
            metrics[key] = round(statistics.mean(row[key] for row in rows), 4)
    budget_keys = (
        f"usr@{budget}c",
        f"covhit@{budget}c-0.8",
        f"chunks@{budget}c",
        f"chars@{budget}c",
    )
    for key in budget_keys:
        metrics[key] = round(statistics.mean(row[key] for row in rows), 4)
    return metrics


def build_evaluator_provenance(testset_path: Path) -> Dict[str, Any]:
    """새 결과 JSON에 evaluator와 입력 파일 provenance를 기록한다."""

    return {
        "name": EVALUATOR_NAME,
        "version": EVALUATOR_VERSION,
        "primary_metric": "UnionSpanRecall",
        "ranking_metric": "coverage_ndcg",
        "legacy_ndcg_retained": True,
        "coverage_precision_added": True,
        "coverage_precision_role": "diagnostic",
        "precision_diagnostic_only": True,
        "idcg_method": "exact_subset_dp",
        "max_exact_idcg_intervals": MAX_EXACT_IDCG_INTERVALS,
        "testset_path": provenance_path(testset_path),
        "testset_sha256": sha256(testset_path),
        "corpus_path": provenance_path(CORPUS),
        "corpus_sha256": sha256(CORPUS),
    }


def score_run(
    run: dict,
    testset: List[dict],
    corpus: Dict[str, dict],
    k_values: Sequence[int],
    budget: int,
    testset_path: Path,
) -> dict:
    """run 전체를 채점하고 evaluator provenance를 포함한다."""

    retrieved_by_id = {
        item["id"]: item.get("retrieved", [])
        for item in run["items"]
    }
    rows = [
        score_item(
            item,
            retrieved_by_id.get(item["id"], []),
            corpus,
            k_values,
            budget,
        )
        for item in testset
    ]

    for k in k_values:
        invalid_relation = [
            row["id"]
            for row in rows
            if row[f"dochit@{k}"] < row[f"evhit@{k}"]
        ]
        if invalid_relation:
            raise ValueError(f"DocHit < EvidenceHit @{k}: {invalid_relation[:3]}")

    segments: Dict[str, Dict[str, Any]] = {
        "전체": aggregate(rows, k_values, budget)
    }
    for source in ("v1", "v2_prose"):
        subset = [row for row in rows if row["eval_source"] == source]
        if subset:
            segments[f"src:{source}"] = aggregate(subset, k_values, budget)
    for gold_type, _ in Counter(row["gold_type"] for row in rows).most_common():
        segments[f"type:{gold_type}"] = aggregate(
            [row for row in rows if row["gold_type"] == gold_type],
            k_values,
            budget,
        )
    for topic, _ in Counter(row["topic"] for row in rows).most_common():
        segments[f"topic:{topic}"] = aggregate(
            [row for row in rows if row["topic"] == topic],
            k_values,
            budget,
        )

    return {
        "run_name": run.get("run_name", "(unnamed)"),
        "retriever": run.get("retriever"),
        "config": run.get("config", {}),
        "corpus_sha256": run.get("corpus_sha256"),
        "evaluator": build_evaluator_provenance(testset_path),
        "k_values": list(k_values),
        "budget": budget,
        "budget_text_field": BUDGET_TEXT_FIELD,
        "metrics": segments["전체"],
        "metrics_by_segment": segments,
        "items": rows,
    }


# =============================================================================
# 회귀 검증
# =============================================================================
def unchanged_metric_keys(k_values: Sequence[int], budget: int) -> List[str]:
    """기존 평가기와 정확히 같아야 하는 문항별 metric key를 반환한다."""

    keys: List[str] = []
    for k in k_values:
        keys.extend(
            [
                f"usr@{k}",
                f"covhit@{k}-0.5",
                f"covhit@{k}-0.8",
                f"evhit@{k}",
                f"mrr@{k}",
                f"dochit@{k}",
                f"docprec@{k}",
                f"precision@{k}",
            ]
        )
    keys.extend(
        [
            f"usr@{budget}c",
            f"covhit@{budget}c-0.8",
            f"chunks@{budget}c",
            f"chars@{budget}c",
        ]
    )
    return keys


def regression_check_against_legacy(
    coverage_result: Mapping[str, Any],
    legacy_result: Mapping[str, Any],
    k_values: Sequence[int],
    budget: int,
) -> None:
    """변경 금지 지표와 legacy nDCG가 기존 평가기와 정확히 같은지 검증한다."""

    coverage_rows = {row["id"]: row for row in coverage_result["items"]}
    legacy_rows = {row["id"]: row for row in legacy_result["items"]}
    if set(coverage_rows) != set(legacy_rows):
        raise ValueError("legacy regression item ID 집합 불일치")

    exact_keys = unchanged_metric_keys(k_values, budget)
    for item_id, new_row in coverage_rows.items():
        old_row = legacy_rows[item_id]
        for key in exact_keys:
            if new_row[key] != old_row[key]:
                raise ValueError(
                    f"legacy regression 실패: {item_id}/{key}: "
                    f"new={new_row[key]!r}, old={old_row[key]!r}"
                )
        for k in k_values:
            if new_row[f"legacy_ndcg@{k}"] != old_row[f"ndcg@{k}"]:
                raise ValueError(
                    f"legacy nDCG regression 실패: {item_id}/@{k}"
                )

    new_metrics = coverage_result["metrics"]
    old_metrics = legacy_result["metrics"]
    for key in exact_keys:
        if new_metrics[key] != old_metrics[key]:
            raise ValueError(
                f"aggregate regression 실패: {key}: "
                f"new={new_metrics[key]!r}, old={old_metrics[key]!r}"
            )
    for k in k_values:
        if new_metrics[f"legacy_ndcg@{k}"] != old_metrics[f"ndcg@{k}"]:
            raise ValueError(f"aggregate legacy nDCG regression 실패: @{k}")


def validate_coverage_ndcg_range(
    result: Mapping[str, Any],
    k_values: Sequence[int],
) -> None:
    """모든 문항과 aggregate coverage_ndcg가 0~1인지 검증한다."""

    for row in result["items"]:
        for k in k_values:
            score = row[f"coverage_ndcg@{k}"]
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{row['id']}: coverage_ndcg@{k}={score} 범위 위반")
    for k in k_values:
        score = result["metrics"][f"coverage_ndcg@{k}"]
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"aggregate coverage_ndcg@{k}={score} 범위 위반")


def validate_coverage_precision_consistency(
    result: Mapping[str, Any],
    run: Mapping[str, Any],
    testset: Sequence[Mapping[str, Any]],
    k_values: Sequence[int],
) -> None:
    """CoveragePrecision 범위, legacy Precision 상한, gain 합과 USR를 검증한다."""

    rows_by_id = {row["id"]: row for row in result["items"]}
    retrieved_by_id = {
        item["id"]: item.get("retrieved", [])
        for item in run["items"]
    }
    for item in testset:
        item_id = item["id"]
        row = rows_by_id[item_id]
        retrieved = retrieved_by_id[item_id]
        for k in k_values:
            coverage_precision = row[f"coverage_precision@{k}"]
            legacy_precision = row[f"precision@{k}"]
            if not 0.0 <= coverage_precision <= 1.0:
                raise ValueError(
                    f"{item_id}: coverage_precision@{k}={coverage_precision} 범위 위반"
                )
            if coverage_precision > legacy_precision + FLOAT_TOLERANCE:
                raise ValueError(
                    f"{item_id}: coverage_precision@{k}={coverage_precision} > "
                    f"precision@{k}={legacy_precision}"
                )

            top = list(retrieved[:k])
            gain_sum = sum(marginal_coverage_gains(item, top))
            usr = union_span_recall(dict(item), top)
            if not math.isclose(
                gain_sum,
                usr,
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    f"{item_id}: marginal gain 합과 USR@{k} 불일치: "
                    f"gain_sum={gain_sum}, usr={usr}"
                )

    for k in k_values:
        aggregate_score = result["metrics"][f"coverage_precision@{k}"]
        if not 0.0 <= aggregate_score <= 1.0:
            raise ValueError(
                f"aggregate coverage_precision@{k}={aggregate_score} 범위 위반"
            )



# =============================================================================
# paired bootstrap
# =============================================================================
def paired_bootstrap(
    base_rows: List[dict],
    other_rows: List[dict],
    key: str,
    n_iter: int = BOOTSTRAP_N,
) -> Dict[str, float]:
    """기존 방식의 paired bootstrap 신뢰구간을 계산한다."""

    base = {row["id"]: row[key] for row in base_rows}
    other = {row["id"]: row[key] for row in other_rows}
    ids = [item_id for item_id in base if item_id in other]
    differences = [other[item_id] - base[item_id] for item_id in ids]
    observed = statistics.mean(differences)
    random_generator = random.Random(BOOTSTRAP_SEED)
    count = len(differences)
    means = []
    for _ in range(n_iter):
        means.append(
            sum(differences[random_generator.randrange(count)] for _ in range(count))
            / count
        )
    means.sort()
    low = means[int(0.025 * n_iter)]
    high = means[int(0.975 * n_iter) - 1]
    probability_better = sum(1 for mean in means if mean > 0) / n_iter
    return {
        "delta": observed,
        "ci_low": low,
        "ci_high": high,
        "p_better": probability_better,
    }


# =============================================================================
# 출력
# =============================================================================
def print_single(result: dict, k_values: Sequence[int], budget: int) -> None:
    """coverage-aware metric을 우선하는 단일 run 표를 출력한다."""

    official_k = OFFICIAL_K if OFFICIAL_K in k_values else max(k_values)
    print(f"\n== {result['run_name']} ==  retriever={result.get('retriever')}")
    if result.get("config"):
        print(f"   config: {json.dumps(result['config'], ensure_ascii=False)}")

    print("\n[근거 확보]  USR = UnionSpanRecall, CH = CoverageHit")
    columns = [f"USR@{k}" for k in k_values] + [
        f"CH@{official_k}-0.5",
        f"CH@{official_k}-0.8",
        f"EvHit@{official_k}",
    ]
    header = f"{'세그먼트':16s} {'n':>4s} " + " ".join(
        f"{column:>10s}" for column in columns
    )
    print(header)
    print("-" * len(header))
    for name, metrics in result["metrics_by_segment"].items():
        values = [metrics[f"usr@{k}"] for k in k_values] + [
            metrics[f"covhit@{official_k}-0.5"],
            metrics[f"covhit@{official_k}-0.8"],
            metrics[f"evhit@{official_k}"],
        ]
        tag = "  ※참고" if metrics["n"] < 10 and name != "전체" else ""
        print(
            f"{name:16s} {metrics['n']:>4d} "
            + " ".join(f"{value:>10.3f}" for value in values)
            + tag
        )

    print("\n[순위 품질]  CovnDCG = coverage-aware nDCG")
    rank_header = (
        f"{'세그먼트':16s} {'n':>4s} "
        f"{'MRR@'+str(official_k):>10s} {'CovnDCG@'+str(official_k):>12s}"
    )
    print(rank_header)
    print("-" * len(rank_header))
    for name, metrics in result["metrics_by_segment"].items():
        tag = "  ※참고" if metrics["n"] < 10 and name != "전체" else ""
        print(
            f"{name:16s} {metrics['n']:>4d} "
            f"{metrics[f'mrr@{official_k}']:>10.3f} "
            f"{metrics[f'coverage_ndcg@{official_k}']:>12.3f}{tag}"
        )

    print("\n[진단]  CoveragePrecision, Legacy Precision, LegacyNDCG는 diagnostic only")
    diagnostic_header = (
        f"{'세그먼트':16s} {'n':>4s} "
        f"{'LegacyNDCG@'+str(official_k):>15s} "
        f"{'CovP@'+str(official_k):>10s} "
        f"{'LegacyP@'+str(official_k):>10s} "
        f"{'DocHit@'+str(official_k):>10s} "
        f"{'DocPrec@'+str(official_k):>10s}"
    )
    print(diagnostic_header)
    print("-" * len(diagnostic_header))
    for name, metrics in result["metrics_by_segment"].items():
        tag = "  ※참고" if metrics["n"] < 10 and name != "전체" else ""
        print(
            f"{name:16s} {metrics['n']:>4d} "
            f"{metrics[f'legacy_ndcg@{official_k}']:>15.3f} "
            f"{metrics[f'coverage_precision@{official_k}']:>10.3f} "
            f"{metrics[f'precision@{official_k}']:>10.3f} "
            f"{metrics[f'dochit@{official_k}']:>10.3f} "
            f"{metrics[f'docprec@{official_k}']:>10.3f}{tag}"
        )

    metrics = result["metrics"]
    print(f"\n[실서비스 - {budget}자 예산, {BUDGET_TEXT_FIELD} 기준]")
    print(
        f"  UnionSpanRecall@{budget}c = {metrics[f'usr@{budget}c']:.3f}"
        f"   CoverageHit@{budget}c-0.8 = {metrics[f'covhit@{budget}c-0.8']:.3f}"
        f"   ChunksUsed = {metrics[f'chunks@{budget}c']:.2f}"
        f"   (평균 {metrics[f'chars@{budget}c']:.0f}자)"
    )
    print("\n※ 주 지표는 UnionSpanRecall@K, 공식 순위 지표는 coverage_ndcg@K.")
    print("※ CoveragePrecision = Top-K 중 새로운 canonical Gold 문자를 실제로 추가한 청크 비율.")
    print("※ Legacy Precision = gold_chunks에 등록된 청크 비율.")
    print("※ CoveragePrecision, Legacy Precision, LegacyNDCG는 diagnostic only.")
    print("※ Top-ranked 수치도 실제 사례 검토와 함께 해석할 것.")


def print_compare(results: List[dict], k_values: Sequence[int], budget: int) -> None:
    """USR와 coverage-aware nDCG를 포함한 run 비교표를 출력한다."""

    official_k = OFFICIAL_K if OFFICIAL_K in k_values else max(k_values)
    base = results[0]
    print(f"\n== 비교 (기준선: {base['run_name']}) ==")
    print(
        f"   ΔUSR은 paired bootstrap {BOOTSTRAP_N}회 · 95% 신뢰구간. "
        "CI가 0을 포함하면 유의하지 않음\n"
    )

    for segment in ["전체", "src:v1", "src:v2_prose"]:
        if segment not in base["metrics_by_segment"]:
            continue
        base_metrics = base["metrics_by_segment"][segment]
        base_rows = [row for row in base["items"] if _in_seg(row, segment)]
        print(f"[{segment}]  n={base_metrics['n']}")
        header = (
            f"  {'run':26s} "
            + " ".join(f"{'USR@'+str(k):>8s}" for k in k_values)
            + f" {'CH@'+str(official_k)+'-.8':>9s}"
            + f" {'MRR@'+str(official_k):>8s}"
            + f" {'CovnDCG':>9s} {'USR@bud':>8s}"
            + f" | {'ΔUSR@'+str(official_k):>10s} {'95% CI':>18s} {'P(>0)':>7s}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for result in results:
            metrics = result["metrics_by_segment"].get(segment)
            if not metrics:
                continue
            line = (
                f"  {result['run_name'][:26]:26s} "
                + " ".join(f"{metrics[f'usr@{k}']:>8.3f}" for k in k_values)
                + f" {metrics[f'covhit@{official_k}-0.8']:>9.3f}"
                + f" {metrics[f'mrr@{official_k}']:>8.3f}"
                + f" {metrics[f'coverage_ndcg@{official_k}']:>9.3f}"
                + f" {metrics[f'usr@{budget}c']:>8.3f}"
            )
            if result is base:
                line += f" | {'(기준선)':>12s}"
            else:
                other_rows = [row for row in result["items"] if _in_seg(row, segment)]
                bootstrap = paired_bootstrap(
                    base_rows,
                    other_rows,
                    f"usr@{official_k}",
                )
                significant = (
                    ""
                    if bootstrap["ci_low"] <= 0 <= bootstrap["ci_high"]
                    else "  *"
                )
                line += (
                    f" | {bootstrap['delta'] * 100:>+9.1f}p"
                    f" [{bootstrap['ci_low'] * 100:>+6.1f}, "
                    f"{bootstrap['ci_high'] * 100:>+6.1f}]"
                    f" {bootstrap['p_better']:>7.3f}{significant}"
                )
            print(line)
        print()
    print("  * = 95% 신뢰구간이 0을 포함하지 않음")


def _in_seg(row: dict, segment: str) -> bool:
    """문항이 출력용 segment에 속하는지 판정한다."""

    if segment == "전체":
        return True
    kind, value = segment.split(":", 1)
    return {
        "src": row["eval_source"],
        "type": row["gold_type"],
        "topic": row["topic"],
    }[kind] == value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """coverage-aware evaluator CLI 인자를 파싱한다."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed chunk rankings with UnionSpanRecall and exact "
            "coverage-aware nDCG while regression-checking legacy metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", nargs="+", help="채점할 run 파일")
    parser.add_argument(
        "--compare",
        nargs="+",
        help="비교할 run 파일들 (첫 번째가 기준선)",
    )
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_K)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--out", default=None, help="결과 출력 폴더")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "입력·Gold·coverage_ndcg·coverage_precision·regression을 검증하고 "
            "파일을 쓰지 않음"
        ),
    )
    parser.add_argument(
        "--testset",
        default=str(DEFAULT_TESTSET),
        help="평가할 testset JSON; 상대경로는 PROJECT_ROOT 기준",
    )
    parser.add_argument(
        "--corpus",
        default=str(CORPUS),
        help=(
            "평가에 쓸 corpus JSONL; 상대경로는 PROJECT_ROOT 기준. "
            "기본값은 정리본 v2 이며, v1 코퍼스로 만든 과거 run 을 채점할 때 지정한다"
        ),
    )
    args = parser.parse_args(argv)
    paths = args.compare or args.run or []
    if not paths:
        parser.error("--run 또는 --compare 중 하나는 필요합니다")
    if any(k <= 0 for k in args.k):
        parser.error("--k 값은 모두 양수여야 합니다")
    if args.budget <= 0:
        parser.error("--budget은 양수여야 합니다")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    global CORPUS
    """입력 검증, 채점, regression, 출력 순서로 평가를 실행한다."""

    args = parse_args(argv)
    run_paths = [
        resolve_project_path(path)
        for path in (args.compare or args.run or [])
    ]
    k_values = sorted(set(args.k))
    testset_path = resolve_project_path(args.testset)
    corpus_path = resolve_project_path(args.corpus)
    if corpus_path != CORPUS:
        CORPUS = corpus_path
        print(f"[corpus] override → {provenance_path(CORPUS)}")
    testset = load_testset(testset_path)
    corpus = load_corpus()
    validate_gold_annotations(testset)

    print(f"[load] testset: {testset_path}")
    print(f"[load] 평가셋 {len(testset)}문항 / 코퍼스 {len(corpus)}청크")
    print("[ok] Gold canonical interval / duplicate provenance 검증 통과")

    results: List[dict] = []
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        errors, warnings = validate_run(run, testset, corpus, max(k_values))
        label = run.get("run_name", run_path.name) if isinstance(run, dict) else run_path.name
        for warning in warnings:
            print(f"  [warn] {label}: {warning}")
        if errors:
            for error in errors:
                print(f"  [ERROR] {label}: {error}")
            raise SystemExit(f"\n[fatal] {run_path} 검증 실패 - 채점을 중단합니다.")
        print(f"  [ok] {label} run 검증 통과")

        result = score_run(
            run,
            testset,
            corpus,
            k_values,
            args.budget,
            testset_path,
        )
        validate_coverage_ndcg_range(result, k_values)
        validate_coverage_precision_consistency(
            result,
            run,
            testset,
            k_values,
        )
        legacy_result = legacy_evaluator.score_run(
            run,
            testset,
            corpus,
            k_values,
            args.budget,
        )
        regression_check_against_legacy(
            result,
            legacy_result,
            k_values,
            args.budget,
        )
        result["evaluator"]["legacy_regression_passed"] = True
        result["evaluator"]["coverage_precision_validation_passed"] = True
        print(
            f"  [ok] {label} coverage metric 정합성 / legacy regression 통과"
        )

        results.append(result)

    if args.validate_only:
        print("\n[validate-only] 모든 검증 통과; 결과 파일을 생성하지 않았습니다.")
        return

    output_dir = (
        resolve_project_path(args.out)
        if args.out
        else PROJECT_ROOT / "results_retriever"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        output_path = output_dir / f"score_coverageaware_{result['run_name']}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print_single(result, k_values, args.budget)
        print(f"saved → {output_path}")

    if len(results) > 1:
        print_compare(results, k_values, args.budget)


if __name__ == "__main__":
    main()
