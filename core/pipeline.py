"""RAG 파이프라인 — 한 질문을 처리하는 단일 진입점.

베이스라인 흐름:
  ① retrieve  — FAISS dense 검색 (top RETRIEVE_K)
  ② generate  — 상위 TOP_K_GEN 문서를 근거로 LLM 답변

리랭커·하이브리드 검색 등 확장은 ① 과 ② 사이에 단계를 끼워 넣으면 된다.
리트리버는 한 번만 로드해 인스턴스 수명 동안 재사용한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import TOP_K_GEN
from core.generator import GenerationResult, generate
from core.retriever import DenseRetriever


class RAGPipeline:
    """리트리버 + 제너레이터를 묶은 최소 RAG 파이프라인."""

    def __init__(self) -> None:
        self.retriever = DenseRetriever()

    def run(self, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """한 질문 처리.

        Returns:
            (answer, doc_ids, meta)
              - answer  : 생성된 답변
              - doc_ids : LLM 에 넘긴 문서들의 doc_id 리스트
              - meta    : 디버그/평가용 메타데이터
        """
        # ① retrieve
        candidates = self.retriever.search(question)
        top_k = candidates[:TOP_K_GEN]

        contexts = [c["page_content"] for c in top_k]
        doc_ids = [c["doc_id"] for c in top_k]

        # ② generate
        result: GenerationResult = generate(question, contexts)

        meta = {
            "n_candidates": len(candidates),
            "top_k_doc_ids": doc_ids,
            "latency_s": round(result.latency, 3),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        return result.answer, doc_ids, meta


# =============================================================================
# ★ 최종 파이프라인 (2026-09-19 팀 확정 사양)
#
#   질문 → Dense 50 + BM25 50 → RRF(k=20, 5:5) → Cohere 가 상위 20 재정렬
#        → Top3 → V2_4 프롬프트로 생성
#
#   설정은 config.py 의 FINAL_* 만 읽는다. 위 RAGPipeline(Dense 단독)은 기준선 비교용으로
#   그대로 둔다. 각 단계 결과를 전부 돌려주는 이유는 (1) 오프라인 run 파일과 순위 대조,
#   (2) 정성 평가 때 "왜 이 답이 나왔나"를 단계별로 추적하기 위해서다.
# =============================================================================
import time as _time

import config as _cfg
from core.generator import resolve_preset
from core.reranker import CohereReranker
from core.retriever import BM25Retriever, DenseIndexRetriever, HybridRetriever, load_corpus


class FinalPipeline:
    """확정 사양 RAG 파이프라인. 무거운 것(인덱스·BM25·클라이언트)은 생성 시 한 번만 로드."""

    def __init__(self, *, prompt_preset: str = "v2_4", verbose: bool = True) -> None:
        t0 = _time.time()
        self.corpus = load_corpus(_cfg.FINAL_CORPUS)
        if verbose:
            print(f"[load] 코퍼스 {len(self.corpus)}청크")
        dense = DenseIndexRetriever(_cfg.FINAL_INDEX_DIR)
        if verbose:
            print(f"[load] FAISS {_cfg.FINAL_INDEX_DIR.name}")
        bm25 = BM25Retriever(self.corpus, k1=_cfg.FINAL_BM25_K1, b=_cfg.FINAL_BM25_B,
                             text_field=_cfg.FINAL_CTX_FIELD)
        if verbose:
            print(f"[load] BM25 noun/k1={_cfg.FINAL_BM25_K1}/b={_cfg.FINAL_BM25_B} 토큰화 완료")
        self.retriever = HybridRetriever(
            dense, bm25,
            dense_k=_cfg.FINAL_DENSE_K, bm25_k=_cfg.FINAL_BM25_K,
            rrf_k=_cfg.FINAL_RRF_K, w_dense=_cfg.FINAL_RRF_W_DENSE, w_bm25=_cfg.FINAL_RRF_W_BM25,
            pool=_cfg.FINAL_HYBRID_POOL, out_k=_cfg.FINAL_HYBRID_POOL,
        )
        self.reranker = CohereReranker(_cfg.FINAL_RERANK_MODEL,
                                       candidate_k=_cfg.FINAL_CANDIDATE_K,
                                       text_field=_cfg.FINAL_CTX_FIELD)
        self.prompt_version, self.placement = resolve_preset(prompt_preset)
        self.prompt_preset = prompt_preset
        if verbose:
            print(f"[load] 리랭커 {_cfg.FINAL_RERANK_MODEL} c{_cfg.FINAL_CANDIDATE_K} / "
                  f"프롬프트 {prompt_preset} / 준비 {_time.time() - t0:.1f}s")

    # 컨텍스트 조립 — scripts/run_generation.py build_context() 의 top_k 분기와 동일
    def build_context(self, ranked_ids: List[str]) -> List[str]:
        out: List[str] = []
        for cid in ranked_ids:
            text = self.corpus.get(cid, {}).get(_cfg.FINAL_CTX_FIELD, "")
            if not text:
                continue
            if len(out) >= _cfg.FINAL_TOP_K_GEN:
                break
            out.append(text)
        return out

    def retrieve(self, question: str) -> Dict[str, List[str]]:
        """생성 없이 검색·리랭킹까지만. 회귀 대조와 검색 지표 계산에 쓴다."""
        r = self.retriever.search(question)
        r["reranked"] = self.reranker.rerank(question, r["hybrid"], self.corpus)
        return r

    def run(self, question: str, *, skip_generation: bool = False) -> Dict[str, Any]:
        t0 = _time.time()
        r = self.retrieve(question)
        t_ret = _time.time() - t0
        ctx_ids = r["reranked"][: _cfg.FINAL_TOP_K_GEN]
        out: Dict[str, Any] = {
            "question": question,
            "dense": r["dense"], "bm25": r["bm25"], "hybrid": r["hybrid"], "reranked": r["reranked"],
            "context_chunk_ids": ctx_ids,
            "retrieval_latency_s": round(t_ret, 3),
        }
        if skip_generation:
            return out
        contexts = self.build_context(r["reranked"])
        g: GenerationResult = generate(
            question, contexts,
            prompt_version=self.prompt_version, placement=self.placement,
            model=_cfg.FINAL_MODEL, temperature=_cfg.FINAL_TEMPERATURE,
            max_tokens=_cfg.FINAL_MAX_TOKENS, top_p=_cfg.FINAL_TOP_P,
        )
        out.update({
            "answer": g.answer,
            "finish_reason": g.finish_reason,
            "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
            "generation_latency_s": round(g.latency, 3),
            "context_chars": sum(len(c) for c in contexts),
        })
        return out

    def config_snapshot(self) -> Dict[str, Any]:
        """결과 파일에 함께 저장할 실행 사양. 나중에 '무슨 설정으로 낸 숫자인지' 를 남긴다."""
        import hashlib
        def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
        return {
            "corpus": _cfg.FINAL_CORPUS.name, "corpus_sha256": sha(_cfg.FINAL_CORPUS),
            "index": _cfg.FINAL_INDEX_DIR.name, "index_sha256": sha(_cfg.FINAL_INDEX_DIR / "index.faiss"),
            "embedding_model": _cfg.EMBEDDING_MODEL,
            "dense_k": _cfg.FINAL_DENSE_K, "bm25_k": _cfg.FINAL_BM25_K,
            "bm25": {"tokenizer": _cfg.FINAL_BM25_TOKENIZER, "k1": _cfg.FINAL_BM25_K1, "b": _cfg.FINAL_BM25_B},
            "rrf": {"k": _cfg.FINAL_RRF_K, "w_dense": _cfg.FINAL_RRF_W_DENSE, "w_bm25": _cfg.FINAL_RRF_W_BM25,
                    "pool": _cfg.FINAL_HYBRID_POOL},
            "reranker": {"provider": _cfg.FINAL_RERANK_PROVIDER, "model": _cfg.FINAL_RERANK_MODEL,
                         "candidate_k": _cfg.FINAL_CANDIDATE_K},
            "top_k_gen": _cfg.FINAL_TOP_K_GEN, "ctx_field": _cfg.FINAL_CTX_FIELD,
            "prompt": {"preset": self.prompt_preset, "version": self.prompt_version, "placement": self.placement},
            "generation": {"model": _cfg.FINAL_MODEL, "temperature": _cfg.FINAL_TEMPERATURE,
                           "top_p": _cfg.FINAL_TOP_P, "max_tokens": _cfg.FINAL_MAX_TOKENS},
        }


class BaselinePipeline:
    """기준선 — 같은 코퍼스·인덱스에서 Dense 단독 Top5 + 프롬프트 v1.

    최초 RAGPipeline(RETRIEVE_K=5, TOP_K_GEN=5, v1)의 동작을 **최종 코퍼스 위에서** 재현한다.
    같은 chunk_id 를 쓰므로 검색·생성 지표 모두 FinalPipeline 과 직접 비교된다.
    이 기준선 대비 개선폭이 "검색·리랭커·프롬프트가 기여한 몫"이다.
    """

    TOP_K = 5

    def __init__(self, *, verbose: bool = True) -> None:
        self.corpus = load_corpus(_cfg.FINAL_CORPUS)
        self.dense = DenseIndexRetriever(_cfg.FINAL_INDEX_DIR)
        self.prompt_version, self.placement = resolve_preset("v1")
        self.prompt_preset = "v1"
        if verbose:
            print(f"[load] 기준선 — Dense 단독 Top{self.TOP_K} / 프롬프트 v1")

    def run(self, question: str, *, skip_generation: bool = False) -> Dict[str, Any]:
        t0 = _time.time()
        d = self.dense.search(question, _cfg.FINAL_DENSE_K)     # 검색 지표 비교를 위해 50개 저장
        ctx_ids = d[: self.TOP_K]
        out: Dict[str, Any] = {"question": question, "dense": d, "context_chunk_ids": ctx_ids,
                               "retrieval_latency_s": round(_time.time() - t0, 3)}
        if skip_generation:
            return out
        contexts = [self.corpus[c][_cfg.FINAL_CTX_FIELD] for c in ctx_ids if c in self.corpus]
        g: GenerationResult = generate(question, contexts,
                                       prompt_version=self.prompt_version, placement=self.placement,
                                       model=_cfg.FINAL_MODEL, temperature=_cfg.FINAL_TEMPERATURE,
                                       max_tokens=_cfg.FINAL_MAX_TOKENS, top_p=_cfg.FINAL_TOP_P)
        out.update({"answer": g.answer, "finish_reason": g.finish_reason,
                    "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
                    "generation_latency_s": round(g.latency, 3),
                    "context_chars": sum(len(c) for c in contexts)})
        return out

    def config_snapshot(self) -> Dict[str, Any]:
        import hashlib
        return {
            "mode": "baseline_dense5_v1",
            "corpus": _cfg.FINAL_CORPUS.name,
            "corpus_sha256": hashlib.sha256(_cfg.FINAL_CORPUS.read_bytes()).hexdigest(),
            "index": _cfg.FINAL_INDEX_DIR.name, "embedding_model": _cfg.EMBEDDING_MODEL,
            "dense_k": _cfg.FINAL_DENSE_K, "top_k_gen": self.TOP_K, "ctx_field": _cfg.FINAL_CTX_FIELD,
            "reranker": {"provider": None},
            "prompt": {"preset": "v1", "version": "v1", "placement": "front"},
            "generation": {"model": _cfg.FINAL_MODEL, "temperature": _cfg.FINAL_TEMPERATURE,
                           "top_p": _cfg.FINAL_TOP_P, "max_tokens": _cfg.FINAL_MAX_TOKENS},
        }
