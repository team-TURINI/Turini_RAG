from __future__ import annotations

from collections.abc import Callable
from typing import Any

import config as cfg
from core.generator import GenerationResult, generate, generate_portfolio_aware, resolve_preset
from multiturn.rag_adapter import RAGContextResult


GeneralGenerate = Callable[..., GenerationResult]
PortfolioGenerate = Callable[..., GenerationResult]


class MultiTurnGeneratorAdapter:
    """검색 결과를 원 질문으로 생성하되 portfolio는 V2_P 경로에만 전달한다."""

    def __init__(
        self,
        *,
        general_prompt_preset: str = "v2_4j",   # 국가 기준 절 포함 (2026-09-20)
        portfolio_profile: str = "v2_p",
        general_generate: GeneralGenerate = generate,
        portfolio_generate: PortfolioGenerate = generate_portfolio_aware,
    ) -> None:
        self.general_prompt_preset = general_prompt_preset
        self.portfolio_profile = portfolio_profile
        self._general_generate = general_generate
        self._portfolio_generate = portfolio_generate

    def generate_general(self, rag_result: RAGContextResult) -> GenerationResult:
        prompt_version, placement = resolve_preset(self.general_prompt_preset)
        return self._general_generate(
            rag_result.original_question,
            list(rag_result.retrieved_context),
            prompt_version=prompt_version,
            placement=placement,
            model=cfg.FINAL_MODEL,
            temperature=cfg.FINAL_TEMPERATURE,
            max_tokens=cfg.FINAL_MAX_TOKENS,
            top_p=cfg.FINAL_TOP_P,
        )

    def generate_portfolio(
        self,
        rag_result: RAGContextResult,
        portfolio_context: dict[str, Any],
    ) -> GenerationResult:
        return self._portfolio_generate(
            rag_result.original_question,
            list(rag_result.retrieved_context),
            portfolio_context,
            profile=self.portfolio_profile,
            model=cfg.FINAL_MODEL,
            temperature=cfg.FINAL_TEMPERATURE,
            max_tokens=cfg.FINAL_MAX_TOKENS,
            top_p=cfg.FINAL_TOP_P,
        )
