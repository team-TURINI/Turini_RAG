"""실제 Query Rewrite → Final RAG → V2_P generation을 수동 검증한다.

OpenAI, Cohere, FAISS를 실제 사용하므로 정상 팀원 환경에서만 실행한다. Codex의 로컬
검증에서는 실행하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from core.generator import get_portfolio_profile  # noqa: E402
from core.pipeline import FinalPipeline  # noqa: E402
from multiturn.generation_adapter import MultiTurnGeneratorAdapter  # noqa: E402
from multiturn.orchestrator import MultiTurnOrchestrator  # noqa: E402
from multiturn.query_rewriter import QueryRewriter  # noqa: E402
from multiturn.rag_adapter import MultiTurnRAGAdapter  # noqa: E402
from multiturn.state import ConversationState  # noqa: E402


CURRENT_QUESTION = "그럼 내 포트폴리오에서는 어떤 의미야?"
PORTFOLIO_FIXTURE = {
    "sample_id": "P-01",
    "diagnosis": {
        "risk_type": "안정형",
        "style_score": 4,
        "level_type": "초급",
        "level_score": 33.3,
        "weak_tags": ["분산투자 기본 원리", "위험-수익 관계", "ETF의 개념"],
    },
    "portfolio": {
        "input_mode": "실제",
        "allocation": {
            "국내주식": 0.55,
            "해외주식": 0.0,
            "주식형 ETF·펀드": 0.15,
            "채권": 0.0,
            "현금성자산": 0.30,
            "금": 0.0,
        },
        "investment_horizon": "3~5년",
        "total_amount_krw": 12000000,
    },
}


def build_state() -> ConversationState:
    state = ConversationState(portfolio_context=PORTFOLIO_FIXTURE)
    state.add_user_message("금리가 오르면 채권 가격은 왜 내려가?")
    state.add_assistant_message(
        "기존 채권의 상대적 매력이 낮아져 가격이 조정될 수 있습니다."
    )
    return state


def check_environment() -> None:
    if not cfg.OPENAI_API_KEY or not cfg.COHERE_API_KEY:
        raise SystemExit(".env에 OPENAI_API_KEY와 COHERE_API_KEY가 모두 필요합니다.")
    required = [
        cfg.FINAL_CORPUS,
        cfg.FINAL_INDEX_DIR / "index.faiss",
        cfg.FINAL_INDEX_DIR / "index.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"필수 corpus/index 파일이 없습니다:\n{formatted}")


def main() -> int:
    check_environment()
    state = build_state()
    pipeline = FinalPipeline(verbose=True)
    generator_adapter = MultiTurnGeneratorAdapter()
    orchestrator = MultiTurnOrchestrator(
        QueryRewriter(),
        MultiTurnRAGAdapter(pipeline),
        generator_adapter,
    )

    result = orchestrator.generate_turn(
        CURRENT_QUESTION,
        state,
        scenario_name="multiturn-v2-p-smoke",
    )
    rewrite = result.rewrite_result

    print("\n[Query Understanding]")
    print(f"original_question: {rewrite.original_question}")
    print(f"retrieval_query: {rewrite.retrieval_query}")
    print(f"is_followup: {rewrite.is_followup}")
    print(f"needs_portfolio: {rewrite.needs_portfolio}")
    print(f"route: {rewrite.route}")

    print("\n[Portfolio]")
    print(f"portfolio_available: {state.portfolio_context is not None}")
    print(f"sample_id: {state.portfolio_context.get('sample_id') if state.portfolio_context else None}")

    if result.rag_result is None:
        print("\n[Retrieval] route가 rag가 아니므로 검색·생성을 수행하지 않았습니다.")
        return 0

    rag = result.rag_result
    print("\n[Reranked IDs: first 10]")
    for rank, chunk_id in enumerate(rag.reranked_ids[:10], start=1):
        print(f"{rank:02d}. {chunk_id}")

    print("\n[Top3 Context]")
    for rank, context in enumerate(rag.retrieved_context, start=1):
        print(f"\n--- context {rank} ---")
        print(context)

    print("\n[Generation]")
    print(f"status: {result.generation_status}")
    print(f"profile: {result.generation_profile}")
    print(f"base_prompt_preset: {get_portfolio_profile('v2_p').base_preset}")
    if result.generation_result is None:
        print("answer: (generation skipped)")
    else:
        generated = result.generation_result
        print(f"answer: {generated.answer}")
        print(f"latency_s: {generated.latency:.3f}")
        print(f"input_tokens: {generated.input_tokens}")
        print(f"output_tokens: {generated.output_tokens}")
        print(f"finish_reason: {generated.finish_reason}")

    print("\n[Checks]")
    print(f"Retriever query: {rag.retrieval_query}")
    print(f"Generator question: {rag.original_question}")
    print("- portfolio는 QueryRewriter/Retriever 입력이 아니라 V2_P generation에만 사용")
    print("- profile=v2_p, needs_portfolio=true, route=rag인지 확인")
    print("- 답변이 채권 비중 0을 보존하고 보유 채권이나 없는 종목을 만들지 않는지 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
