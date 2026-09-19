"""Query Rewriter → FinalPipeline retrieval/context 연결을 수동 검증한다.

실제 OpenAI embedding/query rewrite와 Cohere reranker를 호출한다. Generator는 호출하지
않는다. Codex 검증에서는 실행하지 않고 사용자가 직접 실행한다.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from core.pipeline import FinalPipeline  # noqa: E402
from multiturn.orchestrator import MultiTurnOrchestrator  # noqa: E402
from multiturn.query_rewriter import QueryRewriter  # noqa: E402
from multiturn.rag_adapter import MultiTurnRAGAdapter  # noqa: E402
from multiturn.state import ConversationState  # noqa: E402


CURRENT_QUESTION = "그럼 둘의 추적 오차는 왜 생겨?"


def build_state() -> ConversationState:
    state = ConversationState()
    state.add_user_message("인덱스 ETF가 뭐야?")
    state.add_assistant_message("특정 지수의 성과를 추종하는 ETF입니다.")
    state.add_user_message("액티브 ETF와는 어떻게 달라?")
    state.add_assistant_message("운용 재량과 지수 추종 방식에서 차이가 있습니다.")
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
    pipeline = FinalPipeline(verbose=True)
    orchestrator = MultiTurnOrchestrator(
        QueryRewriter(),
        MultiTurnRAGAdapter(pipeline),
    )

    result = orchestrator.prepare_turn(
        CURRENT_QUESTION,
        build_state(),
        scenario_name="multiturn-rag-adapter-smoke",
    )
    rewrite = result.rewrite_result

    print("\n[Query Understanding]")
    print(f"original_question: {rewrite.original_question}")
    print(f"retrieval_query: {rewrite.retrieval_query}")
    print(f"is_followup: {rewrite.is_followup}")
    print(f"needs_portfolio: {rewrite.needs_portfolio}")
    print(f"route: {rewrite.route}")

    if result.rag_result is None:
        print("\n[Retrieval] route가 rag가 아니므로 FinalPipeline.retrieve() 미호출")
        return 0

    rag = result.rag_result
    print("\n[Reranked chunk IDs: first 10]")
    for rank, chunk_id in enumerate(rag.reranked_ids[:10], start=1):
        print(f"{rank:02d}. {chunk_id}")

    print("\n[Top3 retrieved context]")
    for rank, context in enumerate(rag.retrieved_context, start=1):
        print(f"\n--- context {rank} ---")
        print(context)

    print("\n[Expected qualitative checks]")
    print("- is_followup=true, needs_portfolio=false, route=rag")
    print("- retrieval_query가 인덱스 ETF와 액티브 ETF의 추적 오차를 독립적으로 표현")
    print("- 위 retrieval_query가 FinalPipeline.retrieve() 입력으로 사용됨")
    print("- context는 reranked 순서의 Top3이며 Generator는 호출되지 않음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
