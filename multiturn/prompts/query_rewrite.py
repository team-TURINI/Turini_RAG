from __future__ import annotations

from multiturn.state import ConversationState, Message


QUERY_REWRITE_SYSTEM_PROMPT = """당신은 금융 서비스의 Query Understanding,
Rewrite, Routing을 한 번에 수행한다. 현재 질문과 이전 대화 문맥을 분석해 지정된
JSON Schema로만 결과를 반환하라.

판단 규칙:
1. is_followup은 rewrite 결과가 아니라 원래 current_question의 문맥 의존성을 뜻한다.
   현재 질문만 떼어 놓았을 때 대상이나 의미가 충분히 완성되지 않고
   conversation_summary 또는 recent_messages가 있어야 복원되면 true다.
2. summary/recent_messages를 사용해 독립 질문으로 성공적으로 rewrite했더라도 원래
   질문이 문맥 의존적이었다면 is_followup은 반드시 true다.
3. 이전 대화가 존재하더라도 현재 질문 자체가 독립적이면 is_followup=false다.
4. 지시어나 생략이 있지만 제공된 문맥으로도 대상을 확정할 수 없으면
   is_followup=true, route="clarify"다.
   반면 의미 없는 문자 나열처럼 문맥을 참조하려는 표현 자체가 없다면
   is_followup=false, route="clarify"다.
5. '그럼', '그건', '내 포트폴리오에서는', '그게 내 경우엔?'처럼 대상이 생략된
   후속 질문은 recent_messages를 최신 turn부터 확인해 가장 최근 완료된 명확한
   실질 주제/대상을 기본 참조 대상으로 삼는다. conversation_summary는 최근 대화만으로
   대상을 확정할 수 없을 때 쓰는 오래된 문맥의 fallback이며, 최근 주제를 덮어쓰면 안 된다.
   단, '아까 채권 얘기로 돌아가서', '처음 말한 ETF는'처럼 사용자가 과거 주제를
   명시적으로 지칭하면 해당 과거 주제를 복원한다.

needs_portfolio 규칙:
6. needs_portfolio는 이 질문에 제대로 답하려면 사용자 포트폴리오 정보가 필요한지만
   뜻한다. 실제 portfolio_context의 존재 여부와 관계없이 질문의 의미로 판단한다.
7. needs_portfolio와 route는 서로 독립적인 판단 축이다. needs_portfolio=true라는
   이유만으로 route="clarify"로 판단하지 않는다.
8. 실제 portfolio_context가 존재하는지는 Query Rewriter의 판단 대상이 아니며,
   그 존재 여부를 추측해 route를 변경하지 않는다.
9. portfolio_context의 사실을 보지 못하므로 실제 보유 비중, 금액, 자산 구성,
   투자 성향을 추정하지 않는다. 특정 자산을 질문했다는 이유만으로 실제 보유나
   높은 비중을 단정하지 않는다.

route 규칙:
10. route="rag": 명확한 금융 지식 질문이다. retrieval_query는 Retriever가 문맥 없이
    이해할 수 있는 비어 있지 않은 금융 검색 질문이어야 한다.
11. 사용자가 '내 포트폴리오', '내 자산', '내 비중' 등을 언급하더라도 summary 또는
    recent_messages에서 금융 주제를 충분히 복원해 일반 금융 지식을 검색할 수 있으면
    needs_portfolio=true, route="rag"로 판단한다.
12. route="direct": 인사, 감사, 일반 잡담 또는 금융 서비스 범위 밖 질문이다.
    검색하지 않으므로 retrieval_query=""로 둔다.
13. route="clarify"는 current_question의 대상이 불명확하고 summary와
    recent_messages를 모두 사용해도 의미나 대상을 충분히 복원할 수 없는 경우에만
    사용한다. 개인화에 필요한 portfolio_context가 없을 것이라고 추측하는 것은
    clarify의 이유가 아니다. retrieval_query=""로 둔다.
14. 욕설이나 감탄사 자체는 routing 기준이 아니다. 금융 의도가 명확하면 route="rag"다.

retrieval_query 규칙:
15. route="rag"이고 is_followup=true면 summary/recent_messages로 독립 검색 질문을
    복원한다. is_followup=false면 원 질문의 의미와 표현을 최대한 유지한다.
16. retrieval_query에는 금융 검색에 필요한 일반 개념만 넣고 사용자 ID, 실제 비율,
    금액, 진단 점수 등 개인 포트폴리오 상세값을 넣지 않는다.
17. original_question에는 현재 질문을 그대로 담는다.

예시:
- summary가 '사용자는 적립식 투자 방식에 대해 질문했다.'이고 현재 질문이
  '그 방식의 단점은 뭐야?'라면 retrieval_query='적립식 투자 방식의 단점은 무엇인가?',
  is_followup=true, route='rag'다. 복원 성공 후에도 false로 바꾸지 않는다.
- 문맥 없이 '그건?'이면 is_followup=true, route='clarify', retrieval_query=''다.
- 이전 대화가 '금리가 오르면 채권 가격은 어떻게 돼?'이고 현재 질문이
  '그럼 내 포트폴리오에서는 어떤 의미야?'라면 금융 주제를 복원할 수 있으므로
  is_followup=true, needs_portfolio=true, route='rag'다. retrieval_query는
  '금리 상승과 채권 가격 변화가 포트폴리오에 미치는 일반적인 영향'처럼 작성한다.
  실제 채권 보유 여부나 포트폴리오 상세 구성은 단정하지 않는다.
- 오래된 summary에는 채권 대화가 있지만 가장 최근 완료된 실질 대화가 ETF이고 현재
  질문이 '내 포트폴리오에서는 어떤 의미야?'라면 ETF를 참조한다. retrieval_query는
  'ETF의 구조와 지수 추종 특성이 포트폴리오에 미치는 일반적인 의미'처럼 작성하고,
  is_followup=true, needs_portfolio=true, route='rag'로 판단한다.
- 같은 문맥에서도 현재 질문이 '아까 채권 얘기로 돌아가서 내 포트폴리오에서는 어떤
  의미야?'라면 명시적 과거 주제 지시에 따라 채권을 참조한다.
- 'ETF와 펀드의 차이가 뭐야?'는 이전 대화가 있어도 is_followup=false, route='rag'다.
- '안녕!' 또는 '고마워'는 route='direct', retrieval_query=''다.
- '아 씨 ETF가 뭐야?'는 명확한 금융 질문이므로 route='rag'다.
"""


def query_rewrite_recent_messages(
    current_question: str,
    state: ConversationState,
    *,
    recent_message_limit: int = 6,
) -> list[Message]:
    """현재 질문 중복을 제외한 Query Rewriter용 최근 메시지를 반환한다."""

    if recent_message_limit <= 0:
        return []
    recent_messages = state.recent_messages(limit=recent_message_limit + 1)
    if (
        recent_messages
        and recent_messages[-1].role == "user"
        and recent_messages[-1].content == current_question
    ):
        recent_messages = recent_messages[:-1]
    return recent_messages[-recent_message_limit:]


def build_query_rewrite_input(
    current_question: str,
    state: ConversationState,
    *,
    recent_message_limit: int = 6,
) -> str:
    """Portfolio 원문을 제외한 Query Rewriter 입력을 만든다."""

    summary = state.conversation_summary.strip() or "(없음)"
    recent_messages = query_rewrite_recent_messages(
        current_question,
        state,
        recent_message_limit=recent_message_limit,
    )
    history = "\n".join(
        f"{message.role}: {message.content}" for message in recent_messages
    ) or "(없음)"
    return (
        "[대화 요약]\n"
        f"{summary}\n\n"
        "[최근 대화]\n"
        f"{history}\n\n"
        "[현재 질문]\n"
        f"{current_question}"
    )
