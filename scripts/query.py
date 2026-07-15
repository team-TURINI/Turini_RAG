"""대화형 CLI — 질문 하나를 검색·생성해 눈으로 확인.

실험 중 설정(k값·모델)을 바꾼 뒤 답변 품질을 빠르게 확인하는 용도.

실행:
  python scripts/query.py                 # 대화형 루프
  python scripts/query.py "ETF 가 뭐야?"   # 단일 질문 후 종료
  python scripts/query.py --show-context   # 검색된 문서도 함께 출력
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL, RETRIEVE_K, TOP_K_GEN
from core.pipeline import RAGPipeline


def _answer_once(pipeline: RAGPipeline, question: str, show_context: bool) -> None:
    candidates = pipeline.retriever.search(question)
    if show_context:
        print("\n── 검색된 문서 (top {}) ──".format(min(TOP_K_GEN, len(candidates))))
        for c in candidates[:TOP_K_GEN]:
            title = c["metadata"].get("name") or c["metadata"].get("title") or ""
            print(f"  [{c['rank']}] score={c['score']:.4f}  {title}")
            print(f"      {c['page_content'][:120].replace(chr(10), ' ')}...")
        print()

    answer, doc_ids, meta = pipeline.run(question)
    print("── 답변 ──")
    print(answer)
    print(
        f"\n(latency {meta['latency_s']}s | in {meta['input_tokens']} / "
        f"out {meta['output_tokens']} tokens | docs {doc_ids})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 베이스라인 대화형 질의")
    parser.add_argument("question", nargs="?", help="질문 (생략 시 대화형 루프)")
    parser.add_argument("--show-context", action="store_true", help="검색된 문서도 출력")
    args = parser.parse_args()

    print(f"== RAG baseline | model={MODEL} | retrieve_k={RETRIEVE_K} top_k_gen={TOP_K_GEN} ==")
    print("인덱스 로딩 중...")
    pipeline = RAGPipeline()
    print("준비 완료.\n")

    if args.question:
        _answer_once(pipeline, args.question, args.show_context)
        return

    print("질문을 입력하세요. 종료: 빈 줄 입력 또는 Ctrl-C\n")
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not question:
            print("종료합니다.")
            break
        _answer_once(pipeline, question, args.show_context)
        print()


if __name__ == "__main__":
    main()
