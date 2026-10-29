"""Download public quiz data, apply corrections, and build an isolated FAISS index."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QUIZ_CORPUS,
    QUIZ_INDEX_DIR,
)

SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/team-TURINI/Turini-e2e/main/public/data/"
)
SOURCE_FILES = (
    "quizData_864_FINAL.json",
    "quizData_overrides.json",
    "diagnostic_quiz_level_style_24.json",
)
EMBEDDING_USD_PER_MILLION_TOKENS = 0.13


def _download_json(filename: str) -> tuple[Any, str]:
    request = urllib.request.Request(
        SOURCE_BASE_URL + filename,
        headers={"User-Agent": "Turini-RAG-quiz-index-builder"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    raise TypeError(f"퀴즈 필드의 예상하지 못한 값 형식: {type(value).__name__}")


def _quiz_embedding_text(
    *,
    quiz_id: str,
    category: str,
    difficulty: str,
    quiz_type: str,
    question: str,
    choices: list[str],
    answer: str,
    explanation: str,
    key_concept: str = "",
    learning_goal: str = "",
    weakness_tag: str = "",
) -> str:
    lines = [
        f"퀴즈 ID: {quiz_id}",
        f"분야: {category}",
        f"난이도: {difficulty}",
        f"유형: {quiz_type}",
        f"문제: {question}",
    ]
    if choices:
        lines.append("선택지:\n" + "\n".join(choices))
    lines.extend(
        [
            f"정답: {answer or '미제공'}",
            f"해설: {explanation or '미제공'}",
        ]
    )
    if key_concept:
        lines.append(f"핵심 개념: {key_concept}")
    if learning_goal:
        lines.append(f"학습 목표: {learning_goal}")
    if weakness_tag:
        lines.append(f"취약 태그: {weakness_tag}")
    return "\n".join(lines)


def _main_quiz_row(row: dict[str, Any]) -> dict[str, Any]:
    quiz_id = _display(row.get("id"))
    question = _display(row.get("question"))
    answer = _display(row.get("answer"))
    explanation = _display(row.get("explanation"))
    if not quiz_id or not question or not answer or not explanation:
        raise ValueError(f"필수 정답/해설 필드가 비어 있는 퀴즈: {quiz_id or '(ID 없음)'}")

    raw_choices = row.get("choices") or []
    if not isinstance(raw_choices, list):
        raise TypeError(f"{quiz_id}: choices는 list여야 합니다.")
    choices = [_display(value) for value in raw_choices if _display(value)]
    category = _display(row.get("category"))
    difficulty = _display(row.get("difficulty"))
    quiz_type = _display(row.get("type"))
    key_concept = _display(row.get("key_concept"))
    learning_goal = _display(row.get("learning_goal"))
    embedding_text = _quiz_embedding_text(
        quiz_id=quiz_id,
        category=category,
        difficulty=difficulty,
        quiz_type=quiz_type,
        question=question,
        choices=choices,
        answer=answer,
        explanation=explanation,
        key_concept=key_concept,
        learning_goal=learning_goal,
    )
    return {
        "chunk_id": f"quiz:{quiz_id}",
        "doc_id": quiz_id,
        "quiz_id": quiz_id,
        "title": question,
        "source_name": _display(row.get("source_name")),
        "source_url": _display(row.get("source_url")),
        "category": category,
        "difficulty": difficulty,
        "quiz_type": quiz_type,
        "question": question,
        "choices": choices,
        "answer": answer,
        "explanation": explanation,
        "key_concept": key_concept,
        "learning_goal": learning_goal,
        "verification_status": _display(row.get("verification_status")),
        "embedding_text": embedding_text,
    }


def _diagnostic_row(row: dict[str, Any]) -> dict[str, Any]:
    base_id = _display(row.get("diagnostic_quiz_id"))
    question_no = row.get("question_no")
    if not base_id or type(question_no) is not int:
        raise ValueError("진단 퀴즈에는 diagnostic_quiz_id와 정수 question_no가 필요합니다.")
    question = _display(row.get("question_text"))
    if not question:
        raise ValueError(f"{base_id}/{question_no}: question_text가 비어 있습니다.")
    quiz_id = f"{base_id}-{question_no}"
    choices = [
        f"{number}. {_display(row.get(f'choice_{number}'))}"
        for number in range(1, 7)
        if _display(row.get(f"choice_{number}"))
    ]
    raw_answer = row.get("answer")
    answer = ""
    if raw_answer is not None:
        if type(raw_answer) is int:
            if not 1 <= raw_answer <= 6:
                raise ValueError(f"{quiz_id}: answer 번호가 1~6 범위를 벗어났습니다.")
            answer = f"{raw_answer}. {_display(row.get(f'choice_{raw_answer}'))}".strip()
        else:
            answer = _display(raw_answer)
    explanation = _display(row.get("explanation"))
    category = _display(row.get("category"))
    difficulty = _display(row.get("difficulty"))
    quiz_type = _display(row.get("question_type"))
    weakness_tag = _display(row.get("weakness_tag"))
    embedding_text = _quiz_embedding_text(
        quiz_id=quiz_id,
        category=category,
        difficulty=difficulty,
        quiz_type=quiz_type,
        question=question,
        choices=choices,
        answer=answer,
        explanation=explanation,
        weakness_tag=weakness_tag,
    )
    return {
        "chunk_id": f"diagnostic:{quiz_id}",
        "doc_id": quiz_id,
        "quiz_id": quiz_id,
        "title": question,
        "source_name": "diagnostic_quiz_level_style_24.json",
        "source_url": "",
        "category": category,
        "difficulty": difficulty,
        "quiz_type": quiz_type,
        "question": question,
        "choices": choices,
        "answer": answer,
        "explanation": explanation,
        "key_concept": "",
        "learning_goal": "",
        "weakness_tag": weakness_tag,
        "embedding_text": embedding_text,
    }


def normalize_quiz_data(
    quiz_rows: Any,
    overrides: Any,
    diagnostic_rows: Any,
) -> list[dict[str, Any]]:
    if not isinstance(quiz_rows, list) or not all(isinstance(row, dict) for row in quiz_rows):
        raise TypeError("quizData_864_FINAL.json은 객체 목록이어야 합니다.")
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, dict)
        for key, value in overrides.items()
    ):
        raise TypeError("quizData_overrides.json은 ID → 수정사항 객체여야 합니다.")
    if not isinstance(diagnostic_rows, list) or not all(
        isinstance(row, dict) for row in diagnostic_rows
    ):
        raise TypeError("diagnostic_quiz_level_style_24.json은 객체 목록이어야 합니다.")

    ids = {_display(row.get("id")) for row in quiz_rows}
    if "" in ids or len(ids) != len(quiz_rows):
        raise ValueError("주 퀴즈 데이터에 빈 ID 또는 중복 ID가 있습니다.")
    unknown_override_ids = set(overrides) - ids
    if unknown_override_ids:
        raise ValueError(f"overrides에 원본에 없는 ID가 있습니다: {sorted(unknown_override_ids)}")

    allowed_override_fields = {"question", "explanation"}
    for quiz_id, patch in overrides.items():
        unexpected = set(patch) - allowed_override_fields
        if unexpected:
            raise ValueError(f"{quiz_id}: 지원하지 않는 override 필드: {sorted(unexpected)}")

    normalized: list[dict[str, Any]] = []
    for row in quiz_rows:
        effective = {**row, **overrides.get(_display(row.get("id")), {})}
        normalized.append(_main_quiz_row(effective))
    normalized.extend(_diagnostic_row(row) for row in diagnostic_rows)

    chunk_ids = [row["chunk_id"] for row in normalized]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("정규화된 퀴즈에 중복 chunk_id가 있습니다.")
    return normalized


def _estimate_cost(rows: list[dict[str, Any]]) -> tuple[int, float]:
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = sum(len(encoding.encode(row["embedding_text"])) for row in rows)
    estimated_usd = token_count * EMBEDDING_USD_PER_MILLION_TOKENS / 1_000_000
    print(f"[cost] 문항 {len(rows):,}개 / 예상 입력 토큰 {token_count:,}")
    print(
        f"[cost] {EMBEDDING_MODEL} 단가 ${EMBEDDING_USD_PER_MILLION_TOKENS:.2f}"
        f"/1M tokens 기준 예상 비용: ${estimated_usd:.4f} USD"
    )
    return token_count, estimated_usd


def _enforce_cost_approval(estimated_usd: float, approved_usd: float | None) -> None:
    if approved_usd is None:
        raise SystemExit(
            "임베딩 API는 호출하지 않았습니다. 비용 상한 승인 후 "
            "`--approve-cost-usd <상한>`을 지정해 다시 실행하세요."
        )
    if approved_usd < 0:
        raise ValueError("--approve-cost-usd는 0 이상이어야 합니다.")
    if estimated_usd > approved_usd:
        raise SystemExit(
            f"예상 비용 ${estimated_usd:.4f}가 승인 상한 ${approved_usd:.4f}를 "
            "초과해 임베딩 API를 호출하지 않았습니다."
        )


def _write_corpus(rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    if QUIZ_CORPUS.exists():
        existing = QUIZ_CORPUS.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(
                f"기존 퀴즈 코퍼스를 덮어쓰지 않습니다: {QUIZ_CORPUS}"
            )
        return
    QUIZ_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    QUIZ_CORPUS.write_text(content, encoding="utf-8")


def build_index(rows: list[dict[str, Any]]) -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정돼 있지 않습니다.")
    if (QUIZ_INDEX_DIR / "index.faiss").exists() or (
        QUIZ_INDEX_DIR / "index.pkl"
    ).exists():
        raise FileExistsError(
            f"기존 퀴즈 인덱스를 덮어쓰지 않습니다: {QUIZ_INDEX_DIR}"
        )

    documents = [
        Document(
            page_content=row["embedding_text"],
            metadata={
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "quiz_id": row["quiz_id"],
                "title": row["title"],
                "quiz_type": row["quiz_type"],
                "source_name": row["source_name"],
            },
        )
        for row in rows
    ]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(QUIZ_INDEX_DIR))
    print(f"[faiss] 저장 완료: {QUIZ_INDEX_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-cost-usd",
        type=float,
        help="임베딩 API 호출 비용 상한 (이 상한보다 예상액이 높으면 중단)",
    )
    args = parser.parse_args()

    source_data: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for filename in SOURCE_FILES:
        source_data[filename], source_hashes[filename] = _download_json(filename)
    rows = normalize_quiz_data(
        source_data["quizData_864_FINAL.json"],
        source_data["quizData_overrides.json"],
        source_data["diagnostic_quiz_level_style_24.json"],
    )
    _, estimate = _estimate_cost(rows)
    if args.approve_cost_usd is None:
        print("[dry-run] 임베딩 API 호출 없이 비용만 산정했습니다.")
        return
    _enforce_cost_approval(estimate, args.approve_cost_usd)
    for filename, digest in source_hashes.items():
        print(f"[source] {filename} sha256={digest}")
    print(f"[source] overrides 병합 {len(source_data['quizData_overrides.json'])}건")

    _write_corpus(rows)
    build_index(rows)
    print(
        f"[done] {len(rows)}개 퀴즈 별도 인덱스 생성 완료 "
        f"(최초 임베딩 예상 ${estimate:.4f} USD)"
    )


if __name__ == "__main__":
    main()
