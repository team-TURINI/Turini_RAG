"""FAISS 벡터스토어 로드 (런타임 전용).

빌드(임베딩 호출 + 인덱스 저장)는 scripts/build_index.py 가 담당.
본 모듈은 이미 빌드된 인덱스를 디스크에서 로드만 한다.

인덱스 디렉토리: <INDEX_DIR>/<DOMAIN>/<INDEX_NAME>/
  - index.faiss : 벡터 인덱스 (binary)
  - index.pkl   : docstore + id 매핑 (pickle)

INDEX_NAME 에는 청킹 설정이 들어간다 (예: chunk_recursive_500_50). 따라서
config 의 CHUNK_SIZE 를 바꾸면 자동으로 그 설정의 인덱스를 바라본다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import DOMAIN, EMBEDDING_MODEL, INDEX_DIR, INDEX_NAME, OPENAI_API_KEY


def load_faiss_vectorstore(name: Optional[str] = None) -> FAISS:
    """`{INDEX_DIR}/{DOMAIN}/{name}/` 의 FAISS 인덱스 로드.

    Args:
        name: 인덱스 디렉토리명. 생략하면 config.INDEX_NAME (현재 청킹 설정).

    allow_dangerous_deserialization=True 이유:
        FAISS 저장 파일(index.pkl)에 pickle 이 포함됨 — LangChain 이 명시적
        옵트인을 요구. 이 코드가 직접 만든 파일만 로드하는 전제이므로 허용.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정돼 있지 않습니다 (.env 또는 환경변수)."
        )
    index_path = INDEX_DIR / DOMAIN / (name or INDEX_NAME)
    if not (index_path / "index.faiss").exists():
        available = sorted(
            p.name for p in (INDEX_DIR / DOMAIN).glob("*") if (p / "index.faiss").exists()
        ) if (INDEX_DIR / DOMAIN).exists() else []
        raise FileNotFoundError(
            f"FAISS 인덱스가 없습니다: {index_path}\n"
            f"  빌드된 인덱스: {available or '(없음)'}\n"
            f"  `python scripts/build_index.py` 로 빌드한 뒤 다시 시도하세요."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_faiss_from_dir(index_dir: Union[str, Path]) -> FAISS:
    """디렉토리 경로를 직접 받아 FAISS 인덱스를 로드한다.

    load_faiss_vectorstore() 는 `{INDEX_DIR}/{DOMAIN}/{INDEX_NAME}` 규칙에 묶여 있어
    초기 베이스라인(chunk_recursive_500_50)만 열 수 있다. 최종 파이프라인의 인덱스는
    `vectorstores/fixed_450_70_v2/` 처럼 평평한 구조라 이 함수로 연다.
    scripts/run_dense_baseline.py 가 여는 방식과 동일하다.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 가 설정돼 있지 않습니다 (.env 또는 환경변수).")
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {index_dir}")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
