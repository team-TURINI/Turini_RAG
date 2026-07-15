"""FAISS 인덱스 빌드 (Dense 전용).

첫 실행 시 1회 (OpenAI 임베딩 API 호출 비용 발생).
이후 vectorstores/ 를 그대로 두면 리트리버는 로드만 한다.

원본 데이터 형식 (JSONL, 1줄 = 1청크):
  - 공통: doc_id, chunk_id, source_name, source_url, title, section_title,
          topic, text, embedding_text, ...

산출:
  <INDEX_DIR>/<DOMAIN>/chunk_record/{index.faiss, index.pkl}

실행:
  python scripts/build_index.py
  # 데이터 경로/패턴 override
  CORPUS_GLOB="data/01_*.jsonl,data/02_*.jsonl" python scripts/build_index.py

------------------------------------------------------------
실험 포인트:
  - PAGE_CONTENT_SOURCE : 'embedding_text' vs 'text' vs 'title+text'
  - CORPUS_GLOB         : 어떤 JSONL 을 인덱싱할지
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 import path 에 추가 (scripts/ 하위에서 실행 가능하도록)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import DOMAIN, EMBEDDING_MODEL, INDEX_DIR, OPENAI_API_KEY, PROJECT_ROOT

# ============================================================
# 코퍼스 선택 — 기본: data/ 하위 모든 .jsonl
# ============================================================
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
CORPUS_GLOB = os.getenv("CORPUS_GLOB", "")


def _resolve_corpus_files() -> List[Path]:
    if CORPUS_GLOB:
        files: List[Path] = []
        for pat in [p.strip() for p in CORPUS_GLOB.split(",") if p.strip()]:
            files.extend(sorted(PROJECT_ROOT.glob(pat)))
        return files
    return sorted(DEFAULT_DATA_DIR.glob("*.jsonl"))


# ============================================================
# page_content — 검색에 들어가는 단일 텍스트.
#   embedding_text 가 있으면 그대로, 없으면 title + text 로 fallback.
# ============================================================
def _row_to_text(row: dict) -> str:
    et = (row.get("embedding_text") or "").strip()
    if et:
        return et
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


# ============================================================
# 메타데이터로 보존할 필드 (표시·평가용).
# ============================================================
META_FIELDS = [
    "doc_id",
    "chunk_id",
    "chunk_index",
    "total_chunks",
    "source_name",
    "source_url",
    "source_file",
    "title",
    "section_title",
    "topic",
    "published_date",
    # FSS(금융사) 전용 — 있을 때만
    "company_name",
    "sector_name",
    "sector_code",
    "fin_co_no",
    "homepage_url",
    # tax_deduction 전용
    "subsection_type",
]


def _row_to_metadata(row: dict) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for f in META_FIELDS:
        v = row.get(f)
        if v not in (None, "", []):
            meta[f] = v
    # 표시용 name — FSS 는 회사명, 그 외엔 title
    meta["name"] = row.get("company_name") or row.get("title") or ""
    return meta


# ============================================================
# JSONL 로더 (UTF-8 BOM 허용)
# ============================================================
def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name}:{ln} JSON parse error: {e}")
    return rows


def load_source_documents() -> List[Document]:
    files = _resolve_corpus_files()
    if not files:
        raise SystemExit(
            f"인덱싱 대상 JSONL 을 찾을 수 없습니다.\n"
            f"기본 경로: {DEFAULT_DATA_DIR}/*.jsonl\n"
            f"또는 CORPUS_GLOB 환경변수로 패턴 지정 (콤마 구분, 루트 기준 상대경로)."
        )

    docs: List[Document] = []
    for path in files:
        rows = _load_jsonl(path)
        n_added = 0
        for row in rows:
            text = _row_to_text(row)
            if not text.strip():
                continue
            docs.append(Document(page_content=text, metadata=_row_to_metadata(row)))
            n_added += 1
        print(f"  - {path.name}: {n_added} chunks")

    print(f"[load] {len(files)} files → {len(docs)} chunks total")
    return docs


# ============================================================
# FAISS 빌드
# ============================================================
def build_faiss(docs: List[Document]) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    print(f"[faiss] embedding ({EMBEDDING_MODEL}) — OpenAI API 호출 시작...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(docs, embeddings)
    out_dir = INDEX_DIR / DOMAIN / "chunk_record"
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    print(f"[faiss] saved → {out_dir}")


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== building FAISS index into: {INDEX_DIR} ==")
    docs = load_source_documents()
    print()
    build_faiss(docs)
    print("\n== done ==")
    print("이제 다음 명령으로 질의:")
    print("  python scripts/query.py")


if __name__ == "__main__":
    main()
