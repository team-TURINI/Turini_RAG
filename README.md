# RAG Baseline

RAG 실험을 위한 **최소 구성** 베이스라인. 구성요소는 **리트리버(FAISS Dense)** 와
**제너레이터(OpenAI LLM)** 둘뿐이다. 하이브리드 검색·리랭커·의도분류·쿼리재작성·
멀티턴·서빙(API)·인증 등은 모두 제외했다 — 여기서부터 하나씩 붙여가며 성능 변화를
측정하는 것이 목적이다.

## 구조

```
RAG_baseline/
├── config.py              # 실험 노브 (k값·모델·생성 파라미터) 단일 진입점
├── core/
│   ├── vectorstore.py     # FAISS 인덱스 로드
│   ├── retriever.py       # DenseRetriever — FAISS similarity search
│   ├── generator.py       # generate() — contexts + 질문 → 답변
│   └── pipeline.py        # RAGPipeline — retrieve → generate
├── scripts/
│   ├── build_index.py     # data/*.jsonl → FAISS 인덱스 빌드 (1회)
│   ├── query.py           # 대화형 CLI (눈으로 확인)
│   └── evaluate.py        # 배치 평가 (Hit/MRR/nDCG@K)
└── data/                  # 코퍼스 (JSONL, 1줄 = 1청크)
```

파이프라인 흐름: `질문 → FAISS 검색(top RETRIEVE_K) → 상위 TOP_K_GEN 문서로 LLM 답변`

## 설치

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY 입력
```

## 사용

```bash
# 1) 인덱스 빌드 (최초 1회 — OpenAI 임베딩 API 비용 발생)
python scripts/build_index.py

# 2) 대화형 질의
python scripts/query.py
python scripts/query.py "ETF 가 뭐야?" --show-context

# 3) 배치 평가 (data/goldset.json 필요)
python scripts/evaluate.py --tag baseline
python scripts/evaluate.py --tag baseline --generate   # 답변 생성까지 (LLM 비용)
```

## 실험 방법

`config.py` 값 하나만 바꾸고 스크립트를 재실행한다 (controlled experiment).

| 노브 | 의미 |
|------|------|
| `EMBEDDING_MODEL` | 임베딩 모델 (바꾸면 `build_index.py` 재빌드 필수) |
| `RETRIEVE_K` | FAISS 에서 가져올 후보 수 |
| `TOP_K_GEN` | LLM 에 넘길 최종 문서 수 |
| `MODEL` | 생성 LLM |
| `TEMPERATURE` / `MAX_TOKENS` / `TOP_P` | 생성 파라미터 |

```bash
# 예: RETRIEVE_K 스윕
# config.py 에서 RETRIEVE_K = 5 → 10 수정
python scripts/evaluate.py --tag k10
# results/ 의 baseline 결과와 비교
```

## 확장 지점

리랭커·하이브리드 검색 등을 붙일 때:
- 새 모듈을 `core/` 에 추가 (예: `core/reranker.py`)
- `core/pipeline.py` 의 `retrieve → generate` 사이에 단계를 끼워 넣기
- `config.py` 에 on/off 토글과 파라미터 추가

## goldset 형식 (`data/goldset.json`)

```json
[
  {
    "query_id": "Q001",
    "question": "보수적 투자자에게 맞는 채권 ETF 알려줘",
    "relevant_ids": ["chunk_...", "chunk_..."]
  }
]
```

`relevant_ids` 는 검색 결과의 `doc_id`(= 원본 `chunk_id`)와 매칭된다.
