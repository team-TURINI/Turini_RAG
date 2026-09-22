# RAG Baseline

> **▶ 최종 파이프라인을 바로 돌리려면 [「최종 파이프라인 실행」](#최종-파이프라인-실행-2026-09-19-확정) 절로.**
> 아래 「구조」·「사용」은 초기 Dense 단독 베이스라인 설명이다.

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
│   ├── prepare_docs.py    # raw/ + chunking_data/ → data/docs.jsonl (자르기 전 문서)
│   ├── build_index.py     # docs.jsonl → 자르기 → FAISS 인덱스
│   ├── query.py           # 대화형 CLI (눈으로 확인)
│   └── evaluate.py        # 배치 평가 (Hit/MRR/nDCG@K)
└── data/                  # 코퍼스 — 구조와 규칙은 data/PREPARE_RULES.md
```

파이프라인 흐름: `질문 → FAISS 검색(top RETRIEVE_K) → 상위 TOP_K_GEN 문서로 LLM 답변`
인덱싱 흐름: `raw/ → docs.jsonl → [CHUNK_SIZE 로 자르기] → FAISS`

## 최종 파이프라인 실행 (2026-09-19 확정, 같은 날 국가 필터 추가)

```
질문 → [국가 판별: 기본 KR] → [KR·GLOBAL 문서만] Dense 50 + BM25(noun/1.2/0.75) 50 → RRF k=20 → Cohere c20 → Top3 → gpt-4.1-mini (V2_4j)
```

전 파라미터는 `config.py` 의 **`FINAL_*` 섹션** 하나에 있고, 근거는 `docs/final_pipeline_spec.md` 와
`docs/jurisdiction_report.md`(국가 필터) 에 있다. 실행 코드는 `core/pipeline.py` 의 `FinalPipeline`.
국가 필터는 코퍼스에 섞인 미국 자료(SEC·FRED·yfinance)가 한국 사용자 질문에 쓰이지 않게 **리랭커 앞**에서 거른다
(`core/jurisdiction.py`). 끄려면 `FINAL_JURISDICTION_FILTER = False`.

### 1. 설치

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY, COHERE_API_KEY 둘 다 입력
```

### 2. 로컬 파일 (git 에 없음 — 드라이브에서 받아 같은 경로에 둘 것)

| 경로 | 내용 |
|---|---|
| `data/chunking_data/fixed_450_70/clean_chunks_450_70_v2.jsonl` | 정리본 코퍼스 (1,698청크) |
| `vectorstores/fixed_450_70_v2/` | FAISS 인덱스 (`index.faiss` + `index.pkl`) |
| `data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v3.json` | 평가셋 v3 (157문항, 국가 라벨·재라벨 반영) |
| `data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v3_retrieval.json` | 검색 채점용 155문항 (회피 정답 2문항 제외) |
| `data/gen_contexts/split.json` | dev 107 / holdout 50 분할 |
| `data/jurisdiction.json` | 문서별 국가 라벨 — `python scripts/label_jurisdiction.py` 로 직접 생성 가능 (`data/docs_metadata_cleaned.jsonl` 필요) |

없으면 `run_pipeline.py` 가 어느 파일이 없는지 알려주고 멈춘다.

### 3. 실행

```bash
# 157문항 전체 — 검색·리랭킹·생성 (Trial 키면 Cohere 분당 10회 제한으로 약 22분)
python scripts/run_pipeline.py --tag my_run

# holdout 50 만 / 검색만(무료) / 파일럿 5문항
python scripts/run_pipeline.py --tag my_run --split holdout
python scripts/run_pipeline.py --tag chk    --skip-generation
python scripts/run_pipeline.py --tag smoke  --limit 5

# 기준선 (같은 코퍼스, Dense 단독 Top5 + 프롬프트 v1)
python scripts/run_pipeline.py --tag base --mode baseline
```

중단돼도 같은 명령을 다시 실행하면 이어서 진행된다.

산출물 `results_e2e/<tag>/`:

```
config.json         실행 사양 + 코퍼스·인덱스 sha256
ret_dense.json      ┐
ret_bm25.json       │ 단계별 검색 순위 (runs/*.json 과 같은 형식)
ret_hybrid.json     │
ret_reranked.json   ┘
generation.json     답변 — 판정기·answer_spec 호환
```

### 4. 평가

```bash
C2=data/chunking_data/fixed_450_70/clean_chunks_450_70_v2.jsonl
TS=data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v3.json
TSR=data/testset/rag_testset_retriever_v1v2_fixed450_team_eval_v3_retrieval.json
R=results_e2e/my_run

# 검색 — 회피 정답 문항을 뺀 run 파일을 만든 뒤 USR · CovnDCG · MRR (무료)
python scripts/eval_jurisdiction.py --filter-run $R/ret_reranked.json          # → ret_reranked_r155.json
python scripts/eval_retriever_coverageaware.py --run $R/ret_reranked_r155.json --corpus $C2 --testset $TSR

# 국가 — 미국 청크 누출 · 답변 내 미국 용어 · 회피 (무료)
python scripts/eval_jurisdiction.py --run $R/ret_reranked.json --gen $R/generation.json

# 생성 — correctness · completeness · faithfulness (gpt-4.1 판정, 자료/모범답안 정보 차단)
python scripts/run_judge.py --run $R/generation.json --corpus $C2 --testset $TS --out $R

# 생성 — answer_relevance · context_relevance
python scripts/eval_rag_triad.py --gen $R/generation.json --corpus $C2 \
       --metrics answer_relevance,context_relevance --out $R

# 규정 준수 7종 · 예시문구 반복 · 토큰 → answer_spec 은 report 에서 즉석 계산
```

| 축 | 지표 | 도구 |
|---|---|---|
| 검색 | USR@k · CovnDCG@10 · MRR@10 | `eval_retriever_coverageaware.py` |
| 정확도 | correctness · completeness | `run_judge.py` |
| 근거성 | faithfulness (규정 고지 문구는 판정 제외) | `run_judge.py` |
| 관련성 | answer_relevance · context_relevance | `eval_rag_triad.py` |
| 규정 준수 | 준수율 7종 · sample_verbatim | `answer_spec.py` |
| 국가 | JP@k · US 누출 · 미국 용어 · 회피 | `eval_jurisdiction.py` |

### 5. 멀티턴 (대화형)

`multiturn/` 은 팀원(승윤)이 만든 질문 재작성·라우팅·요약·포트폴리오 생성 부품이고, `multiturn/session.py` 가
그것을 한 대화로 잇는다 — 답변 문장 확정(direct/clarify/portfolio_required 는 정형 문구), state 갱신, 요약 압축.
검색은 같은 `FinalPipeline` 을 쓰므로 국가 필터가 재작성된 질문에 그대로 적용된다.

```python
from multiturn.session import build_session
s = build_session()                              # 포트폴리오 있으면 build_session(portfolio={...})
print(s.ask("채권이 뭐야?").answer)
r = s.ask("그럼 종류는?")                          # 후속 질문 → 재작성돼 검색됨
print(r.rewrite.retrieval_query, r.answer)
```

```bash
python scripts/run_multiturn.py --tag smoke      # scripts/scenarios/multiturn_smoke.json 6개 대화 22턴
                                                  # → results_multiturn/smoke/{turns.json, transcript.md}
python -m unittest discover -s tests -q          # 단위 테스트 (API 키 불필요)
```

### 6. 한 문항만 돌려보기

```python
from core.pipeline import FinalPipeline
p = FinalPipeline()                 # 인덱스·BM25·Cohere 로드 (약 40초, 1회)
out = p.run("펀드는 예금처럼 원금이 보장되나요?")
print(out["answer"])
print(out["context_chunk_ids"])     # 생성에 쓴 청크 3개
print(out["reranked"][:10])         # 리랭커 순위
```

---

## 설치 (초기 베이스라인)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY 입력
```

## 사용

```bash
# 0) 자르기 전 문서 복원 (최초 1회 — API 비용 없음)
python scripts/prepare_docs.py

# 1) 인덱스 빌드 (OpenAI 임베딩 API 비용 발생 — 전체 1회 약 $0.07)
python scripts/build_index.py --dry-run    # 자르기 결과만 미리 확인 (무료)
python scripts/build_index.py

# 2) 대화형 질의
python scripts/query.py
python scripts/query.py "ETF 가 뭐야?" --show-context

# 3) 배치 평가 (data/testset/rag_testset_merged.json 사용)
python scripts/evaluate.py --oracle                 # API 없이 채점 검증 + 설정별 상한
python scripts/evaluate.py --budget 2500            # 컨텍스트 예산 고정 비교 병행
python scripts/evaluate.py --generate               # 답변 생성까지 (LLM 비용)
```

## 실험 방법

`config.py` 값 하나만 바꾸고 스크립트를 재실행한다 (controlled experiment).

| 노브 | 의미 |
|------|------|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `SPLITTER` | 청킹 (바꾸면 재빌드 — 설정별로 인덱스가 분리 저장됨) |
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

# 예: 청크 사이즈 스윕 (config.py 수정 없이 환경변수로)
for s in 200 300 500 800 1200; do
  CHUNK_SIZE=$s CHUNK_OVERLAP=$((s/10)) python scripts/build_index.py
  CHUNK_SIZE=$s CHUNK_OVERLAP=$((s/10)) python scripts/evaluate.py --tag c$s
done
```

## 확장 지점

리랭커·하이브리드 검색 등을 붙일 때:
- 새 모듈을 `core/` 에 추가 (예: `core/reranker.py`)
- `core/pipeline.py` 의 `retrieve → generate` 사이에 단계를 끼워 넣기
- `config.py` 에 on/off 토글과 파라미터 추가

## 평가 방식

정답은 `chunk_id` 가 아니라 **정답 구간(gold_span)** 이다. 청킹 설정을 바꾸면 chunk_id
가 전부 달라지므로 ID 대조로는 채점이 불가능하기 때문이다.

```json
{ "id": "v2_054", "question": "해외주식 팔면 세금을 얼마나 내나요?",
  "doc_id": "doc_475f69f7c25b", "gold_span": "해외주식 양도소득세율은 ...",
  "segment": "prose" }
```

채점: 정답 구간을 문장 단위로 쪼개고, **top-K 청크들의 합집합**이 그 문장들을 얼마나
덮는지 본다. 단일 청크 커버리지로 채점하면 작은 청크가 산술적으로 불리해진다.

세그먼트별로 분해해 본다 — `prose` 124 / `section` 28 / `short` 9.
청크 사이즈의 실제 대상은 `prose` 이고, `section` 은 정답이 기존 섹션 청킹에서 나와
순환 편향이 있으므로 참고용이다. 자세한 내용은 `data/testset/README.md` 참조.
