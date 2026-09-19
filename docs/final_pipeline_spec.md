# 최종 RAG 파이프라인 확정 사양

**코드 반영** `config.py` 「★ 최종 파이프라인 확정 사양」 섹션 — **파이프라인은 이 섹션만 읽음**
**목적** 청킹 → 임베딩 → 검색 → 리랭킹 → 생성을 한 파이프라인으로 묶어 테스트셋으로 정량·정성 평가

---

## 0. 한 장 요약

```
문서 575개
   ↓ ① 청킹        긴 문서만 450/70, 짧은 문서는 통째로        → 1,698청크 (정리본 v2)
   ↓ ② 임베딩      text-embedding-3-large                   → FAISS
   ↓ ③ 검색        Dense 50 + BM25(noun/1.2/0.75) 50
                   → RRF k=20, 5:5                          → 상위 50
   ↓ ④ 리랭킹      Cohere rerank-v4.0-pro, candidate_k=20   → 상위 20 재정렬
   ↓ ⑤ 컨텍스트    Top3                                      → 약 1,000 토큰
   ↓ ⑥ 생성        gpt-4.1-mini, 프롬프트 V2_4               → 답변
```

---

## 1. 단계별 사양과 근거

### ① 청킹

| 항목 | 값 | `config.py` |
|---|---|---|
| 코퍼스 | `clean_chunks_450_70_v2.jsonl` (1,698청크) | `FINAL_CORPUS` |
| 규칙 | 긴 일반 문서만 450자 / overlap 70, 나머지는 통째로 | — (사전 생성) |
| 정리본 | 중복 35 + 광고·네비 25 = 57청크 제거, gold 전량 보존 | — |

**현재 코퍼스가 이미 이 규칙과 일치함 → 재청킹하지 않음.**

```
문서 575개 = 청크 1개(통째로) 484  +  여러 청크(분할) 91
  통째로   whole 456 · faq 17 · size 11      중앙 299자 / 최대 549자
  분할     size 66 · section 24 · atomic 1
```

**정리본(v2) 채택 근거** — 중복 청크가 있으면 같은 근거가 두 번 검색되고 정밀도 지표가 왜곡됨.
팀원 Cohere 실험은 원본(1,755)에서 됐으므로 **본 파이프라인에서 v2 기준 수치를 다시 냄.**

### ② 임베딩

| 항목 | 값 | `config.py` |
|---|---|---|
| 모델 | `text-embedding-3-large` | `EMBEDDING_MODEL` |
| 인덱스 | `vectorstores/fixed_450_70_v2/` | `FINAL_INDEX_DIR` |
| 필드 | `embedding_text` (제목 + 본문) | `FINAL_CTX_FIELD` |

### ③ 검색 — Hybrid

| 항목 | 값 | `config.py` | 근거 |
|---|---|---|---|
| Dense 후보 | 50 | `FINAL_DENSE_K` | 지우 최종 설정 |
| BM25 후보 | 50 | `FINAL_BM25_K` | 〃 |
| BM25 토크나이저 | Kiwi `noun` = {NNG, NNP, SL, SN, SH} | `FINAL_BM25_TOKENIZER` | 27조합 전수 · `all` 대비 +9.6%p |
| BM25 k1 / b | 1.2 / 0.75 | `FINAL_BM25_K1/B` | 27조합 전수 · 융합 후 `b=0.75` 우세 |
| 결합 | RRF, `k=20`, 가중치 5:5 | `FINAL_RRF_K/W_*` | RRF 는 크기 불변 → 5:5 ≡ 1:1 |
| pool | 50 | `FINAL_HYBRID_POOL` | 리랭커에 넉넉히 넘기기 위해 |


### ④ 리랭킹 — Cohere

| 항목 | 값 | `config.py` |
|---|---|---|
| 제공자 · 모델 | Cohere `rerank-v4.0-pro` | `FINAL_RERANK_PROVIDER/MODEL` |
| candidate_k | **20** | `FINAL_CANDIDATE_K` |

**c=20 근거 (팀원 실험)** — 검색 지표는 c30 이 소폭 우세하나(USR@10 .9474 vs .9410),
**생성 단계 correctness 에서 유의한 차이 없음** (Top3: 1.9873 vs 1.9936, CI 가 0 포함).
품질이 같으면 리랭커 처리 후보가 33% 적은 쪽을 택함.

**리랭커 채택 근거** — Top1·Top3 처럼 적은 컨텍스트에서 답변 정확도가 유의하게 향상됨.
Hybrid → Cohere c30: USR@1 +31.5%p, MRR@10 +23.8%p.

### ⑤ 컨텍스트

| 항목 | 값 | `config.py` | 근거 |
|---|---|---|---|
| 개수 | 리랭커 재정렬 **Top3** | `FINAL_TOP_K_GEN` | budget(5.7개) 대비 입력 토큰 39% 절감, 품질 동일 |


### ⑥ 생성

| 항목 | 값 | `config.py` |
|---|---|---|
| 모델 | `gpt-4.1-mini` | `FINAL_MODEL` |
| temperature / top_p / max_tokens | 0.1 / 0.9 / 768 | `FINAL_TEMPERATURE/TOP_P/MAX_TOKENS` |
| 프롬프트 | **V2_4** = v2 시스템 + 질문 뒤 「출력 전 확인」 블록 | `FINAL_PROMPT_VERSION="v2"`, `FINAL_INSTRUCTION_PLACEMENT="check"` |

**코드** — `core.generator.PROMPT_PRESETS["v2_4"]` → `("v2", "check")`

**V2_4 유저 템플릿**

```
# 금융상품 자료
{retrieved_context}

# 사용자 질문
{user_question}

# 출력 전 확인
최종 답변을 내기 전 질문에 직접 답했는지, 150~250자의 한 문단인지, 필요한 원금 손실 고지와
투자 권유 아님 문구가 있는지, 모든 숫자에 단위·기준이 있는지 확인하고 어긋난 부분만 고치세요.
점검 과정은 쓰지 말고 답변만 출력하세요.
```

---

## 2. 평가 설계

### 2-1. 테스트셋 — 선택지 c

| 층 | 문항 | 이유 |
|---|---|---|
| **검색 지표** | 157 전체 | 검색 튜닝이 157 로 됐으므로 깨끗한 holdout 없음. **"개발셋 기준"임을 명시** |
| **생성 지표** | **holdout 50** | 프롬프트가 dev 107 로 튜닝됨. holdout 은 어떤 결정에도 안 씀 |

```
dev 107 / holdout 50   data/gen_contexts/split.json   seed 20260811 · 주제별 층화
```

**추후** — 신규 40문항(미사용 문서에서 생성)이 승인되면 전 구간 깨끗한 검증을 얹음.

### 2-2. 정량 지표

| 층 | 지표 | 도구 |
|---|---|---|
| 검색 | USR@1/3/10 · CovnDCG@10 · MRR@10 · CovHit@10 | `eval_retriever_coverageaware.py` |
| 생성 정확도 | correctness · completeness | `run_judge.py` (자료 차단) |
| 생성 근거성 | faithfulness | `run_judge.py` (모범답안 차단) |
| 생성 관련성 | answer_relevance · context_relevance | `eval_rag_triad.py` (AR·CR 전용) |
| 규정 준수 | 준수율 7종 · sample_verbatim | `answer_spec.py` |
| 비용 | 입력·출력 토큰 · 지연 · API 호출 수 | 실행 로그 |
| 판정 모델 | `gpt-4.1-2025-04-14`, temp 0 | `FINAL_JUDGE_MODEL` |

**기준선** — Dense 단독(`RETRIEVE_K=5`, `TOP_K_GEN=5`, 프롬프트 v1)을 같은 문항에 돌려
**베이스라인 → 최종** 개선폭을 paired bootstrap(2,000회, seed 20260805)으로 검정.

### 2-3. 정성 평가 (설계 확정 필요)

| 항목 | 안 |
|---|---|
| 표본 | 실패·경계 문항 우선 + 무작위, 약 30문항 |
| 방식 | 블라인드 쌍 비교 (베이스라인 vs 최종, 구성 미표시) |
| 루브릭 | 정확성 · 근거 충실 · 가독성 · 규정 고지 · 유해 답변 |
| 기록 | 문항별 판정 + 코멘트 → 정량 결과와 대조 |




