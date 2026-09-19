# RAG 파이프라인 구축·평가 — 전체 진행 요약

**기준일** 2026-09-19 · **작성** 예진
**상태** 정량 평가 완료 → 정성 평가 대기

---

## 1. 최종 확정 사양 (한 장)

```
문서 575개
  ↓ ① 청킹      긴 문서만 450/70, 짧은 문서는 통째로        → 1,698청크 (정리본 v2)
  ↓ ② 임베딩    text-embedding-3-large                    → FAISS
  ↓ ③ 검색      Dense 50 + BM25(noun / k1=1.2 / b=0.75) 50
                → RRF k=20, 5:5                           → 상위 50
  ↓ ④ 리랭킹    Cohere rerank-v4.0-pro, candidate_k=20    → 상위 20 재정렬
  ↓ ⑤ 컨텍스트  Top3                                       → 약 1,000 토큰
  ↓ ⑥ 생성      gpt-4.1-mini · temp 0.1 · 프롬프트 V2_4    → 답변
```

| 단계 | 값 | 근거 문서 |
|---|---|---|
| 청킹 | 450/70 정리본 v2 (중복 35 + 광고 25 제거) | change_audit_v2.md |
| BM25 | noun 토크나이저 / k1=1.2 / b=0.75 | bm25_retune_posfix_report.md |
| Hybrid | RRF k=20, 5:5, pool 50 | retriever_2nd_report.md |
| 리랭커 | Cohere c20 | 팀원 실험 (c20 ≈ c30, 후보 33% 절감) |
| 컨텍스트 | Top3 | rerun_retuned_bm25_experiment.md |
| 프롬프트 | V2_4 = v2 + 「출력 전 확인」 블록 | prompt_experiment_report.md |
| 코드 | config.py FINAL_* 섹션 · core/pipeline.py FinalPipeline | final_pipeline_spec.md |

---

## 2. 최종 결과 — 기준선 vs 최종

**기준선** = 같은 코퍼스에서 Dense 단독 Top5 + 프롬프트 v1 (최초 베이스라인 재현)
**통계** = paired bootstrap 2,000회, 95% CI. ✓ = 유의

### 검색 (157문항)

| 지표 | 기준선 | 최종 | 차이 | 유의 |
|---|---|---|---|---|
| USR@1 | 0.4020 | **0.6844** | **+28.2%p** | ✓ |
| USR@3 | 0.6176 | **0.8637** | +24.6%p | ✓ |
| USR@10 | 0.8685 | **0.9447** | +7.6%p | ✓ |
| CovnDCG@10 | 0.6346 | **0.8399** | +20.5%p | ✓ |
| MRR@10 | 0.6212 | **0.8474** | +22.6%p | ✓ |

→ **1등에 정답이 오는 비율 40% → 68%.** 리랭커 효과는 얕게 자를수록 큼.

### 생성 — holdout 50 (프롬프트 튜닝에 안 쓴 깨끗한 표본)

| 지표 | 기준선 | 최종 | 차이 | 유의 |
|---|---|---|---|---|
| correctness (0~2) | 1.8400 | **1.9400** | +0.100 | ✗ (n=50) |
| completeness | 0.9000 | **0.9750** | +0.075 | 경계 |
| faithfulness | 1.0000 | 0.9783 | −0.022 | ✓ 관문 통과 |
| answer_relevance | 0.7294 | 0.7037 | −0.026 | ✓ 하락 |
| context_relevance | 0.1308 | **0.2351** | +0.104 | ✓ |
| **준수율 (전체)** | 0.5678 | **0.9593** | **+39.2%p** | ✓ |
| 권유 아님 명시 | **0.0000** | **1.0000** | +100%p | ✓ |
| 위험 고지 | 0.1562 | **1.0000** | +84.4%p | ✓ |
| 길이 150~250자 | 0.3750 | **0.9375** | +56.3%p | ✓ |
| 예시문구 반복 | 0.0000 | 0.3600 | +36%p | ⚠ |
| 입력 토큰 | 1,469 | **1,374** | −95 | ✓ |
| 출력 토큰 | 151 | **120** | −31 | ✓ |

### 해석

- **정확도** — 올랐으나 유의하지 않음. 기준선이 이미 1.84 로 높아 천장 효과
- **실질 성과** — 규정 준수(권유 아님 0→100%, 위험 고지 16→100%) + 검색 품질 + 비용 절감
- **faithfulness** — 고지 문구를 감점 대상에서 제외(관문 b) 후 −0.022. 관문(−0.05) 안
- **answer_relevance 하락** — 고지 문장이 붙어 질문 무관 문장 비율이 늘어난 구조적 결과
- ⚠ **예시문구 반복 36%** — V2_4 「출력 전 확인」 블록의 부작용. 답변 3개 중 1개가 같은 문장으로 끝남. 정성평가에서 확인 필요

---

## 3. 진행 경과 — 단계별

### 3-1. 검색 — BM25 재튜닝과 토크나이저 사건

**한 일**
- 팀 그리드와 내 그리드의 BM25 수치가 달라 원인 추적
- 27조합(토크나이저 3 × k1 3 × b 3) 전수 재실행, 라벨 v2·정리 코퍼스 3단계에서 재검증
- 하이브리드 융합 이후 지표로 k1·b 최종 확정

**결과**
- 원인 = noun 모드의 **품사 태그 집합이 서로 달랐음** (같은 이름, 다른 정의)
  - 팀: NNB/NR/NP 포함, SL/SN/SH 제외 → 0.757
  - 나: SL/SN/SH 포함, NNB/NR/NP 제외 → 0.846
  - ETF·IPO·PER 같은 영문 약어(SL)가 도메인 최고 IDF 토큰인데 팀 정의는 이를 버림
- 팀 그리드 숫자 6개를 소수점 3자리까지 재현 → 파이프라인은 동일, 정의만 달랐음
- k1·b 는 BM25 단독으로는 지표별 최적이 갈렸으나 **융합 후 공식 지표(CovnDCG@10)로 재면 1.2/0.75 가 1등**

**결정** — noun = {NNG, NNP, SL, SN, SH} / k1=1.2 / b=0.75. 팀 그리드 스크립트 태그 집합 교정 요청

### 3-2. 검색 — 리랭커·컨텍스트

**한 일**
- 지우 최종 설정(Hybrid k_rrf=20, pool=50)에 재튜닝 BM25 + bge 리랭커 c50 재측정
- 컨텍스트 6조건(hyb/rr × top1/top3/budget) 생성 실험 재실행

**결과**
- bge c50: USR@10 0.9698 (Dense 대비 +10.1%p 유의)
- 리랭커는 "새로 찾는" 장치가 아니라 "위로 올리는" 장치 — USR@50 변화 0, USR@1 +21.7%p
- **근거를 적게 넘길수록 리랭커 이득이 커짐** (top1 +0.134 → top3 +0.070 → budget +0.057, 전부 유의)
- 리랭커 + Top3 = budget(5.7개) 대비 입력 토큰 39% 절감, 품질 동일

**결정** — 컨텍스트 Top3. 리랭커는 팀원 실험 결과 Cohere c20 채택

### 3-3. 프롬프트 — GPT-4.1 가이드 기반 실험

**한 일**
- 가이드 대조 → 격차 4개 도출 → 인자 5개 설계 (few-shot · 리터럴 지시 · 규칙 순서 · 지시 앞뒤 · 구조 템플릿)
- 기준선 v1/v2/v3 비교 (dev 107) → v2 채택 → 인자 5개 단독 스크리닝
- 별도로 v1 에서 출발해 가이드를 통째로 적용한 v5_guide → 실측 반영 v5_guide_r2

**결과**

| 갈래 | 내용 | 결과 |
|---|---|---|
| A | 이미 강한 v2 에 가이드 인자 하나씩 | 개선 0건 · 악화 1건(리터럴 지시) |
| B | 기본 v1 에서 가이드 전면 적용 | 준수율 0.61 → 0.88 → (2차) 0.96 |

- v1 → v2 로 준수율 0.6098 → 0.9716. 특히 권유 아님 명시 0% → 100%
- 핵심 메커니즘: v1 에도 규칙은 있었으나 Instructions(금지)에만 있고 Output Format(출력 요구)에 없어 출력에 안 나옴
- v3 는 예시 문장을 79% 문항에서 그대로 복사 → 가이드가 경고한 실패 모드 재현 → 기각
- correctness 는 전 버전 1.96~1.97 로 차이 없음 (천장)
- 남은 실패 4문항은 전부 컨텍스트 절단·라벨 문제 — 프롬프트로 못 고침

**결정** — v2 유지. 팀 최종은 v2 + 확인 블록(V2_4)

### 3-4. 평가 도구 정리

**발견한 결함**

| 파일 | 결함 | 조치 |
|---|---|---|
| eval_generation.py | 회피 정규식이 문구 변형을 놓쳐 회피율 항상 0 | answer_spec 정규식으로 교체 ✅ |
| eval_generation.py | correctness·groundedness 를 한 호출에 자료+모범답안 동시 노출 → 판정 오염, groundedness 1.98~2.00 포화 | run_judge.py 로 일원화 |
| eval_rag_triad.py (신규) | 판정 누락 10~19% 를 "근거 없음" 처리 → faithfulness 과소 | faithfulness 제거, AR·CR 전용으로 |
| run_judge.py | --run 방식 출력 못 읽음 / 경로 하드코딩 / 결과 파일명이 검색 run 이름이라 덮어씀 | 3건 수정 ✅ |
| run_generation.py | --split 이 검색 run 파일에서 작동 안 함 (holdout 격리 실패) / 코퍼스·평가셋 기록 없음 | 2건 수정 ✅ |
| RefOverlap | 한국어 의역에 0 (0.009~0.022) | 폐기 |

**신규 지표**
- sample_verbatim — 프롬프트 예시 문구를 그대로 복사한 비율 (가이드 경고 대응)
- answer_relevance · context_relevance — RAG triad
- faithfulness 관문 b — 규정 고지 문구("원금 손실 가능", "투자 권유 아님")는 no_claim 으로 분류. 단, 문구 안에 수치·상품명이 있으면 판정 대상

**정리 후 지표 체계**

| 축 | 지표 | 도구 |
|---|---|---|
| 정확도 | correctness · completeness | run_judge.py (자료 차단) |
| 근거성 | faithfulness | run_judge.py (모범답안 차단) |
| 관련성 | answer_relevance · context_relevance | eval_rag_triad.py |
| 규정 준수 | 준수율 7종 · sample_verbatim | answer_spec.py |
| 검색 | USR · CovnDCG · MRR · CovHit | eval_retriever_coverageaware.py |
| 비용 | 토큰 · 지연 | 실행 로그 |

### 3-5. 평가셋 라벨 문제 — 원인 규명

**현상** — 답변 정확도가 검색 상한(gold 커버리지)을 최대 +42.5%p 초과
**규명**
- 회피분을 빼면 +25.9%p
- 라벨상 gold 0 인 83문항의 Context Relevance(라벨 무관)를 재니 절반(49%)이 실제로 답할 수 있는 자료를 받았음
- 모델이 사전 지식을 알면서도 쓰지 않은 사례 확인 (소장펀드 2015년 종료 → "확인할 수 없습니다")

**결론** — 원인은 모델 사전 지식이 아니라 **gold 라벨 누락**. 다중 정답 기준(회의 안건 1)을 뒷받침

### 3-6. 파이프라인 통합

**한 일**
- core/retriever.py — BM25Retriever · HybridRetriever(RRF) · DenseIndexRetriever 추가
- core/reranker.py — CohereReranker (신규, Trial 키 속도 제한 대응)
- core/pipeline.py — FinalPipeline · BaselinePipeline (기존 RAGPipeline 유지)
- scripts/run_pipeline.py — 테스트셋 E2E 실행, 단계별 산출물 저장, resume
- 오프라인 스크립트 로직을 **그대로 옮김** (새로 짜지 않음)

**회귀 검증** — 파이프라인 출력 vs 기존 run 파일, 157문항

| 단계 | 순위 완전 일치 | 판정 |
|---|---|---|
| BM25 | 157/157 | ✅ |
| Dense | 114/157 (집합 146/157) | ✅ 임베딩 API 비결정성 — 동점 근처(거리차 0.0001)만 흔들림, 지표 영향 0.001 이하 |
| Hybrid | 125/157 | ✅ Dense 노이즈 상속 |
| Reranked | 구조 검증 157/157 | ✅ |

**Cohere c20 첫 수치 (v2 코퍼스)** — USR@10 0.9447, MRR@10 0.8474. 팀원 원본 코퍼스 수치(.9410 / .8484)와 거의 동일 → 코퍼스 정리가 결과를 흔들지 않음

---

## 4. 확인된 사실 / 주의점

**확인된 사실**
- 파이프라인은 자료에 근거해 동작함 — 사전 지식을 알아도 자료에 없으면 회피
- 검색 튜닝은 157 전체로 했으므로 검색 지표는 "개발셋 기준". 생성 지표만 holdout 50 이 깨끗
- 가이드 기반 프롬프트 개선은 기본선에서는 크게 작동(+35%p), 이미 강한 프롬프트에서는 0
- 필수 고지 문구는 근거성·관련성 지표를 구조적으로 깎음 → 관문 b 로 대응

**주의점**
- ⚠ **V2_4 예시문구 반복 36%** — 확인 블록이 예시 문장을 그대로 쓰게 유도. 고치려면 "그대로 반복하지 말 것" 한 줄 (v5_guide 에서 그 방식으로 0% 달성)
- ⚠ Cohere Trial 키 — 분당 10회. 157문항 17분. 반복 실행하려면 Production 키 필요
- ⚠ 판정기 변경 전후 faithfulness 수치는 직접 비교 불가 (관문 b 적용 시점 2026-09-19)
- ⚠ 코드 다수가 미커밋 상태 — core/ 변경분·신규 스크립트 전부

---

## 5. 남은 것

| # | 작업 | 상태 | 필요 결정 |
|---|---|---|---|
| 1 | **정성 평가** | ⬜ | 평가자 수 · 문항 수 · 루브릭 · 블라인드 여부 |
| 2 | 최종 보고서 | ⬜ 1 이후 | |
| 3 | 코드 커밋 | ⬜ **유실 위험** | |
| 4 | 신규 검증 문항 40개 | ⬜ | 회의 승인 |
| 5 | 다중 정답 라벨 반영 | ⬜ | 회의 안건 1 |
| 6 | V2_4 예시 반복 대응 | ⬜ | 팀 확정 사양 변경 여부 |

**정성 평가 제안** — 실패·경계 문항 우선 + 무작위 약 30문항, 기준선 vs 최종 블라인드 쌍 비교, 루브릭 5축(정확성·근거 충실·가독성·규정 고지·유해 답변)

---

## 6. 산출물

**결과**
```
results_e2e/final_v1/      최종 — config · ret_{dense,bm25,hybrid,reranked} · generation · judge
results_e2e/baseline_v1/   기준선 — config · ret_dense · generation · judge
results_e2e/_triad/        관련성 지표 (두 결과)
```

**문서**

| 문서 | 내용 |
|---|---|
| docs/final_pipeline_spec.md | 최종 사양 단일 출처 |
| docs/bm25_retune_posfix_report.md | BM25 27조합 + 하이브리드 검증 |
| docs/team_notice_bm25_tokenizer.md | 토크나이저 태그 집합 불일치 원인 |
| docs/rerun_retuned_bm25_experiment.md | 리랭커·컨텍스트 재실험 + 라벨 문제 규명 |
| docs/prompt_experiment_design.md / _report.md | 프롬프트 실험 설계·결과 |
| docs/retriever_2nd_report.md | 2차 리트리버 보고서 (회의 안건 6건) |

**코드**

| 파일 | 역할 |
|---|---|
| config.py | FINAL_* 확정 사양 |
| core/pipeline.py | FinalPipeline · BaselinePipeline |
| core/retriever.py · core/reranker.py | 검색·리랭킹 |
| scripts/run_pipeline.py | E2E 실행 |
| scripts/run_judge.py | 정확도·근거성 판정 (정보 차단) |
| scripts/eval_rag_triad.py | 관련성 |
| scripts/answer_spec.py | 규정 준수 |
| scripts/report_gen_metrics.py | 지표 통합 표 |
