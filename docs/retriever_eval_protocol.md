# 리트리버 평가셋 검증 결과 및 실험 프로토콜

**작성일** 2026-08-05
**대상** `rag_testset_retriever_v1v2_fixed450_team_eval.json` (157문항) + `fixed_450_70` 코퍼스
**결론** **사용 가능함.** 구조 정합성은 전부 통과했고, 보완이 필요한 것은 주로 **채점 방식 합의**임

---

## 1. 검증 결과 — 전부 통과

> **2026-08-10 갱신** — 메타데이터 정제 코퍼스(`clean_chunks_450_70.jsonl`)로 교체하면서
> 평가셋 gold 를 **재매핑**하고 Dense 인덱스를 **재빌드**했음. 아래는 재매핑 후 수치임.

| 항목 | 결과 |
|---|---|
| `clean_chunks_450_70.jsonl` 라인 수 = manifest = FAISS 벡터 수 | **1,755 / 1,755 / 1,755** |
| 코퍼스 ↔ FAISS 인덱스 `chunk_id` 집합 | **완전 일치** (양쪽 누락 0) |
| 원본 `docs_metadata_cleaned.jsonl` SHA256 ↔ manifest 기록 | **일치** → 재현 가능 |
| gold_chunks 의 `chunk_id` 가 코퍼스에 존재 | **256/256** |
| gold 청크 offset == 코퍼스 청크 offset | **256/256** |
| `overlap_chars` == 실제 교집합 길이 | **256/256** |
| `gold_coverage` == overlap / gold_length | **256/256** |
| `union_gold_coverage` | **157/157 = 1.0** |
| 앵커(`span_start`/`span_end`) 로 정답 구간 재확인 | **157/157** |
| 정제 전후 정답 텍스트 동일성 | **157/157** (메타데이터는 문서 앞부분에만 있었음) |
| `id` / `question` 중복 | **0 / 0** |
| chunk_id 중복, 빈 텍스트 | **0 / 0** |
| 인덱스 `page_content` = `embedding_text` | **1,755/1,755** |

**구성**

```
평가셋 157문항 = v1 57 + v2_prose 100
gold 청크 타입(주 타입 기준): size 121 / section 27 / faq 5 / whole 4
코퍼스 1,755청크 = size 909 / whole 456 / section 200 / atomic 173 / faq 17
```

### 1-1. 재매핑이 필요했던 이유 (기록)

메타데이터 서문을 제거하면 **문서 앞부분이 잘려 뒤쪽 문자 offset 이 전부 밀린다.**
`chunk_id` 는 `doc_id + type + index` 기반이라 살아남지만, offset 기반 정보는 틀어진다.

| 항목 | 정제 전 → 후 |
|---|---|
| gold `chunk_id` 존재 | 261/261 → **256/256** (문제 없었음) |
| **gold offset** | **36문항에서 −167 ~ −256자 밀림** |
| **`gold_chunks` 의 offset·overlap·coverage** | **72개 항목 틀어짐** (17개 문서) |
| FAISS 인덱스 | 1,767 벡터 → **재빌드 필요했음** |

채점기가 offset 으로 커버리지를 계산하므로 방치했다면 **채점이 조용히 틀렸을 것임.**

재매핑은 평가셋에 보존된 **앵커(`span_start`/`span_end`)** 로 수행함
(`scripts/remap_testset.py`). 앵커는 사람이 정한 문자열이라 정제 전후로 안 변함.

> **함정** — 단순 `find` 로는 `rag_etf_12` 가 232자 → 1,633자로 잘못 잡혔음. 시작 앵커가
> 문서에 2회 등장해 엉뚱한 조합이 걸린 것. 29문항이 같은 상황이라 **모든 시작 후보를
> 순회해 원래 길이에 가장 가까운 조합**을 선택하도록 구현함.

---

## 2. 보완 제안 (우선순위 순)

### 🔴 P1. 채점 방식을 먼저 합의해야 함 — 가장 중요

**문제** — 정답이 여러 청크에 걸친 문항이 많음.

```
gold 청크가 2개 이상인 문항          : 89 / 157
그중 단일 청크로 100% 커버 불가       : 46 문항
단일 청크 최고 커버리지 평균          : 0.938
```

`gold_chunks` 중 **하나만 맞으면 Hit** 으로 채점하면, 이 50문항은 정답의 일부만
가져와도 만점 처리됨. 단일 최고 커버리지 평균이 0.938 이므로 **청크 하나만 건져도
93% 맞힌 것처럼 보임.** 리트리버 간 차이가 이 지점에서 뭉개짐.

**해결 — 근거를 "얼마나" 가져왔는지를 주 지표로 둠**

| 지표 | 정의 | 용도 |
|---|---|---|
| **`UnionSpanRecall@K`** | top-K gold 청크가 정답 구간을 덮은 문자 비율 (**구간 합집합**) | **주 지표** |
| `CoverageHit@K-τ` | `UnionSpanRecall@K >= τ` 인 문항 비율 (τ = 0.5 / **0.8**) | 주요 보조 |
| `EvidenceHit@K` | top-K 에 gold 청크가 하나라도 있으면 1 | 보조 (관대함) |
| `MRR@K` / `nDCG@K` | 첫 gold 청크 순위 / `gold_coverage` 등급 nDCG | 순위 품질 |
| `DocHit@K` | top-K 에 정답 `doc_id` 청크가 **하나라도 있으면 1** (binary) | 진단 |
| `DocPrecision@K` / `Precision@K` | top-K 중 정답 문서 / gold 청크의 **비율** | 진단 |

> **⚠ `gold_coverage` 를 단순 합산하면 안 됨.** 청크끼리 70자 오버랩이 있어 겹치는
> 구간을 두 번 셈. 실제로 **157문항 중 85건(54%)** 이 단순 합산 시 1.0 을 초과함.
> 반드시 **문자 구간의 합집합**으로 계산할 것 (채점기가 그렇게 구현되어 있음).

관계: `DocHit >= EvidenceHit >= CoverageHit@K-0.5 >= CoverageHit@K-0.8`
(채점기가 assert 로 검사함)

**실서비스 관점 지표도 병기** — 청킹 실험에서 쓰던 문자 예산 기준을 이어서 봄.

| 지표 | 정의 |
|---|---|
| `UnionSpanRecall@2500c` | 순위대로 2,500자까지 채웠을 때의 커버리지 |
| `CoverageHit@2500c-0.8` | 위 값이 0.8 이상인 문항 비율 |
| `ChunksUsed@2500c` | 예산 안에 들어간 평균 청크 수 (진단용) |

예산 계산은 **`embedding_text` 길이 기준**임 (Dense 인덱스가 이 필드로 만들어졌으므로).

### 🟠 P2. BM25 입력 필드를 Dense 와 맞춰야 함

Dense 인덱스는 **`embedding_text`** 로 임베딩됨. 실측 결과:

```
embedding_text = "{title}\n\n{text}"      (1,755/1,755 이 text 로 끝남)
FAISS page_content == embedding_text      (1,755/1,755)
```

**BM25 가 `text` 만 사용하면 제목이 빠져 불공정한 비교가 됨.** 제목에 "채권 FAQ",
"ETF 기초 가이드" 같은 핵심어가 들어 있어 키워드 검색에서 특히 영향이 큼.

→ **Dense·BM25·Reranker 모두 `embedding_text` 를 대상 텍스트로 쓸 것.**

### ✅ P3. gold 청크 메타데이터 오염 — **해결됨** (2026-08-10)

정답 근거 청크 앞부분에 `source_name:` / `source_url:` 덩어리가 붙어 있던 문제
(`rag_bond_01` 0.42, `v2_004` 0.55, `rag_fund_12` 0.80). 코퍼스 전체 17개 청크가 해당됐음.

**정제된 코퍼스에서 재검사 결과 0건.** 메타데이터 서문이 제거되었음.

부수 효과 — 문서 앞부분이 잘리면서 gold offset 이 밀렸고, 평가셋을 재매핑했음(1-1 참조).
`topic:bond` 는 `USR@10` 이 0.619 → 0.553 으로 내려갔는데, 메타데이터 청크가 사라진 자리를
다른 문서 청크가 차지한 결과로 보임. **정제는 옳은 방향이나 bond 는 여전히 최약점임.**

### 🟡 P4. v1 을 넣은 목적(구조 다양성) 대비 표본이 얇음

| 주 gold 타입 | 문항 수 | 1문항의 비중 |
|---|---:|---:|
| size | 121 | 0.8%p |
| section | 27 | 3.7%p |
| **faq** | **5** | **20%p** |
| **whole** | **4** | **25%p** |

v1 57문항 중 **21문항은 여전히 `size` 타입**이라, 구조 다양성 확보 효과는 실제로
section 27 + faq 5 + whole 4 = 36문항임.

→ **faq·whole 은 구조별 성능 비교에 쓸 수 없음.** 1문항이 20~25%p 라 어떤 차이도
노이즈임. "전체 성능에 다양한 구조가 포함됨" 정도로만 해석하고, **구조별 분해 결론은
`size`(121)와 `section`(27)까지만** 낼 것.

### 🟡 P5. 본문이 완전히 동일한 중복 청크 38개

원문에 같은 문단이 반복 수록된 데이터에서 발생함 (정제로 49 → 38 개로 줄었음). 검색 결과 상위를 중복이 차지하면
top-K 한 칸이 낭비됨. 실제로 이전 분석에서 1·2위가 사실상 같은 텍스트인 사례를 확인함.

→ 이번 실험은 조건이 동일하므로 비교에는 무해함. **중복 제거는 다음 코퍼스 버전 과제.**

### 🟡 P6. topic 편중

```
stock 71 / fund 33 / bond 20 / etf 13 / bank 12 / pension_savings 6 / market_data 2
```

stock 이 45%임. **topic 별 성능 비교는 stock·fund·bond 까지만** 유효하고,
`market_data`(2), `pension_savings`(6) 은 결론 불가.

### 🟢 P7. manifest 경로 표기 정정

`source_corpus_path` 가 `data/cleaned/docs_metadata_cleaned.jsonl` 로 적혀 있으나 실제
파일은 `data/docs_metadata_cleaned.jsonl` 임. SHA256 은 일치하므로 **내용은 동일**하고
경로 표기만 틀림. 혼선 방지를 위해 정정 권장.

### 🟢 P8. gold 청크 19개가 복수 문항에 공유됨

예) `rag_bond_01` 과 `rag_bond_03` 이 같은 청크를 gold 로 씀. 문제는 아니나
**문항이 완전히 독립적이지는 않다**는 점을 인지할 것.

---

## 3. 팀원 통일 필수 사항

각자 실험한 뒤 성능을 비교하려면 **아래 항목이 하나라도 다르면 비교가 무의미해짐.**

### 3-1. 입력 (고정)

| # | 항목 | 값 |
|---|---|---|
| 1 | 평가셋 | `rag_testset_retriever_v1v2_fixed450_team_eval.json` (157문항) |
| 2 | 코퍼스 | `data/chunking_data/fixed_450_70/clean_chunks_450_70.jsonl` (1,755청크) |
| 3 | Dense 인덱스 | `vectorstores/fixed_450_70/` (공유 파일 그대로, 재빌드 금지) |
| 4 | 임베딩 모델 | `text-embedding-3-large` |
| 5 | 검색 대상 텍스트 | **`embedding_text`** (Dense·BM25·Reranker 공통) |
| 6 | 쿼리 | **`question` 원문만** |

> **파일이 같은지 반드시 해시로 확인할 것**
> ```bash
> python -c "import hashlib;print(hashlib.sha256(open('data/chunking_data/fixed_450_70/clean_chunks_450_70.jsonl','rb').read()).hexdigest())"
> python -c "import hashlib;print(hashlib.sha256(open('vectorstores/fixed_450_70/index.faiss','rb').read()).hexdigest())"
> ```
> 이 값을 결과 파일에 같이 기록하면 나중에 "다른 파일로 돌린 결과" 사고를 막을 수 있음.

### 3-2. 절대 금지 — 데이터 누수

**`answer`, `gold_chunks`, `gold_start/end`, `span_start/end`, `union_gold_coverage`
는 채점 전용임.** 검색 쿼리나 Reranker 입력에 절대 넣지 말 것.

특히 Reranker 입력은 **`question` + 검색된 청크 텍스트**만 사용할 것.

### 3-3. 검색 조건 (고정)

| # | 항목 | 값 |
|---|---|---|
| 7 | Top-K | **K = 1, 3, 5, 10 전부 산출** |
| 8 | retrieve_k (저장 후보 수) | **50 권장** — 한 번 저장하면 @10/@20/@30/@50 을 재검색 없이 비교 가능. 최소 10. **Reranker 후보 수는 실험 변수** — 3-3-1 참조 |
| 9 | 동점 처리 | 점수 동일 시 `chunk_id` 오름차순 (구현체별 순서 차이 제거) |
| 10 | 랜덤 시드 | 사용하는 라이브러리 전부 고정 |

### 3-3-1. 후보 수는 고정 대상이 아니라 튜닝 대상임

역할이 단계마다 다르므로 일괄 고정하지 않음. **저장은 50 으로 통일**하되, Reranker/Hybrid 가
실제로 사용할 후보 수는 각자 스윕함.

| 방식 | 후보 수의 역할 | 스윕 가치 |
|---|---|---|
| Dense / BM25 단독 | 단순히 몇 개를 반환할지 | **없음.** 20 이든 100 이든 **top-10 은 완전히 동일**하므로 점수가 안 변함 |
| Hybrid | 결합에 넣을 **후보 풀 크기** | **있음.** 풀이 달라지면 결합 결과의 top-5 가 바뀜 |
| Reranker | 재정렬할 **후보 풀 크기** | **있음.** 상한과 비용이 함께 바뀜 |

**권장 스윕값** — Hybrid·Reranker 에서 `20 / 50 / 100`

**Reranker 의 상한** — Reranker 는 후보 풀 안의 것만 재정렬하므로 다음이 성립함.

```
Reranker 의 UnionSpanRecall@10  ≤  기반 리트리버의 UnionSpanRecall@(후보 수)
```

풀을 키우면 상한이 올라가지만 노이즈와 **비용·시간이 비례해 증가**함. 풀 크기를 정하기
전에 기반 리트리버의 `UnionSpanRecall@20`, `@50` 을 먼저 재볼 것 (실측치는 3-6-4).

> **비교할 때 두 축을 섞지 말 것**
>
> | 무엇을 비교하나 | 고정할 것 |
> |---|---|
> | 방법 비교 (Dense vs Hybrid vs Reranker) | 후보 수를 **같게** |
> | 후보 수 튜닝 | 방법을 **같게** 하고 10/20/30/50 |
>
> 그리고 후보 수를 **반드시 `run_name` 과 `config` 에 기록할 것.** 기록만 되어 있으면
> 값이 달라도 나중에 축을 분리해 해석할 수 있음.

### 3-4. BM25 세부 (한국어라 특히 중요)

| # | 항목 | 합의 필요 |
|---|---|---|
| 11 | 토크나이저 | Kiwi / Mecab / 공백분리 중 **하나로 통일** |
| 12 | 불용어·정규화 | 적용 여부와 목록 통일 |
| 13 | BM25 파라미터 | `k1`, `b` 값 통일 (기본값 사용 시 라이브러리 버전도 기록) |

한국어 BM25 는 토크나이저에 따라 점수가 크게 달라짐. **토크나이저가 다르면 그건
"BM25 비교"가 아니라 "토크나이저 비교"가 됨.**

### 3-5. Hybrid 세부

| # | 항목 | 합의 필요 |
|---|---|---|
| 14 | 결합 방식 | RRF / 가중합 중 선택 (RRF 권장 — 점수 스케일 정규화 불필요) |
| 15 | 파라미터 | RRF 의 `k` 값, 가중합의 α 값 |

**결합 방식 자체가 실험 변수라면** 그 사실을 명시하고, Dense·BM25 결과는 동일하게
고정한 뒤 결합만 바꿔서 비교할 것.

### 3-6. 채점 (공용 스크립트 사용 — 각자 구현 금지)

**`scripts/eval_retriever.py` 로만 채점할 것.** 각자 구현하면 반올림·동점·클립 처리
차이만으로 수치가 흔들려 실제 성능 차이와 구분이 안 됨.

| # | 항목 | 값 |
|---|---|---|
| 16 | 채점 스크립트 | **`scripts/eval_retriever.py`** (공용) |
| 17 | 주 지표 | `UnionSpanRecall@K` — gold 청크와 정답 구간의 **문자 구간 합집합** 비율 |
| 18 | 보조 지표 | `CoverageHit@K-0.5/-0.8`, `EvidenceHit@K`, `MRR@10`, `nDCG@10`, `DocHit@K`(binary), `DocPrecision@K`, `Precision@K`, `@2500c` 3종 |
| 19 | 세그먼트 분해 | `eval_source`(v1/v2), gold `chunk_type`, `topic`(10문항 이상만) |

> 공식 보고값은 **`@10` 기준**(`MRR@10`, `nDCG@10`). `@1/3/5` 도 계산은 하되 최종 표는 `@10` 으로 통일.
> 청크 오버랩이 70자라 gold 청크끼리 겹치는 구간이 있어 단순 합은 중복 계산됨.

### 3-6-1. 사용법

```bash
# 1) 검색 실행 → run 파일 생성 (Dense 예시 / 팀 공용 기준선)
python scripts/run_dense_baseline.py --fetch-k 20

# 2) 채점
python scripts/eval_retriever.py --run runs/dense_baseline.json

# 3) 여러 방식 비교 (첫 번째가 기준선)
python scripts/eval_retriever.py --compare runs/dense_baseline.json runs/bm25.json runs/hybrid.json

# 4) run 파일 형식만 검사 (채점 없이, 제출 전 셀프체크)
python scripts/eval_retriever.py --run runs/bm25.json --validate-only
```

**`run_dense_baseline.py` 는 run 파일 형식의 참조 구현임.** BM25·Hybrid·Reranker 를
만들 때 이 파일의 출력 형식과 쿼리·동점 처리 규칙을 그대로 따를 것.

### 3-6-2. 채점기가 자동으로 잡아주는 오류

제출 전에 `--validate-only` 로 확인할 것. 아래가 하나라도 걸리면 채점이 중단됨.

| 검사 | 걸리는 경우 |
|---|---|
| 문항 누락 / 초과 | 157문항이 아님 |
| `chunk_id` 미존재 | **다른 코퍼스로 검색함** |
| `corpus_sha256` 불일치 | **다른 코퍼스 파일** |
| `retrieved` 내 중복 | 같은 청크를 두 번 반환 |
| id 중복 | 같은 문항이 두 번 |
| `retrieved` 길이 < K | 경고 (해당 K 지표가 불리해짐) |

### 3-6-3. 기준선 (참고값)

정제 코퍼스(`clean_chunks_450_70.jsonl`) + 재빌드 인덱스, `retrieve_k=50` 기준.
`USR` = UnionSpanRecall, `CH` = CoverageHit.

**전체 157문항**

| run | USR@1 | USR@3 | USR@5 | **USR@10** | CH@10-0.8 | EvHit@10 | MRR@10 | nDCG@10 | DocHit@10 | USR@2500c | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_k50` | 0.384 | 0.589 | 0.694 | **0.859** | 0.847 | 0.892 | 0.597 | 0.609 | 0.905 | 0.719 | 530ms |
| `bm25_k50` | 0.215 | 0.453 | 0.600 | **0.736** | 0.707 | 0.790 | 0.423 | 0.464 | 0.866 | 0.614 | **13ms** |

BM25 는 Dense 대비 `USR@10` **−12.2%p** (paired bootstrap 95% CI `[-19.7, -4.6]`, 유의).
다만 latency 는 **40배 빠름**.

**Dense 세그먼트별**

| 세그먼트 | n | USR@5 | USR@10 | CH@10-0.8 | EvHit@10 | MRR@10 | DocHit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| src:v1 | 57 | 0.612 | 0.806 | 0.807 | 0.842 | 0.540 | 0.842 |
| src:v2_prose | 100 | 0.740 | 0.888 | 0.870 | 0.920 | 0.629 | 0.940 |
| type:size | 121 | 0.693 | 0.850 | 0.835 | 0.893 | 0.605 | 0.909 |
| type:section | 27 | 0.778 | 0.922 | 0.926 | 0.926 | 0.599 | 0.926 |
| topic:stock | 71 | 0.733 | 0.899 | 0.887 | 0.930 | 0.619 | 0.944 |
| topic:fund | 33 | 0.876 | 0.936 | 0.939 | 0.939 | 0.754 | 0.939 |
| **topic:bond** | 20 | **0.357** | **0.553** | 0.500 | 0.700 | 0.359 | 0.700 |
| topic:etf | 13 | 0.462 | 0.846 | 0.846 | 0.846 | 0.367 | 0.923 |
| topic:bank | 12 | 0.653 | 0.833 | 0.833 | 0.833 | 0.630 | 0.833 |

**눈에 띄는 지점**
- **`topic:bond` 가 압도적 최약점** — `USR@10` 0.553, `MRR@10` 0.359. 메타데이터 정제
  이후 오히려 내려갔음(0.619 → 0.553). **유사 문서 중복**(채권 설명 문서가
  `doc_9e4ecd4a0eb2` / `doc_d0ab541c0e80` / `doc_ba29b232e816` 등 여러 개)이 주원인으로 보임.
  BM25·Reranker 가 이 구간을 개선하는지가 최대 관전 포인트임
- `topic:etf` 는 `USR@5` 0.462 → `USR@10` 0.846 으로 급등. **정답이 5~10위에 몰려 있다는
  뜻**이라 Reranker 로 끌어올릴 여지가 큼
- `type:whole`(4문항)·`type:faq`(5문항)은 표본이 작아 **결론 근거로 쓰지 말 것**

### 3-6-4. 후보 수 saturation — Reranker 후보 정할 때 볼 것

```bash
python scripts/eval_retriever.py --run runs/dense_k50.json --k 10 20 30 50
```

**실측 결과 (UnionSpanRecall, n=157)**

| 후보 수 | Dense | BM25 |
|---|---:|---:|
| @10 | 0.859 | 0.736 |
| @20 | 0.921 | 0.839 |
| @30 | 0.939 | 0.865 |
| @50 | **0.953** | **0.906** |

> **아직 완전히 saturate 되지는 않았음.** Dense 는 @30 → @50 이 **+1.4%p** 로 꺾이는
> 조짐이 보이나 BM25 는 **+4.1%p** 로 여전히 오름. Reranker 후보는 **30~50 사이**에서
> 각자 이득/비용을 확인한 뒤 정할 것.

### 3-7. 결과 제출 형식 (통일)

**문항별 검색 결과 `chunk_id` 리스트를 반드시 저장할 것.** 그래야 나중에 채점 기준이
바뀌어도 재실행 없이 다시 채점할 수 있음.

```json
{
  "run_name": "bm25_kiwi_k1.2_b0.75",
  "retriever": "bm25",
  "config": { "tokenizer": "kiwi", "k1": 1.2, "b": 0.75, "retrieve_k": 50 },
  "corpus_sha256": "...",
  "index_sha256": "...",
  "testset": "rag_testset_retriever_v1v2_fixed450_team_eval.json",
  "items": [
    { "id": "rag_fund_01", "retrieved": ["chunk_id_1", "chunk_id_2", "..."] }
  ]
}
```

### 3-8. 해석 기준 (미리 합의)

| # | 기준 |
|---|---|
| 20 | **고정 임계값을 쓰지 않음.** `--compare` 의 paired bootstrap 95% CI 로 판단. CI 가 0 을 포함하면 유의하지 않음 |
| 21 | 구조별 결론은 `size`(121) / `section`(27) 까지만. faq(5)·whole(4) 는 참고용 |
| 22 | topic 별 결론은 stock·fund·bond 까지만 |
| 23 | 임베딩 API 재호출 시 미세한 비결정성이 있으므로, 동일 설정 재실행 결과를 1회 이상 비교해 노이즈 하한을 실측할 것 |

---

## 4. 요약

**써도 됨.** 코퍼스·인덱스·gold 매핑이 전부 정합하고 원본 해시까지 일치해 재현 가능함.

**실험 시작 전에 반드시 정할 것 두 가지**

1. **채점 코드** — `EvidenceHit` 단독으로 채점하면 50문항이 과대평가됨.
   `UnionSpanRecall@K` 를 주 지표로 하고 `scripts/eval_retriever.py` 를 공유할 것
2. **`embedding_text` 사용 + BM25 토크나이저** — 이게 어긋나면 성능 비교가 아니라
   설정 비교가 됨. 후보 수는 고정 대상이 아니라 **튜닝 대상**이며, 기록만 하면 됨(3-3-1)

**결과 해석 시 주의**

- 유의성은 paired bootstrap 95% CI 로 판단 (고정 임계값 없음)
- faq(5문항)·whole(4문항)·market_data(2문항)·pension(6문항)은 결론 불가
