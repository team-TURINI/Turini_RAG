# 리트리버 평가셋 검증 결과 및 실험 프로토콜

**작성일** 2026-08-05
**대상** `rag_testset_retriever_v1v2_fixed450_team_eval.json` (157문항) + `fixed_450_70` 코퍼스
**결론** **사용 가능함.** 구조 정합성은 전부 통과했고, 보완이 필요한 것은 주로 **채점 방식 합의**임

---

## 1. 검증 결과 — 전부 통과

| 항목 | 결과 |
|---|---|
| `chunks.jsonl` 라인 수 = manifest = FAISS 벡터 수 | **1,767 / 1,767 / 1,767** |
| chunks.jsonl ↔ FAISS 인덱스 `chunk_id` 집합 | **완전 일치** (양쪽 누락 0) |
| 원본 `docs.jsonl` SHA256 ↔ manifest 기록 | **일치** → 재현 가능 |
| gold_chunks 의 `chunk_id` 가 코퍼스에 존재 | **261/261** |
| `union_gold_coverage` | **157/157 = 1.0** |
| gold offset ↔ 원문 `span_start` 대조 | **157/157 일치** |
| `id` / `question` 중복 | **0 / 0** |
| chunk_id 중복, 빈 텍스트 | **0 / 0** |
| 인덱스 `page_content` = `embedding_text` | **1,767/1,767** |
| 인덱스 metadata 에 `chunk_id`·`text` 보존 | **1,767/1,767** |

**구성**

```
평가셋 157문항 = v1 57 + v2_prose 100
gold 청크 타입(주 타입 기준): size 121 / section 27 / faq 5 / whole 4
코퍼스 1,767청크 = size 921 / whole 456 / section 200 / atomic 173 / faq 17
```

---

## 2. 보완 제안 (우선순위 순)

### 🔴 P1. 채점 방식을 먼저 합의해야 함 — 가장 중요

**문제** — 정답이 여러 청크에 걸친 문항이 많음.

```
gold 청크가 2개 이상인 문항          : 93 / 157
그중 단일 청크로 100% 커버 불가       : 50 문항
단일 청크 최고 커버리지 평균          : 0.932
```

`gold_chunks` 중 **하나만 맞으면 Hit** 으로 채점하면, 이 50문항은 정답의 일부만
가져와도 만점 처리됨. 단일 최고 커버리지 평균이 0.932 이므로 **청크 하나만 건져도
93% 맞힌 것처럼 보임.** 리트리버 간 차이가 이 지점에서 뭉개짐.

**권장 — 두 지표를 함께 산출하고 주 지표를 합집합으로 둘 것**

| 지표 | 정의 | 용도 |
|---|---|---|
| `Hit@K` | top-K 에 gold 청크가 하나라도 있으면 1 | 보조 (관대함) |
| **`Coverage@K`** | **top-K ∩ gold 의 `gold_coverage` 합** (1.0 상한) | **주 지표** |
| `MRR@K` | 첫 gold 청크의 순위 역수 | 순위 품질 |
| `nDCG@K` | `gold_coverage` 를 등급 관련도로 사용 | 순위 품질 |

`gold_coverage` 가 이미 청크별로 저장돼 있으므로 합산만 하면 됨. 상한 1.0 으로 클립할 것
(gold 청크끼리 오버랩 구간이 겹칠 수 있음).

추가로 권장하는 노이즈 지표 — Recall 계열만 보면 잘게 쪼개 많이 던지는 쪽이 유리해 보임.

| 지표 | 정의 |
|---|---|
| `Precision@K` | top-K 중 gold 청크의 비율 |
| `DocHit@K` | top-K 중 정답 `doc_id` 에서 온 비율 |

### 🟠 P2. BM25 입력 필드를 Dense 와 맞춰야 함

Dense 인덱스는 **`embedding_text`** 로 임베딩됨. 실측 결과:

```
embedding_text = "{title}\n\n{text}"      (1,767/1,767 이 text 로 끝남, 길이차 중앙 23자)
FAISS page_content == embedding_text      (1,767/1,767)
```

**BM25 가 `text` 만 사용하면 제목이 빠져 불공정한 비교가 됨.** 제목에 "채권 FAQ",
"ETF 기초 가이드" 같은 핵심어가 들어 있어 키워드 검색에서 특히 영향이 큼.

→ **Dense·BM25·Reranker 모두 `embedding_text` 를 대상 텍스트로 쓸 것.**

### 🟠 P3. gold 청크 3건이 메타데이터로 오염됨

정답 근거 청크의 앞부분이 출처 정보 덩어리임.

| 문항 | gold_coverage | 청크 앞부분 |
|---|---:|---|
| `rag_bond_01` | **0.42** | `source_name: KDI` `source_url: https://eiec.kdi.re.kr/...` |
| `v2_004` | 0.55 | `source_name: SEC Investor.gov` `source_url: ...` |
| `rag_fund_12` | 0.80 | `source_name: KCIE` `source_url: ...` |

코퍼스 전체로는 **17개 청크**가 이 문제를 가짐. 원본 `raw/sy/*.txt` 등에 본문 앞
메타데이터 서문이 붙어 있고 그것이 `docs.jsonl` 에 그대로 들어간 결과임.

- **단기** — 이번 실험은 전 팀원이 같은 코퍼스를 쓰므로 **비교에는 지장 없음**.
  다만 `rag_bond_01` 은 gold_coverage 0.42 로 이미 낮아 gold 재검토 권장
- **중기** — `prepare_docs.py` 에서 메타데이터 서문 제거 후 재청킹

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

### 🟡 P5. 본문이 완전히 동일한 중복 청크 49개

원문에 같은 문단이 반복 수록된 데이터에서 발생함. 검색 결과 상위를 중복이 차지하면
top-K 한 칸이 낭비됨. 실제로 이전 분석에서 1·2위가 사실상 같은 텍스트인 사례를 확인함.

→ 이번 실험은 조건이 동일하므로 비교에는 무해함. **중복 제거는 다음 코퍼스 버전 과제.**

### 🟡 P6. topic 편중

```
stock 71 / fund 33 / bond 20 / etf 13 / bank 12 / pension_savings 6 / market_data 2
```

stock 이 45%임. **topic 별 성능 비교는 stock·fund·bond 까지만** 유효하고,
`market_data`(2), `pension_savings`(6) 은 결론 불가.

### 🟢 P7. manifest 경로 표기 정정

`source_corpus_path` 가 `data/normalized/docs.jsonl` 로 적혀 있으나 실제 파일은
`data/docs.jsonl` 임. SHA256 은 일치하므로 **내용은 동일**하고 경로 표기만 틀림.
혼선 방지를 위해 정정 권장.

### 🟢 P8. gold 청크 21개가 복수 문항에 공유됨

예) `rag_bond_01` 과 `rag_bond_03` 이 같은 청크를 gold 로 씀. 문제는 아니나
**문항이 완전히 독립적이지는 않다**는 점을 인지할 것.

---

## 3. 팀원 통일 필수 사항

각자 실험한 뒤 성능을 비교하려면 **아래 항목이 하나라도 다르면 비교가 무의미해짐.**

### 3-1. 입력 (고정)

| # | 항목 | 값 |
|---|---|---|
| 1 | 평가셋 | `rag_testset_retriever_v1v2_fixed450_team_eval.json` (157문항) |
| 2 | 코퍼스 | `data/chunking_data/fixed_450_70/chunks.jsonl` (1,767청크) |
| 3 | Dense 인덱스 | `vectorstores/fixed_450_70/` (공유 파일 그대로, 재빌드 금지) |
| 4 | 임베딩 모델 | `text-embedding-3-large` |
| 5 | 검색 대상 텍스트 | **`embedding_text`** (Dense·BM25·Reranker 공통) |
| 6 | 쿼리 | **`question` 원문만** |

> **파일이 같은지 반드시 해시로 확인할 것**
> ```bash
> python -c "import hashlib;print(hashlib.sha256(open('data/chunking_data/fixed_450_70/chunks.jsonl','rb').read()).hexdigest())"
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
| 8 | fetch_k | **20 이상 동일하게** (Reranker 는 후보를 넉넉히 받아야 하므로 반드시 통일) |
| 9 | 동점 처리 | 점수 동일 시 `chunk_id` 오름차순 (구현체별 순서 차이 제거) |
| 10 | 랜덤 시드 | 사용하는 라이브러리 전부 고정 |

> **fetch_k 는 특히 주의.** Reranker 를 top-50 후보에서 재정렬한 사람과 top-10 에서
> 재정렬한 사람의 결과는 비교 불가함.

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
차이만으로 1~2%p 가 흔들려 실제 성능 차이와 구분이 안 됨.

| # | 항목 | 값 |
|---|---|---|
| 16 | 채점 스크립트 | **`scripts/eval_retriever.py`** (공용) |
| 17 | 주 지표 | `Coverage@K` — gold 청크와 정답 구간의 **문자 구간 합집합** 비율 |
| 18 | 보조 지표 | `Hit@K`, `MRR@K`, `nDCG@K`, `Precision@K`, `DocHit@K` |
| 19 | 세그먼트 분해 | `eval_source`(v1/v2), gold `chunk_type`, `topic`(10문항 이상만) |

> `Coverage@K` 는 `gold_coverage` 단순 합이 아니라 **구간 합집합**으로 계산함.
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

### 3-6-3. Dense 기준선 (참고값)

공유 인덱스로 측정한 값임. 각자 구현이 이 근처에서 시작하면 harness 가 정상이라는 뜻임.

| 세그먼트 | n | Cov@1 | Cov@3 | Cov@5 | Cov@10 | Hit@5 | MRR@5 | P@5 | DocHit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **전체** | 157 | 0.376 | 0.598 | **0.714** | 0.866 | 0.764 | 0.577 | 0.223 | 0.612 |
| src:v1 | 57 | 0.362 | 0.564 | 0.646 | 0.801 | 0.684 | 0.525 | 0.172 | 0.435 |
| src:v2_prose | 100 | 0.384 | 0.618 | 0.754 | 0.903 | 0.810 | 0.607 | 0.252 | 0.712 |
| type:size | 121 | 0.375 | 0.595 | 0.720 | 0.868 | 0.777 | 0.590 | 0.243 | 0.673 |
| type:section | 27 | 0.357 | 0.667 | 0.778 | 0.922 | 0.815 | 0.563 | 0.178 | 0.511 |
| topic:stock | 71 | 0.388 | 0.601 | 0.748 | 0.899 | 0.789 | 0.603 | 0.245 | 0.741 |
| topic:fund | 33 | 0.432 | 0.812 | 0.864 | 0.932 | 0.909 | 0.713 | 0.297 | 0.618 |
| topic:bond | 20 | 0.247 | 0.359 | 0.484 | 0.619 | 0.600 | 0.399 | 0.160 | 0.520 |
| topic:etf | 13 | 0.154 | 0.385 | 0.462 | 0.923 | 0.462 | 0.276 | 0.092 | 0.277 |
| topic:bank | 12 | 0.526 | 0.526 | 0.653 | 0.833 | 0.750 | 0.621 | 0.167 | 0.267 |

참고 — 랜덤 검색 하한은 Cov@5 = 0.006 임. Dense 는 0.714 로 하한 대비 충분히 유의미함.

**눈에 띄는 지점**
- `topic:bond`(0.484)와 `topic:etf`(0.462)가 유독 낮음. 이전 분석에서 확인한
  **유사 문서 중복**(같은 주제를 여러 문서가 설명) 영향으로 보임. BM25·Reranker 가
  이 구간을 개선하는지가 관전 포인트임
- `type:whole`(4문항)·`type:faq`(5문항)은 표본이 작아 채점표에는 나오지만
  **결론 근거로 쓰지 말 것**

### 3-7. 결과 제출 형식 (통일)

**문항별 검색 결과 `chunk_id` 리스트를 반드시 저장할 것.** 그래야 나중에 채점 기준이
바뀌어도 재실행 없이 다시 채점할 수 있음.

```json
{
  "run_name": "bm25_kiwi_k1.2_b0.75",
  "retriever": "bm25",
  "config": { "tokenizer": "kiwi", "k1": 1.2, "b": 0.75, "fetch_k": 20 },
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
| 20 | **157문항 기준 1문항 = 0.64%p.** 2%p 미만 차이는 노이즈로 간주 |
| 21 | 구조별 결론은 `size`(121) / `section`(27) 까지만. faq(5)·whole(4) 는 참고용 |
| 22 | topic 별 결론은 stock·fund·bond 까지만 |
| 23 | 임베딩 API 재호출 시 미세한 비결정성이 있으므로, 동일 설정 재실행 결과를 1회 이상 비교해 노이즈 하한을 실측할 것 |

---

## 4. 요약

**써도 됨.** 코퍼스·인덱스·gold 매핑이 전부 정합하고 원본 해시까지 일치해 재현 가능함.

**실험 시작 전에 반드시 정할 것 두 가지**

1. **채점 코드** — `Hit` 단독으로 채점하면 50문항이 과대평가됨. `Coverage@K` 를 주
   지표로 하고 코드를 공유할 것
2. **`embedding_text` 사용 + fetch_k + BM25 토크나이저** — 이게 어긋나면 성능 비교가
   아니라 설정 비교가 됨

**결과 해석 시 주의**

- 1문항 = 0.64%p, 2%p 미만은 노이즈
- faq(5문항)·whole(4문항)·market_data(2문항)·pension(6문항)은 결론 불가
