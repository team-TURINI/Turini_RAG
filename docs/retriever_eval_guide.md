# 리트리버 평가 사용 가이드

**대상** 각자 Dense / BM25 / Hybrid / Reranker 를 실험한 뒤 성능을 비교할 팀원
**규칙 문서** 통일해야 할 조건은 `docs/retriever_eval_protocol.md` 참조. 이 문서는 **실행 방법**만 다룸

---

## 0. 3분 요약

```bash
# 1) 검색 실행 → run 파일 생성
python scripts/run_dense_baseline.py          # retrieve_k=50 저장

# 2) 채점
python scripts/eval_retriever.py --run runs/dense_baseline.json

# 3) 여러 방식 비교 (첫 번째가 기준선, paired bootstrap 으로 유의성 판단)
python scripts/eval_retriever.py --compare runs/dense_baseline.json runs/bm25_kiwi_k1.2_b0.75.json
```

**각자 할 일은 1번뿐임.** 자기 리트리버로 검색해서 `run 파일` 만 만들면, 채점과 비교는
공용 스크립트가 처리함.

---

## 1. 준비

### 1-1. 환경

```bash
.venv\Scripts\activate          # cmd 기준
pip install -r requirements.txt
```

### 1-2. 필요한 파일 3개

| 파일 | 경로 | 비고 |
|---|---|---|
| 평가셋 | `data/testset/rag_testset_retriever_v1v2_fixed450_team_eval.json` | 157문항 |
| 코퍼스 | `data/chunking_data/fixed_450_70/clean_chunks_450_70.jsonl` | 1,755청크 |
| Dense 인덱스 | `vectorstores/fixed_450_70/` | 구글드라이브 공유본 |

### 1-3. 파일이 같은지 먼저 확인 (필수)

```bash
python -c "import hashlib;print(hashlib.sha256(open('data/chunking_data/fixed_450_70/clean_chunks_450_70.jsonl','rb').read()).hexdigest())"
```

**기대값**
```
clean_chunks_450_70.jsonl : fbfe0b0f8cfe20212a71f6e60f84815e8db1a6b6b27232d4ae78c77954cf6fe1
index.faiss  : a90e7d6aa461a9e8d462b544cda506bb53af379fe4b39616d72d5274db15e250
```

다르면 코퍼스가 다른 것임. **다시 받아야 함.** 채점기가 자동으로도 잡아주지만 미리
확인하는 편이 빠름.

### 1-4. BM25 라이브러리 버전 확인

`requirements.txt` 에 **버전 고정**으로 들어 있음. `pip install -r requirements.txt` 만
하면 됨.

```
rank_bm25==0.2.2
kiwipiepy==0.23.2
```

**버전을 임의로 올리지 말 것.** 토크나이즈 결과와 기본 파라미터가 달라지면
"BM25 비교"가 아니라 "라이브러리 버전 비교"가 됨. 확인:

```bash
pip list | findstr /I "rank-bm25 kiwipiepy"
```

---

## 2. run 파일 만들기 — 각자 할 일

### 2-1. 형식

```json
{
  "run_name": "bm25_kiwi_k1.2_b0.75",
  "retriever": "bm25",
  "config": { "tokenizer": "kiwi", "k1": 1.2, "b": 0.75, "retrieve_k": 50 },
  "corpus_sha256": "fbfe0b0f...",
  "items": [
    { "id": "rag_fund_01", "retrieved": ["chunk_id_1위", "chunk_id_2위", "..."] }
  ]
}
```

- `items` 는 **157문항 전부** 있어야 함
- `retrieved` 는 **순위 오름차순**(1등이 맨 앞), 길이는 `retrieve_k`. **50 권장**
  (한 번 저장해두면 @10/@20/@30/@50 을 재검색 없이 비교 가능). 최소 10
- `run_name` 은 설정을 알아볼 수 있게. 파일명에도 쓰임

### 2-2. 참조 구현

**`scripts/run_dense_baseline.py` 를 열어볼 것.** 짧고, 지켜야 할 규칙이 전부 들어 있음.
새 리트리버는 이 구조를 복사해서 검색 부분만 바꾸면 됨.

### 2-3. BM25 — 바로 쓸 수 있는 스크립트가 있음

```bash
python scripts/run_bm25.py                              # 기본 k1=1.2, b=0.75
python scripts/run_bm25.py --k1 1.5 --b 0.6 --run-name bm25_k1.5_b0.6
```

파라미터를 바꿔가며 실험할 때는 `--run-name` 을 다르게 줄 것. run 파일이 그 이름으로
저장되어 나중에 한 번에 비교할 수 있음.

**코드를 고쳐야 한다면 `scripts/run_bm25.py` 를 복사해서 시작할 것.** 지켜야 할 규칙이
주석으로 들어 있음. 핵심만 옮기면 다음과 같음.

```python
# ⚠ text 가 아니라 embedding_text. Dense 인덱스가 이 필드로 만들어졌음
corpus_tokens = [tokenize(c["embedding_text"]) for c in chunks]
bm25 = BM25Okapi(corpus_tokens, k1=1.2, b=0.75)

scores = bm25.get_scores(tokenize(r["question"]))        # ⚠ question 원문만
# 동점 처리: 점수 내림차순, 같으면 chunk_id 오름차순
ranked = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))[:RETRIEVE_K]   # 50
```

> **동점 정렬 방향 주의**
> - BM25 는 점수가 **높을수록** 좋음 → `key=lambda x: (-score, chunk_id)`
> - FAISS 기본은 **L2 거리**라 **낮을수록** 좋음 → `key=lambda x: (score, chunk_id)`
> 방향을 뒤집으면 성능이 랜덤 수준으로 떨어짐. 결과가 이상하면 여기부터 확인할 것.

### 2-4. Hybrid 예시 (RRF)

```python
K_RRF = 60   # 팀 합의값으로 고정

def rrf(dense_ids, bm25_ids, k=K_RRF, top=RETRIEVE_K):   # top=50 으로 저장
    score = {}
    for rank, cid in enumerate(dense_ids, start=1):
        score[cid] = score.get(cid, 0) + 1 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        score[cid] = score.get(cid, 0) + 1 / (k + rank)
    return [cid for cid, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:top]
```

- Dense·BM25 의 run 파일을 먼저 만들어 두고, **그 결과를 읽어서 결합**하는 편이 좋음.
  같은 Dense 결과 위에서 결합 방식만 바꿔 비교할 수 있음
- **결합 방식 자체가 실험 변수라면** Dense·BM25 를 고정하고 결합만 바꿀 것

### 2-5. Reranker

- 입력은 **`question` + 검색된 청크 텍스트**만
- 청크 텍스트는 코퍼스의 **`embedding_text`** 를 쓸 것 (Dense·BM25 와 동일하게)
- 재정렬 후 `retrieved` 길이는 **10 이상** 유지 (K=10 까지 채점하므로)
- **후보 풀 크기는 튜닝 대상임.** `retrieve_k=50` 으로 저장해두고 10/20/30/50 을 비교 → 2-7 참조

### 2-7. 후보 수는 고정값이 아니라 실험 변수임

역할이 단계마다 다름.

| 방식 | 후보 수의 역할 | 스윕할 가치 |
|---|---|---|
| Dense / BM25 단독 | 몇 개를 반환할지 | **없음** — 20 이든 100 이든 top-10 이 동일해 점수가 안 변함 |
| Hybrid | 결합에 넣을 **후보 풀** | **있음** — 풀이 바뀌면 결합 top-5 가 바뀜 |
| Reranker | 재정렬할 **후보 풀** | **있음** — 상한과 비용이 함께 바뀜 |

**Reranker 의 천장**

```
Reranker 의 UnionSpanRecall@10  ≤  기반 리트리버의 UnionSpanRecall@(후보 수)
```

후보에 없는 정답은 아무리 잘 재정렬해도 못 살림. 풀을 키우면 천장이 올라가지만
**노이즈와 비용·시간이 비례해 증가**함. 풀 크기를 정하기 전에 기반 리트리버의
`UnionSpanRecall@20`, `@50` 을 먼저 재보면 판단이 쉬움 (실측치는 6-5 참조).

```bash
# 후보를 넉넉히 뽑아두고 K 를 크게 줘서 천장 확인
python scripts/run_dense_baseline.py --retrieve-k 50 --run-name dense_k50
python scripts/eval_retriever.py --run runs/dense_k50.json --k 10 20 30 50
```

**비교할 때 두 축을 섞지 말 것**

| 무엇을 비교하나 | 고정할 것 |
|---|---|
| 방법 비교 (Dense vs Hybrid vs Reranker) | 후보 수를 **같게** |
| 후보 수 튜닝 | 방법을 **같게** 하고 10/20/30/50 |

`retrieve_k` 와 Reranker 후보 수를 `run_name`·`config` 에 반드시 기록할 것. 기록만 되어
있으면 값이 달라도 나중에 축을 분리해 해석할 수 있음.

### 2-6. 절대 금지

```
answer, gold_chunks, gold_start/end, span_start/end, union_gold_coverage
```

**전부 채점 전용임.** 검색 쿼리나 Reranker 입력에 넣으면 데이터 누수라 점수가 의미를
잃음. 특히 `answer` 를 쿼리에 섞으면 성능이 비현실적으로 높게 나옴.

---

## 3. 제출 전 셀프체크

```bash
python scripts/eval_retriever.py --run runs/bm25_kiwi_k1.2_b0.75.json --validate-only
```

```
[ok] bm25_kiwi_k1.2_b0.75 검증 통과
```

이게 안 나오면 채점이 안 됨. 잡아주는 오류는 다음과 같음.

| 메시지 | 원인 | 조치 |
|---|---|---|
| `평가셋 문항 N건 누락` | 157문항을 다 안 돌림 | 전체 순회 확인 |
| `코퍼스에 없는 chunk_id` | 다른 코퍼스로 검색함 | `clean_chunks_450_70.jsonl` 다시 받기 |
| `corpus_sha256 불일치` | 코퍼스 파일이 다름 | 〃 |
| `retrieved 안에 중복 chunk_id` | 같은 청크를 두 번 반환 | dedup 후 재실행 |
| `retrieved 길이가 K 보다 짧음` | (경고) `retrieve_k` 가 K 보다 작음 | 50 권장 |

---

## 4. 채점

```bash
python scripts/eval_retriever.py --run runs/bm25_kiwi_k1.2_b0.75.json
```

출력은 세 블록으로 나옴. `USR` = UnionSpanRecall, `CH` = CoverageHit.

```
[근거 확보]
세그먼트             n    USR@1    USR@3    USR@5   USR@10  CH@10-0.5  CH@10-0.8  EvHit@10
전체               157    0.384    0.589    0.694    0.859      0.860      0.847     0.892
src:v1              57    0.372    0.542    0.612    0.806      0.807      0.807     0.842
src:v2_prose       100    0.390    0.616    0.740    0.888      0.890      0.870     0.920
type:faq             5    0.600    0.600    0.600    0.800      0.800      0.800     0.800  ※참고

[순위 품질 · 진단]
세그먼트             n   MRR@10  nDCG@10  DocHit@10  DocPrec@10     P@10
전체               157    0.597    0.609      0.905       0.553    0.139

[실서비스 — 2500자 예산, embedding_text 기준]
  UnionSpanRecall@2500c = 0.719   CoverageHit@2500c-0.8 = 0.694   ChunksUsed = 5.73
```

- `※참고` 는 n<10 세그먼트. 출력은 하되 **결론 근거로 쓰지 말 것**
- 결과 JSON 은 `results_retriever/score_<run_name>.json` 에 저장됨. 문항별 점수가 들어
  있어 실패 문항을 따로 볼 수 있음

---

## 5. 비교 — paired bootstrap

```bash
python scripts/eval_retriever.py --compare runs/dense_k50.json runs/bm25_k50.json
```

```
[전체]  n=157
  run                    USR@1   USR@3   USR@5  USR@10  CH@10-.8  MRR@10  USR@bud |  ΔUSR@10           95% CI   P(>0)
  dense_k50              0.384   0.589   0.694   0.859     0.847   0.597    0.719 |     (기준선)
  bm25_k50               0.215   0.453   0.600   0.736     0.707   0.423    0.614 |    -12.2p  [-19.7, -4.6]   0.001  *

  * = 95% 신뢰구간이 0 을 포함하지 않음 (차이가 유의)
```

- **첫 번째 파일이 기준선.** Dense 를 먼저 두는 것을 권장함
- **Δ 는 문항 단위 paired bootstrap 2,000회의 95% 신뢰구간**임. 시드 고정이라 재현됨
- **CI 가 0 을 포함하면 유의하지 않음.** "몇 %p 이상이면 의미 있다" 같은 고정 임계값을
  쓰지 않는 이유는, `UnionSpanRecall` 이 연속값이라 문항 수만으로 노이즈 폭을 정할 수
  없기 때문임
- 전체 / v1 / v2 세 구간으로 나눠 출력됨. **n 이 작은 구간일수록 CI 가 넓게 나옴**
  (실제로 v1 은 CI 가 v2 보다 훨씬 넓음)

---

## 6. 결과 읽는 법

### 6-1. 정답의 기준

이 평가셋의 원래 정답 기준은 **원문 Gold span**(`doc_id` + `gold_start`~`gold_end`) 임.
`gold_chunks`, `gold_coverage` 는 그 Gold 가 fixed_450 에서 어떤 청크에 걸리는지 **미리
계산해 둔 보조정보**이며, 채점기는 이 캐시를 재사용함.

157문항 전부 아래가 검증되어 있음.

```
gold 청크 offset == 코퍼스 청크 offset      261/261
overlap_chars   == 실제 교집합 길이          261/261
gold_coverage   == overlap / gold_length     261/261
union_gold_coverage == 합집합 / gold_length  157/157
```

### 6-2. 지표 하나하나

**1. DocHit@K — 정답 문서 자체를 찾았는지**

top-K 안에 정답 `doc_id` 의 청크가 **하나라도 있으면 1** (binary). 같은 문서를 찾긴 했지만
그 안에서 엉뚱한 부분을 찾았을 수도 있어 **가장 관대한 지표**임. 문서 선택 자체는
성공했는지 진단하는 용도. **높을수록 좋음**

**2. EvidenceHit@K — 실제 정답 근거를 하나라도 찾았는지**

top-K 안에 gold 청크가 **하나라도 있으면 1**. DocHit 보다 엄격함. 다만 Gold 가 여러
청크에 걸린 경우 그중 일부만 찾아도 1 이므로 **이것만 보면 부족함**. **높을수록 좋음**

**3. UnionSpanRecall@K — 정답 근거를 실제로 얼마나 가져왔는지 (주 지표)**

top-K 의 gold 청크들이 정답 원문 구간을 **몇 % 덮었는지**. 청크끼리 70자 overlap 이
있으므로 `gold_coverage` 를 단순 합산하지 않고 **문자 구간의 합집합**으로 계산함.

```
Gold 근거 전체 = 300자
Top5 가 합쳐서 240자를 커버
→ UnionSpanRecall@5 = 0.80
```

1.0 이면 필요한 근거를 전부 확보한 것. EvidenceHit 과 달리 **"조금 찾음"과 "거의 다
찾음"을 구분**할 수 있음. **높을수록 좋음**

> 단순 합산이 왜 틀리냐면 — 실제로 **157문항 중 89건(57%)** 이 `gold_coverage` 를 단순
> 합산하면 1.0 을 초과함. 겹치는 구간을 두 번 세기 때문임.

**4. CoverageHit@K-0.8 — 충분한 양의 근거를 확보했는지**

각 문항의 `UnionSpanRecall@K` 가 기준 이상인지 보는 binary 지표.

```
CoverageHit@10-0.8 = UnionSpanRecall@10 이 0.8 이상이면 1, 아니면 0
```

전체 문항 중 **Gold 근거의 80% 이상을 확보한 문항 비율**임. `EvidenceHit=1` 인데 실제로
근거를 20% 만 찾은 경우를 구분할 수 있음. `0.5` 와 `0.8` 둘 다 산출하되 **0.8 을 주요
보조 지표로 볼 것**. **높을수록 좋음**

**5. MRR@10 — 첫 정답 근거를 얼마나 빨리 찾았는지**

첫 gold 청크가 몇 등에 나왔는지. `1위 → 1.0`, `2위 → 0.5`, `5위 → 0.2`.
**첫 근거의 위치에만** 관심이 있고 이후 청크는 반영 안 하므로 UnionSpanRecall 과 같이
봐야 함. **높을수록 좋음**

**6. nDCG@10 — 중요한 근거를 앞쪽에 배치했는지**

MRR 보다 ranking 전체를 자세히 봄. Gold 를 많이 포함한 청크일수록 relevance 를 높게
주므로, Gold 90% 짜리를 1위에 둔 쪽이 10% 짜리를 1위에 둔 쪽보다 높게 평가됨.
**높을수록 좋음**

**7. Precision@K / DocPrecision@K — 불필요한 청크가 얼마나 적은지**

`Precision@K` 는 top-K 중 gold 청크의 비율, `DocPrecision@K` 는 정답 문서 청크의 비율.
Gold 가 여러 overlapping chunk 로 구성될 수 있어 **핵심 지표라기보다 보조 진단용**임.
**높을수록 좋음**

**8~10. 실서비스 관점 — 2,500자 예산**

| 지표 | 뜻 |
|---|---|
| `UnionSpanRecall@2500c` | 순위대로 청크를 넣다가 2,500자에 도달했을 때의 커버리지 |
| `CoverageHit@2500c-0.8` | 위 값이 0.8 이상인 문항 비율 |
| `ChunksUsed@2500c` | 예산 안에 들어간 평균 청크 수 (**진단용, 높고 낮음의 좋고 나쁨 없음**) |

실제 LLM 입력 상황에 가장 가까운 지표임. 좋은 근거를 앞에 놓지 못하면 중요한 청크가
context 밖으로 밀려 점수가 낮아짐.

```
A: USR@2500c = 0.90 / ChunksUsed = 5
B: USR@2500c = 0.90 / ChunksUsed = 8
→ 같은 근거량을 A 가 더 압축된 결과로 확보한 것
```

> 예산 계산은 **`embedding_text` 길이 기준**임 (Dense 인덱스도 이 필드로 만들어짐).

### 6-3. 지표 관계와 우선순위

```
DocHit          "맞는 문서라도 찾았나?"
   ↓ 더 엄격
EvidenceHit     "실제 근거를 하나라도 찾았나?"
   ↓ 더 자세히
UnionSpanRecall "필요한 근거를 얼마나 가져왔나?"
CoverageHit     "충분하다고 볼 정도(80%)까지 가져왔나?"
MRR             "첫 근거를 얼마나 빨리 가져왔나?"
nDCG            "좋은 근거들을 전체적으로 앞에 배치했나?"
```

항상 `DocHit >= EvidenceHit >= CoverageHit@K-0.5 >= CoverageHit@K-0.8` 이 성립함
(채점기가 assert 로 검사함).

**한 지표로 승자를 정하지 말 것.** 우선순위는 다음과 같음.

| 목적 | 지표 |
|---|---|
| **근거 확보 능력** | `UnionSpanRecall@K` → `CoverageHit@K-0.8` → `EvidenceHit@K` |
| **순위 품질** | `MRR@10` → `nDCG@10` |
| **실서비스 관점** | `UnionSpanRecall@2500c` → `CoverageHit@2500c-0.8` → `ChunksUsed@2500c` |
| **진단용** | `DocHit@K` → `Precision@K` |

### 6-4. 기준선 (harness 정상 여부 자가진단)

`retrieve_k=50`, 전체 157문항 기준.

| run | USR@1 | USR@3 | USR@5 | **USR@10** | CH@10-0.8 | EvHit@10 | MRR@10 | nDCG@10 | DocHit@10 | USR@2500c | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_k50` | 0.384 | 0.589 | 0.694 | **0.859** | 0.847 | 0.892 | 0.597 | 0.609 | 0.905 | 0.719 | 530ms |
| `bm25_k50` | 0.215 | 0.453 | 0.600 | **0.736** | 0.707 | 0.790 | 0.423 | 0.464 | 0.866 | 0.614 | **13ms** |
| 랜덤 하한 | 0.000 | 0.006 | 0.006 | 0.016 | — | 0.006 | 0.003 | — | — | — | — |

자기 구현이 이 근처에서 시작하면 harness 는 정상임. **0.1 이하가 나오면** 동점 정렬
방향이나 텍스트 필드를 잘못 썼을 가능성이 큼(2-3 참조).

BM25 는 Dense 보다 `USR@10` 기준 **12.2%p 낮음** (paired bootstrap 95% CI `[-19.7, -4.6]`,
유의함). 다만 **latency 는 13ms 대 530ms 로 40배 빠름** (Dense 는 임베딩 API 왕복 때문).

### 6-5. 후보 수 saturation — Reranker 후보 정할 때 볼 것

`retrieve_k=50` 으로 한 번 저장해두면 재검색 없이 아래를 볼 수 있음.

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

**아직 완전히 saturate 되지는 않았음.** Dense 는 @30 → @50 이 +1.4%p 로 꺾이는 조짐이
보이나 BM25 는 +4.1%p 로 여전히 오름. Reranker 후보는 **30~50 사이**에서 각자 이득/비용을
확인한 뒤 정할 것.

즉 `Reranker 의 USR@10 ≤ 기반 리트리버의 USR@(후보 수)` 라는 천장이 후보를 늘릴수록
계속 올라가는 상태임. 후보를 늘리면 비용·시간도 비례해 늘어나므로 **어디서 이득이
꺾이는지 각자 확인한 뒤 결정할 것.**

### 6-6. 판단 기준

| 기준 | 내용 |
|---|---|
| 유의성 | **고정 임계값을 쓰지 않음.** `--compare` 의 paired bootstrap 95% CI 로 판단. CI 가 0 을 포함하면 유의하지 않음 |
| 구조별 결론 | `size`(121)·`section`(27)까지만. **faq(5)·whole(4)은 참고용** |
| topic 별 결론 | stock(71)·fund(33)·bond(20)까지. etf(13)·bank(12)는 참고용 |
| 작은 세그먼트 | 출력은 되지만 `※참고` 표시됨. 삭제하지 않는 이유는 경향 확인용 |

### 6-4. 눈여겨볼 구간

Dense 기준선에서 **`topic:bond` 가 압도적 최약점**임 (`USR@10` 0.553, `MRR@10` 0.359).

원인은 **유사 문서 중복**으로 보임. 코퍼스에 같은 주제를 설명하는 문서가 여러 개 있어,
검색이 내용상 맞는 문단을 가져와도 정답 라벨이 지정한 문서가 아니면 0점 처리됨.
예를 들어 "ETF 하나만 사도 분산투자가 되나요?" 는 상위 5개가 전부 ETF 분산투자를
정확히 설명하지만 서로 다른 문서라 점수를 못 받음. 채권도 설명 문서가
`doc_9e4ecd4a0eb2` / `doc_d0ab541c0e80` / `doc_ba29b232e816` 등 여러 개임.

즉 이 구간의 낮은 점수는 **리트리버 성능만의 문제가 아니라 코퍼스·라벨 특성**이
섞여 있음. 절대값보다 **방식 간 상대 비교**로 읽을 것.

**BM25 나 Reranker 가 이 구간을 개선하는지**가 이번 실험의 관전 포인트임.

---

## 7. 자주 하는 실수

| # | 실수 | 증상 | 확인 |
|---|---|---|---|
| 1 | `text` 로 검색 (BM25) | Dense 보다 이유 없이 낮음 | `embedding_text` 를 쓰고 있나 |
| 2 | 동점 정렬 방향 반대 | 점수가 랜덤 수준(0.01 근처) | BM25 는 `-score`, FAISS 는 `+score` |
| 3 | 후보 수를 기록 안 함 | 방법 차이인지 후보 풀 차이인지 구분 불가 | `run_name`·`config` 에 기록 |
| 4 | `answer` 를 쿼리에 포함 | 점수가 비현실적으로 높음(0.95+) | 쿼리는 `question` 만 |
| 5 | 인덱스를 직접 재빌드 | `index_sha256` 경고 | 공유본 그대로 쓸 것 |
| 6 | 채점을 직접 구현 | 남의 수치와 미묘하게 어긋남 | `eval_retriever.py` 사용 |
| 7 | run 파일 미제출 | 나중에 재채점 불가 | `runs/` 째로 공유 |

> **4번은 특히 조심할 것.** 성능이 갑자기 0.95 이상 나오면 실력이 아니라 누수임.

---

## 8. 결과 공유

1. **`runs/<run_name>.json`** — 검색 결과 (필수). 채점 기준이 바뀌어도 재실행 없이
   다시 채점할 수 있음
2. **`results_retriever/score_<run_name>.json`** — 채점 결과
3. 실행에 쓴 스크립트

`runs/` 와 `results_retriever/` 는 gitignore 대상이므로 **드라이브로 공유할 것.**

---

## 9. 파일 요약

| 파일 | 용도 |
|---|---|
| `scripts/run_dense_baseline.py` | Dense 실행 + run 파일 형식 참조 구현 |
| `scripts/run_bm25.py` | BM25 실행 (Kiwi 토크나이저) |
| `scripts/eval_retriever.py` | **공용 채점기** (검증·채점·비교) |
| `docs/retriever_eval_protocol.md` | 통일 규칙 23항목, 평가셋 검증 결과 |
| `docs/retriever_eval_guide.md` | 이 문서 (실행 방법) |

**데이터는 레포에 없음.** 아래는 드라이브에서 받을 것.

| 받을 것 | 두는 위치 |
|---|---|
| 평가셋 `rag_testset_retriever_v1v2_fixed450_team_eval.json` | `data/testset/` |
| 코퍼스 `chunks.jsonl` (+ `manifest.json`) | `data/chunking_data/fixed_450_70/` |
| Dense 인덱스 `index.faiss`, `index.pkl` | `vectorstores/fixed_450_70/` |
