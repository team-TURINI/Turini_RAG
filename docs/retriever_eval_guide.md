# 리트리버 평가 사용 가이드

**대상** 각자 Dense / BM25 / Hybrid / Reranker 를 실험한 뒤 성능을 비교할 팀원
**규칙 문서** 통일해야 할 조건은 `docs/retriever_eval_protocol.md` 참조. 이 문서는 **실행 방법**만 다룸

---

## 0. 3분 요약

```bash
# 1) 검색 실행 → run 파일 생성
python scripts/run_dense_baseline.py --fetch-k 20

# 2) 채점
python scripts/eval_retriever.py --run runs/dense_baseline.json

# 3) 여러 방식 비교 (첫 번째가 기준선)
python scripts/eval_retriever.py --compare runs/dense_baseline.json runs/bm25.json
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
| 코퍼스 | `data/chunking_data/fixed_450_70/chunks.jsonl` | 1,767청크 |
| Dense 인덱스 | `vectorstores/fixed_450_70/` | 구글드라이브 공유본 |

### 1-3. 파일이 같은지 먼저 확인 (필수)

```bash
python -c "import hashlib;print(hashlib.sha256(open('data/chunking_data/fixed_450_70/chunks.jsonl','rb').read()).hexdigest())"
```

**기대값**
```
chunks.jsonl : 4cef51096632aba7ad487be6fb6fc1b293fc296d44f093462d539d6a39a1cf70
index.faiss  : e2860671d1bfc513f2c0407d390d0e12070399be16c5cf352300b0c0d66f457a
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
  "config": { "tokenizer": "kiwi", "k1": 1.2, "b": 0.75, "fetch_k": 20 },
  "corpus_sha256": "4cef5109...",
  "items": [
    { "id": "rag_fund_01", "retrieved": ["chunk_id_1위", "chunk_id_2위", "..."] }
  ]
}
```

- `items` 는 **157문항 전부** 있어야 함
- `retrieved` 는 **순위 오름차순**(1등이 맨 앞), 길이는 `fetch_k`(=20)
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
ranked = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))[:FETCH_K]
```

> **동점 정렬 방향 주의**
> - BM25 는 점수가 **높을수록** 좋음 → `key=lambda x: (-score, chunk_id)`
> - FAISS 기본은 **L2 거리**라 **낮을수록** 좋음 → `key=lambda x: (score, chunk_id)`
> 방향을 뒤집으면 성능이 랜덤 수준으로 떨어짐. 결과가 이상하면 여기부터 확인할 것.

### 2-4. Hybrid 예시 (RRF)

```python
K_RRF = 60   # 팀 합의값으로 고정

def rrf(dense_ids, bm25_ids, k=K_RRF, top=FETCH_K):
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
- **후보는 `fetch_k`=20 에서 재정렬.** top-50 에서 재정렬한 사람과는 비교 불가함
- 재정렬 후에도 `retrieved` 길이는 20 을 유지할 것 (K=10 지표까지 계산해야 하므로)

### 2-6. 절대 금지

```
answer, gold_chunks, gold_start/end, span_start/end, union_gold_coverage
```

**전부 채점 전용임.** 검색 쿼리나 Reranker 입력에 넣으면 데이터 누수라 점수가 의미를
잃음. 특히 `answer` 를 쿼리에 섞으면 성능이 비현실적으로 높게 나옴.

---

## 3. 제출 전 셀프체크

```bash
python scripts/eval_retriever.py --run runs/bm25_kiwi.json --validate-only
```

```
[ok] bm25_kiwi_k1.2_b0.75 검증 통과
```

이게 안 나오면 채점이 안 됨. 잡아주는 오류는 다음과 같음.

| 메시지 | 원인 | 조치 |
|---|---|---|
| `평가셋 문항 N건 누락` | 157문항을 다 안 돌림 | 전체 순회 확인 |
| `코퍼스에 없는 chunk_id` | 다른 코퍼스로 검색함 | `chunks.jsonl` 다시 받기 |
| `corpus_sha256 불일치` | 코퍼스 파일이 다름 | 〃 |
| `retrieved 안에 중복 chunk_id` | 같은 청크를 두 번 반환 | dedup 후 재실행 |
| `retrieved 길이가 K 보다 짧음` | (경고) `fetch_k` 가 작음 | 20 이상으로 |

---

## 4. 채점

```bash
python scripts/eval_retriever.py --run runs/bm25_kiwi.json
```

출력 예시 (Dense 기준선)

```
세그먼트          n   Cov@1   Cov@3   Cov@5  Cov@10   Hit@5   MRR@5  nDCG@5     P@5  DocHit@5
전체            157   0.376   0.598   0.714   0.866   0.764   0.577   0.554   0.223   0.612
src:v1           57   0.362   0.564   0.646   0.801   0.684   0.525   0.520   0.172   0.435
src:v2_prose    100   0.384   0.618   0.754   0.903   0.810   0.607   0.574   0.252   0.712
type:size       121   0.375   0.595   0.720   0.868   0.777   0.590   0.552   0.243   0.673
type:section     27   0.357   0.667   0.778   0.922   0.815   0.563   0.600   0.178   0.511
topic:stock      71   0.388   0.601   0.748   0.899   0.789   0.603   0.572   0.245   0.741
topic:bond       20   0.247   0.359   0.484   0.619   0.600   0.399   0.345   0.160   0.520
```

결과 JSON 은 `results_retriever/score_<run_name>.json` 에 저장됨. 문항별 점수까지 들어
있어 나중에 실패 문항을 따로 볼 수 있음.

---

## 5. 비교

```bash
python scripts/eval_retriever.py --compare runs/dense_baseline.json runs/bm25_kiwi.json runs/hybrid_rrf.json
```

```
run                    Cov@1   Cov@3   Cov@5  Cov@10   Hit@5   MRR@5     P@5    ΔCov@5
dense_baseline         0.376   0.598   0.714   0.866   0.764   0.577   0.223     +0.0p
bm25_kiwi              ...                                                       +3.1p  ▲
hybrid_rrf             ...                                                       +5.4p  ▲
```

- **첫 번째 파일이 기준선.** Dense 를 먼저 두는 것을 권장함
- `ΔCov@5` 는 기준선 대비 차이. **2%p 미만이면 ▲▼ 마커가 안 붙음** — 노이즈이기 때문임
- 전체 / v1 / v2 세 구간으로 나눠서 출력됨

---

## 6. 결과 읽는 법

### 6-1. 지표

| 지표 | 뜻 | 비고 |
|---|---|---|
| **Coverage@K** | top-K 가 정답 구간을 덮은 **문자 비율** | **주 지표** |
| Hit@K | top-K 에 gold 청크가 하나라도 있으면 1 | 관대함. 단독 판단 금지 |
| MRR@K | 첫 gold 청크 순위의 역수 | 순위 품질 |
| nDCG@K | gold_coverage 를 등급으로 쓴 nDCG | 순위 품질 |
| Precision@K | top-K 중 gold 청크 비율 | 노이즈 |
| DocHit@K | top-K 중 정답 문서에서 온 비율 | 문서를 맞혔나 |

> **Hit@K 만 보면 안 되는 이유** — 157문항 중 93문항이 gold 청크 2개 이상이고, 그중
> 50문항은 청크 하나로 100% 커버가 불가능함. Hit 로만 채점하면 정답 일부만 가져와도
> 만점이라 리트리버 간 차이가 뭉개짐.

### 6-2. 기준선 (harness 정상 여부 자가진단)

| run | Cov@1 | Cov@3 | **Cov@5** | Cov@10 | Hit@5 | MRR@5 | P@5 | latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_baseline` | 0.376 | 0.598 | **0.714** | 0.866 | 0.764 | 0.577 | 0.223 | 561ms |
| `bm25_kiwi_k1.2_b0.75` | 0.212 | 0.450 | **0.591** | 0.728 | 0.656 | 0.398 | 0.171 | 12ms |
| (랜덤 하한) | 0.000 | 0.006 | 0.006 | 0.016 | 0.006 | 0.003 | 0.001 | — |

자기 구현이 위 수치 근처에서 시작하면 harness 는 정상임. **0.1 이하가 나오면**
동점 정렬 방향이나 텍스트 필드를 잘못 썼을 가능성이 큼(2-3 참조).

**현재 BM25 는 Dense 보다 12.3%p 낮음.** 파라미터 튜닝과 Hybrid 결합의 출발점임.
다만 latency 는 12ms 대 561ms 로 BM25 가 압도적임(Dense 는 임베딩 API 왕복 때문).

### 6-3. 판단 기준

| 기준 | 내용 |
|---|---|
| 노이즈 | 157문항 기준 1문항 = **0.64%p**. **2%p 미만 차이는 노이즈** |
| 구조별 결론 | `size`(121)·`section`(27)까지만. **faq(5)·whole(4)은 결론 근거로 쓰지 말 것** |
| topic 별 결론 | stock(71)·fund(33)·bond(20)까지. etf(13)·bank(12)는 참고용 |

### 6-4. 눈여겨볼 구간

Dense 기준선에서 **`topic:bond`(0.484)와 `topic:etf`(0.462)가 유독 낮음.**

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
| 3 | `fetch_k` 를 다르게 | Reranker 결과가 비교 불가 | 전원 20 |
| 4 | `answer` 를 쿼리에 포함 | 점수가 비현실적으로 높음(0.95+) | 쿼리는 `question` 만 |
| 5 | 인덱스를 직접 재빌드 | `index_sha256` 경고 | 공유본 그대로 쓸 것 |
| 6 | 채점을 직접 구현 | 남의 수치와 1~2%p 어긋남 | `eval_retriever.py` 사용 |
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
