# data/

이 폴더에는 코퍼스 JSONL 파일을 넣습니다. **실제 데이터는 레포에 올리지 않습니다**
(`.gitignore` 로 `data/*.jsonl` 제외). 각자 로컬에 데이터를 배치한 뒤 인덱스를 빌드하세요.

## 형식 (JSONL, 1줄 = 1청크)

```json
{
  "doc_id": "doc_fund_pre_info",
  "chunk_id": "chunk_...",
  "title": "현명한 펀드투자방법 - 펀드가입 전 정보수집",
  "section_title": "...",
  "topic": "fund_pre_subscription",
  "source_name": "금융투자협회 펀드다모아",
  "source_url": "https://...",
  "text": "본문 ...",
  "embedding_text": "임베딩에 넣을 정제 텍스트 (없으면 title+text 사용)"
}
```

- 인덱싱에 실제로 들어가는 텍스트: `embedding_text` (없으면 `title` + `text` fallback)
  → `scripts/build_index.py` 의 `_row_to_text()`
- 검색 결과 식별자(`doc_id`)로는 `chunk_id`(없으면 `doc_id`)가 쓰입니다.
- 그 외 필드는 메타데이터로 보존됩니다 (`_row_to_metadata()`).

## 빌드

```bash
python scripts/build_index.py                       # data/*.jsonl 전체
CORPUS_GLOB="data/01_*.jsonl" python scripts/build_index.py   # 일부만
```

## 평가용 goldset (선택)

리트리버 지표(Hit/MRR/nDCG)를 재려면 `data/goldset.json` 이 필요합니다.

```json
[
  { "query_id": "Q001", "question": "질문", "relevant_ids": ["chunk_..."] }
]
```
