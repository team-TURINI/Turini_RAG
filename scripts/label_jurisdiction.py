"""문서 관할(국가) 라벨 — KR / US / GLOBAL.

코퍼스·인덱스 파일은 건드리지 않는다. doc_id → 라벨 **사이드카** 하나만 만든다.
검색 시점에 core/jurisdiction.py 가 이 파일을 읽어 필터한다.

규칙 (문서 단위, 위에서부터 먼저 맞는 것)
  1. source = SEC Investor.gov, yfinance                    → US
  2. source = FRED
       제목·본문 앞 300자에 미국·연준·연은·S&P·나스닥… → US   (미국 지표 해설)
       제목에 원자재·환율·국제…                          → GLOBAL (국가 무관 시장)
       나머지 (CPI란·GDP란·인플레이션과 구매력 등 개념)   → GLOBAL
  3. 그 외 출처 (금감원·한은·KB Think·투교협·KCIE·KDI·KOFIA·KRX…) → KR
       본문에 미국 언급이 많아도 "한국 투자자가 해외주식 사는 법" 류라 KR 유지 (검토 목록에 기록)

산출
  data/jurisdiction.json          {doc_id: {jurisdiction, rule}}  + 집계
  docs/jurisdiction_labels.md     규칙·건수·GLOBAL 전체 목록·검토 대상 목록

실행: python scripts/label_jurisdiction.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config as cfg  # noqa: E402

META = PROJECT_ROOT / "data" / "docs_metadata_cleaned.jsonl"
OUT_JSON = PROJECT_ROOT / "data" / "jurisdiction.json"
OUT_MD = PROJECT_ROOT / "docs" / "jurisdiction_labels.md"

US_SOURCES = {"SEC Investor.gov", "yfinance"}
US_RE = re.compile(r"미국|연준|연은|Fed\b|FOMC|S&P|나스닥|다우|Baa|Aaa|달러 인덱스|미 국채|미국채|재무부|"
                   r"금융상황지수|금융스트레스지수|국가활동지수|Chicago Fed|St\. Louis Fed")
GLOBAL_RE = re.compile(r"브렌트|WTI|구리|천연가스|금 가격|금값|원자재|유가|환율|엔/|유로|위안|VIX|글로벌|세계|국제")
KR_MENTION_RE = re.compile(r"미국|SEC|나스닥|S&P|연준|달러|월가|뉴욕")


def label(d: dict) -> tuple[str, str]:
    src, title, head = d["source_name"], d["title"], d["text"][:300]
    if src in US_SOURCES:
        return "US", f"source={src}"
    if src == "FRED":
        if US_RE.search(title) or US_RE.search(head):
            return "US", "FRED+미국 지표"
        if GLOBAL_RE.search(title):
            return "GLOBAL", "FRED+원자재·환율·국제"
        return "GLOBAL", "FRED 개념 설명"
    return "KR", f"source={src}"


def main() -> None:
    docs = [json.loads(l) for l in META.open(encoding="utf-8") if l.strip()]
    out, by_rule, lists = {}, Counter(), defaultdict(list)
    for d in docs:
        j, rule = label(d)
        out[d["doc_id"]] = {"jurisdiction": j, "rule": rule, "source_name": d["source_name"], "title": d["title"]}
        by_rule[(j, rule)] += 1
        lists[j].append(d)

    # 검토 목록 — 한국 출처인데 미국 언급 밀도 높은 문서 (라벨은 KR 유지, 사람이 한 번 보라고)
    review = []
    for d in docs:
        if out[d["doc_id"]]["jurisdiction"] != "KR":
            continue
        n = len(KR_MENTION_RE.findall(d["text"]))
        if n * 1000 / max(1, len(d["text"])) >= 2:
            review.append((d, n))

    # 코퍼스 청크 커버리지 — 라벨 없는 doc_id 가 있으면 KR 로 두되 경고
    chunk_cnt, missing = Counter(), set()
    with cfg.FINAL_CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            j = out.get(c["doc_id"], {}).get("jurisdiction")
            if j is None:
                missing.add(c["doc_id"]); j = "KR"
            chunk_cnt[j] += 1

    doc_cnt = Counter(v["jurisdiction"] for v in out.values())
    OUT_JSON.write_text(json.dumps({
        "version": "2026-09-19", "default": "KR", "labels": ["KR", "US", "GLOBAL"],
        "n_docs": len(out), "doc_counts": dict(doc_cnt), "chunk_counts": dict(chunk_cnt),
        "docs": out}, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# 문서 관할 라벨 (jurisdiction)", "",
         f"- 생성: `scripts/label_jurisdiction.py` → `data/jurisdiction.json` (메타 {len(docs)}행 · 고유 doc_id {len(out)}건, 코퍼스 sha 불변)",
         "- 금감원 금융상품통합비교공시 173행은 doc_id 하나를 공유 (전부 KR) — 규칙별 건수는 행 기준",
         f"- 문서: KR {doc_cnt['KR']} · US {doc_cnt['US']} · GLOBAL {doc_cnt['GLOBAL']}",
         f"- 청크(v2 코퍼스 {sum(chunk_cnt.values())}): KR {chunk_cnt['KR']} · US {chunk_cnt['US']} · GLOBAL {chunk_cnt['GLOBAL']}",
         f"- 라벨 없는 doc_id: {len(missing)}" + (f" — {sorted(missing)[:10]}" if missing else ""), "",
         "## 규칙별 건수", "", "| 라벨 | 규칙 | 문서 |", "|---|---|---|"]
    for (j, rule), n in sorted(by_rule.items(), key=lambda x: (-x[1])):
        L.append(f"| {j} | {rule} | {n} |")
    L += ["", "## GLOBAL 전체 목록 (국가 무관 — 항상 허용)", "", "| doc_id | 제목 | 규칙 |", "|---|---|---|"]
    for d in lists["GLOBAL"]:
        L.append(f"| {d['doc_id']} | {d['title']} | {out[d['doc_id']]['rule']} |")
    L += ["", "## US 목록 (FRED 제외)", "", "| doc_id | 출처 | 제목 |", "|---|---|---|"]
    for d in lists["US"]:
        if d["source_name"] != "FRED":
            L.append(f"| {d['doc_id']} | {d['source_name']} | {d['title']} |")
    L += ["", f"- FRED US {sum(1 for d in lists['US'] if d['source_name']=='FRED')}건은 제목이 '미국 …' 인 지표 해설 — 목록 생략",
          "", "## 검토 목록 — KR 로 뒀지만 미국 언급이 많은 문서", "",
          "한국 투자자 관점(해외주식 계좌 개설, 미국 주식 거래시간, 환헤지 ETF 등)이라 KR 유지. 다르게 볼 문서가 있으면 규칙에 예외 추가.", "",
          "| doc_id | 출처 | 제목 | 미국 언급 |", "|---|---|---|---|"]
    for d, n in review:
        L.append(f"| {d['doc_id']} | {d['source_name']} | {d['title']} | {n}회/{len(d['text'])}자 |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[문서] {dict(doc_cnt)}   [청크] {dict(chunk_cnt)}   라벨 없음 {len(missing)}")
    print(f"[저장] {OUT_JSON}\n[저장] {OUT_MD}")


if __name__ == "__main__":
    main()
