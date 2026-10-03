# RAG 서비스 컨테이너 — 쪼개 배포할 때 "무거운 쪽"만 담는다.
#   앱 백엔드·포트폴리오 생성 LLM 은 이 이미지에 들어가지 않는다 (가벼우므로 다른 곳에).
#
# 메모리: 코퍼스·FAISS·BM25(Kiwi)·클라이언트를 올리면 약 900MB 를 쓴다.
#   512MB 환경(렌더 무료 등)에서는 뜨지 않는다. RAM 2GB 이상인 곳에 올릴 것.
#
# 빌드·실행
#   docker build -t turini-rag .
#   docker run -p 7860:7860 --env-file .env -e RAG_API_KEY=... -e HF_DATASET=... -e HF_TOKEN=... turini-rag
#
# 포트는 $PORT 로 받는다. Cloud Run 은 기동 때 PORT(기본 8080)를 주입해 아래 ENV 기본값을 덮어쓴다.
# 7860 은 로컬 실행·HF Spaces 기본값으로 남겨둔 fallback 이다.
FROM python:3.11-slim

WORKDIR /app

# 빌드 캐시가 살도록 의존성을 먼저 깐다. 실험용 프로바이더(anthropic·google-genai)는 서빙에 불필요해 뺀다.
COPY requirements.txt requirements-serve.txt ./
RUN grep -viE "^(anthropic|google-genai)" requirements.txt > requirements-runtime.txt \
    && pip install --no-cache-dir -r requirements-runtime.txt -r requirements-serve.txt

COPY config.py ./
COPY core/ ./core/
COPY multiturn/ ./multiturn/
COPY serve/ ./serve/
COPY scripts/answer_spec.py ./scripts/

# 코퍼스·FAISS 인덱스·관할 라벨은 git 에 없다. 넣는 방법은 두 가지 (docs/deploy_plan.md §3-1):
#   (A) 이미지에 포함 — 아래 COPY 3줄 주석 해제. Cloud Run 권장(Artifact Registry 는 비공개,
#       HF 계정·토큰 불필요, 기동 때 27MB 다운로드 없음). .dockerignore 는 이 경로들을 막지 않는다.
#   (B) 기동 때 HF 비공개 Dataset 에서 내려받기 — HF_DATASET · HF_TOKEN (serve/bootstrap.py)
# COPY data/chunking_data/fixed_450_70/clean_chunks_450_70_v2.jsonl ./data/chunking_data/fixed_450_70/
# COPY data/jurisdiction.json ./data/
# COPY vectorstores/fixed_450_70_v2/ ./vectorstores/fixed_450_70_v2/

ENV PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# 워커 1개 — 코퍼스·인덱스가 워커마다 통째로 올라가므로 늘리면 메모리가 배로 든다.
CMD uvicorn serve.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
