# GCP 배포 결과 — 2026-10-03

- 기준 Git commit: ecff98836f47ad59810425a364cbd5fc1c5bcfd2
- 프로젝트: turini (941433084012)
- 리전: asia-northeast3
- 서비스: turini-rag
- revision: turini-rag-00001-plx
- URL: https://turini-rag-941433084012.asia-northeast3.run.app
- status.url: https://turini-rag-ss3o2mzysq-du.a.run.app
- Cloud Build: a3d631e6-2347-4eee-a849-afc840cee3b7 / SUCCESS
- 인증된 GET /health: 200, ready=true, boot_s=22.0, corpus_chunks=1698
- IAM 인증 후 X-API-Key 없는 POST /chat: 401 (모델 호출 없음)

## 배포 설정

- 메모리 2Gi / CPU 1
- concurrency 1 / timeout 120초
- min instances 1 / max instances 3
- startup CPU boost 활성
- 비공개 Cloud Run: no-allow-unauthenticated
- 실행 및 빌드 계정: 941433084012-compute@developer.gserviceaccount.com
- Secret Manager 환경변수: OPENAI_API_KEY, COHERE_API_KEY, RAG_API_KEY (각 버전 1)
- 코퍼스와 인덱스는 이미지에 포함. HF_DATASET/HF_TOKEN 불필요.

## 로컬 변경

- .gcloudignore 신규: 업로드 목록 명시, .env와 실험 산출물 제외
- Dockerfile: 런타임 데이터 COPY 3줄 활성
- 업로드 대상: 64개 파일, 28,540,424 bytes (압축 전)
- 기존 검증 데이터와 .env 복사: gitignored

## 추가한 IAM 권한

- 각 시크릿에 실행 계정의 roles/secretmanager.secretAccessor
- run-sources-turini-asia-northeast3 버킷에 빌드 계정의 roles/storage.objectViewer
- turini 프로젝트에 빌드 계정의 roles/run.builder (사용자 승인 후 적용)

## 실행 명령

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project=turini --quiet
gcloud secrets add-iam-policy-binding <SECRET_NAME> --project=turini --member=serviceAccount:941433084012-compute@developer.gserviceaccount.com --role=roles/secretmanager.secretAccessor
gcloud storage buckets add-iam-policy-binding gs://run-sources-turini-asia-northeast3 --member=serviceAccount:941433084012-compute@developer.gserviceaccount.com --role=roles/storage.objectViewer
gcloud projects add-iam-policy-binding turini --member=serviceAccount:941433084012-compute@developer.gserviceaccount.com --role=roles/run.builder
gcloud run deploy turini-rag --project=turini --source . --region=asia-northeast3 --memory=2Gi --cpu=1 --concurrency=1 --min-instances=1 --max-instances=3 --timeout=120 --no-allow-unauthenticated --service-account=941433084012-compute@developer.gserviceaccount.com --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:1,COHERE_API_KEY=COHERE_API_KEY:1,RAG_API_KEY=RAG_API_KEY:1 --quiet
```

최초 배포는 빌드 계정의 소스 버킷 읽기 권한 부족으로 실패했다. 권한 보완 후 재배포 성공.

## 남은 앱 연결

앱 백엔드의 호출 계정에 해당 서비스의 roles/run.invoker를 부여해야 한다. 요청은 Google ID token(Authorization: Bearer)과 공유 비밀(X-API-Key)을 함께 사용한다. RAG_API_URL은 위 URL을 사용한다.

이번 확인은 배포·기동·인증 검증이다. 실제 금융 질문의 /chat 모델 호출과 앱의 state 왕복 통합 검증은 아직 수행하지 않았다. 앱 호출 계정 생성, 렌더 연결, 예산 알림 설정도 아직 수행하지 않았다.

API 키·토큰을 출력하거나 커밋하지 않았으며 Git commit/push도 수행하지 않았다.
