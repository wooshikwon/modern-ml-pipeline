# 배포 가이드 (Deployment Guide)

MMP 프로젝트를 Docker 이미지로 빌드하고 컨테이너 레지스트리에 푸시하는 방법을 안내합니다.

---

## 1. 개요

### MMP의 범위

```text
mmp init → 실험/학습 → Docker 이미지 빌드 → GCR/ECR 푸시
```

MMP는 **프로젝트 생성부터 컨테이너 레지스트리 푸시까지** 지원합니다.

| MMP가 제공하는 것 | 설명 |
|------------------|------|
| `Dockerfile` | 프로덕션 이미지 빌드 (학습/추론/서빙 통합) |
| `docker-compose.yml` | 로컬 개발 및 테스트 |
| `configs/`, `recipes/` | 환경별 설정 파일 |

### MMP 범위 외

Kubernetes 매니페스트(Deployment, CronJob, ConfigMap 등)는 **MMP에서 제공하지 않습니다**. 조직의 GitOps 레포지토리에서 플랫폼팀/SRE가 별도 관리합니다.

```text
[MMP 프로젝트]                          [GitOps 레포 - 별도 관리]
my-ml-project/                          k8s-deployments/
├── Dockerfile         ──빌드/푸시──→   └── my-ml-project/
├── docker-compose.yml                      ├── dev/
├── configs/                                ├── staging/
└── recipes/                                └── prod/
```

### 단일 이미지 컨셉

빌드된 이미지는 **단일 이미지**로 학습, 추론, API 서빙을 모두 지원합니다. 실행 시 command만 다르게 지정합니다:

```bash
mmp serve-api --run-id abc123 -c configs/prod.yaml           # API 서빙
mmp batch-inference --run-id abc123 -d gs://bucket/data.csv  # 배치 추론
mmp train -r recipes/model.yaml -d gs://bucket/train.csv     # 학습
```

---

## 2. Docker 이미지 빌드

### 기본 빌드

```bash
docker build -t my-model:latest .
```

### Extras 선택

Dockerfile의 `RUN pip install` 라인에서 필요한 extras를 조합합니다:

```dockerfile
# 사용 가능한 extras:
#   standard     : XGBoost, scikit-learn, Optuna, SHAP (기본)
#   ml-extras    : LightGBM, CatBoost
#   cloud-extras : BigQuery, GCS, S3
#   torch-extras : PyTorch, LSTM, TabNet
#   causal       : CausalML
#   all          : 위 전체 포함

# 클라우드 배포 + Gradient Boosting (기본 권장)
RUN pip install ".[ml-extras,cloud-extras]"
```

**권장 조합:**

| 시나리오 | extras |
|----------|--------|
| 클라우드 배포 (XGBoost, LightGBM) | `ml-extras,cloud-extras` |
| 클라우드 배포 (딥러닝) | `cloud-extras,torch-extras` |
| 온프레미스 배포 (XGBoost만) | `standard` |
| 전체 기능 | `all` |

### 이미지 태그 전략

```bash
docker build -t my-model:v1.0.0 .
docker build -t my-model:v1.0.0-$(git rev-parse --short HEAD) .
```

---

## 3. 로컬 테스트 (Docker Compose)

프로덕션 배포 전 로컬에서 동일한 이미지로 테스트할 수 있습니다.

### 환경변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
MODEL_RUN_ID=abc123def456
CONFIG_PATH=configs/production.yaml
RECIPE_PATH=recipes/model.yaml
TRAIN_DATA_PATH=data/train.csv
MLFLOW_TRACKING_URI=./mlruns
EOF
```

### 실행

```bash
# API 서버
docker-compose up api

# 학습
docker-compose run --rm train

# 배치 추론
MODEL_RUN_ID=abc123 INFERENCE_DATA_PATH=data/test.csv docker-compose run --rm inference

# MLflow UI (선택)
docker-compose --profile mlflow up mlflow
```

---

## 4. 이미지 레지스트리 푸시

### GCR (Google Container Registry)

```bash
PROJECT_ID=my-gcp-project
IMAGE_TAG=v1.0.0

docker build -t gcr.io/${PROJECT_ID}/mmp:${IMAGE_TAG} .
docker push gcr.io/${PROJECT_ID}/mmp:${IMAGE_TAG}
```

### ECR (AWS)

```bash
AWS_REGION=ap-northeast-2
AWS_ACCOUNT_ID=123456789012

aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mmp:${IMAGE_TAG} .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mmp:${IMAGE_TAG}
```

---

## 5. Kubernetes 연동 참고사항

> **Note**: Kubernetes 배포는 MMP 범위 외입니다. 아래는 플랫폼팀에 전달할 참고 정보입니다.

### MMP 이미지 실행 명령어

플랫폼팀이 k8s 매니페스트 작성 시 사용할 command/args:

| 용도 | command | args 예시 |
|------|---------|-----------|
| API 서빙 | `mmp serve-api` | `--run-id abc123 -c /app/configs/prod.yaml` |
| 배치 추론 | `mmp batch-inference` | `--run-id abc123 -d gs://bucket/data.csv` |
| 학습 | `mmp train` | `-r /app/recipes/model.yaml -d gs://bucket/train.csv` |

### Health Check 엔드포인트

API 서빙 시 사용하는 헬스체크:

| 엔드포인트 | 용도 | 응답 |
|------------|------|------|
| `GET /health` | Liveness Probe | `{"status": "ok"}` (항상 200) |
| `GET /ready` | Readiness Probe | 모델 로드 완료 시 200, 미완료 시 503 |

### 환경변수

| 환경변수 | 용도 | 기본값 |
|----------|------|--------|
| `MODEL_RUN_ID` | 서빙/추론할 모델의 MLflow run ID | - |
| `CONFIG_PATH` | Config 파일 경로 | `configs/production.yaml` |
| `RECIPE_PATH` | Recipe 파일 경로 | `recipes/model.yaml` |
| `TRAIN_DATA_PATH` | 학습 데이터 경로 | `data/train.csv` |
| `INFERENCE_DATA_PATH` | 추론 데이터 경로 | - |
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `./mlruns` |

### 리소스 권장 사양

**API 서버:**

| 모델 크기 | Memory |
|-----------|--------|
| 소형 (XGBoost) | 512Mi - 2Gi |
| 중형 (LightGBM) | 1Gi - 4Gi |
| 대형 (LSTM) | 2Gi - 8Gi |

**배치 추론/학습:**

| 데이터 크기 | Memory |
|-------------|--------|
| < 100K rows | 1Gi |
| 100K - 1M rows | 4Gi |
| > 1M rows | 8Gi |

---

## 참고

- [API 서빙 가이드](./API_SERVING_GUIDE.md): REST API 엔드포인트 상세
- [CLI 레퍼런스](./CLI_REFERENCE.md): `mmp serve-api`, `mmp batch-inference` 명령어
- [환경 설정 가이드](./ENVIRONMENT_SETUP.md): Config 파일 작성법
