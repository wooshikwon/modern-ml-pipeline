# Modern ML Pipeline

YAML 설정 기반의 머신러닝 파이프라인 CLI 도구입니다.

코드를 수정하지 않고 **YAML 설정 파일**만으로 모델 학습부터 API 서빙까지 처리합니다. 프로젝트 생성(`mmp init`)부터 실험, Docker 이미지 빌드, 컨테이너 레지스트리(GCR/ECR) 푸시까지 일관된 워크플로우를 제공합니다.


## 주요 특징

- **설정 기반 (Config-driven)**: YAML만으로 실험을 정의하고 환경 간 이식 가능
- **단일 이미지 배포**: 학습, 추론, API 서빙을 하나의 Docker 이미지로 지원 (GCR/ECR 푸시까지)
- **클라우드 데이터 연동**: GCS/S3/BigQuery에서 직접 데이터 로드
- **자동 실험 추적**: MLflow와 연동되어 모든 실험 결과와 모델이 자동 저장
- **Data Leakage 방지**: Train/Validation/Test/Calibration 4단계 분할 자동 처리
- **즉시 서빙**: 학습 완료 후 명령어 한 줄로 REST API 서버 기동


## 빠른 시작

### 1. 설치

**요구사항**: Python 3.10, 3.11, 3.12, 또는 3.13

#### 기본 설치

```bash
pip install modern-ml-pipeline        # pip
pipx install modern-ml-pipeline       # pipx (CLI 전역 설치, 권장)
```

#### 시나리오별 추가 설치

기본 설치 후 필요한 extras를 추가합니다:

| 시나리오 | pip | pipx inject |
|----------|-----|-------------|
| BigQuery/GCS/S3 | `pip install 'modern-ml-pipeline[cloud-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[cloud-extras]' --force` |
| LightGBM, CatBoost | `pip install 'modern-ml-pipeline[ml-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[ml-extras]' --force` |
| PyTorch (LSTM 등) | `pip install 'modern-ml-pipeline[torch-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[torch-extras]' --force` |
| 전체 기능 | `pip install 'modern-ml-pipeline[all]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[all]' --force` |

상세 설치 옵션은 [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md)를 참고하세요.


### 2. 프로젝트 생성

```bash
mmp init my-project
cd my-project
```

생성되는 디렉토리 구조:

```text
my-project/
├── configs/             # 환경별 설정 (dev.yaml, prod.yaml)
├── recipes/             # 실험 레시피
├── data/                # 데이터 파일 (CSV, SQL)
├── Dockerfile           # 프로덕션 배포용 (학습/추론/서빙 통합)
├── docker-compose.yml   # 로컬 실행 및 테스트
└── ...
```

> **Tip**: `Dockerfile`과 `docker-compose.yml`이 기본 포함되어, 로컬 개발부터 클라우드 배포까지 동일한 환경에서 실행할 수 있습니다.


### 3. 학습 전 준비

#### 3-1. 데이터 준비

학습 데이터를 `data/` 디렉토리에 CSV 또는 SQL 파일로 준비합니다.

> Task별 데이터 형식은 [Task 가이드](./docs/user/TASK_GUIDE.md)를 참고하세요.

---

#### 3-2. Config 파일 생성 (인프라 설정)

```bash
mmp get-config
```

대화형 인터페이스를 통해 MLflow, 스토리지, DB 연결 등을 설정하고 `configs/{env}.yaml` 파일을 생성합니다.

---

#### 3-3. Recipe 파일 생성 (실험 설정)

```bash
mmp get-recipe
```

Task, 모델, 전처리 등을 선택하고 `recipes/{name}.yaml` 파일을 생성합니다. 생성된 파일에서 **데이터 컬럼 정보만 수정**하면 됩니다:

```yaml
# recipes/my-recipe.yaml
task_choice: classification

data:
  data_interface:
    entity_columns: [user_id]      # [필수] ID 컬럼
    target_column: is_fraud        # [필수] 예측 대상

model:
  class_path: xgboost.XGBClassifier
```

> 상세 옵션은 [Task 가이드](./docs/user/TASK_GUIDE.md), [설정 스키마](./docs/user/SETTINGS_SCHEMA.md)를 참고하세요.

---

#### 3-4. 환경변수 (.env)

`mmp get-config` 실행 시 `.env.{env_name}.template` 파일이 함께 생성됩니다. 이 템플릿을 복사하여 실제 값을 입력합니다:

```bash
# 템플릿 파일을 복사하여 실제 환경변수 파일 생성
cp .env.local.template .env.local

# .env.local 파일 편집하여 값 입력
```

```bash
# .env.local 예시
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

CLI 명령어(`train`, `batch-inference`, `serve-api`)는 Config 파일명에서 환경 이름을 추출하여 대응되는 `.env.{env_name}` 파일을 자동으로 로드합니다:

| Config 파일 | 자동 로드되는 .env 파일 |
|-------------|------------------------|
| `configs/local.yaml` | `.env.local` |
| `configs/dev.yaml` | `.env.dev` |
| `configs/prod.yaml` | `.env.prod` |

> **Note**: 모델 서빙/추론 시 `run_id`는 CLI `--run-id` 인자로 전달합니다. Docker 실행 시에는 `MODEL_RUN_ID` 환경변수를 사용합니다.


### 4. 학습 및 실험

Recipe 파일을 여러 개 만들어 다양한 모델과 하이퍼파라미터를 실험할 수 있습니다. MLflow가 설정되어 있다면 모든 실험 결과가 자동으로 기록되어 `mlflow ui` 명령어로 성능 비교가 가능합니다. 상세 설정은 [MLflow 가이드](./docs/user/MLFLOW_GUIDE.md)를 참고하세요.

#### 로컬 실행

```bash
# CSV 파일로 학습
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml -d data/train.csv

# SQL 파일로 학습 (BigQuery/PostgreSQL 등)
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml -d data/query.sql

# SQL에 Jinja 변수 사용 시
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml -d data/query.sql -p '{"start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

#### Docker 실행

```bash
# 이미지 빌드 (최초 1회)
docker build -t my-model:latest .

# Docker로 학습 실행
docker-compose run --rm train
```

학습 완료 시 출력되는 `run_id`를 추론/서빙에 사용합니다.


### 5. 추론

#### 배치 추론

대량의 데이터를 한 번에 예측합니다.

```bash
mmp batch-inference -c configs/dev.yaml --run-id <run_id> -d data/test.csv

# 결과 파일 경로 직접 지정 (날짜별 디렉토리, CSV/JSON 포맷 지원)
mmp batch-inference --run-id <run_id> -o gs://bucket/2025/01/09/result.parquet
```

#### 실시간 API 서빙

REST API 서버를 기동하여 실시간 예측 요청을 처리합니다.

```bash
# 기본 포트 8000
mmp serve-api -c configs/dev.yaml --run-id <run_id>

# 포트/호스트 지정
mmp serve-api -c configs/dev.yaml --run-id <run_id> --host 0.0.0.0 --port 8080
```

```bash
# API 호출
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.5, "feature_2": 100}'
```

API 엔드포인트 상세는 [API 서빙 가이드](./docs/user/API_SERVING_GUIDE.md)를 참고하세요.


### 6. 배포 및 운영

#### 배포 흐름

```text
mmp init → 실험 → docker build → (( CI/CD로 GCR/ECR 푸시 → k8s에서 실행 ))
```

```bash
# 이미지 빌드 및 푸시 (CI/CD에서 자동화)
docker build -t gcr.io/my-project/mmp:v1 .
docker push gcr.io/my-project/mmp:v1
```

> **MMP의 범위**: 프로젝트 생성 → 실험 → 이미지 빌드
>
> **MMP 범위 외**: CI/CD, k8s 매니페스트, ConfigMap은 각 조직에서 별도 구성

상세 가이드는 [배포 및 운영 가이드](./docs/user/DEPLOYMENT_GUIDE.md)를 참고하세요.


## 지원 Task

| Task | 설명 | 활용 사례 |
|------|------|----------|
| Classification | 범주형 분류 | 사기 탐지, 이탈 예측 |
| Regression | 연속값 예측 | 집값 예측, 매출 예측 |
| Timeseries | 시계열 예측 | 일별 매출, 트래픽 예측 |
| Clustering | 비지도 군집화 | 고객 세분화 |
| Causal | 인과 추론 | 프로모션 효과 분석 |

각 Task별 데이터 형식과 모델 설정은 [Task 가이드](./docs/user/TASK_GUIDE.md)를 참고하세요.


## 지원 모델

| 라이브러리 | 모델 |
|------------|------|
| Scikit-learn | RandomForest, LogisticRegression, KMeans 등 |
| XGBoost | XGBClassifier, XGBRegressor |
| LightGBM | LGBMClassifier, LGBMRegressor |
| CatBoost | CatBoostClassifier, CatBoostRegressor |
| PyTorch | LSTM, TabNet |
| statsmodels | ARIMA, ExponentialSmoothing |
| CausalML | T-Learner, S-Learner |

```bash
mmp list models   # 사용 가능한 모델 목록
mmp list metrics  # 사용 가능한 메트릭 목록
```


## 문서

### 사용자 문서

| 순서 | 문서 | 설명 |
|------|------|------|
| 1 | [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md) | 설치, DB 연결, Cloud 설정 |
| 2 | [Task 가이드](./docs/user/TASK_GUIDE.md) | Task별 데이터 형식, 모델, Recipe 설정 |
| 3 | [설정 스키마](./docs/user/SETTINGS_SCHEMA.md) | Config/Recipe YAML 작성법 |
| 4 | [CLI 레퍼런스](./docs/user/CLI_REFERENCE.md) | 명령어 상세 옵션 |
| 5 | [MLflow 가이드](./docs/user/MLFLOW_GUIDE.md) | 실험 추적, UI 설정 |
| 6 | [API 서빙 가이드](./docs/user/API_SERVING_GUIDE.md) | REST API 서버 사용법 |
| 7 | [배포 및 운영 가이드](./docs/user/DEPLOYMENT_GUIDE.md) | 이미지 빌드, CI/CD, 운영 설정 |
| 8 | [전처리 레퍼런스](./docs/user/PREPROCESSOR_REFERENCE.md) | 전처리 상세 (선택) |
| 9 | [로컬 개발 환경](./docs/user/LOCAL_DEV_ENVIRONMENT.md) | Docker 기반 로컬 개발 (선택) |

### 개발자 문서

시스템 확장이나 기여를 원하시면 [개발자 문서](./docs/developer/)를 참고하세요.


## 도움말

```bash
mmp --help              # 전체 명령어 도움말
mmp train --help        # 특정 명령어 사용법
```

---

**Version**: 1.1.26 | **License**: Apache 2.0 | **Python**: 3.10 - 3.13
