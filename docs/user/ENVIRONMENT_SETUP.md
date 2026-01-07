# 환경 설정 가이드 (Environment Setup)

Modern ML Pipeline을 사용하기 위한 설치와 환경 설정 방법을 안내합니다.

---

## 1. 설치하기

### 기본 요구사항

- **Python 3.11 또는 3.12**
- **pipx** (CLI 도구 설치용)

> pipx가 없다면: `brew install pipx && pipx ensurepath` (macOS) 또는 `pip install pipx && pipx ensurepath`

### 패키지 설치

**기본 설치**

```bash
# Scikit-learn 모델만 사용 시
pipx install git+https://github.com/wooshikwon/modern-ml-pipeline.git
```

**Extras와 함께 설치**

```bash
# XGBoost, LightGBM 등 추가 모델 사용 시
pipx install "git+https://github.com/wooshikwon/modern-ml-pipeline.git#egg=modern-ml-pipeline[ml-extras]"

# 클라우드 스토리지(BigQuery, S3, GCS) 사용 시
pipx install "git+https://github.com/wooshikwon/modern-ml-pipeline.git#egg=modern-ml-pipeline[cloud-extras]"

# 모든 기능 사용 시
pipx install "git+https://github.com/wooshikwon/modern-ml-pipeline.git#egg=modern-ml-pipeline[all]"
```

**기존 설치에 Extras 추가**

이미 설치된 환경에 extras를 추가하려면 `--force` 옵션으로 재설치합니다:

```bash
pipx install --force "git+https://github.com/wooshikwon/modern-ml-pipeline.git#egg=modern-ml-pipeline[ml-extras,cloud-extras]"
```

**개발 환경 설치 (소스 코드)**

```bash
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline
uv sync --all-extras  # 전체 의존성 설치
```

### 추가 패키지 (Extras) 선택 가이드

| Extras 이름 | 언제 필요한가요? | 포함된 주요 라이브러리 |
|-------------|-------------------|----------------------|
| `ml-extras` | XGBoost, LightGBM, CatBoost 모델 사용 시 | `xgboost`, `lightgbm`, `catboost` |
| `torch-extras` | 딥러닝 모델(FT-Transformer, LSTM 등) 사용 시 | `torch`, `rtdl`, `pytorch-tabnet` |
| `cloud-extras` | BigQuery, AWS S3, GCS 데이터 사용 시 | `google-cloud-bigquery`, `boto3`, `s3fs` |
| `feature-store` | Feast Feature Store 연동 시 | `feast`, `redis` |
| `all` | 모든 기능 사용 시 | 위 전체 포함 |

---

## 2. 주요 연결 설정

### 환경 변수 (.env) 설정

프로젝트 루트에 `.env` 파일을 만들면 민감한 정보를 안전하게 관리할 수 있습니다.

```bash
# .env 파일 예시

# MLflow 연결 정보
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=my-project

# 데이터베이스 연결 정보
DATABASE_URI=postgresql://user:password@localhost:5432/mydb

# GCP 인증 (BigQuery/GCS 사용 시)
GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
GCP_PROJECT_ID=my-gcp-project

# AWS 인증 (S3 사용 시)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

Config 파일에서는 `${ENV_VAR_NAME}` 형태로 환경 변수를 참조합니다.

---

### MLflow 설정

실험 추적 및 모델 저장을 위한 MLflow 연결 설정입니다.

**Config 예시 (configs/dev.yaml):**

```yaml
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  experiment_name: "my-experiment"
```

**로컬 파일 저장 (MLflow 서버 없이):**

```yaml
mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "local-experiment"
```

---

### 데이터베이스 (SQL) 설정

SQL 기반 데이터를 학습에 사용할 때 필요합니다.

#### PostgreSQL

```yaml
data_source:
  adapter_type: "sql"
  config:
    connection_uri: "${DATABASE_URI}"
    # 형식: postgresql://USER:PASSWORD@HOST:PORT/DB_NAME
```

#### BigQuery

`cloud-extras` 설치와 GCP 서비스 계정 인증이 필요합니다.

```yaml
data_source:
  adapter_type: "bigquery"
  config:
    connection_uri: "bigquery://${GCP_PROJECT_ID}"
    project_id: "${GCP_PROJECT_ID}"
    dataset_id: "my_dataset"
    location: "US"
    use_pandas_gbq: true
    query_timeout: 300
```

**인증 설정:**

```bash
# 서비스 계정 JSON 파일 경로 설정
export GOOGLE_APPLICATION_CREDENTIALS=./secrets/service-account.json
```

---

### 클라우드 스토리지 설정

모델 아티팩트나 대용량 데이터를 S3/GCS에 저장할 때 사용합니다.

#### AWS S3

```yaml
data_source:
  adapter_type: "storage"
  config:
    # base_path는 프로젝트 루트 레벨까지만 지정
    base_path: "s3://my-bucket/my-project/"
    storage_options:
      aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
      aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
      region_name: "us-east-1"
```

#### Google Cloud Storage (GCS)

```yaml
data_source:
  adapter_type: "storage"
  config:
    # base_path는 프로젝트 루트 레벨까지만 지정
    base_path: "gs://my-bucket/my-project/"
    storage_options:
      project: "${GCP_PROJECT_ID}"
      token: "${GOOGLE_APPLICATION_CREDENTIALS}"
```

#### 클라우드 스토리지 경로 사용

`base_path`를 프로젝트 루트로 설정하면 **로컬과 동일한 CLI 경로**를 사용할 수 있습니다:

```bash
# 로컬과 클라우드 모두 동일한 경로 지정
mmp train ... -d data/train.csv
mmp train ... -d sql/query.sql

# Config에 따라 자동 변환:
# - 로컬: data/train.csv
# - S3:   s3://my-bucket/my-project/data/train.csv
# - GCS:  gs://my-bucket/my-project/data/train.csv
```

**경로 변환 규칙:**

| 환경 | Config base_path | CLI -d 옵션 | 최종 경로 |
|------|------------------|-------------|----------|
| 로컬 | (없음) | `data/train.csv` | `data/train.csv` |
| S3 | `s3://bucket/project/` | `data/train.csv` | `s3://bucket/project/data/train.csv` |
| GCS | `gs://bucket/project/` | `data/train.csv` | `gs://bucket/project/data/train.csv` |

전체 URL 직접 지정도 가능합니다:

```bash
mmp train ... -d s3://other-bucket/other-path/train.csv
```

---

## 3. Feature Store 설정 (선택)

Feature Store(Feast)는 **선택 사항**입니다. 다음 경우에만 필요합니다:

| 사용 시나리오 | Feature Store 필요 여부 |
|--------------|------------------------|
| CSV/SQL에서 직접 데이터 로드 | 불필요 |
| 학습/추론에 동일한 피처 사용 | 불필요 |
| 실시간 서빙에서 피처 자동 조회 | **필요** |
| Point-in-Time Join으로 Data Leakage 방지 | **필요** |

### Feast 구성 요소

| 구성 요소 | 역할 | 사용 시점 |
|----------|------|----------|
| **Offline Store** | 과거 피처 저장 (PostgreSQL, BigQuery, File) | 학습, 배치 추론 |
| **Online Store** | 최신 피처 저장 (Redis, DynamoDB, SQLite) | 실시간 서빙 |
| **Registry** | 피처 메타데이터 관리 | 항상 |

### Config 예시 (Feast 사용 시)

```yaml
feature_store:
  provider: "feast"
  feast_config:
    project: "my_feature_store"
    registry: "./feast/data/registry.db"
    provider: "local"

    # Offline Store: 학습/배치 추론용
    offline_store:
      type: "postgres"
      host: "localhost"
      port: 5432
      database: "features"
      user: "mluser"
      password: "${DB_PASSWORD}"

    # Online Store: 실시간 서빙용
    online_store:
      type: "redis"
      connection_string: "redis://localhost:6379"
```

### Recipe에서 Fetcher 설정

Feature Store를 사용하려면 Recipe에 fetcher를 설정합니다:

```yaml
data:
  loader:
    source_uri: "sql/transactions.sql"

  data_interface:
    entity_columns: [user_id]
    target_column: is_fraud
    timestamp_column: event_time

  # Feature Store fetcher 설정
  fetcher:
    timestamp_column: event_time
    feature_views:
      user_features:
        join_key: user_id
        features: [avg_amount, transaction_count_7d, age]
      merchant_features:
        join_key: merchant_id
        features: [avg_fraud_rate, category]
```

### Fetcher 없이 사용 (기본)

Feature Store가 필요 없으면 fetcher 설정을 생략합니다. 자동으로 `pass_through` fetcher가 사용됩니다:

```yaml
data:
  loader:
    source_uri: "data/train.csv"

  data_interface:
    entity_columns: [user_id]
    target_column: is_fraud
    # fetcher 설정 생략 → 피처 증강 없음
```

---

## 4. 설정 검증하기

모든 설정이 올바른지 확인하려면 `system-check` 명령어를 사용하세요.

```bash
# Config 파일의 연결 정보 검증
mmp system-check --config configs/dev.yaml

# Recipe 포함 전체 검증
mmp system-check --config configs/dev.yaml --recipe recipes/my-model.yaml

# 해결 방법 함께 출력
mmp system-check --config configs/dev.yaml --actionable
```

**성공 시 출력 예시:**

```text
시스템 연결 검사 결과:
  PackageDependencies: 패키지 설치 완료
  MLflow: 연결됨 (http://localhost:5000)
  Database: 연결됨
```

---

## 5. 연결 테스트 (Docker)

로컬에서 MLflow, PostgreSQL, Redis 등을 테스트하려면 Docker를 사용할 수 있습니다.

### mmp-local-dev 사용하기

별도로 제공되는 개발 환경 레포지토리를 사용하면 편리합니다.

```bash
# 레포지토리 다운로드
git clone https://github.com/wooshikwon/mmp-local-dev.git
cd mmp-local-dev

# 설정 파일 준비
cp .env.example .env
# .env 파일에서 POSTGRES_PASSWORD 설정

# 실행
docker-compose up -d
```

**제공되는 서비스:**

| 서비스 | 포트 | 용도 |
|--------|------|------|
| MLflow | 5000 | 실험 추적 UI |
| PostgreSQL | 5432 | Offline Store / 데이터 저장소 |
| Redis | 6379 | Online Store (실시간 서빙용) |

**연결 확인:**

```bash
# MLflow
curl http://localhost:5000/health

# PostgreSQL
PGPASSWORD=<password> psql -h localhost -p 5432 -U mluser -d mlpipeline -c "SELECT 1;"

# Redis
redis-cli -h localhost -p 6379 ping
```

상세 설정은 [로컬 개발 환경](./LOCAL_DEV_ENVIRONMENT.md)을 참고하세요.
