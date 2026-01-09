# 환경 설정 가이드 (Environment Setup)

Modern ML Pipeline을 사용하기 위한 설치와 환경 설정 방법을 안내합니다.

---

## 1. 설치하기

### 요구사항

- **Python 3.10, 3.11, 3.12, 또는 3.13**

### 설치 전 준비

```bash
# Python 3.10+ 설치
brew install python@3.10              # macOS (Homebrew)
sudo apt install python3.10           # Ubuntu/Debian
pyenv install 3.10.14                 # pyenv 사용 시

# pipx 설치 (CLI 전역 설치 도구, 권장)
brew install pipx && pipx ensurepath  # macOS (Homebrew)
pip install pipx && pipx ensurepath   # Linux/Windows
```

---

### 기본 설치

MMP 코어 기능(XGBoost, scikit-learn, Optuna, SHAP 등)만 설치합니다.

```bash
# pip
pip install modern-ml-pipeline

# pipx (권장)
pipx install modern-ml-pipeline
```

> **Note**: pipx는 CLI 도구를 격리된 환경에 전역 설치합니다. Python 버전을 명시하려면:
> ```bash
> pipx install --python python3.10 modern-ml-pipeline
> ```

---

### 시나리오별 추가 설치 (Extras)

기본 설치 후 필요한 extras를 추가합니다.

| 시나리오 | pip | pipx inject |
|----------|-----|-------------|
| **LightGBM, CatBoost** | `pip install 'modern-ml-pipeline[ml-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[ml-extras]' --force` |
| **BigQuery, GCS, S3** | `pip install 'modern-ml-pipeline[cloud-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[cloud-extras]' --force` |
| **PyTorch (LSTM, TabNet)** | `pip install 'modern-ml-pipeline[torch-extras]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[torch-extras]' --force` |
| **Feast Feature Store** | `pip install 'modern-ml-pipeline[feature-store]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[feature-store]' --force` |
| **전체 기능** | `pip install 'modern-ml-pipeline[all]'` | `pipx inject modern-ml-pipeline 'modern-ml-pipeline[all]' --force` |

**여러 extras 동시 추가:**

```bash
# pip
pip install 'modern-ml-pipeline[ml-extras,cloud-extras]'

# pipx inject
pipx inject modern-ml-pipeline 'modern-ml-pipeline[ml-extras,cloud-extras]' --force
```

---

### Extras와 함께 처음부터 설치

처음 설치할 때 extras를 함께 지정할 수도 있습니다.

```bash
# pip
pip install 'modern-ml-pipeline[all]'

# pipx
pipx install 'modern-ml-pipeline[all]'
pipx install --python python3.10 'modern-ml-pipeline[ml-extras,cloud-extras]'
```

---

### 개발 환경 설치 (소스 코드)

MMP를 직접 개발하거나 수정하려면 소스 코드에서 설치합니다.

```bash
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline
uv sync --all-extras  # 전체 의존성 설치
```

---

### Extras 상세 가이드

| Extras 이름 | 언제 필요한가요? | 포함된 주요 라이브러리 |
|-------------|-----------------|----------------------|
| `ml-extras` | LightGBM, CatBoost 모델 사용 시 | `lightgbm`, `catboost` |
| `torch-extras` | 딥러닝 모델 (LSTM, TabNet, FT-Transformer) 사용 시 | `torch`, `pytorch-tabnet`, `rtdl-revisiting-models` |
| `cloud-extras` | BigQuery, S3, GCS 연동 시 | `sqlalchemy-bigquery`, `gcsfs`, `s3fs` |
| `feature-store` | Feast Feature Store 연동 시 | `feast` |
| `causal` | CausalML 인과추론 모델 사용 시 | `causalml` |
| `all` | 모든 기능 사용 시 | 위 전체 포함 |

**모델별 필요 extras:**

| 모델 | 필요한 extras |
|------|--------------|
| XGBoost, RandomForest, LogisticRegression | (기본 설치에 포함) |
| LightGBM, CatBoost | `ml-extras` |
| LSTM, TabNet, FT-Transformer | `torch-extras` |
| ARIMA, ExponentialSmoothing | (기본 설치에 포함) |
| T-Learner, S-Learner | `causal` |

---

### 설치 확인

```bash
# 버전 확인
mmp --version

# 사용 가능한 모델 목록
mmp list models

# 시스템 점검 (의존성 확인)
mmp system-check -c configs/dev.yaml --actionable
```

---

## 2. Config 파일과 환경변수 설정

MMP는 **Config 파일에 직접 값을 설정**하는 방식을 권장합니다. 민감한 정보(인증서 경로, 비밀번호 등)만 환경 변수로 분리합니다.

### 2-1. Config와 .env 파일 생성

`mmp get-config` 명령어를 실행하면 두 가지 파일이 생성됩니다:

```bash
mmp get-config
# 대화형 인터페이스로 환경 설정...
```

| 생성 파일 | 설명 |
|----------|------|
| `configs/{env_name}.yaml` | 인프라 설정 (MLflow, DB, Storage 등) |
| `.env.{env_name}.template` | 환경변수 템플릿 (민감 정보용) |

### 2-2. 환경변수 파일 활성화

생성된 템플릿을 복사하여 실제 값을 입력합니다:

```bash
# 템플릿을 실제 환경변수 파일로 복사
cp .env.local.template .env.local

# 파일 편집하여 실제 값 입력
vi .env.local
```

**템플릿 예시 (.env.local.template):**

```bash
# BigQuery/GCS 인증
GOOGLE_APPLICATION_CREDENTIALS=

# AWS S3 인증
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# PostgreSQL 인증
DB_USER=
DB_PASSWORD=
```

**입력 후 (.env.local):**

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
DB_USER=mluser
DB_PASSWORD=secretpassword
```

### 2-3. 자동 로딩 메커니즘

CLI 명령어(`train`, `batch-inference`, `serve-api`, `system-check`)는 **Config 파일명에서 환경 이름을 추출**하여 대응되는 `.env.{env_name}` 파일을 자동으로 로드합니다:

| Config 파일 | 자동 로드되는 .env 파일 |
|-------------|------------------------|
| `configs/local.yaml` | `.env.local` |
| `configs/dev.yaml` | `.env.dev` |
| `configs/prod.yaml` | `.env.prod` |
| `configs/staging.yaml` | `.env.staging` |

**예시:**

```bash
# configs/dev.yaml 사용 시 → .env.dev 자동 로드
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv

# configs/prod.yaml 사용 시 → .env.prod 자동 로드
mmp batch-inference -c configs/prod.yaml --run-id abc123 -d data/test.csv
```

### 2-4. Config에서 환경변수 참조

Config YAML 파일에서 환경변수는 `${VAR_NAME}` 또는 `${VAR_NAME:기본값}` 문법으로 참조합니다:

```yaml
# configs/dev.yaml
data_source:
  adapter_type: sql
  config:
    # 환경변수 참조 (기본값 없음)
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"

    # 환경변수 참조 (기본값 있음)
    connection_uri: "postgresql://${DB_USER:postgres}:${DB_PASSWORD:}@localhost:5432/mydb"
```

이 설정은 `.env.dev` 파일의 값으로 대체됩니다:

```bash
# .env.dev
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
DB_USER=mluser
DB_PASSWORD=secretpassword
```

### 2-5. 권장 패턴

```yaml
# configs/dev.yaml - 권장 방식
mlflow:
  tracking_uri: ./mlruns              # 직접 값 (민감하지 않음)
  experiment_name: mmp-dev            # 직접 값

data_source:
  adapter_type: sql
  config:
    project_id: my-gcp-project                              # 직접 값
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"   # 인증서만 환경변수
```

> **Tip**: `.env.*` 파일은 `.gitignore`에 추가하여 Git에 커밋되지 않도록 합니다. 템플릿 파일(`.env.*.template`)만 커밋하세요.

---

## 3. 환경별 Config 예시

### Dev 환경 (로컬 MLflow)

```yaml
# configs/dev.yaml
environment:
  name: dev

mlflow:
  tracking_uri: ./mlruns              # 로컬 파일 저장
  experiment_name: mmp-dev

data_source:
  name: BigQuery
  adapter_type: sql
  config:
    connection_uri: bigquery://my-project
    project_id: my-project
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
    location: US
    use_pandas_gbq: true

output:
  inference:
    enabled: true
    adapter_type: storage
    config:
      base_path: gs://my-bucket/dev/predictions
```

### Staging/Prod 환경 (원격 MLflow 서버)

```yaml
# configs/staging.yaml
environment:
  name: staging

mlflow:
  tracking_uri: https://mlflow.example.com   # 원격 MLflow 서버
  experiment_name: mmp-staging

data_source:
  name: BigQuery
  adapter_type: sql
  config:
    connection_uri: bigquery://my-project
    project_id: my-project
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
    location: US
    use_pandas_gbq: true

output:
  inference:
    enabled: true
    adapter_type: storage
    config:
      base_path: gs://my-bucket/staging/predictions
```

---

## 4. 데이터 소스 설정

### PostgreSQL

```yaml
data_source:
  adapter_type: sql
  config:
    connection_uri: postgresql://user:password@localhost:5432/mydb
```

### BigQuery

`cloud-extras` 설치와 GCP 서비스 계정 인증이 필요합니다.

```yaml
data_source:
  name: BigQuery
  adapter_type: sql
  config:
    connection_uri: bigquery://my-project
    project_id: my-project
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
    location: US
    use_pandas_gbq: true
```

**인증 설정:**

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## 5. 추론 결과 저장 설정

배치 추론 결과를 저장할 위치를 설정합니다.

### GCS (Google Cloud Storage)

`cloud-extras` 설치가 필요합니다.

```yaml
output:
  inference:
    enabled: true
    adapter_type: storage
    config:
      base_path: gs://my-bucket/predictions
```

### S3 (AWS)

`cloud-extras` 설치가 필요합니다.

```yaml
output:
  inference:
    enabled: true
    adapter_type: storage
    config:
      base_path: s3://my-bucket/predictions
```

### 로컬 파일

```yaml
output:
  inference:
    enabled: true
    adapter_type: storage
    config:
      base_path: ./artifacts/predictions
```

### BigQuery 테이블

```yaml
output:
  inference:
    enabled: true
    adapter_type: sql
    config:
      connection_uri: bigquery://my-project
      table_name: predictions
      write_disposition: WRITE_APPEND
```

---

## 6. Feature Store 설정 (선택)

Feature Store(Feast)는 **선택 사항**입니다. 다음 경우에만 필요합니다:

| 사용 시나리오 | Feature Store 필요 여부 |
|--------------|------------------------|
| CSV/SQL에서 직접 데이터 로드 | 불필요 |
| 학습/추론에 동일한 피처 사용 | 불필요 |
| 실시간 서빙에서 피처 자동 조회 | **필요** |
| Point-in-Time Join으로 Data Leakage 방지 | **필요** |

### Config 예시 (Feast 사용 시)

```yaml
feature_store:
  provider: feast
  feast_config:
    project: my_feature_store
    registry: ./feast/data/registry.db
    provider: local

    offline_store:
      type: postgres
      host: localhost
      port: 5432
      database: features
      user: mluser
      password: "${DB_PASSWORD}"

    online_store:
      type: redis
      connection_string: redis://localhost:6379
```

---

## 7. 설정 검증하기

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
  MLflow: 연결됨 (./mlruns)
  Database: 연결됨
```

---

## 8. 연결 테스트 (Docker)

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
