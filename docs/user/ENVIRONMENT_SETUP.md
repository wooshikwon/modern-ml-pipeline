# 환경 설정 가이드 (Environment Setup)

Modern ML Pipeline을 사용하기 위한 설치와 환경 설정 방법을 안내합니다.

---

## 1. 설치하기

### 기본 요구사항

- **Python 3.11 또는 3.12**
- **pipx** (권장)

### 설치 전 준비

```bash
# 1. Python 3.11 설치
brew install python@3.11              # macOS (Homebrew)
# sudo apt install python3.11         # Ubuntu/Debian
# pyenv install 3.11.10               # pyenv 사용 시

# 2. pipx 설치
brew install pipx && pipx ensurepath  # macOS
pip install pipx && pipx ensurepath   # Linux/Windows
```

### 패키지 설치

**기본 설치 (pipx 권장)**

```bash
# Homebrew Python 사용 시
pipx install --python python3.11 modern-ml-pipeline

# pyenv 사용 시
pyenv shell 3.11.10 && pipx install modern-ml-pipeline
# 또는: pipx install --python ~/.pyenv/versions/3.11.10/bin/python modern-ml-pipeline
```

**Extras와 함께 설치**

```bash
# XGBoost, LightGBM, CatBoost 모델 사용 시
pipx install --python python3.11 "modern-ml-pipeline[ml-extras]"

# BigQuery, S3, GCS 사용 시
pipx install --python python3.11 "modern-ml-pipeline[cloud-extras]"

# 딥러닝 모델(LSTM, TabNet 등) 사용 시
pipx install --python python3.11 "modern-ml-pipeline[torch-extras]"

# 모든 기능 사용 시 (권장)
pipx install --python python3.11 "modern-ml-pipeline[all]"
```

**Extras 변경 (재설치)**

설치 후 extras를 변경하려면 `--force` 옵션으로 재설치합니다:

```bash
pipx install --force --python python3.11 "modern-ml-pipeline[ml-extras,cloud-extras]"
```

**pip 설치 (대체 방법)**

기존 환경에 직접 설치하려면 pip을 사용합니다:

```bash
pip install modern-ml-pipeline
pip install "modern-ml-pipeline[all]"
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
| `ml-extras` | LightGBM, CatBoost 모델 사용 시 | `lightgbm`, `catboost` |
| `torch-extras` | 딥러닝 모델(LSTM, TabNet 등) 사용 시 | `torch` |
| `cloud-extras` | BigQuery, S3, GCS 사용 시 | `sqlalchemy-bigquery`, `gcsfs`, `s3fs` |
| `feature-store` | Feast Feature Store 연동 시 | `feast` |
| `all` | 모든 기능 사용 시 | 위 전체 포함 |

---

## 2. Config 파일 설정

MMP는 **Config 파일에 직접 값을 설정**하는 방식을 권장합니다. 민감한 정보(인증서 경로 등)만 환경 변수로 분리합니다.

### 권장 패턴

```yaml
# configs/dev.yaml - 권장 방식: 직접 값 지정
mlflow:
  tracking_uri: ./mlruns              # 직접 값
  experiment_name: mmp-dev            # 직접 값

data_source:
  adapter_type: sql
  config:
    project_id: my-gcp-project        # 직접 값
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"  # 인증서만 환경변수
```

### 환경 변수가 필요한 경우

인증 정보처럼 Git에 커밋하면 안 되는 값만 환경 변수로 처리합니다:

```bash
# 필수 환경 변수 (민감 정보)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
```

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
