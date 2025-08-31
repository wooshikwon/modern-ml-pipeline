# Modern ML Pipeline CLI 사용자 가이드

## 🚀 빠른 시작

Modern ML Pipeline은 환경별 설정을 분리하여 개발/스테이징/프로덕션 환경을 쉽게 전환할 수 있는 ML 파이프라인 도구입니다.

### 설치

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 및 설정
git clone <repository-url>
cd modern-ml-pipeline
uv sync
```

### 1분 퀵스타트

```bash
# 1. 프로젝트 초기화
mmp init --project-name my-ml-project
cd my-ml-project

# 2. 개발 환경 설정
mmp get-config --env-name dev

# 3. 환경변수 설정
cp .env.dev.template .env.dev
# .env.dev 파일을 편집하여 실제 값 입력

# 4. 연결 테스트
mmp system-check --env-name dev

# 5. Recipe 생성 (대화형)
mmp get-recipe

# 6. 학습 실행
mmp train --recipe-file recipes/model.yaml --env-name dev
```

## 📋 주요 개념

### Recipe vs Config

- **Recipe**: 모델과 데이터 처리 로직을 정의 (환경 무관)
- **Config**: 환경별 연결 정보와 설정 (DB, MLflow, Storage 등)

```yaml
# Recipe: 논리적 구조만 정의
model:
  loader:
    adapter: "sql"  # 어댑터 이름만
    source_uri: "sql/query.sql"  # SQL 파일 경로만

# Config: 실제 연결 정보
data_adapters:
  adapters:
    sql:
      connection_uri: "${DB_CONNECTION_URI}"  # 실제 DB 연결
```

### 환경 분리

하나의 Recipe로 여러 환경에서 실행:

```bash
# 개발 환경
mmp train --recipe-file recipes/xgboost.yaml --env-name dev

# 스테이징 환경  
mmp train --recipe-file recipes/xgboost.yaml --env-name staging

# 프로덕션 환경
mmp train --recipe-file recipes/xgboost.yaml --env-name prod
```

## 🔧 상세 명령어

### mmp init

프로젝트 구조를 초기화합니다.

```bash
# 기본 초기화
mmp init --project-name my-project

# Docker 개발 환경 포함
mmp init --project-name my-project --with-mmp-dev
```

생성되는 구조:
```
my-project/
├── configs/       # 환경별 설정
├── recipes/       # ML 레시피
├── sql/          # SQL 쿼리
├── data/         # 데이터 디렉토리
└── .gitignore
```

### mmp get-config

환경별 설정 파일을 생성합니다.

```bash
# 대화형 모드 (권장)
mmp get-config --env-name dev

# 템플릿 사용 (빠른 설정)
mmp get-config --env-name prod --template prod --non-interactive
```

템플릿 옵션:
- `local`: 로컬 개발 (PostgreSQL + 로컬 스토리지)
- `dev`: 개발 서버 (PostgreSQL + MLflow)
- `prod`: 프로덕션 (BigQuery + GCS + MLflow)

### mmp system-check

시스템 연결 상태를 검사합니다.

```bash
# 기본 검사
mmp system-check --env-name dev

# 실행 가능한 해결책 제시
mmp system-check --env-name dev --actionable
```

검사 항목:
- ✅ MLflow 서버 연결
- ✅ PostgreSQL/BigQuery 연결
- ✅ Redis 연결 (Feature Store)
- ✅ 스토리지 접근

### mmp get-recipe

대화형으로 ML Recipe를 생성합니다.

```bash
mmp get-recipe
```

선택 옵션:
1. 모델 선택 (XGBoost, LightGBM, CatBoost 등)
2. 데이터 소스 (SQL, CSV, Parquet)
3. 특성 처리 파이프라인
4. 하이퍼파라미터 설정

### mmp train

학습 파이프라인을 실행합니다.

```bash
# 기본 실행
mmp train --recipe-file recipes/model.yaml --env-name dev

# 파라미터 전달
mmp train -r recipes/model.yaml -e prod --params '{"date": "2024-01-01"}'
```

### mmp batch-inference

배치 추론을 실행합니다.

```bash
mmp batch-inference --run-id <mlflow-run-id> --env-name prod
```

### mmp serve-api

모델 서빙 API를 실행합니다.

```bash
mmp serve-api --run-id <mlflow-run-id> --env-name dev --port 8080
```

## 🌍 환경 설정

### .env 파일 구조

각 환경별로 `.env.{env_name}` 파일이 필요합니다:

```bash
# .env.dev
ENV_NAME=dev
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=ml_dev

MLFLOW_TRACKING_URI=http://localhost:5002
GCP_PROJECT=my-project-dev
```

### 환경변수 치환

Config 파일에서 `${VAR:default}` 형식 사용:

```yaml
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:./mlruns}"
  
database:
  host: "${DB_HOST:localhost}"
  port: "${DB_PORT:5432}"
```

### 환경 전환

두 가지 방법으로 환경 전환 가능:

1. **명령어 파라미터** (권장)
```bash
mmp train --recipe-file recipes/model.yaml --env-name prod
```

2. **환경변수 설정**
```bash
export ENV_NAME=prod
mmp train --recipe-file recipes/model.yaml
```

## 🔍 문제 해결

### 일반적인 문제

#### .env 파일을 찾을 수 없음

```bash
# 템플릿에서 복사
cp .env.dev.template .env.dev

# 필수 값 편집
vim .env.dev
```

#### Config 파일을 찾을 수 없음

```bash
# Config 생성
mmp get-config --env-name dev
```

#### 데이터베이스 연결 실패

```bash
# 상세 진단
mmp system-check --env-name dev --actionable

# Docker 환경 시작 (mmp-local-dev 사용 시)
cd ../mmp-local-dev
docker-compose up -d postgres
```

#### MLflow 서버 연결 실패

```bash
# 로컬 MLflow 서버 시작
mlflow ui --host 0.0.0.0 --port 5002

# 또는 Docker 사용
cd ../mmp-local-dev
docker-compose up -d mlflow
```

### 디버깅 팁

1. **로그 레벨 조정**
```bash
export LOG_LEVEL=DEBUG
mmp train --recipe-file recipes/model.yaml --env-name dev
```

2. **설정 검증**
```python
# Python에서 직접 확인
from src.settings import load_settings_by_file
settings = load_settings_by_file("recipes/model.yaml", env_name="dev")
print(settings)
```

3. **환경변수 확인**
```bash
# 현재 환경변수 출력
env | grep -E "(DB_|MLFLOW_|GCP_)"
```

## 📚 예제

### 예제 1: XGBoost 분류 모델

```yaml
# recipes/xgboost_classifier.yaml
name: "customer_churn_prediction"
model:
  class_path: "xgboost.XGBClassifier"
  loader:
    adapter: "sql"
    source_uri: "sql/train_features.sql"
  data_interface:
    task_type: "classification"
    target_column: "churned"
  hyperparameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.3
```

실행:
```bash
mmp train --recipe-file recipes/xgboost_classifier.yaml --env-name dev
```

### 예제 2: 시계열 예측

```yaml
# recipes/timeseries_forecast.yaml
name: "sales_forecast"
model:
  class_path: "lightgbm.LGBMRegressor"
  loader:
    adapter: "sql"
    source_uri: "sql/timeseries_features.sql"
    entity_schema:
      entity_columns: ["store_id", "product_id"]
      timestamp_column: "date"
  data_interface:
    task_type: "regression"
    target_column: "sales"
```

### 예제 3: Feature Store 사용

```yaml
# configs/prod.yaml에 Feature Store 설정
feature_store:
  provider: "feast"
  feast_config:
    project: "ml_features"
    registry: "gs://my-bucket/feast/registry.pb"
    online_store:
      type: "redis"
      connection_string: "${REDIS_HOST:localhost}:${REDIS_PORT:6379}"
```

## 🔄 마이그레이션 가이드

### 기존 프로젝트에서 마이그레이션

1. **Config 분리**
```bash
# 기존 config를 환경별로 분리
cp config/config.yaml configs/local.yaml
```

2. **환경 설정 생성**
```bash
mmp get-config --env-name local
```

3. **Recipe 수정**
```yaml
# 환경 특정 정보 제거
# Before:
loader:
  connection_uri: "postgresql://user:pass@localhost/db"
  
# After:
loader:
  adapter: "sql"
  source_uri: "sql/query.sql"
```

4. **실행 명령어 수정**
```bash
# Before:
python train.py --config config/config.yaml

# After:
mmp train --recipe-file recipes/model.yaml --env-name local
```

## 🤝 기여하기

버그 리포트나 기능 제안은 GitHub Issues를 통해 제출해주세요.

## 📝 라이센스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.