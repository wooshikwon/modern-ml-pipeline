# �� Modern ML Pipeline

**차세대 MLOps 플랫폼 - 학습부터 배포까지 자동화된 머신러닝 파이프라인**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/dependency-uv-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 프로젝트 소개

Modern ML Pipeline은 **YAML 설정만으로 머신러닝 모델을 학습하고 배포**할 수 있는 통합 MLOps 플랫폼입니다.

### 🎯 핵심 특징

- **🔧 Zero-Code ML**: YAML 레시피만으로 모든 ML 모델 실험 가능
- **⚡ 자동 최적화**: Optuna 기반 하이퍼파라미터 자동 튜닝
- **🏗️ 완전한 재현성**: 동일한 결과 100% 보장
- **🌍 멀티 환경**: LOCAL → DEV → PROD 단계적 확장
- **🚀 즉시 배포**: 학습된 모델 바로 API 서빙

---

## 🚀 빠른 시작 (5분 설정)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline

# Python 환경 설정 (uv 권장)
uv venv && uv sync
# 또는 pip 사용시: pip install -r requirements.txt
```

### 2. 프로젝트 초기화

```bash
# 새 프로젝트 구조 생성
uv run python main.py init

# 생성된 파일 확인
ls config/    # base.yaml, data_adapters.yaml
ls recipes/   # example_recipe.yaml
```

### 3. 첫 번째 레시피 생성 (`guide` 명령어)

```bash
# sklearn의 RandomForestClassifier에 대한 레시피 템플릿을 생성합니다.
uv run python main.py guide sklearn.ensemble.RandomForestClassifier > recipes/my_first_model.yaml

# 생성된 파일을 열어 source_uri, target_column 등을 당신의 데이터에 맞게 수정하세요.
```

### 4. 모델 검증 및 학습

```bash
# 수정된 레시피 파일의 유효성을 검증합니다.
uv run python main.py validate --recipe-file recipes/my_first_model.yaml

# 모델 학습을 실행합니다.
uv run python main.py train --recipe-file recipes/my_first_model.yaml

# 학습 결과 확인 (MLflow UI - 자동 실행)
# 로컬 파일 모드: MLflow UI가 자동으로 백그라운드에서 실행됩니다
# 브라우저에서 자동으로 열리거나 콘솔 메시지의 URL로 접속하세요
# 수동 실행: mlflow ui --backend-store-uri ./mlruns
```

### 5. 모델 배포 및 추론

```bash
# 학습에서 나온 run-id 사용 (예: abc123def456)
RUN_ID="your-run-id-here"

# 배치 추론
uv run python main.py batch-inference --run-id $RUN_ID

# 실시간 API 서빙
uv run python main.py serve-api --run-id $RUN_ID
# API 테스트: curl http://localhost:8000/predict -X POST -H 'Content-Type: application/json' -d '{"user_id": 1, "event_ts": "2024-01-01T00:00:00"}'
```

### 6. Docker 이미지로 실행

```bash
# 이미지 빌드 (서빙용)
docker build -t mmp-api --target serve .

# 모델 서빙 (포트 8000 노출)
docker run --rm -p 8000:8000 mmp-api --run-id $RUN_ID

# 이미지 빌드 (학습용)
docker build -t mmp-train --target train .

# 학습 실행 (레시피 파일 경로 지정)
docker run --rm mmp-train --recipe-file recipes/recipe_example.yaml
```

---

## 📖 기본 사용법

### CLI 명령어 전체 목록

```bash
# 프로젝트 관리
uv run python main.py init [--dir ./my-project]     # 새 프로젝트 초기화
uv run python main.py validate --recipe-file <path> # 설정 파일 검증

# 레시피 가이드
uv run python main.py guide <model_class_path>       # 모델에 맞는 레시피 템플릿 생성

# 모델 개발
uv run python main.py train --recipe-file <path>    # 모델 학습
uv run python main.py train --recipe-file <path> --context-params '{"date":"2024-01-01"}'  # 동적 파라미터

# 모델 추론
uv run python main.py batch-inference --run-id <id> # 배치 추론
uv run python main.py serve-api --run-id <id>       # 실시간 API

# 시스템 검증
uv run python main.py test-contract                 # 인프라 연결 테스트
```

### Recipe 파일 작성법

Recipe는 모델의 모든 논리를 정의하는 YAML 파일입니다:

```yaml
# recipes/my_model.yaml
model:
  # 모델 클래스 (sklearn, xgboost, lightgbm 등 모든 Python 패키지)
  class_path: "sklearn.ensemble.RandomForestClassifier"
  
  # 하이퍼파라미터 (고정값 또는 최적화 범위)
  hyperparameters:
    n_estimators: 100              # 고정값
    max_depth: {type: "int", low: 3, high: 10}  # 자동 최적화 범위
  
  # 데이터 로딩
  loader:
    name: "default_loader"
    source_uri: "data/my_dataset.parquet"  # 파일 경로 또는 SQL
    adapter: storage
  
  # 데이터 전처리
  preprocessor:
    name: "default_preprocessor"
    params:
      exclude_cols: ["id", "timestamp"]
  
  # 모델 설정
  data_interface:
    task_type: "classification"    # classification, regression, causal
    target_col: "target"

# 자동 하이퍼파라미터 최적화 (선택사항)
hyperparameter_tuning:
  enabled: true
  n_trials: 50
  metric: "roc_auc"
  direction: "maximize"
```

---

## 🔧 고급 기능

### 1. 동적 SQL 템플릿 (Jinja2)

```sql
-- recipes/sql/dynamic_query.sql.j2
SELECT user_id, feature1, feature2, target
FROM my_table 
WHERE date = '{{ target_date }}'
LIMIT {{ limit | default(1000) }}
```

```bash
# 템플릿 파라미터와 함께 실행
uv run python main.py train \
  --recipe-file recipes/templated_model.yaml \
  --context-params '{"target_date": "2024-01-01", "limit": 5000}'
```

### 2. Feature Store 연동

```yaml
# recipes/feature_store_model.yaml
model:
  augmenter:
    type: "feature_store"
    features:
      - feature_namespace: "user_demographics"
        features: ["age", "country"]
      - feature_namespace: "user_behavior"
        features: ["click_rate", "conversion_rate"]
```

### 3. 환경별 설정 관리

```bash
# 환경 변수로 설정 전환
APP_ENV=local   uv run python main.py train ...  # 로컬 파일 기반
APP_ENV=dev     uv run python main.py train ...  # PostgreSQL + Redis  
APP_ENV=prod    uv run python main.py train ...  # BigQuery + Redis Labs
```

---

## 🌍 실행 모드별 가이드

### 🚀 독립 실행 모드 (기본값 - 권장)
**MLflow Graceful Degradation 적용 - 외부 서버 없이도 완전 동작**

```bash
# 즉시 실행 가능 - 외부 서버나 Docker 불필요  
uv run python main.py train --recipe-file recipes/example.yaml
```

- **MLflow**: 로컬 파일 (`./mlruns`) + 자동 UI 실행
- **Feature Store**: PassThrough 모드 (외부 의존성 없음)  
- **데이터**: 로컬 파일 (CSV, Parquet 등)
- **특징**: 설치 즉시 실행, 인터넷 연결 불필요, 빠른 실험

### 🔧 고급 기능 모드 (mmp-local-dev 연동)
**선택적 고급 기능 - Feature Store 및 공유 MLflow 서버 사용**

```bash
# 1. mmp-local-dev 설치 및 실행 (선택사항)
git clone https://github.com/wooshikwon/mmp-local-dev.git ../mmp-local-dev  
cd ../mmp-local-dev && docker-compose up -d

# 2. 환경 변경 후 실행
echo "APP_ENV=dev" > .env
echo "MLFLOW_TRACKING_URI=http://localhost:5002" >> .env
uv run python main.py train --recipe-file recipes/example.yaml
```

- **MLflow**: 공유 서버 (http://localhost:5002)
- **Feature Store**: Feast (PostgreSQL + Redis)
- **데이터**: 실시간 피처 조회, 팀 공유 실험  
- **특징**: 프로덕션 유사 환경, 팀 협업, Feature Store 테스트

### ☁️ 클라우드 연결 모드 (프로덕션)
```bash
# 환경변수로 클라우드 서버 연결
APP_ENV=prod MLFLOW_TRACKING_URI=https://your-mlflow-server.com \
uv run python main.py train --recipe-file recipes/prod_model.yaml
```

- **MLflow**: 클라우드 서버 (GCP/AWS/Azure)
- **Feature Store**: 프로덕션 Feast 클러스터
- **데이터**: BigQuery, Snowflake 등 대규모 DW
- **MLflow**: 클라우드 스토리지
- **특징**: 확장성, 안정성

---

## 📊 지원하는 ML 프레임워크

### 분류 (Classification)
```yaml
# scikit-learn
class_path: "sklearn.ensemble.RandomForestClassifier"
class_path: "sklearn.linear_model.LogisticRegression"

# XGBoost
class_path: "xgboost.XGBClassifier"

# LightGBM  
class_path: "lightgbm.LGBMClassifier"
```

### 회귀 (Regression)
```yaml
class_path: "sklearn.ensemble.RandomForestRegressor"
class_path: "sklearn.linear_model.LinearRegression"
class_path: "xgboost.XGBRegressor"
class_path: "lightgbm.LGBMRegressor"
```

### 인과추론 (Causal Inference)
```yaml
# CausalML
class_path: "causalml.inference.meta.XGBTRegressor"
class_path: "causalml.inference.meta.TRegressor"
```

---

## 🔐 환경변수 및 비밀 관리

Modern ML Pipeline은 **config YAML에는 연결 정보만, 실제 비밀은 환경변수로 주입**하는 보안 패턴을 사용합니다.

### 📋 기본 사용법

#### 1. .env 파일 설정
```bash
# 기본 .env 파일 생성
cp .env.example .env

# 필요한 환경변수 설정
cat >> .env << EOF
APP_ENV=local
MLFLOW_TRACKING_URI=http://localhost:5002
POSTGRES_PASSWORD=mysecretpassword
EOF
```

#### 2. Config YAML에서 환경변수 참조
```yaml
# config/prod.yaml
data_adapters:
  adapters:
    sql:
      connection_uri: "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:5432/${POSTGRES_DB}"

mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}

feature_store:
  feast_config:
    online_store:
      connection_string: ${REDIS_CONNECTION_STRING}
```

### 🌍 인프라별 설정 예시

#### ☁️ Google Cloud Platform
```bash
# 환경변수 설정
export GCP_PROJECT_ID="my-ml-project"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export BIGQUERY_DATASET="ml_pipeline_data"

# config에서 참조
# connection_uri: "bigquery://${GCP_PROJECT_ID}/${BIGQUERY_DATASET}"
```

#### ☁️ Amazon Web Services  
```bash
# 환경변수 설정 (권장: IAM Role 사용)
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
export S3_BUCKET="my-ml-data-bucket"

# config에서 참조
# storage_options:
#   key: ${AWS_ACCESS_KEY_ID}
#   secret: ${AWS_SECRET_ACCESS_KEY}
```

#### 🔵 Microsoft Azure
```bash
# 환경변수 설정
export AZURE_STORAGE_ACCOUNT="mystorageaccount"  
export AZURE_STORAGE_KEY="your-storage-key"
export AZURE_SQL_PASSWORD="your-password"

# config에서 참조
# storage_options:
#   account_name: ${AZURE_STORAGE_ACCOUNT}
#   account_key: ${AZURE_STORAGE_KEY}
```

### 🔒 보안 Best Practices

#### 1. 로컬 개발
```bash
# .env 파일 사용 (자동 로딩됨)
echo "POSTGRES_PASSWORD=dev-password" >> .env
echo "API_KEY=dev-api-key" >> .env
```

#### 2. 컨테이너/CI-CD
```bash
# 환경변수로 직접 주입
docker run -e POSTGRES_PASSWORD="$VAULT_PASSWORD" my-ml-app
export POSTGRES_PASSWORD="$(kubectl get secret db-secret -o jsonpath='{.data.password}' | base64 -d)"
```

#### 3. 클라우드 네이티브 (권장)
```bash
# 서비스 계정/IAM Role 자동 인증 (환경변수 불필요)
gcloud auth application-default login  # GCP
# EC2/EKS의 IAM Role 자동 사용        # AWS  
# Managed Identity 자동 사용           # Azure
```

### 🎯 환경변수 사용 패턴

**필수 변수** (기본값 없음):
```yaml
connection_uri: "postgresql://user:${POSTGRES_PASSWORD}@host/db"
```

**선택적 변수** (기본값 제공):
```yaml  
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI:./mlruns}  # 기본값: 로컬 파일
  experiment_name: ${EXPERIMENT_NAME:Default-Experiment}
```

### 📚 지원하는 모든 환경변수

전체 환경변수 목록과 설정 예시는 [.env.example](/.env.example) 파일을 참조하세요:

- **기본 설정**: `APP_ENV`, `LOG_LEVEL`
- **MLflow**: `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`
- **데이터베이스**: `POSTGRES_*`, `BIGQUERY_*`, `SNOWFLAKE_*`
- **클라우드 스토리지**: `GCS_*`, `S3_*`, `AZURE_*`
- **Feature Store**: `REDIS_*`, `FEAST_*`
- **보안**: `API_SECRET_KEY`, `JWT_SECRET_KEY`
- **모니터링**: `SENTRY_DSN`, `DATADOG_API_KEY`

---

## 🐛 트러블슈팅

### 자주 발생하는 문제

**1. MLflow 자동 전환 확인**
```bash
# Graceful Degradation 동작 확인 - 서버 없이도 정상 동작
curl http://localhost:5002/health
# 환경변수 확인
echo $MLFLOW_TRACKING_URI
```

**2. 데이터 파일을 찾을 수 없음**
```bash
# 현재 경로 확인
pwd
# 데이터 파일 경로 확인 (프로젝트 루트 기준)
ls data/my_dataset.parquet
```

**3. 패키지 의존성 오류**
```bash
# 필요한 패키지 추가 설치
uv add scikit-learn xgboost lightgbm
# 또는: pip install scikit-learn xgboost lightgbm
```

**4. Feature Store 연결 오류**
```bash
# Redis 연결 확인
redis-cli ping
# PostgreSQL 연결 확인  
psql -h localhost -p 5432 -U mlpipeline_user -d mlpipeline_db
```

### 로그 확인

```bash
# 상세 로그 출력
export LOG_LEVEL=DEBUG
uv run python main.py train --recipe-file recipes/my_model.yaml

# 로그 파일 위치
tail -f logs/modern_ml_pipeline.log
```

---

## 📚 추가 문서

- **[개발자 가이드](docs/DEVELOPER_GUIDE.md)**: 심화 사용법 및 커스터마이징
- **[인프라 가이드](docs/INFRASTRUCTURE_STACKS.md)**: 환경별 인프라 설정
- **[Blueprint](blueprint.md)**: 시스템의 핵심 설계 원칙과 실제 코드 구현을 연결한 기술 청사진

---

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📞 지원 및 문의

- **이슈 제보**: [GitHub Issues](https://github.com/wooshikwon/modern-ml-pipeline/issues)
- **문서**: [Wiki](https://github.com/wooshikwon/modern-ml-pipeline/wiki)
- **이메일**: [your-email@example.com](mailto:your-email@example.com)
