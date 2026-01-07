# API 서빙 가이드 (API Serving Guide)

학습된 모델을 REST API 서버로 배포하는 방법을 안내합니다.

---

## 1. API 서버 시작

```bash
mmp serve-api --run-id <mlflow_run_id> --config configs/dev.yaml
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | 서빙할 모델의 MLflow Run ID |
| `--config` | `-c` | O | Config 파일 경로 |
| `--host` | - | - | 바인딩 호스트 (기본: `0.0.0.0`) |
| `--port` | - | - | 바인딩 포트 (기본: `8000`) |

**출력 예시:**

```text
mmp v1.0.0

[1/3] Loading settings        done  run_id: abc123...
[2/3] Checking package deps   done  verified
[3/3] Starting API server     done

      API Server:   http://0.0.0.0:8000
      API Docs:     http://0.0.0.0:8000/docs
      Health Check: http://0.0.0.0:8000/health
```

---

## 2. API 입력 필드 이해하기

API 요청 시 어떤 필드를 제공해야 하는지 이해하는 것이 중요합니다.

### 입력 필드 결정 규칙

학습 시 사용한 데이터 소스에 따라 API 입력 요구사항이 달라집니다:

| 학습 시 설정 | API 입력 요구사항 |
|-------------|-------------------|
| **SQL/CSV에서 직접 로드한 피처** | 모든 피처 컬럼을 API 요청에 포함해야 함 |
| **Feature Store (fetcher)로 증강한 피처** | entity_columns만 제공하면 됨 (피처는 자동 조회) |

### 예시 1: SQL/CSV 기반 학습

학습 시 Recipe:

```yaml
data:
  loader:
    source_uri: "data/transactions.csv"
  data_interface:
    entity_columns: [transaction_id]
    target_column: is_fraud
    feature_columns: [amount, merchant_category, user_age, hour_of_day]
```

**API 요청 시 필요한 필드:**

```json
{
  "transaction_id": "TX123",
  "amount": 150.50,
  "merchant_category": "electronics",
  "user_age": 32,
  "hour_of_day": 14
}
```

모든 피처 컬럼(`amount`, `merchant_category`, `user_age`, `hour_of_day`)을 직접 제공해야 합니다.

### 예시 2: Feature Store 기반 학습

학습 시 Recipe:

```yaml
data:
  loader:
    source_uri: "sql/transactions.sql"
  data_interface:
    entity_columns: [user_id]
    target_column: is_fraud
    timestamp_column: trans_date_trans_time
  fetcher:
    feature_service: "fraud_detection_features"
    features:
      - "user_transaction_features:avg_amount"
      - "user_transaction_features:transactions_7d"
      - "user_demographics:age"
```

**API 요청 시 필요한 필드:**

```json
{
  "user_id": "U12345"
}
```

`entity_columns`인 `user_id`만 제공하면, 나머지 피처(`avg_amount`, `transactions_7d`, `age`)는 Feature Store Online Store에서 자동으로 조회됩니다.

### 필수 입력 필드 확인

`/model/schema` 엔드포인트에서 필수 입력 필드를 확인할 수 있습니다:

```bash
curl http://localhost:8000/model/schema | jq '.required_columns'
```

---

## 3. API 엔드포인트

### `GET /health` (Liveness)

프로세스 생존 여부를 확인합니다. Kubernetes livenessProbe용으로, 모델 로드 상태와 무관하게 항상 200을 반환합니다.

```bash
curl http://localhost:8000/health
```

**응답:**

```json
{
  "status": "ok"
}
```

---

### `GET /ready` (Readiness)

모델이 로드되어 트래픽을 받을 준비가 되었는지 확인합니다. Kubernetes readinessProbe용입니다.

```bash
curl http://localhost:8000/ready
```

**응답 (성공):**

```json
{
  "status": "ready",
  "model_uri": "runs:/abc123def456/model",
  "model_name": "xgboost.XGBClassifier"
}
```

**응답 (모델 미로드):** `503 Service Unavailable`

---

### `POST /predict`

단일 샘플 예측을 수행합니다.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.50,
    "merchant_category": "electronics",
    "user_age": 32,
    "hour_of_day": 14
  }'
```

**응답:**

```json
{
  "prediction": 0.85,
  "model_uri": "runs:/abc123def456/model"
}
```

---

### `POST /predict/batch`

여러 샘플을 한 번에 예측합니다.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"amount": 150.50, "merchant_category": "electronics", "user_age": 32, "hour_of_day": 14},
      {"amount": 25.00, "merchant_category": "grocery", "user_age": 45, "hour_of_day": 10},
      {"amount": 999.99, "merchant_category": "jewelry", "user_age": 28, "hour_of_day": 23}
    ]
  }'
```

**응답:**

```json
{
  "predictions": [
    {"prediction": 0.85, "transaction_id": "TX001"},
    {"prediction": 0.12, "transaction_id": "TX002"},
    {"prediction": 0.92, "transaction_id": "TX003"}
  ],
  "model_uri": "runs:/abc123def456/model",
  "sample_count": 3
}
```

---

### `GET /model/info`

모델 메타데이터와 학습 정보를 조회합니다.

```bash
curl http://localhost:8000/model/info
```

**응답:**

```json
{
  "model_uri": "runs:/abc123def456/model",
  "model_class_path": "xgboost.XGBClassifier",
  "hyperparameter_optimization": {
    "enabled": true,
    "engine": "optuna",
    "best_params": {"n_estimators": 200, "max_depth": 6},
    "best_score": 0.92,
    "total_trials": 50
  },
  "training_methodology": {
    "train_test_split_method": "random",
    "train_ratio": 0.7,
    "validation_strategy": "holdout"
  },
  "api_schema": {
    "input_fields": ["amount", "merchant_category", "user_age", "hour_of_day"],
    "entity_columns": ["transaction_id"],
    "task_type": "classification"
  }
}
```

---

### `GET /model/schema`

API 요청/응답 스키마를 조회합니다.

```bash
curl http://localhost:8000/model/schema
```

**응답:**

```json
{
  "prediction_request_schema": {
    "type": "object",
    "properties": {
      "amount": {"type": "number"},
      "merchant_category": {"type": "string"},
      "user_age": {"type": "integer"},
      "hour_of_day": {"type": "integer"}
    },
    "required": ["amount", "merchant_category", "user_age", "hour_of_day"]
  },
  "required_columns": ["amount", "merchant_category", "user_age", "hour_of_day"],
  "entity_columns": ["transaction_id"],
  "task_type": "classification",
  "schema_generation_method": "datainterface_based"
}
```

---

### `GET /metrics` (Prometheus)

Prometheus 형식의 메트릭을 반환합니다. `serving.metrics_enabled: true` (기본값) 시 활성화됩니다.

```bash
curl http://localhost:8000/metrics
```

**응답 (일부):**

```text
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",path="/predict",status="2xx"} 150
http_request_duration_seconds_bucket{le="0.1",method="POST",path="/predict"} 120
```

---

## 4. OpenAPI 문서

Swagger UI에서 API를 테스트하고 스키마를 확인할 수 있습니다:

```text
http://localhost:8000/docs
```

---

## 5. 에러 응답

### 필수 컬럼 누락 (422)

```json
{
  "detail": "필수 컬럼 누락: ['amount', 'user_age']"
}
```

### 잘못된 데이터 타입 (422)

```json
{
  "detail": "컬럼 'amount'은 숫자형이어야 합니다."
}
```

### 비스칼라 값 (422)

```json
{
  "detail": "컬럼 'features'에 비스칼라 값이 포함되어 있습니다."
}
```

### 모델 미로드 (503)

```json
{
  "detail": "모델이 준비되지 않았습니다."
}
```

### 요청 타임아웃 (504)

```json
{
  "detail": "Request timeout (30s)"
}
```

---

## 6. 운영 설정

Config 파일에서 API 서빙 관련 설정을 구성할 수 있습니다.

```yaml
serving:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  model_stage: "None"           # MLflow Model Registry 스테이지 (기본: None)
  request_timeout_seconds: 30   # 요청 타임아웃 (기본: 30초)
  metrics_enabled: true         # Prometheus /metrics 활성화 (기본: true)
  cors:
    enabled: false              # CORS 활성화 (기본: false)
    allow_origins: ["*"]
    allow_methods: ["*"]
    allow_headers: ["*"]
```

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `model_stage` | `None` | MLflow Model Registry 스테이지 (아래 표 참고) |
| `request_timeout_seconds` | `30` | 예측 요청 타임아웃 (초) |
| `metrics_enabled` | `true` | Prometheus `/metrics` 엔드포인트 활성화 |
| `cors.enabled` | `false` | CORS 활성화 (브라우저 직접 호출 시 필요) |

### model_stage 설정

MLflow Model Registry에 등록된 모델의 스테이지를 지정합니다.

| 값 | 설명 |
|----|------|
| `None` | 스테이지 미지정. Model Registry 미사용 시 기본값 |
| `Staging` | 테스트/검증 환경용 모델 |
| `Production` | 프로덕션 배포 승인된 모델 |
| `Archived` | 보관/비활성화된 모델 |

```yaml
# 로컬 개발/테스트 환경
serving:
  model_stage: "None"

# 프로덕션 환경 (승인된 모델만 서빙)
serving:
  model_stage: "Production"
```

### X-Request-ID 트레이싱

모든 요청에 대해 `X-Request-ID` 헤더를 지원합니다:

- 요청에 `X-Request-ID`가 있으면 그대로 사용
- 없으면 UUID 자동 생성
- 응답 헤더에 `X-Request-ID` 포함

```bash
curl -H "X-Request-ID: my-trace-123" http://localhost:8000/health
# 응답 헤더: X-Request-ID: my-trace-123
```

---

## 7. 전처리 자동 적용

학습 시 설정한 전처리(스케일링, 인코딩 등)는 서빙 시 자동으로 적용됩니다:

1. **입력**: 원본 값으로 요청
2. **내부 처리**: 학습된 전처리기가 자동 변환
3. **출력**: 예측 결과 반환

```bash
# 원본 값으로 요청 (전처리는 자동)
curl -X POST http://localhost:8000/predict \
  -d '{"amount": 150.50, "category": "electronics"}'
```

---

## 8. Task별 응답 형식

### Classification

```json
{
  "prediction": 1,
  "probability": 0.85,
  "model_uri": "runs:/abc123/model"
}
```

### Regression

```json
{
  "prediction": 245000.50,
  "model_uri": "runs:/abc123/model"
}
```

### Timeseries

```json
{
  "prediction": [120.5, 125.3, 130.1, 128.7],
  "model_uri": "runs:/abc123/model"
}
```

---

## 9. 배포 예시

### MLflow 서버 없이 배포 (로컬 Artifact)

MLflow 서버 연결 없이 로컬 artifact만으로 배포할 수 있습니다:

```bash
# 1. 학습 (로컬 mlruns/에 저장)
MLFLOW_TRACKING_URI=./mlruns mmp train \
  -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv

# 2. Docker 이미지 빌드 (mlruns/ 포함)
docker build -t my-model-api .

# 3. 추론 (동일 환경변수로 로컬 artifact 사용)
docker run -e MLFLOW_TRACKING_URI=./mlruns \
  -e RUN_ID=abc123 -p 8000:8000 my-model-api
```

```dockerfile
FROM python:3.11-slim

RUN pip install modern-ml-pipeline

COPY configs/ /app/configs/
COPY mlruns/ /app/mlruns/

WORKDIR /app

ENV MLFLOW_TRACKING_URI=./mlruns

CMD ["mmp", "serve-api", "--run-id", "${RUN_ID}", "-c", "configs/prod.yaml", "--port", "8000"]
```

### MLflow 서버 연결 배포

MLflow 서버가 있는 환경에서는 artifact를 이미지에 포함하지 않아도 됩니다:

```dockerfile
FROM python:3.11-slim

RUN pip install modern-ml-pipeline

COPY configs/ /app/configs/

WORKDIR /app

CMD ["mmp", "serve-api", "--run-id", "${RUN_ID}", "-c", "configs/prod.yaml", "--port", "8000"]
```

```bash
docker run -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e RUN_ID=abc123 -p 8000:8000 my-model-api
```

### Kubernetes 배포

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mmp-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mmp-api
        image: mmp-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: RUN_ID
          value: "abc123def456"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"  # 또는 ./mlruns (로컬)
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## 참고

- [CLI 레퍼런스](./CLI_REFERENCE.md): `mmp serve-api` 명령어 상세
- [환경 설정 가이드](./ENVIRONMENT_SETUP.md): MLflow, Feature Store 연결 설정
- [로컬 개발 환경](./LOCAL_DEV_ENVIRONMENT.md): Docker 기반 로컬 테스트
