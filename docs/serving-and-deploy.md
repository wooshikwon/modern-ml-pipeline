# API 서빙 & 배포 가이드

MMP v1.4.5 기준. 모델 서빙, 엔드포인트, Docker 빌드, Kubernetes 배포를 다룬다.

---

## 1. API 서빙 시작

```bash
mmp serve-api --run-id <mlflow_run_id> --config configs/prod.yaml
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | 서빙할 모델의 MLflow Run ID |
| `--config` | `-c` | O | Config 파일 경로 |
| `--host` | - | - | 바인딩 호스트 (기본: `0.0.0.0`) |
| `--port` | - | - | 바인딩 포트 (기본: `8000`) |

서버 시작 시 AppContext가 모델 메타데이터(스키마, 전처리기, Feature Store 설정 등)를 캐싱하여 요청당 오버헤드를 제거한다. 이후 예측 요청은 캐시된 컨텍스트를 재사용하므로 첫 요청부터 빠르게 응답한다.

```text
mmp v1.4.5

[1/3] Loading settings        done  run_id: abc123...
[2/3] Checking package deps   done  verified
[3/3] Starting API server     done

      API Server:   http://0.0.0.0:8000
      API Docs:     http://0.0.0.0:8000/docs
      Health Check: http://0.0.0.0:8000/health
```

---

## 2. 엔드포인트 레퍼런스

### `GET /health` (Liveness)

프로세스 생존 여부 확인. 모델 로드 상태와 무관하게 항상 200 반환. K8s `livenessProbe`용.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `GET /ready` (Readiness)

모델 로드 완료 여부 확인. K8s `readinessProbe`용. 미로드 시 503.

```bash
curl http://localhost:8000/ready
```

```json
{"status": "ready", "model_uri": "runs:/abc123def456/model", "model_name": "xgboost.XGBClassifier"}
```

### `POST /predict`

단일 샘플 예측. 학습 시 전처리(스케일링, 인코딩)는 자동 적용된다.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.50, "merchant_category": "electronics", "user_age": 32, "hour_of_day": 14}'
```

```json
{"prediction": 0.85, "model_uri": "runs:/abc123def456/model"}
```

### `POST /predict/batch`

여러 샘플을 한 번에 예측.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"amount": 150.50, "merchant_category": "electronics", "user_age": 32, "hour_of_day": 14},
      {"amount": 25.00, "merchant_category": "grocery", "user_age": 45, "hour_of_day": 10}
    ]
  }'
```

```json
{
  "predictions": [
    {"prediction": 0.85, "transaction_id": "TX001"},
    {"prediction": 0.12, "transaction_id": "TX002"}
  ],
  "model_uri": "runs:/abc123def456/model",
  "sample_count": 2
}
```

### `GET /model/info`

모델 메타데이터 조회 (하이퍼파라미터, 학습 방법론, 스키마).

```bash
curl http://localhost:8000/model/info
```

### `GET /model/schema`

필수 입력 필드 확인.

```bash
curl http://localhost:8000/model/schema | jq '.required_columns'
```

### `GET /metrics` (Prometheus)

Prometheus 형식 메트릭. `serving.metrics_enabled: true` (기본값) 시 활성화.

```bash
curl http://localhost:8000/metrics
```

```text
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",path="/predict",status="2xx"} 150
http_request_duration_seconds_bucket{le="0.1",method="POST",path="/predict"} 120
```

### API 입력 필드 결정 규칙

| 학습 시 설정 | API 입력 요구사항 |
|-------------|-------------------|
| SQL/CSV에서 직접 로드한 피처 | 모든 피처 컬럼을 요청에 포함 |
| Feature Store (fetcher)로 증강 | `entity_columns`만 제공 (피처는 자동 조회) |

### Task별 응답 형식

| Task | `prediction` 필드 |
|------|-------------------|
| Classification | `1` (+ `probability: 0.85`) |
| Regression | `245000.50` |
| Timeseries | `[120.5, 125.3, 130.1, 128.7]` |

### 에러 코드

| 코드 | 원인 |
|------|------|
| 422 | 필수 컬럼 누락, 잘못된 데이터 타입, 비스칼라 값 |
| 503 | 모델 미로드 |
| 504 | 요청 타임아웃 (기본 30초) |

---

## 3. 서빙 설정

```yaml
serving:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  model_stage: "None"           # None | Staging | Production | Archived
  request_timeout_seconds: 30
  metrics_enabled: true         # Prometheus /metrics 활성화
  cors:
    enabled: false
    allow_origins: ["*"]
    allow_methods: ["*"]
    allow_headers: ["*"]
```

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `model_stage` | `None` | MLflow Model Registry 스테이지 |
| `request_timeout_seconds` | `30` | 예측 요청 타임아웃 (초) |
| `metrics_enabled` | `true` | Prometheus `/metrics` 활성화 |
| `cors.enabled` | `false` | CORS (브라우저 직접 호출 시 필요) |

프로덕션에서는 `model_stage: "Production"`으로 승인된 모델만 서빙한다.

### X-Request-ID 트레이싱

모든 요청/응답에 `X-Request-ID` 헤더 지원. 요청에 없으면 UUID 자동 생성.

```bash
curl -H "X-Request-ID: my-trace-123" http://localhost:8000/health
```

---

## 4. Docker 빌드

### 단일 이미지 패턴

하나의 이미지로 학습, 추론, 서빙을 모두 실행한다. command만 변경.

```bash
docker build -t my-model:v1.4.5 .
```

| 용도 | command |
|------|---------|
| API 서빙 | `mmp serve-api --run-id <run_id> -c /app/configs/prod.yaml` |
| 배치 추론 | `mmp batch-inference --run-id <run_id> -d gs://bucket/data.csv` |
| 학습 | `mmp train -r /app/recipes/model.yaml -d gs://bucket/train.csv` |

### Extras 선택

Dockerfile에서 필요한 extras를 조합한다.

```dockerfile
# 클라우드 + Gradient Boosting (권장)
RUN pip install ".[ml-extras,cloud-extras]"

# 클라우드 + 딥러닝
RUN pip install ".[cloud-extras,torch-extras]"
```

### 레지스트리 푸시

```bash
# GCR
docker build -t gcr.io/my-project/mmp:v1.4.5 .
docker push gcr.io/my-project/mmp:v1.4.5

# ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t <account>.dkr.ecr.<region>.amazonaws.com/mmp:v1.4.5 .
docker push <account>.dkr.ecr.<region>.amazonaws.com/mmp:v1.4.5
```

---

## 5. Kubernetes 배포

### ConfigMap으로 런타임 설정 변경

이미지 재빌드 없이 설정만 변경할 때 ConfigMap을 마운트한다.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mmp-config
data:
  prod.yaml: |
    environment:
      name: production
    mlflow:
      tracking_uri: http://mlflow:5000
    serving:
      model_stage: "Production"
      metrics_enabled: true
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mmp-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mmp-serving
  template:
    metadata:
      labels:
        app: mmp-serving
    spec:
      containers:
      - name: mmp
        image: gcr.io/my-project/mmp:v1.4.5
        command: ["mmp", "serve-api", "--run-id", "$(MODEL_RUN_ID)", "-c", "/app/configs/prod.yaml"]
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_RUN_ID
          valueFrom:
            configMapKeyRef:
              name: mmp-env
              key: run_id
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/configs
      volumes:
      - name: config
        configMap:
          name: mmp-config
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mmp-serving
spec:
  selector:
    app: mmp-serving
  ports:
  - port: 80
    targetPort: 8000
```

### 환경변수 레퍼런스

| 환경변수 | 용도 | 기본값 |
|----------|------|--------|
| `MODEL_RUN_ID` | 서빙/추론할 모델 ID | - |
| `CONFIG_PATH` | Config 파일 경로 | `configs/production.yaml` |
| `RECIPE_PATH` | Recipe 파일 경로 | `recipes/model.yaml` |
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `./mlruns` |

---

## 6. 모니터링

### Prometheus 메트릭 수집

`/metrics` 엔드포인트를 Prometheus가 스크레이핑하도록 설정한다.

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'mmp-serving'
    scrape_interval: 15s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['mmp-serving:80']
```

Pod annotation 방식:

```yaml
# Deployment template metadata
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

### 수집되는 메트릭

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `http_requests_total` | counter | HTTP 요청 수 (method, path, status별) |
| `http_request_duration_seconds` | histogram | 요청 처리 시간 분포 |

### Health Check 요약

| 엔드포인트 | K8s Probe | 동작 |
|------------|-----------|------|
| `GET /health` | livenessProbe | 프로세스 생존 확인. 항상 200 |
| `GET /ready` | readinessProbe | 모델 로드 완료 시 200, 미로드 시 503 |
