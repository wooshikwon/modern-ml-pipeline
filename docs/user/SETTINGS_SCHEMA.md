# 설정 스키마 (Settings Schema)

Modern ML Pipeline은 설정을 **인프라(Config)**와 **실험(Recipe)** 두 가지로 분리하여 관리합니다.


## 1. 설정 구조 개요

| 구분 | 파일 위치 | 역할 | 변경 빈도 |
|------|----------|------|----------|
| **Config** | `configs/*.yaml` | "어디서 실행할까?" (DB, MLflow, Cloud) | 환경별 1회 설정 |
| **Recipe** | `recipes/*.yaml` | "무엇을 학습할까?" (모델, 데이터, 전처리) | 실험할 때마다 변경 |


## 2. Config 스키마 (인프라 설정)

환경(Local, Dev, Prod)에 따라 달라지는 인프라 연결 정보를 정의합니다.

### 기본 구조

```yaml
environment:
  name: "local"

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "my-project"

data_source:
  adapter_type: "sql"
  config:
    connection_uri: "${DATABASE_URI}"
```

### 주요 섹션 상세

#### `environment`

```yaml
environment:
  name: "local"  # 환경 이름 (로깅/추적용)
```

#### `mlflow`

실험 추적 서버 설정입니다.

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"  # 또는 "file://./mlruns"
  experiment_name: "my-experiment"
  s3_endpoint_url: "http://minio:9000"   # MinIO 사용 시
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `tracking_uri` | O | MLflow 서버 주소 또는 로컬 경로 |
| `experiment_name` | X | 실험 그룹 이름 |
| `s3_endpoint_url` | X | S3 호환 스토리지 엔드포인트 |

#### `data_source`

데이터를 읽어올 원천 저장소입니다.

| `adapter_type` | 설명 | 필수 config 항목 |
|----------------|------|------------------|
| `storage` | 로컬 파일, S3, GCS | `base_path` |
| `sql` | PostgreSQL, BigQuery 등 | `connection_uri` |
| `feature_store` | Feast 피처 스토어 | `repo_path`, `feature_service` |

**Storage 예시:**

```yaml
data_source:
  adapter_type: "storage"
  config:
    base_path: "data/"           # 로컬 경로
    # base_path: "s3://bucket/"  # S3
    # base_path: "gs://bucket/"  # GCS
```

**SQL 예시:**

```yaml
data_source:
  adapter_type: "sql"
  config:
    connection_uri: "postgresql://user:pass@localhost:5432/db"
    # BigQuery: "bigquery://project-id"
```

#### `serving` (선택)

```yaml
serving:
  enabled: true
  port: 8000
  workers: 4
```


## 3. Recipe 스키마 (실험 설정)

실제 머신러닝 모델 학습에 필요한 모든 정보를 정의합니다.

### 기본 구조

```yaml
name: "my-experiment-v1"
task_choice: "classification"

data:
  loader:
    source_uri: "data/train.csv"
  data_interface:
    target_column: "label"

model:
  class_path: "xgboost.XGBClassifier"

evaluation:
  metrics: ["accuracy", "f1"]
```

### 주요 섹션 상세

#### `task_choice` (필수)

해결하려는 문제 유형을 지정합니다.

| 값 | 설명 | 예시 모델 |
|----|------|----------|
| `classification` | 분류 문제 | XGBClassifier, LGBMClassifier |
| `regression` | 회귀 문제 | XGBRegressor, LinearRegression |
| `timeseries` | 시계열 예측 | ARIMA, LSTMTimeSeries |
| `clustering` | 군집화 | KMeans, GaussianMixture |
| `causal` | 인과추론 | XGBTRegressor, LRSRegressor |

#### `data`

```yaml
data:
  loader:
    source_uri: "data/train.csv"           # CSV 파일
    # source_uri: "sql/my_query.sql"   # SQL 쿼리 파일
    # source_uri: "sql/query.sql.j2"   # Jinja2 템플릿

  data_interface:
    target_column: "label"                  # 예측할 컬럼 (필수)
    feature_columns: ["col1", "col2"]       # 사용할 피처 (선택, 미지정시 자동)
    entity_columns: ["user_id"]             # ID 컬럼 (선택)
    timestamp_column: "created_at"          # 시계열용 (timeseries 필수)
    sequence_length: 30                     # LSTM용 시퀀스 길이

  split:
    train: 0.7
    validation: 0.15
    test: 0.15
    calibration: 0.0                        # 확률 보정용 (선택)
```

#### `model` (필수)

```yaml
model:
  class_path: "xgboost.XGBClassifier"       # 전체 모델 경로
  hyperparameters:
    tuning_enabled: false                   # true: Optuna 튜닝
    values:                                 # 고정 하이퍼파라미터
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
```

**사용 가능한 모델 확인:**

```bash
mmp list models
```

출력된 `class_path` 값을 그대로 사용합니다:

```text
Classification:
  - xgboost.XGBClassifier                         (xgboost)
  - sklearn.ensemble.RandomForestClassifier       (scikit-learn)
  - src.models.custom.ft_transformer.FTTransformerClassifier (rtdl_revisiting_models)
```

#### `preprocessor` (선택)

전처리 단계를 리스트로 정의합니다. 순서대로 실행됩니다.

```yaml
preprocessor:
  steps:
    - type: "simple_imputer"              # 결측치 처리
      strategy: "mean"
    - type: "standard_scaler"             # 정규화
    - type: "one_hot_encoder"             # 원-핫 인코딩
      columns: ["category", "city"]
```

**사용 가능한 전처리기:**

| type | 설명 | 주요 파라미터 |
|------|------|--------------|
| `simple_imputer` | 결측치 대치 | `strategy`: mean, median, most_frequent |
| `forward_fill` | 앞 값으로 채우기 | - |
| `backward_fill` | 뒤 값으로 채우기 | - |
| `constant_fill` | 상수로 채우기 | `fill_value` |
| `drop_missing` | 결측 행 삭제 | - |
| `interpolation` | 보간법 | `method`: linear, polynomial |
| `standard_scaler` | Z-score 정규화 | - |
| `min_max_scaler` | 0-1 스케일링 | - |
| `robust_scaler` | 이상치 견고 스케일링 | - |
| `one_hot_encoder` | 원-핫 인코딩 | `columns` |
| `ordinal_encoder` | 순서형 인코딩 | `columns` |
| `catboost_encoder` | CatBoost 인코딩 | `columns` |
| `kbins_discretizer` | 구간화 | `n_bins`, `strategy` |
| `polynomial_features` | 다항 피처 생성 | `degree` |
| `tree_based_feature_generator` | 트리 기반 피처 생성 | - |

```bash
mmp list preprocessors  # 전체 목록 확인
```

> **Note**
>
> 상세 파라미터와 모델별 권장 조합은 [PREPROCESSOR_REFERENCE.md](./PREPROCESSOR_REFERENCE.md) 참조

#### `evaluation` (선택)

```yaml
evaluation:
  optimization_metric: "f1"               # 튜닝 최적화 기준
  metrics: ["accuracy", "f1", "roc_auc"]  # 평가할 메트릭 목록
```

**Task별 사용 가능한 메트릭:**

| Task | 메트릭 |
|------|--------|
| `classification` | accuracy, precision, recall, f1, roc_auc |
| `regression` | r2_score, mean_squared_error |
| `timeseries` | mse, rmse, mae, mape, smape |
| `clustering` | silhouette_score, inertia, n_clusters, bic, aic |
| `causal` | ate, ate_std, treatment_effect_significance |

```bash
mmp list metrics  # 전체 목록 확인
```


## 4. 전체 Recipe 예시

### Classification (XGBoost)

```yaml
name: "fraud-detection-v1"
task_choice: "classification"

data:
  loader:
    source_uri: "data/transactions.csv"
  data_interface:
    target_column: "is_fraud"
    entity_columns: ["transaction_id"]
  split:
    train: 0.7
    validation: 0.15
    test: 0.15

model:
  class_path: "xgboost.XGBClassifier"
  hyperparameters:
    tuning_enabled: true
    values:
      n_estimators: 100

preprocessor:
  steps:
    - type: "simple_imputer"
      strategy: "median"
    - type: "standard_scaler"
    - type: "one_hot_encoder"
      columns: ["merchant_category"]

evaluation:
  optimization_metric: "roc_auc"
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
```

### Timeseries (LSTM)

```yaml
name: "sales-forecast-v1"
task_choice: "timeseries"

data:
  loader:
    source_uri: "data/sales.csv"
  data_interface:
    target_column: "sales"
    timestamp_column: "date"
    entity_columns: ["store_id"]
    sequence_length: 30

model:
  class_path: "src.models.custom.lstm_timeseries.LSTMTimeSeries"
  hyperparameters:
    values:
      hidden_size: 64
      num_layers: 2
      epochs: 50

evaluation:
  optimization_metric: "rmse"
  metrics: ["mse", "rmse", "mae", "mape"]
```

### Regression (LightGBM)

```yaml
name: "price-prediction-v1"
task_choice: "regression"

data:
  loader:
    source_uri: "data/housing.csv"
  data_interface:
    target_column: "price"

model:
  class_path: "lightgbm.LGBMRegressor"
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05

preprocessor:
  steps:
    - type: "simple_imputer"
    - type: "min_max_scaler"

evaluation:
  metrics: ["r2_score", "mean_squared_error"]
```


## 5. 환경 변수 사용법

Config 파일에서 민감 정보를 환경 변수로 참조할 수 있습니다.

```yaml
# 환경변수 사용
password: "${DATABASE_PASSWORD}"

# 기본값 지정
password: "${DATABASE_PASSWORD:default_pass}"
```


## 6. CLI 명령어 참조

| 명령어 | 설명 |
|--------|------|
| `mmp list models` | 사용 가능한 모델과 class_path 확인 |
| `mmp list preprocessors` | 사용 가능한 전처리기 확인 |
| `mmp list metrics` | Task별 평가 메트릭 확인 |
| `mmp list adapters` | 데이터 어댑터 확인 |
| `mmp get-recipe <task> <model>` | Recipe 템플릿 생성 |
| `mmp get-config <env>` | Config 템플릿 생성 |

```bash
# Recipe 생성 예시
mmp get-recipe classification xgboost -o recipes/my_recipe.yaml

# Config 생성 예시
mmp get-config local -o configs/local.yaml
```
