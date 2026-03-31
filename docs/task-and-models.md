# Task & Model & Preprocessing 통합 가이드

Task 선택 → 모델 선택 → 전처리 조합을 한 곳에서 결정하기 위한 실전 가이드.
상세 스키마는 `AGENT.md` 참조.


## Task 개요

| Task | 모델 수 | DataHandler | 필수 data_interface | 대표 활용 사례 |
|------|---------|-------------|---------------------|---------------|
| **Classification** | 10 | Tabular | `entity_columns`, `target_column` | 사기 탐지, 이탈 예측 |
| **Regression** | 13 | Tabular | `entity_columns`, `target_column` | 집값 예측, 수요 예측 |
| **Timeseries** | 4 | Timeseries / Sequence | `entity_columns`, `timestamp_column`, `target_column` | 일별 매출, 트래픽 예측 |
| **Clustering** | 3 | Tabular | `entity_columns` | 고객 세분화, 이상 패턴 그룹화 |
| **Causal** | 3 | Tabular | `entity_columns`, `target_column`, `treatment_column` | 프로모션 효과, 약물 효과 |

> 전체 34개 모델. `mmp list models`로 최신 목록 확인 가능.


## 공통 설정

### Data Split 전략

| `data.split.strategy` | 설명 | 언제 쓰나 |
|---|---|---|
| `random` (기본값) | sklearn shuffle split | 일반 tabular, clustering |
| `temporal` | 시간순 정렬 후 분할, 미래 누수 방지 | 시계열, 시간 의존성이 있는 분류/회귀 |

`temporal` 사용 시 `temporal_column` 필수:

```yaml
data:
  split:
    strategy: temporal
    temporal_column: created_at
    train: 0.7
    test: 0.15
    validation: 0.15
```

### Performance Monitoring

학습 후 데이터/예측 드리프트를 PSI 기반으로 감시한다.

```yaml
monitoring:
  enabled: true
  data_drift:      # PSI 임계값 (기본값 있음, 커스텀 가능)
  prediction_drift: # PSI 임계값
```

`enabled: true`로 설정하면 train 분포 대비 test/inference 분포의 PSI를 자동 계산하여 MLflow에 기록한다.


---


## 1. Classification (분류)

### 모델 추천

| 상황 | 모델 (`class_path`) | 라이브러리 |
|------|---------------------|-----------|
| **범용 (첫 시도)** | `xgboost.XGBClassifier` | xgboost |
| 대용량 데이터, 빠른 학습 | `lightgbm.LGBMClassifier` | lightgbm |
| 범주형 변수 다수 | `catboost.CatBoostClassifier` | catboost |
| 과적합 방지, 튜닝 최소 | `sklearn.ensemble.RandomForestClassifier` | sklearn |
| 피처 중요도 해석 필요 | `TabNetClassifierWrapper` | pytorch-tabnet |
| 딥러닝 tabular | `FTTransformerClassifier` | rtdl_revisiting_models |
| 베이스라인 | `sklearn.linear_model.LogisticRegression` | sklearn |
| 기타 | `GaussianNB`, `KNeighborsClassifier`, `SVC` | sklearn |

### 필수 data_interface

```yaml
data_interface:
  entity_columns: [user_id]
  target_column: is_fraud       # 0/1 또는 다중 클래스
  feature_columns: null         # null이면 자동 선택
```

### 평가 지표

`accuracy`, `precision`, `recall`, `f1`, `roc_auc`

### 권장 전처리 조합

**트리 모델 (XGBoost, LightGBM, CatBoost, RandomForest)**

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder
      columns: [merchant, city, job]      # high cardinality
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: one_hot_encoder
      columns: [gender, state]            # low cardinality (<10)
```

**딥러닝 모델 (TabNet, FT-Transformer)**

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder
      columns: [merchant, city, job]
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: standard_scaler
```


---


## 2. Regression (회귀)

### 모델 추천

| 상황 | 모델 (`class_path`) | 라이브러리 |
|------|---------------------|-----------|
| **범용 (첫 시도)** | `xgboost.XGBRegressor` | xgboost |
| 대용량, 빠른 학습 | `lightgbm.LGBMRegressor` | lightgbm |
| 해석 중요 | `sklearn.linear_model.LinearRegression` | sklearn |
| 정규화 필요 | `Ridge`, `Lasso`, `ElasticNet` | sklearn |
| 딥러닝 tabular | `FTTransformerRegressor`, `ResNetRegressor` | rtdl_revisiting_models |
| 피처 해석 | `TabNetRegressorWrapper` | pytorch-tabnet |
| 기타 | `KNeighborsRegressor`, `SVR`, `RandomForestRegressor` | sklearn |

### 필수 data_interface

```yaml
data_interface:
  entity_columns: [house_id]
  target_column: price
  feature_columns: null
```

### 평가 지표

`mae`, `mse`, `rmse`, `r2_score`

### 권장 전처리 조합

트리 모델, 딥러닝 모델 조합은 Classification과 동일. 선형 모델은 반드시 스케일링 추가:

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder
      columns: [category_col]
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: standard_scaler          # 선형 모델 필수
```

이상치가 많으면 `robust_scaler` 사용.


### Quantile Regression (분위수 회귀)

점 예측 대신 분위수별 예측을 출력하는 회귀 변형. 불확실성 구간 추정에 사용한다.

**모델**: `mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble` (library: `mmp-custom`)

**핵심 설정**:

| 파라미터 | 설명 |
|----------|------|
| `base_class_path` | 내부 base 모델 (기본값: `lightgbm.LGBMRegressor`) |
| `quantiles` | 예측할 분위수 리스트. 소수점 분위수 지원 (예: `0.995` → `pred_p99.5`) |

```yaml
task_choice: regression
model:
  class_path: mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble
  library: mmp-custom
  hyperparameters:
    tuning_enabled: false
    values:
      base_class_path: lightgbm.LGBMRegressor
      quantiles: [0.5, 0.75, 0.9, 0.95, 0.99, 0.995]   # p99.5 지원
      n_estimators: 100
      learning_rate: 0.1
```

**출력 컬럼**: `pred_p50`, `pred_p75`, `pred_p90`, `pred_p95`, `pred_p99`, `pred_p99.5`

**평가 지표** (자동 계산):

| 지표 | 설명 |
|------|------|
| `pinball_loss_p<N>` | 분위수별 핀볼 손실 |
| `coverage_rate_p<N>` | 실제값이 예측 이하인 비율 (p90이면 ~0.90이 정상) |
| `mae_p<N>` | 분위수별 MAE |
| `mean_pinball_loss` | 전체 분위수 평균 핀볼 손실 |
| `interval_coverage` | [최저, 최고] 분위수 구간 내 실제값 비율 |

**권장 전처리**: 일반 Regression과 동일.


---


## 3. Timeseries (시계열)

### 모델 추천

| 상황 | 모델 (`class_path`) | DataHandler |
|------|---------------------|-------------|
| 단변량, 짧은 시계열 | `ARIMA` | Timeseries |
| 단변량, 계절성 | `ExponentialSmoothing` | Timeseries |
| 단변량, 추세만 | `LinearTrend` | Timeseries |
| 외생 변수 활용, 복잡 패턴 | `LSTMTimeSeries` | Sequence |

> ARIMA/ExponentialSmoothing/LinearTrend는 target만 사용 (univariate). 외생 변수가 중요하면 LSTM을 쓴다.

### 필수 data_interface

```yaml
data_interface:
  entity_columns: [store_id]         # 다중 entity 지원
  timestamp_column: date
  target_column: daily_sales
  feature_columns: [temperature, is_holiday]   # LSTM만 사용
  sequence_length: 14                          # LSTM 윈도우 크기
```

### 평가 지표

`mape`, `mse`, `mae`

### 권장 전처리 조합

**통계 모델 (ARIMA, ExponentialSmoothing, LinearTrend)**: 전처리 불필요 또는 최소.

**LSTM**:

```yaml
preprocessor:
  steps:
    - type: forward_fill
      limit: 3
    - type: backward_fill
      limit: 3
    - type: interpolation
      method: linear
    - type: ordinal_encoder           # 문자열 컬럼이 있으면 필수
      columns: [category_col]
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: standard_scaler
```

> LSTM은 숫자형 데이터만 지원. 문자열 컬럼이 있으면 인코더 필수.

### Split 전략

시계열은 `strategy: temporal` 강력 권장 — 미래 데이터 누수를 방지한다.


---


## 4. Clustering (군집화)

### 모델 추천

| 상황 | 모델 (`class_path`) |
|------|---------------------|
| **범용, K 지정** | `sklearn.cluster.KMeans` |
| 정규분포 가정, 소프트 클러스터 | `sklearn.mixture.GaussianMixture` |
| 대규모 데이터, 자동 군집 수 | `sklearn.cluster.Birch` |

### 필수 data_interface

```yaml
data_interface:
  entity_columns: [customer_id]
  target_column: null              # 불필요
  feature_columns: null
```

### 평가 지표

`silhouette_score`, `inertia`

### 권장 전처리 조합

거리 기반 알고리즘이므로 스케일링 필수:

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder           # 범주형 있으면
      columns: [category]
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: standard_scaler           # 필수
```


---


## 5. Causal Inference (인과 추론)

### 모델 추천

| 상황 | 모델 (`class_path`) |
|------|---------------------|
| **정확도 우선** (T-Learner) | `causalml.inference.meta.XGBTRegressor` |
| 단순한 구조 (S-Learner) | `causalml.inference.meta.LRSRegressor` |
| 비선형 이질적 효과 | `causalml.inference.tree.CausalRandomForestRegressor` |

### 필수 data_interface

```yaml
data_interface:
  entity_columns: [user_id]
  target_column: purchase_amount     # outcome
  treatment_column: received_coupon  # 처치 여부 (0/1)
  feature_columns: [age, gender, past_purchases]
```

### 평가 지표

`ate` (평균 처치 효과), `cate` (조건부 처치 효과)

### 권장 전처리 조합

내부적으로 트리 기반이므로 Classification 트리 모델 조합과 동일:

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder
      columns: [gender, region]
      handle_unknown: use_encoded_value
      unknown_value: -1
```


---


## 전처리기 목록

| 카테고리 | 전처리기 | 용도 |
|----------|---------|------|
| Missing | `simple_imputer` | 결측값 대체 (mean/median/most_frequent/constant) |
| Missing | `drop_missing` | 결측 행/열 삭제 |
| Missing | `forward_fill` | 순방향 채움 (시계열) |
| Missing | `backward_fill` | 역방향 채움 (시계열) |
| Missing | `constant_fill` | 상수값 채움 |
| Missing | `interpolation` | 보간법 (linear/polynomial/spline) |
| Scaler | `standard_scaler` | 표준화 (평균 0, 표준편차 1) |
| Scaler | `min_max_scaler` | Min-Max 정규화 (0-1) |
| Scaler | `robust_scaler` | 로버스트 스케일링 (이상치에 강함) |
| Encoder | `ordinal_encoder` | 순서형 정수 인코딩 |
| Encoder | `one_hot_encoder` | 원-핫 인코딩 |
| Encoder | `catboost_encoder` | 타겟 기반 인코딩 |
| Discretizer | `kbins_discretizer` | 연속형 → 범주형 구간화 |
| Feature Gen | `polynomial_features` | 다항식 피처 생성 |
| Feature Gen | `tree_based_feature_generator` | 트리 기반 피처 생성 |

### 적용 순서

```
1. 결측값 처리  →  2. 인코딩  →  3. 피처 생성 (선택)  →  4. 스케일링
```

반드시 이 순서를 지킨다. 스케일링 후 인코딩하면 스케일이 깨지고, 결측값 처리 전에 인코딩하면 에러가 발생한다.

### 인코더 선택 기준

| 고유값 수 | 권장 인코더 |
|-----------|------------|
| < 10 | `one_hot_encoder` |
| 10 ~ 100 | `ordinal_encoder` |
| > 100 | `ordinal_encoder` 또는 `catboost_encoder` |

### Task별 권장 전처리 요약

| Task | 결측값 | 인코딩 | 스케일링 | 비고 |
|------|--------|--------|---------|------|
| Classification (트리) | `simple_imputer` | `ordinal` + `one_hot` | 불필요 | |
| Classification (DL) | `simple_imputer` | `ordinal` | `standard_scaler` | 문자열 컬럼 인코딩 필수 |
| Regression (트리) | `simple_imputer` | `ordinal` + `one_hot` | 불필요 | |
| Regression (선형) | `simple_imputer` | `ordinal` + `one_hot` | `standard_scaler` 필수 | 이상치 많으면 `robust_scaler` |
| Regression (Quantile) | `simple_imputer` | `catboost_encoder` 권장 | 불필요 | 카탈로그 기본 전처리 참조 |
| Timeseries (통계) | 불필요 | 불필요 | 불필요 | |
| Timeseries (LSTM) | `forward_fill` → `backward_fill` → `interpolation` | `ordinal` (문자열 시) | `standard_scaler` | |
| Clustering | `simple_imputer` | `ordinal` | `standard_scaler` 필수 | 거리 기반이므로 |
| Causal | `simple_imputer` | `ordinal` | 불필요 | 내부 트리 기반 |
