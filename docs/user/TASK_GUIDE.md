# Task 가이드 (Task Guide)

Modern ML Pipeline에서 수행할 수 있는 작업(Task)의 종류와, 각 작업에 필요한 데이터 형식, 지원 모델을 안내합니다.


## 지원 Task 목록

| Task | 설명 | DataHandler | 대표 활용 사례 |
|------|------|-------------|---------------|
| **[Classification](#1-classification-분류)** | 범주형 데이터(0/1, 등급)를 예측 | Tabular | 사기 탐지, 이탈 예측, 상품 카테고리 분류 |
| **[Regression](#2-regression-회귀)** | 연속된 숫자(가격, 수량)를 예측 | Tabular | 집값 예측, 매출 예측, 재고 수요 예측 |
| **[Timeseries](#3-timeseries-시계열)** | 시간의 흐름에 따른 미래 값을 예측 | Timeseries/Sequence | 일별 매출 예측, 서버 트래픽 예측 |
| **[Clustering](#4-clustering-군집화)** | 정답 없이 데이터를 그룹으로 묶음 | Tabular | 고객 세분화, 이상 패턴 그룹화 |
| **[Causal](#5-causal-inference-인과-추론)** | 특정 행동(처치)의 효과를 분석 | Tabular | 마케팅 프로모션 효과 분석, 약물 효과 분석 |


## 1. Classification (분류)

데이터를 정해진 클래스(Category) 중 하나로 분류합니다.

### 데이터 형식

| entity_id | feature_1 | feature_2 | category | **target** |
|-----------|-----------|-----------|----------|------------|
| user_1    | 0.5       | 100       | A        | **0**      |
| user_2    | 0.8       | 200       | B        | **1**      |

### data_interface 설정

| 필드 | 필수 | 설명 |
|------|------|------|
| `entity_columns` | O | 행을 식별하는 ID 컬럼 (학습에서 자동 제외) |
| `target_column` | O | 예측 대상 (0/1 또는 다중 클래스) |
| `feature_columns` | X | 명시하지 않으면 target/entity 제외 전체 사용 |

### 지원 모델

| 라이브러리 | 모델 (`class_path`) | 특징 |
|------------|-------------------|------|
| **XGBoost** | `xgboost.XGBClassifier` | 가장 범용적이고 성능이 우수함 (추천) |
| **LightGBM** | `lightgbm.LGBMClassifier` | 대용량 데이터에서 학습 속도가 매우 빠름 |
| **CatBoost** | `catboost.CatBoostClassifier` | 범주형 변수가 많을 때 별도 전처리 없이 강력함 |
| **Scikit-learn** | `sklearn.ensemble.RandomForestClassifier` | 튜닝 없이도 준수한 성능, 과적합에 강함 |
| **TabNet** | `src.models.custom.tabnet_wrapper.TabNetClassifierWrapper` | 딥러닝 기반, 피처 중요도 해석 가능 |

### 평가 지표 (Metrics)

- `accuracy`, `precision`, `recall`, `f1`, `roc_auc`

### Recipe 예시

```yaml
task_choice: classification
data:
  data_interface:
    entity_columns: [user_id]
    target_column: is_fraud
    feature_columns: null  # 자동 선택
model:
  class_path: xgboost.XGBClassifier
```


## 2. Regression (회귀)

특정 수치를 예측합니다.

### 데이터 형식

| entity_id | feature_1 | feature_2 | **target** |
|-----------|-----------|-----------|------------|
| house_1   | 30        | 2010      | **500000000** |
| house_2   | 24        | 2020      | **800000000** |

### data_interface 설정

| 필드 | 필수 | 설명 |
|------|------|------|
| `entity_columns` | O | 행을 식별하는 ID 컬럼 |
| `target_column` | O | 예측 대상 (연속형 숫자) |
| `feature_columns` | X | 명시하지 않으면 자동 선택 |

### 지원 모델

| 라이브러리 | 모델 (`class_path`) | 특징 |
|------------|-------------------|------|
| **XGBoost** | `xgboost.XGBRegressor` | 강력한 성능, 결측치 자동 처리 |
| **LightGBM** | `lightgbm.LGBMRegressor` | 대규모 데이터셋에 효율적 |
| **Scikit-learn** | `sklearn.linear_model.LinearRegression` | 결과 해석이 중요할 때 (기본 모델) |
| **TabNet** | `src.models.custom.tabnet_wrapper.TabNetRegressorWrapper` | 딥러닝 기반 회귀 |

### 평가 지표 (Metrics)

- `mae` (평균 절대 오차), `mse` (평균 제곱 오차), `rmse`, `r2_score`


## 3. Timeseries (시계열)

과거의 패턴을 학습하여 미래의 값을 예측합니다.

### 데이터 형식

**단일 시계열** (entity 1개):

| **timestamp** | feature_1 | feature_2 | **target** |
|---------------|-----------|-----------|------------|
| 2024-01-01    | 0.5       | 10        | **100**    |
| 2024-01-02    | 0.6       | 12        | **120**    |

**다중 시계열** (entity 여러 개):

| **entity_id** | **timestamp** | feature_1 | **target** |
|---------------|---------------|-----------|------------|
| store_A       | 2024-01-01    | 휴일      | **100**    |
| store_A       | 2024-01-02    | 평일      | **120**    |
| store_B       | 2024-01-01    | 휴일      | **80**     |
| store_B       | 2024-01-02    | 평일      | **90**     |

### data_interface 설정

| 필드 | 필수 | 설명 |
|------|------|------|
| `timestamp_column` | O | 시간 순서를 나타내는 컬럼 (datetime) |
| `target_column` | O | 예측 대상 값 |
| `entity_columns` | X | 다중 시계열일 때 entity 구분 컬럼 |
| `feature_columns` | X | 외생 변수 (exogenous features) |
| `sequence_length` | X | LSTM 등 시퀀스 모델용 윈도우 크기 (기본값: 30) |

### 지원 모델 및 DataHandler

| 모델 | DataHandler | 데이터 처리 방식 |
|------|-------------|-----------------|
| **ARIMA** | Timeseries | target만 사용 (univariate), 피처 무시 |
| **ExponentialSmoothing** | Timeseries | target만 사용 (univariate) |
| **LSTM** | Sequence | sliding window로 시퀀스 생성, 피처 활용 |

### ARIMA vs LSTM 선택 가이드

| 상황 | 추천 모델 |
|------|----------|
| 단변량 시계열 (target만으로 예측) | ARIMA |
| 외생 변수(feature)가 중요한 경우 | LSTM |
| 짧은 시계열 (< 100개 시점) | ARIMA |
| 긴 시계열 + 복잡한 패턴 | LSTM |
| 빠른 학습/추론 필요 | ARIMA |

### Recipe 예시: ARIMA (통계 모델)

```yaml
task_choice: timeseries
data:
  data_interface:
    entity_columns: [store_id]
    timestamp_column: date
    target_column: daily_sales
    feature_columns: null  # ARIMA는 target만 사용
model:
  class_path: src.models.custom.timeseries_wrappers.ARIMA
  hyperparameters:
    values:
      order_p: 1
      order_d: 1
      order_q: 1
```

### Recipe 예시: LSTM (딥러닝)

> **주의**: LSTM은 **숫자형 데이터만 지원**합니다.
> 문자열 컬럼이 있으면 preprocessor에 인코더(`ordinal_encoder` 등)를 추가하세요.

```yaml
task_choice: timeseries
data:
  data_interface:
    entity_columns: [store_id]  # 다중 entity 지원
    timestamp_column: date
    target_column: daily_sales
    feature_columns: [temperature, is_holiday, promotion]
    sequence_length: 14  # 14일치 시퀀스로 학습
model:
  class_path: src.models.custom.lstm_timeseries.LSTMTimeSeries
  hyperparameters:
    values:
      hidden_dim: 64
      num_layers: 2
      epochs: 100
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: standard_scaler
```

### 다중 Entity 시계열 처리

`entity_columns`를 설정하면 각 entity별로 독립적인 시퀀스를 생성합니다.

```text
입력 데이터: store_A 100일 + store_B 100일 = 200행
sequence_length: 14

처리 결과:
- store_A에서 86개 시퀀스 생성 (100 - 14)
- store_B에서 86개 시퀀스 생성 (100 - 14)
- 총 172개 시퀀스 (entity 경계 혼합 없음)
```

### 평가 지표 (Metrics)

- `mape` (평균 절대 비율 오차), `mse`, `mae`


## 4. Clustering (군집화)

정답(Target) 없이 데이터의 유사도에 따라 그룹을 나눕니다.

### 데이터 형식

| entity_id | age | spend_score | visit_count |
|-----------|-----|-------------|-------------|
| cust_1    | 25  | 80          | 10          |
| cust_2    | 40  | 20          | 2           |

### data_interface 설정

| 필드 | 필수 | 설명 |
|------|------|------|
| `entity_columns` | O | 행을 식별하는 ID 컬럼 |
| `target_column` | X | 군집화에서는 불필요 |
| `feature_columns` | X | 군집화에 사용할 피처 |

### 지원 모델

| 모델 (`class_path`) | 특징 |
|-------------------|------|
| `sklearn.cluster.KMeans` | 가장 대중적, 군집의 개수(K)를 지정해야 함 |
| `sklearn.mixture.GaussianMixture` | 데이터가 정규분포를 따른다고 가정할 때 유용 |

### 평가 지표 (Metrics)

- `silhouette_score` (군집 내 응집도와 군집 간 분리도), `inertia`


## 5. Causal Inference (인과 추론)

어떤 행동(Treatment)이 결과(Outcome)에 미친 순수한 영향(Effect)을 분석합니다.

### 데이터 형식

| entity_id | age | **treatment** | **target** |
|-----------|-----|---------------|------------|
| user_1    | 20  | **1** (처치군) | **50000**  |
| user_2    | 30  | **0** (대조군) | **10000**  |

### data_interface 설정

| 필드 | 필수 | 설명 |
|------|------|------|
| `entity_columns` | O | 행을 식별하는 ID 컬럼 |
| `target_column` | O | 결과 변수 (Outcome) |
| `treatment_column` | O | 처치 여부 (0/1) |
| `feature_columns` | X | 공변량 (Covariates) |

### 지원 모델 (CausalML)

| 모델 (`class_path`) | 특징 |
|-------------------|------|
| `causalml.inference.meta.XGBTRegressor` | **T-Learner**: 처치군/대조군 모델을 따로 학습 (정확도 높음) |
| `causalml.inference.meta.LRSRegressor` | **S-Learner**: 처치 여부를 피처로 넣어 하나의 모델 학습 (단순함) |

### Recipe 예시

```yaml
task_choice: causal
data:
  data_interface:
    entity_columns: [user_id]
    target_column: purchase_amount
    treatment_column: received_coupon
    feature_columns: [age, gender, past_purchases]
model:
  class_path: causalml.inference.meta.XGBTRegressor
```

### 평가 지표 (Metrics)

- `ate` (평균 처치 효과), `cate` (조건부 처치 효과)


## DataHandler 자동 선택

모델 Catalog에 정의된 `data_handler` 속성에 따라 DataHandler가 자동 선택됩니다.

| DataHandler | 용도 | 데이터 변환 |
|-------------|------|------------|
| **Tabular** | Classification, Regression, Clustering, Causal | 2D 유지 (피처 선택만) |
| **Timeseries** | ARIMA, ExponentialSmoothing | 2D 유지 + lag/rolling 피처 자동 생성 |
| **Sequence** | LSTM, Transformer | Sliding window로 시퀀스 생성 (2D -> flatten) |

### DataHandler별 특징

**TabularHandler**

- 지정된 `feature_columns` 선택 (미지정시 자동 선택)
- `entity_columns`, `target_column` 자동 제외
- 4-way 분할 지원 (train/validation/test/calibration)

**TimeseriesHandler**

- 시간 피처 자동 생성: year, month, day, dayofweek, quarter, is_weekend
- lag 피처 생성: target_lag_1, target_lag_2, target_lag_3, target_lag_7, target_lag_14
- rolling 피처 생성: target_rolling_mean_3, target_rolling_std_3 등
- ARIMA 등 univariate 모델 사용시 피처 생성 자동 스킵

**SequenceHandler**

- Sliding window로 시퀀스 생성
- `entity_columns` 설정시 entity별로 독립 시퀀스 생성 (경계 혼합 방지)
- `sequence_length` 설정 가능 (기본값: 30)
- 출력: flatten된 2D DataFrame (seq0_feat0, seq0_feat1, ...)


## data_interface 필드 요약

| 필드 | 타입 | 필수 Task | 설명 |
|------|------|----------|------|
| `entity_columns` | list[str] | 전체 | 행 식별 ID (학습에서 제외) |
| `target_column` | str | 분류/회귀/시계열/인과 | 예측 대상 |
| `feature_columns` | list[str] | - | 학습에 사용할 피처 (null이면 자동 선택) |
| `timestamp_column` | str | 시계열 | 시간 컬럼 (datetime) |
| `treatment_column` | str | 인과 | 처치 여부 컬럼 (0/1) |
| `sequence_length` | int | 시계열(LSTM) | 시퀀스 윈도우 크기 (기본값: 30) |


## 모델과 메트릭 확인하기

현재 설치된 환경에서 사용 가능한 전체 목록은 CLI로 즉시 확인할 수 있습니다.

```bash
# 사용 가능한 모든 모델 조회
mmp list models

# Task별 사용 가능한 메트릭 조회
mmp list metrics
```
