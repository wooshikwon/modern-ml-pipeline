# 전처리기 참조 (Preprocessor Reference)

Modern ML Pipeline에서 사용 가능한 전처리기와 사용법을 설명합니다.


## 전처리기 목록

| 타입 | 전처리기명 | 용도 | 적용 범위 |
|------|-----------|------|-----------|
| Missing | `simple_imputer` | 결측값 대체 (mean, median, most_frequent) | targeted |
| Missing | `drop_missing` | 결측값 행/열 삭제 | targeted |
| Missing | `forward_fill` | 순방향 결측값 채움 (시계열) | targeted |
| Missing | `backward_fill` | 역방향 결측값 채움 (시계열) | targeted |
| Missing | `constant_fill` | 상수값으로 결측값 채움 | targeted |
| Missing | `interpolation` | 보간법으로 결측값 채움 (숫자형) | targeted |
| Scaler | `standard_scaler` | 표준화 (평균 0, 표준편차 1) | global (숫자형) |
| Scaler | `min_max_scaler` | Min-Max 정규화 (0-1 범위) | global (숫자형) |
| Scaler | `robust_scaler` | 로버스트 스케일링 (이상치에 강함) | global (숫자형) |
| Encoder | `ordinal_encoder` | 순서형 정수 인코딩 | targeted |
| Encoder | `one_hot_encoder` | 원-핫 인코딩 | targeted |
| Encoder | `catboost_encoder` | 타겟 기반 인코딩 | targeted |
| Discretizer | `kbins_discretizer` | 연속형 → 범주형 변환 | targeted |
| Feature Gen | `polynomial_features` | 다항식 피처 생성 | targeted |
| Feature Gen | `tree_based_feature_generator` | 트리 기반 피처 생성 | targeted |


## 상세 설명

### 1. simple_imputer

결측값을 지정된 전략에 따라 대체합니다.

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median  # mean, median, most_frequent, constant
      create_missing_indicators: false  # true로 설정 시 결측값 지시자 컬럼 생성
```

**파라미터:**

- `strategy`: 결측값 대체 전략
  - `mean`: 평균값 (숫자형만)
  - `median`: 중앙값 (숫자형만)
  - `most_frequent`: 최빈값 (모든 타입)
  - `constant`: 상수값
- `create_missing_indicators`: 결측값 지시자 컬럼 생성 여부


### 2. drop_missing

결측값이 있는 행 또는 열을 삭제합니다.

```yaml
preprocessor:
  steps:
    - type: drop_missing
      axis: rows        # rows 또는 columns
      threshold: 0.0    # 결측값 비율 기준 (0.0 = 하나라도 있으면 삭제)
      columns: null     # 특정 컬럼만 검사 (null = 전체)
```

**파라미터:**

- `axis`: 삭제 대상
  - `rows`: 결측값이 있는 행 삭제
  - `columns`: 결측값이 많은 열 삭제
- `threshold`: 결측값 비율 기준 (0.0~1.0)
  - `0.0`: 결측값이 하나라도 있으면 삭제
  - `0.5`: 결측값이 50% 이상이면 삭제
- `columns`: 검사할 컬럼 목록 (null이면 전체)

**주의사항:**

- 데이터 손실이 발생하므로 신중하게 사용
- 결측값이 많은 컬럼 제거에 유용


### 3. forward_fill

순방향 채움: 이전 유효값으로 결측값을 채웁니다. 시계열 데이터에 적합합니다.

```yaml
preprocessor:
  steps:
    - type: forward_fill
      limit: null       # 연속 결측값 최대 채움 개수 (null = 무제한)
      columns: null     # 적용할 컬럼 (null = 결측값 있는 전체 컬럼)
```

**파라미터:**

- `limit`: 연속 결측값 최대 채움 개수
- `columns`: 적용할 컬럼 목록

**권장 사용:**

- 시계열 데이터 (일별 매출, 센서 데이터 등)
- 이전 값이 유효한 대체값인 경우


### 4. backward_fill

역방향 채움: 다음 유효값으로 결측값을 채웁니다.

```yaml
preprocessor:
  steps:
    - type: backward_fill
      limit: null       # 연속 결측값 최대 채움 개수 (null = 무제한)
      columns: null     # 적용할 컬럼 (null = 결측값 있는 전체 컬럼)
```

**파라미터:**

- `limit`: 연속 결측값 최대 채움 개수
- `columns`: 적용할 컬럼 목록

**권장 사용:**

- 시계열 데이터에서 forward_fill과 함께 사용
- 시작 부분 결측값 처리


### 5. constant_fill

상수값으로 결측값을 채웁니다. 컬럼별로 다른 값 지정 가능합니다.

```yaml
preprocessor:
  steps:
    # 단일 값으로 채우기
    - type: constant_fill
      fill_value: 0
      columns: ['feature1', 'feature2']

    # 컬럼별 다른 값으로 채우기
    - type: constant_fill
      fill_value:
        age: -1
        income: 0
        category: "unknown"
```

**파라미터:**

- `fill_value`: 채울 값
  - 단일 값: 모든 컬럼에 동일 적용
  - Dict: 컬럼별 다른 값 지정 `{column: value}`
- `columns`: 적용할 컬럼 목록 (null이면 결측값 있는 전체)

**권장 사용:**

- 결측값이 특정 의미를 가질 때 (예: "정보 없음" = -1)
- 범주형 컬럼에 "unknown" 값 채우기


### 6. interpolation

보간법으로 결측값을 채웁니다. 숫자형 컬럼에만 적용됩니다.

```yaml
preprocessor:
  steps:
    - type: interpolation
      method: linear     # linear, polynomial, spline 등
      order: null        # polynomial/spline 차수
      limit: null        # 연속 결측값 최대 채움 개수
      columns: null      # 적용할 컬럼 (null = 숫자형 전체)
```

**파라미터:**

- `method`: 보간 방법
  - `linear`: 선형 보간 (기본값)
  - `polynomial`: 다항식 보간 (order 필요)
  - `spline`: 스플라인 보간 (order 필요)
  - `nearest`: 가장 가까운 값
  - `zero`: 0차 보간 (계단식)
- `order`: 다항식/스플라인 차수 (method가 polynomial/spline일 때)
- `limit`: 연속 결측값 최대 채움 개수
- `columns`: 적용할 컬럼 목록

**권장 사용:**

- 시계열 데이터의 숫자형 피처
- 연속적인 패턴이 있는 데이터


### 7. standard_scaler

표준화: 평균 0, 표준편차 1로 변환합니다.

```yaml
preprocessor:
  steps:
    - type: standard_scaler
```

**권장 사용:**

- 딥러닝 모델 (FT-Transformer, LSTM)
- 선형 모델 (LinearRegression, LogisticRegression)
- 클러스터링 (KMeans)


### 8. min_max_scaler

Min-Max 정규화: 0-1 범위로 변환합니다.

```yaml
preprocessor:
  steps:
    - type: min_max_scaler
```

**권장 사용:**

- TabNet
- 신경망 모델


### 9. robust_scaler

중앙값과 사분위수 기반 스케일링으로, 이상치에 덜 민감합니다.

```yaml
preprocessor:
  steps:
    - type: robust_scaler
```

**권장 사용:**

- 거래 금액 등 이상치가 많은 데이터
- 회귀 문제


### 10. ordinal_encoder

범주형 변수를 순서형 정수로 변환합니다.

```yaml
preprocessor:
  steps:
    - type: ordinal_encoder
      columns: ['merchant', 'city', 'job']
      handle_unknown: use_encoded_value
      unknown_value: -1
```

**파라미터:**

- `columns`: 적용할 컬럼 목록 (필수)
- `handle_unknown`: 미지 범주 처리 방식
  - `error`: 에러 발생
  - `use_encoded_value`: unknown_value로 대체
- `unknown_value`: 미지 범주 대체값 (handle_unknown이 use_encoded_value일 때)

**권장 사용:**

- High cardinality 범주형 컬럼 (merchant, city, job 등)
- 트리 기반 모델 (XGBoost, LightGBM, RandomForest)


### 11. one_hot_encoder

범주형 변수를 원-핫 벡터로 변환합니다.

```yaml
preprocessor:
  steps:
    - type: one_hot_encoder
      columns: ['category', 'gender', 'state']
```

**파라미터:**

- `columns`: 적용할 컬럼 목록 (필수)

**주의사항:**

- High cardinality 컬럼에는 사용하지 않음 (차원 폭발 위험)
- Low cardinality 컬럼에만 사용 권장 (10개 미만 고유값)


### 12. catboost_encoder

타겟 기반 인코딩으로, 정보 누수를 방지하면서 범주형 변수를 처리합니다.

```yaml
preprocessor:
  steps:
    - type: catboost_encoder
      columns: ['merchant', 'city']
      sigma: 0.05
```

**파라미터:**

- `columns`: 적용할 컬럼 목록 (필수)
- `sigma`: 노이즈 파라미터 (기본값: 0.05)

**주의사항:**

- 지도 학습 방식으로 타겟 변수(y) 필요
- 과적합 방지를 위해 교차 검증 권장


### 13. kbins_discretizer

연속형 변수를 구간화하여 범주형으로 변환합니다.

```yaml
preprocessor:
  steps:
    - type: kbins_discretizer
      columns: ['age', 'income']
      n_bins: 5
      strategy: quantile  # uniform, quantile, kmeans
      encode: ordinal  # ordinal, onehot
```

**파라미터:**

- `columns`: 적용할 컬럼 목록 (필수)
- `n_bins`: 구간 수 (기본값: 5)
- `strategy`: 구간화 전략
  - `uniform`: 동일 폭
  - `quantile`: 동일 빈도
  - `kmeans`: K-means 클러스터링
- `encode`: 인코딩 방식
  - `ordinal`: 순서형 정수
  - `onehot`: 원-핫 벡터


### 14. polynomial_features

기존 피처들의 고차항(다항식)을 생성합니다.

```yaml
preprocessor:
  steps:
    - type: polynomial_features
      columns: ['lat', 'long']
      degree: 2
      interaction_only: false
```

**파라미터:**

- `columns`: 적용할 컬럼 목록 (필수)
- `degree`: 다항식 차수 (기본값: 2)
- `interaction_only`: 상호작용 항만 생성 (기본값: false)
- `include_bias`: 상수항 포함 (기본값: false)


## 모델별 권장 전처리 조합

### 트리 기반 모델 (XGBoost, LightGBM, RandomForest)

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    # High cardinality: ordinal_encoder
    - type: ordinal_encoder
      columns: ['merchant', 'city', 'job']
      handle_unknown: use_encoded_value
      unknown_value: -1
    # Low cardinality: one_hot_encoder (선택)
    - type: one_hot_encoder
      columns: ['gender', 'state']
```

### 선형 모델 (LinearRegression, LogisticRegression)

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: ordinal_encoder
      columns: ['merchant', 'city', 'job']
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: one_hot_encoder
      columns: ['category', 'gender', 'state']
    - type: standard_scaler
```

### 딥러닝 모델 (TabNet, FT-Transformer, LSTM)

> **주의**: 딥러닝 모델은 **숫자형 데이터만 지원**합니다.
> 문자열(범주형) 컬럼이 있는 경우 반드시 인코더를 추가하세요.

```yaml
# 문자열 컬럼이 없는 경우
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: standard_scaler  # 또는 min_max_scaler

# 문자열 컬럼이 있는 경우 (필수)
preprocessor:
  steps:
    - type: ordinal_encoder  # 또는 onehot_encoder
    - type: simple_imputer
      strategy: median
    - type: standard_scaler
```

### 클러스터링 (KMeans)

```yaml
preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
    - type: standard_scaler  # 필수: 거리 기반 알고리즘
```

### 시계열 모델 (ARIMA, LSTM)

```yaml
preprocessor:
  steps:
    - type: forward_fill     # 시계열 결측값 순방향 채움
      limit: 3
    - type: backward_fill    # 시작 부분 결측값 처리
      limit: 3
    - type: interpolation    # 남은 결측값 보간
      method: linear
    - type: standard_scaler  # LSTM의 경우 스케일링 권장
```


## High Cardinality 컬럼 처리 가이드

| 고유값 수 | 권장 인코더 | 이유 |
|-----------|------------|------|
| < 10 | `one_hot_encoder` | 차원 증가가 적음 |
| 10 - 100 | `ordinal_encoder` | 차원 폭발 방지 |
| > 100 | `ordinal_encoder` 또는 `catboost_encoder` | 트리 모델에 적합 |

**fraud 데이터 예시:**
- Low cardinality: `gender` (2), `state` (50 미만), `category` (15 미만)
- High cardinality: `merchant` (500+), `city` (500+), `job` (300+), `first`, `last`, `street`


## 전처리 순서 권장

1. **결측값 처리** - 항상 첫 번째
   - 일반: `simple_imputer`, `drop_missing`, `constant_fill`
   - 시계열: `forward_fill`, `backward_fill`, `interpolation`
2. **인코딩** (`ordinal_encoder`, `one_hot_encoder`) - 범주형 → 숫자형
3. **피처 생성** (`polynomial_features`) - 선택적
4. **스케일링** (`standard_scaler`, `min_max_scaler`) - 마지막

```yaml
preprocessor:
  steps:
    - type: simple_imputer        # 1. 결측값 처리
      strategy: median
    - type: ordinal_encoder       # 2. 인코딩
      columns: ['merchant', 'city']
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: one_hot_encoder       # 2. 인코딩
      columns: ['gender']
    - type: standard_scaler       # 3. 스케일링 (마지막)
```
