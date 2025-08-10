# Modern ML Pipeline: 개발자 가이드

이 문서는 Modern ML Pipeline을 사용하여 **새로운 모델을 추가하고, 실험하며, 배포하는 전체 과정**을 안내하는 심화 가이드입니다. `README.md`를 통해 기본 설정을 마쳤다고 가정합니다.

## 1. 핵심 철학: "Recipe 중심 개발"

우리 파이프라인의 가장 중요한 설계 철학은 **"모델의 모든 논리적 정의는 YAML 레시피 파일 안에서 완결된다"**는 것입니다.

- **당신의 핵심 작업 공간**: 대부분의 경우, 당신은 레시피 YAML 파일과 관련 SQL 파일만 생성하거나 수정하게 될 것입니다.
- **수정할 필요 없는 엔진**: `src/` 디렉토리의 코드는 파이프라인의 "엔진"입니다. 새로운 모델을 추가하기 위해 **코드 수정이나 팩토리 등록이 전혀 필요 없습니다.**

---

## 1.5. 핵심 컨셉: 동적 레시피 가이드 및 자동 검증

우리 파이프라인은 개발자의 실수를 줄이고 생산성을 극대화하기 위한 두 가지 강력한 지능형 기능을 내장하고 있습니다.

### 1.5.1. `guide` 명령어로 레시피 작성 시작하기

**문제점**: 새로운 모델을 사용할 때마다 어떤 하이퍼파라미터를 사용해야 하는지, 어떤 타입을 써야 하는지 문서를 찾아봐야 하는 번거로움이 있습니다.

**해결책**: `guide` 명령어는 모델의 클래스 경로만 알려주면, 해당 모델에 대한 완벽한 레시피 초안을 즉시 생성해 줍니다.

**상세 사용법**
```bash
# RandomForestClassifier에 대한 가이드 생성
uv run python main.py guide sklearn.ensemble.RandomForestClassifier

# XGBRegressor에 대한 가이드 생성
uv run python main.py guide xgboost.XGBRegressor
```

**출력 예시 (`... guide xgboost.XGBRegressor`)**
```yaml
# xgboost.XGBRegressor 모델을 위한 레시피 가이드
# ...
model:
  class_path: "xgboost.XGBRegressor"
  data_interface:
    task_type: "regression"
    # ...
  hyperparameters:
    n_estimators: 100  # type: <class 'int'>, default: 100
    learning_rate: 0.3 # type: <class 'float'>, default: 0.3
    # ... (그 외 모든 파라미터와 기본값) ...
# ...
evaluation:
  metrics:
    - "mse"
    - "rmse"
    - "mae"
    - "r2"
# ...
```
**원리**: 이 기능은 파이썬의 `inspect` 모듈을 사용해 모델 클래스의 `__init__` 시그니처를 실시간으로 분석하여 동작합니다.

### 1.5.2. 실패-조기(Fail-Fast) 자동 검증

**문제점**: 레시피의 사소한 오타나 논리적 오류(예: 회귀 모델에 분류 지표 사용)는 데이터 로딩과 전처리가 모두 끝난 후에야 발견되어 많은 시간을 낭비하게 만듭니다.

**해결책**: 우리 시스템은 `train` 명령어 실행 즉시, 설정 로딩 단계에서 레시피의 논리적 일관성을 자동으로 검증합니다.

**자동 검증 항목**
1.  **모델-하이퍼파라미터 호환성 검증**: 레시피에 정의된 `hyperparameters`가 `model.class_path`에 명시된 실제 모델이 수용할 수 있는 유효한 파라미터인지 검증합니다.
    - **오류 예시**: `ValueError: 잘못된 하이퍼파라미터: 'invalid_param'은(는) ... 유효한 파라미터가 아닙니다. 사용 가능한 파라미터: [...]`
2.  **태스크-평가지표 호환성 검증**: `data_interface.task_type`과 `evaluation.metrics`에 명시된 평가지표가 서로 호환되는지 검증합니다.
    - **오류 예시**: `ValueError: 평가 지표 'roc_auc'은(는) 'regression' 태스크 타입과 호환되지 않습니다. 'regression'에 사용 가능한 지표: ['mse', 'rmse', ...]`

**원리**: 이 기능은 `Pydantic`의 `model_validator`를 활용하여 구현되었습니다. 시스템은 설정을 단순한 딕셔너리가 아닌, 자체 검증 로직을 가진 똑똑한 `Settings` 객체로 로드하기 때문에 이러한 강력한 사전 검증이 가능합니다.


## 2. 환경별 점진적 발전 경로

### 2.1. LOCAL 환경: 빠른 실험의 성지

**철학**: "제약은 단순함을 낳고, 단순함은 집중을 낳는다"

```bash
# LOCAL 환경 특징
APP_ENV=local uv run python main.py train --recipe-file recipes/my_model.yaml

특징:
✅ 즉시 실행 가능 (외부 의존성 없음)
✅ 파일 기반 데이터 로딩
✅ Pass-through 피처 증강 (Feature Store 비활성화)
❌ API 서빙 시스템적 차단
❌ 복잡한 인프라 기능 제한
```

### 2.2. DEV 환경: 완전한 기능의 실험실

**철학**: "모든 기능이 완전히 작동하는 안전한 실험실"

```bash
# mmp-local-dev 인프라 설정 (최초 1회)
git clone https://github.com/wooshikwon/mmp-local-dev.git ../mmp-local-dev
cd ../mmp-local-dev && docker-compose up -d

# DEV 환경에서 실행
APP_ENV=dev uv run python main.py train --recipe-file recipes/my_model.yaml

특징:
✅ 완전한 Feature Store (PostgreSQL + Redis)
✅ 팀 공유 MLflow 서버
✅ API 서빙 지원 (serving.enabled: true 필요)
✅ 실제 운영과 동일한 아키텍처
```

### 2.3. PROD 환경: 확장성과 안정성의 정점

**철학**: "성능, 안정성, 관측 가능성의 삼위일체"

```bash
# PROD 환경에서 실행
APP_ENV=prod uv run python main.py train --recipe-file recipes/my_model.yaml

특징:
✅ BigQuery/Snowflake 대용량 데이터 처리
✅ Redis Labs 고성능 Feature Store
✅ Cloud Run 서버리스 배포
✅ 무제한 확장성과 운영급 모니터링
```

---

## 3. Recipe 파일 완전 가이드

### 3.1. 기본 Recipe 구조

```yaml
# recipes/models/classification/my_classifier.yaml
name: "my_classifier_experiment"

model:
  # 1. 모델 클래스 동적 임포트
  class_path: "sklearn.ensemble.RandomForestClassifier"
  
  # 2. 하이퍼파라미터 정의
  hyperparameters:
    n_estimators: 100                              # 고정값
    max_depth: {type: "int", low: 3, high: 20}     # 자동 최적화 범위
    min_samples_split: {type: "int", low: 2, high: 10}
    
  # 3. 데이터 로딩 설정 (Point-in-Time 스키마 필수)
  loader:
    name: "default_loader"
    source_uri: "data/classification_dataset.parquet"
    adapter: storage
    entity_schema:
      entity_columns: ["user_id"]
      timestamp_column: "event_ts"
    
  # 4. 피처 증강 설정 (선택사항)
  augmenter:
    type: "pass_through"  # LOCAL: pass_through, DEV/PROD: feature_store
    
  # 5. 전처리 설정 (예시)
  preprocessor:
    name: "default_preprocessor"
    params:
      exclude_cols: ["user_id", "event_ts", "target"]
      
  # 6. 모델 인터페이스 설정
  data_interface:
    task_type: "classification"
    target_column: "target"

# 7. 자동 하이퍼파라미터 최적화 (선택사항)
hyperparameter_tuning:
  enabled: true
  n_trials: 50
  metric: "roc_auc"
  direction: "maximize"
  
# 8. 평가 설정
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  validation:
    method: "train_test_split"
    test_size: 0.2
    stratify: true
    random_state: 42
```

### 3.2. 지원하는 모델 클래스

- Scikit-learn, XGBoost, LightGBM, CatBoost, CausalML 등 주요 라이브러리를 직접 `class_path`로 임포트하여 사용합니다.

---

## 4. 동적 SQL 템플릿 (Jinja2) 가이드

### 4.1. Jinja 템플릿 기본 사용법

허용된 컨텍스트 키만 사용할 수 있습니다: `start_date`, `end_date`, `target_date`, `period`, `include_target`.

**템플릿 파일 작성**
```sql
-- recipes/sql/user_behavior_spine.sql.j2
SELECT 
    user_id,
    event_timestamp,
    CASE 
        WHEN total_spent > 1000 THEN 1 ELSE 0 
    END as include_flag,
    '{{ target_date }}' as snapshot_date
FROM user_events 
WHERE event_date BETWEEN '{{ start_date }}' AND '{{ end_date }}'
```

**Recipe에서 템플릿 사용**
```yaml
# recipes/models/classification/templated_model.yaml
model:
  loader:
    name: "default_loader"
    source_uri: "recipes/sql/user_behavior_spine.sql.j2"  # .j2 확장자
    adapter: sql
    entity_schema:
      entity_columns: ["user_id"]
      timestamp_column: "event_timestamp"

# CLI에서 파라미터 전달
# uv run python main.py train \
#   --recipe-file recipes/models/classification/templated_model.yaml \
#   --context-params '{"target_date": "2024-01-01", "start_date": "2023-12-01", "end_date": "2024-01-01"}'
```

### 4.2. Jinja 템플릿 보안 규칙

- 템플릿 렌더링은 `render_template_from_file|string`을 통해 수행됩니다.
- 컨텍스트 파라미터 화이트리스트를 엄격히 적용합니다.
- 렌더링 결과 SQL에 대해 다음을 검증합니다:
  - SELECT \* 금지
  - DDL/DML 금칙어 차단: DROP/DELETE/UPDATE/INSERT/ALTER/…
  - LIMIT 미존재 경고(정보성)

---

## 5. 통합 어댑터 시스템

### 5.1. SqlAdapter: 모든 SQL 데이터베이스 통합

- 지원 DB: PostgreSQL, BigQuery, Snowflake, MySQL, SQLite
- 실행 전 보안 가드 적용: `SELECT *` 금지, DDL/DML 금칙어 차단, LIMIT 경고

```yaml
# config/base.yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "postgresql://user:pass@localhost:5432/db"
```

### 5.2. StorageAdapter: 통합 파일 시스템

- 지원 스토리지: 로컬, GCS, S3, Azure Blob (fsspec 스킴 기반)

```yaml
model:
  loader:
    source_uri: "file://data/local_file.parquet"
    adapter: storage
```

### 5.3. FeastAdapter: Feature Store 통합

```yaml
model:
  augmenter:
    type: "feature_store"
    features:
      - feature_namespace: "user_demographics"
        features: ["age", "gender", "country", "city"]
      - feature_namespace: "user_behavior"
        features: ["total_purchases", "avg_session_time", "last_login"]
```

---

## 6. 자동 하이퍼파라미터 최적화 (HPO)

- 레시피에서 탐색 공간을 정의하고, 설정에서 시간/병렬화를 제어합니다.
- 내부적으로 `OptunaIntegration`을 사용해 `create_study`와 `suggest_hyperparameters`를 수행합니다.

```yaml
model:
  hyperparameters:
    n_estimators: {type: "int", low: 50, high: 1000}
    learning_rate: {type: "float", low: 0.01, high: 0.3, log: true}

hyperparameter_tuning:
  enabled: true
  n_trials: 100
  metric: "roc_auc"
  direction: "maximize"
```

---

## 7. 고급 Recipe 패턴

- 멀티 태스크, 커스텀 전처리, 앙상블 구성은 표준 레시피 문법으로 표현할 수 있습니다.

---

## 8. 배포 및 모니터링

### 8.1. 모델 배포 전략

```bash
# 1. LOCAL에서 프로토타입
APP_ENV=local uv run python main.py train --recipe-file recipes/my_model.yaml

# 2. DEV에서 완전 검증
APP_ENV=dev uv run python main.py train --recipe-file recipes/my_model.yaml

# 3. PROD에서 운영 배포
APP_ENV=prod uv run python main.py train --recipe-file recipes/my_model.yaml

# 4. API 서빙 시작 (DEV/PROD)
RUN_ID="<your-run-id>"
uv run python main.py serve-api --run-id $RUN_ID
```

### 8.2. API 스키마와 응답

- `/predict` 응답은 `MinimalPredictionResponse` 스키마를 따릅니다.
- 필드: `prediction`, `model_uri`

예시
```bash
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "event_ts": "2024-01-01T00:00:00"}'
# {"prediction": 0.87, "model_uri": "runs:/<run-id>/model"}
```

---

## 9. 트러블슈팅 가이드

### 9.1. 학습 관련 문제

```bash
# 로그 레벨 상승으로 상세 정보 확인
export LOG_LEVEL=DEBUG
uv run python main.py train --recipe-file recipes/my_model.yaml
```

**메모리 부족**
```yaml
model:
  loader:
    # 샘플링 예시 (SELECT * 금지)
    source_uri: |
      SELECT user_id, event_ts, feature_1
      FROM large_table
      LIMIT 10000
    adapter: sql
```

### 9.2. Feature Store 관련 문제

- Feast/Redis/PostgreSQL 연결 상태를 우선 확인하세요.

### 9.3. API 서빙 관련 문제

```bash
# MLflow run 존재 확인
mlflow runs describe --run-id $RUN_ID

# 모델 아티팩트 확인
ls mlruns/0/$RUN_ID/artifacts/model/
```

---

## 10. 성능 최적화 가이드

### 10.1. 학습 성능 최적화

```yaml
# Parquet 형식 권장
model:
  loader:
    source_uri: "data/dataset.parquet"
    adapter: storage
```

```yaml
# Pruning 사용 예시
hyperparameter_tuning:
  enabled: true
  n_trials: 100
  pruning:
    enabled: true
```

### 10.2. 추론 성능 최적화

```bash
# 배치 추론
uv run python main.py batch-inference --run-id $RUN_ID
```

```bash
# API 서빙 (멀티 워커 예시)
# gunicorn serving.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 11. 고급 확장 가이드

- 커스텀 어댑터/평가자/전처리 스텝은 레지스트리 패턴에 따라 확장 가능합니다.

---

이 가이드를 통해 Modern ML Pipeline의 기능을 활용하여 확장 가능한 ML 시스템을 구축할 수 있습니다.
