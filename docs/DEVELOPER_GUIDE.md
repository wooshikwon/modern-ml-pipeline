# Modern ML Pipeline: 개발자 가이드

이 문서는 Modern ML Pipeline을 사용하여 **새로운 모델을 추가하고, 실험하며, 배포하는 전체 과정**을 안내하는 심화 가이드입니다. `README.md`를 통해 기본 설정을 마쳤다고 가정합니다.

## 1. 핵심 철학: "Recipe 중심 개발"

우리 파이프라인의 가장 중요한 설계 철학은 **"모델의 모든 논리적 정의는 YAML 레시피 파일 안에서 완결된다"**는 것입니다.

*   **당신의 핵심 작업 공간:** 대부분의 경우, 당신은 레시피 YAML 파일과 관련 SQL 파일만 생성하거나 수정하게 될 것입니다.
*   **수정할 필요 없는 엔진:** `src/` 디렉토리의 코드는 파이프라인의 "엔진"입니다. 새로운 모델을 추가하기 위해 **코드 수정이나 팩토리 등록이 전혀 필요 없습니다.**

---

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
✅ API 서빙 지원
✅ 실제 운영과 동일한 아키텍처
```

### 2.3. PROD 환경: 확장성과 안정성의 정점

**철학**: "성능, 안정성, 관측 가능성의 완벽한 삼위일체"

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
    
  # 3. 데이터 로딩 설정
  loader:
    name: "default_loader"
    source_uri: "data/classification_dataset.parquet"
    
  # 4. 피처 증강 설정 (선택사항)
  augmenter:
    type: "pass_through"  # LOCAL: pass_through, DEV/PROD: feature_store
    
  # 5. 전처리 설정
  preprocessor:
    name: "default_preprocessor"
    params:
      exclude_cols: ["user_id", "event_timestamp", "target"]
      
  # 6. 모델 인터페이스 설정
  data_interface:
    task_type: "classification"
    target_col: "target"

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

**Scikit-learn 모델**
```yaml
# 분류
class_path: "sklearn.ensemble.RandomForestClassifier"
class_path: "sklearn.linear_model.LogisticRegression"
class_path: "sklearn.svm.SVC"
class_path: "sklearn.naive_bayes.GaussianNB"

# 회귀
class_path: "sklearn.ensemble.RandomForestRegressor"
class_path: "sklearn.linear_model.LinearRegression"
class_path: "sklearn.linear_model.Ridge"
class_path: "sklearn.linear_model.Lasso"
```

**Gradient Boosting 모델**
```yaml
# XGBoost
class_path: "xgboost.XGBClassifier"
class_path: "xgboost.XGBRegressor"

# LightGBM
class_path: "lightgbm.LGBMClassifier"
class_path: "lightgbm.LGBMRegressor"

# CatBoost
class_path: "catboost.CatBoostClassifier"
class_path: "catboost.CatBoostRegressor"
```

**인과추론 모델 (CausalML)**
```yaml
class_path: "causalml.inference.meta.XGBTRegressor"
class_path: "causalml.inference.meta.TRegressor"
class_path: "causalml.inference.meta.SRegressor"
```

---

## 4. 동적 SQL 템플릿 (Jinja2) 완전 가이드

### 4.1. Jinja 템플릿 기본 사용법

**템플릿 파일 작성**
```sql
-- recipes/sql/user_behavior_spine.sql.j2
SELECT 
    user_id,
    session_id,
    event_timestamp,
    '{{ target_date }}' as snapshot_date,
    CASE 
        WHEN total_spent > {{ high_value_threshold | default(1000) }}
        THEN 1 ELSE 0 
    END as target
FROM user_events 
WHERE event_date BETWEEN '{{ start_date }}' AND '{{ end_date }}'
    AND event_type = '{{ event_type | default("purchase") }}'
{% if limit is defined %}
LIMIT {{ limit }}
{% endif %}
```

**Recipe에서 템플릿 사용**
```yaml
# recipes/models/classification/templated_model.yaml
model:
  loader:
    name: "default_loader"
    source_uri: "recipes/sql/user_behavior_spine.sql.j2"  # .j2 확장자

# CLI에서 파라미터 전달
# uv run python main.py train \
#   --recipe-file recipes/models/classification/templated_model.yaml \
#   --context-params '{"target_date": "2024-01-01", "start_date": "2023-12-01", "end_date": "2024-01-01", "high_value_threshold": 500, "limit": 10000}'
```

### 4.2. Jinja 템플릿 고급 기능

**조건부 로직**
```sql
-- recipes/sql/conditional_spine.sql.j2
SELECT user_id, session_id, event_timestamp
{% if include_features %}
    , feature1, feature2, feature3
{% endif %}
{% if task_type == "classification" %}
    , CASE WHEN conversion = 1 THEN 1 ELSE 0 END as target
{% elif task_type == "regression" %}
    , revenue as target
{% endif %}
FROM events
WHERE date = '{{ target_date }}'
```

**반복문과 리스트 처리**
```sql
-- recipes/sql/multi_feature_spine.sql.j2
SELECT 
    user_id,
    event_timestamp,
{% for feature in features %}
    {{ feature }}{{ "," if not loop.last }}
{% endfor %}
FROM feature_table
WHERE user_id IN (
{% for user_id in user_ids %}
    '{{ user_id }}'{{ "," if not loop.last }}
{% endfor %}
)
```

**Context 파라미터 사용 예시**
```bash
# 복잡한 파라미터 전달
uv run python main.py train \
  --recipe-file recipes/templated_advanced.yaml \
  --context-params '{
    "target_date": "2024-01-01",
    "task_type": "classification", 
    "include_features": true,
    "features": ["age", "income", "purchase_history"],
    "user_ids": ["user1", "user2", "user3"]
  }'
```

---

## 5. 통합 어댑터 시스템

### 5.1. SqlAdapter: 모든 SQL 데이터베이스 통합

**지원 데이터베이스**
- PostgreSQL
- BigQuery  
- Snowflake
- MySQL
- SQLite

**설정 예시**
```yaml
# config/base.yaml
data_adapters:
  adapters:
    sql:
      class_name: SqlAdapter
      config:
        connection_uri: "postgresql://user:pass@localhost:5432/db"
        # 또는: "bigquery://project-id/dataset"
        # 또는: "snowflake://user:pass@account/db/schema"
```

### 5.2. StorageAdapter: 통합 파일 시스템

**지원 스토리지**
- 로컬 파일 시스템
- Google Cloud Storage (GCS)
- Amazon S3
- Azure Blob Storage

**설정 예시**
```yaml
# config/base.yaml
data_adapters:
  adapters:
    storage:
      class_name: StorageAdapter
      config:
        # fsspec이 URI 스킴으로 자동 판단
        # "file://", "gs://", "s3://", "abfs://" 모두 지원
```

**사용 예시**
```yaml
# Recipe에서 다양한 스토리지 사용
model:
  loader:
    source_uri: "file://data/local_file.parquet"           # 로컬
    # source_uri: "gs://my-bucket/data/file.parquet"       # GCS
    # source_uri: "s3://my-bucket/data/file.parquet"       # S3
```

### 5.3. FeastAdapter: Feature Store 통합

**Feast 기반 피처 증강**
```yaml
# recipes/feature_store_model.yaml
model:
  augmenter:
    type: "feature_store"
    features:
      # 사용자 인구통계 피처
      - feature_namespace: "user_demographics"
        features: ["age", "gender", "country", "city"]
        
      # 사용자 행동 피처  
      - feature_namespace: "user_behavior"
        features: ["total_purchases", "avg_session_time", "last_login"]
        
      # 상품 피처
      - feature_namespace: "product_features"
        features: ["price", "category", "brand", "rating"]
```

**환경별 Feature Store 설정**
```yaml
# config/base.yaml - LOCAL
feature_store:
  feast_config:
    offline_store:
      type: "file"
    online_store:
      type: "sqlite"

# config/dev.yaml - DEV  
feature_store:
  feast_config:
    offline_store:
      type: "postgresql"
    online_store:
      type: "redis"
      
# config/prod.yaml - PROD
feature_store:
  feast_config:
    offline_store:
      type: "bigquery"
    online_store:
      type: "redis"
```

---

## 6. 자동 하이퍼파라미터 최적화 (HPO)

### 6.1. 기본 HPO 설정

**Recipe 레벨 설정 (실험 논리)**
```yaml
# recipes/hpo_model.yaml
model:
  hyperparameters:
    # Optuna 탐색 범위 정의
    n_estimators: {type: "int", low: 50, high: 1000}
    max_depth: {type: "int", low: 3, high: 20}
    learning_rate: {type: "float", low: 0.01, high: 0.3, log: true}
    subsample: {type: "float", low: 0.5, high: 1.0}

hyperparameter_tuning:
  enabled: true
  n_trials: 100           # "이 실험은 100번 시도할 가치가 있다"
  metric: "roc_auc"       # 최적화할 메트릭
  direction: "maximize"   # 방향 (maximize/minimize)
```

**Config 레벨 설정 (인프라 제약)**
```yaml
# config/dev.yaml - 개발 환경
hyperparameter_tuning:
  enabled: true
  timeout: 600          # 10분 제한 (개발 환경 자원 보호)
  
# config/prod.yaml - 운영 환경
hyperparameter_tuning:
  enabled: true  
  timeout: 7200         # 2시간까지 허용 (운영 환경 자원 활용)
  parallelization:
    n_jobs: 8           # 병렬 처리
```

### 6.2. HPO 결과 추적 및 분석

**MLflow에서 확인 가능한 정보**
```python
# 자동 로깅되는 HPO 관련 메트릭
- best_score: 0.945          # 달성한 최고 점수
- total_trials: 50           # 수행된 총 trial 수  
- optimization_time: 1847    # 총 최적화 시간 (초)
- pruned_trials: 12          # 조기 중단된 trial 수

# 자동 로깅되는 하이퍼파라미터
- n_estimators: 847          # 최적 파라미터들
- max_depth: 12
- learning_rate: 0.0847
```

**PyfuncWrapper에 저장되는 완전한 HPO 결과**
```python
wrapper.hyperparameter_optimization = {
    "enabled": True,
    "engine": "optuna",
    "best_params": {"n_estimators": 847, "max_depth": 12},
    "best_score": 0.945,
    "optimization_history": [...],  # 전체 탐색 과정
    "total_trials": 50,
    "pruned_trials": 12,
    "optimization_time": 1847
}
```

---

## 7. 고급 Recipe 패턴

### 7.1. 멀티 태스크 모델

```yaml
# recipes/multi_task_model.yaml
model:
  class_path: "sklearn.multioutput.MultiOutputRegressor"
  hyperparameters:
    estimator:
      class_path: "xgboost.XGBRegressor"
      hyperparameters:
        n_estimators: {type: "int", low: 100, high: 500}
        
  data_interface:
    task_type: "regression"
    target_col: ["target1", "target2", "target3"]  # 다중 타겟
```

### 7.2. 커스텀 전처리 파이프라인

```yaml
# recipes/custom_preprocessing.yaml
model:
  preprocessor:
    name: "custom_preprocessor"
    params:
      # 범주형 변수 처리
      categorical_features: ["category", "brand", "region"]
      categorical_strategy: "onehot"  # onehot, label, target
      
      # 수치형 변수 처리  
      numerical_features: ["price", "rating", "reviews"]
      numerical_strategy: "standard"  # standard, minmax, robust
      
      # 결측치 처리
      missing_strategy: "median"      # mean, median, mode, drop
      
      # 이상치 처리
      outlier_detection: true
      outlier_method: "iqr"          # iqr, zscore, isolation_forest
```

### 7.3. 앙상블 모델

```yaml
# recipes/ensemble_model.yaml
model:
  class_path: "sklearn.ensemble.VotingClassifier"
  hyperparameters:
    estimators:
      - name: "rf"
        estimator:
          class_path: "sklearn.ensemble.RandomForestClassifier"
          hyperparameters:
            n_estimators: {type: "int", low: 50, high: 200}
      - name: "xgb"  
        estimator:
          class_path: "xgboost.XGBClassifier"
          hyperparameters:
            n_estimators: {type: "int", low: 50, high: 200}
    voting: "soft"
```

---

## 8. 배포 및 모니터링

### 8.1. 모델 배포 전략

**단계별 배포**
```bash
# 1. LOCAL에서 프로토타입
APP_ENV=local uv run python main.py train --recipe-file recipes/my_model.yaml

# 2. DEV에서 완전 검증
APP_ENV=dev uv run python main.py train --recipe-file recipes/my_model.yaml

# 3. PROD에서 운영 배포
APP_ENV=prod uv run python main.py train --recipe-file recipes/my_model.yaml
RUN_ID="prod-run-id"

# 4. API 서빙 시작
APP_ENV=prod uv run python main.py serve-api --run-id $RUN_ID
```

### 8.2. 모델 성능 모니터링

**배치 추론으로 정기 평가**
```bash
# 주간 모델 성능 평가
RUN_ID="latest-model"
uv run python main.py batch-inference \
  --run-id $RUN_ID \
  --context-params '{"evaluation_date": "2024-01-07"}'
```

**A/B 테스트**
```bash
# 모델 A (기존)
uv run python main.py serve-api --run-id $MODEL_A_RUN_ID --port 8000

# 모델 B (신규)  
uv run python main.py serve-api --run-id $MODEL_B_RUN_ID --port 8001
```

---

## 9. 트러블슈팅 가이드

### 9.1. 학습 관련 문제

**하이퍼파라미터 최적화 실패**
```bash
# 로그 레벨 상승으로 상세 정보 확인
export LOG_LEVEL=DEBUG
uv run python main.py train --recipe-file recipes/my_model.yaml

# Optuna 데이터베이스 확인
ls ~/.optuna/  # 기본 SQLite 저장 위치
```

**메모리 부족 오류**
```yaml
# Recipe에서 데이터 크기 제한
model:
  loader:
    source_uri: "SELECT * FROM large_table LIMIT 10000"  # 샘플링
    
hyperparameter_tuning:
  n_trials: 10  # trial 수 감소
```

### 9.2. Feature Store 관련 문제

**Feast 초기화 실패**
```bash
# Feast 설정 확인
feast version
feast repo-config

# Redis 연결 확인
redis-cli ping

# PostgreSQL 연결 확인
psql -h localhost -p 5432 -U mlpipeline_user -d mlpipeline_db -c "SELECT 1;"
```

### 9.3. API 서빙 관련 문제

**모델 로딩 실패**
```bash
# MLflow run 존재 확인
mlflow runs describe --run-id $RUN_ID

# 모델 아티팩트 확인
ls mlruns/0/$RUN_ID/artifacts/model/
```

**동적 스키마 생성 실패**
```python
# SQL 파싱 직접 테스트
from src.utils.system.sql_utils import parse_select_columns

sql = "SELECT user_id, product_id FROM table"
columns = parse_select_columns(sql)
print(columns)  # ['user_id', 'product_id']
```

---

## 10. 성능 최적화 가이드

### 10.1. 학습 성능 최적화

**데이터 로딩 최적화**
```yaml
# Parquet 형식 사용 (CSV 대비 10x 빠름)
model:
  loader:
    source_uri: "data/dataset.parquet"  # ✅ 권장
    # source_uri: "data/dataset.csv"   # ❌ 느림
```

**하이퍼파라미터 최적화 효율화**
```yaml
hyperparameter_tuning:
  enabled: true
  n_trials: 100
  
  # Pruning으로 비효율적 trial 조기 중단
  pruning:
    enabled: true
    algorithm: "MedianPruner"
    n_startup_trials: 5
    n_warmup_steps: 10
```

### 10.2. 추론 성능 최적화

**배치 추론 최적화**
```bash
# 병렬 처리로 배치 크기 조정
export BATCH_SIZE=1000
uv run python main.py batch-inference --run-id $RUN_ID
```

**API 서빙 최적화**
```bash
# Gunicorn으로 다중 워커 실행
gunicorn serving.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

---

## 11. 고급 확장 가이드

### 11.1. 커스텀 어댑터 추가

```python
# src/utils/adapters/custom_adapter.py
from src.interface.base_adapter import BaseAdapter

class CustomAdapter(BaseAdapter):
    def __init__(self, settings, **kwargs):
        self.settings = settings
        
    def read(self, query, **kwargs):
        # 커스텀 데이터 소스에서 읽기
        pass
        
    def write(self, df, destination, **kwargs):
        # 커스텀 데이터 소스에 쓰기
        pass
```

```python
# src/engine/registry.py에 등록
def register_all_adapters():
    # ... 기존 어댑터들 ...
    
    try:
        from src.utils.adapters.custom_adapter import CustomAdapter
        AdapterRegistry.register("custom", CustomAdapter)
    except ImportError:
        logger.warning("CustomAdapter not available")
```

### 11.2. 커스텀 평가 메트릭 추가

```python
# src/components/evaluator.py 확장
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, y_true, y_pred, **kwargs):
        metrics = super().evaluate(y_true, y_pred, **kwargs)
        
        # 커스텀 메트릭 추가
        metrics["custom_metric"] = self._calculate_custom_metric(y_true, y_pred)
        return metrics
        
    def _calculate_custom_metric(self, y_true, y_pred):
        # 커스텀 계산 로직
        return custom_score
```

---

이 가이드를 통해 Modern ML Pipeline의 모든 고급 기능을 활용하여 강력하고 확장 가능한 ML 시스템을 구축할 수 있습니다. 추가 질문이나 도움이 필요하시면 언제든 문의해 주세요!
