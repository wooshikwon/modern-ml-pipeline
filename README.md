# 🚀 모던 ML 파이프라인

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/your-username/modern-ml-pipeline)

**아름다운 콘솔 출력, MLflow 통합, FastAPI 서빙을 제공하는 프로덕션 레디 YAML 기반 ML 파이프라인 프레임워크**

```
███╗   ███╗███╗   ███╗████████╗ 
████╗ ████║████╗ ████║██╔════██╗
██╔████╔██║██╔████╔██║████████╔╝
██║╚██╔╝██║██║╚██╔╝██║██╔═════╝ 
██║ ╚═╝ ██║██║ ╚═╝ ██║██║     
╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  
```

## 🌟 주요 기능

- **📝 YAML 기반 설정** - 간단한 YAML 파일로 ML 워크플로우 정의
- **🏭 팩토리 패턴 아키텍처** - 자동 등록이 가능한 플러그인 컴포넌트
- **🎨 리치 콘솔 인터페이스** - 진행 상황 추적이 가능한 계층적 출력
- **📊 MLflow 통합** - 완전한 실험 추적 및 모델 레지스트리
- **🚀 FastAPI 서빙** - 자동 모델 로딩이 가능한 프로덕션 레디 API 서버  
- **⚡ CLI 우선 설계** - 모든 작업을 위한 대화형 명령줄 인터페이스
- **🔧 컴포넌트 레지스트리** - 어댑터, 프로세서, 모델을 위한 확장 가능한 시스템
- **🌍 멀티 환경 지원** - local, dev, prod 환경에서 원활한 배포
- **🤖 사용자 정의 모델** - FT Transformer 및 사용자 정의 모델 내장 지원
- **🔄 전체 파이프라인 생명주기** - 데이터 로딩부터 모델 서빙까지 하나의 프레임워크로

## ⚡ 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/modern-ml-pipeline.git
cd modern-ml-pipeline

# uv로 설치 (권장)
uv init --python 3.11
uv add --editable .

# 또는 pip로 설치
pip install -e .
```

### 기본 사용법

```bash
# 새로운 ML 프로젝트 초기화
mmp init

# 환경 설정 config YAML 생성
mmp get-config

# conofig YAML 기반 헬스 체크
mmp system-check --config-path configs/my-dev.yaml

# (데이터 준비 후) 모델 학습을 위한 Recipe YAML 생성
mmp get-recipe

# 모델 학습
mmp train --config-path configs/my-dev.yaml --recipe-path recipe/my-recipe.yaml --data-path sql/my-sql.sql --params '{"execution_date":"2025-09-01"}'

# 배치 추론 실행
mmp batch-inference --config-path configs/my-dev.yaml --run-id <run_id> --data-path data/test.csv

# API 서버 시작
mmp serve-api --host 0.0.0.0 --port 8000
```

## 🧠 핵심 개념

### 1. 레시피 기반 설정

YAML 레시피로 ML 워크플로우를 정의하세요:

```yaml
# recipes/classification_model.yaml
name: "고객_이탈_예측"
task_choice: "classification"

data:
  loader:
    source_uri: "data/customer_data.csv"
  data_interface:
    target_column: "churn"
    feature_columns: ["age", "tenure", "monthly_charges"]

model:
  class_path: "sklearn.ensemble.RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

preprocessor:
  steps:
    - type: "scaler"
      columns: ["age", "monthly_charges"]
    - type: "encoder"
      columns: ["category"]
```

### 2. 팩토리 패턴 아키텍처

모든 컴포넌트는 중앙화된 팩토리를 통해 생성됩니다:

```python
from src.factory import Factory
from src.settings import load_settings

# 설정 로드
settings = load_settings("config/local.yaml", "recipes/my_model.yaml")

# 팩토리 인스턴스 생성
factory = Factory(settings)

# 컴포넌트 자동 생성
data_adapter = factory.create_data_adapter()
model = factory.create_model()
preprocessor = factory.create_preprocessor()
evaluator = factory.create_evaluator()
```

### 3. 컴포넌트 레지스트리 시스템

프레임워크는 플러그인 기반 아키텍처를 사용합니다:

```python
# 컴포넌트는 자동으로 등록됩니다
from src.components.adapter import AdapterRegistry
from src.components.evaluator import EvaluatorRegistry

# 사용 가능한 컴포넌트 목록
print(AdapterRegistry.list_adapters())
print(EvaluatorRegistry.list_evaluators())

# 이름으로 컴포넌트 생성
sql_adapter = AdapterRegistry.create("sql", settings)
classifier_evaluator = EvaluatorRegistry.create("classification", data_interface)
```

## ⚡ CLI 명령어

### 프로젝트 관리
```bash
mmp init                    # 프로젝트 구조 초기화
mmp get-config             # 대화형 설정 파일 생성
mmp get-recipe            # 대화형 레시피 생성  
mmp system-check          # 시스템 연결 확인
```

### ML 파이프라인 작업
```bash
mmp train                  # 학습 파이프라인 실행
mmp batch-inference       # 배치 추론 실행
mmp serve-api             # FastAPI 서버 시작
```

### 컴포넌트 탐색
```bash
mmp list adapters         # 사용 가능한 데이터 어댑터 표시
mmp list evaluators      # 사용 가능한 평가기 표시
mmp list preprocessors   # 사용 가능한 전처리기 표시
mmp list models          # 사용 가능한 모델 타입 표시
```

## 🏗️ 아키텍처 개요

```
├── 🖥️ CLI 인터페이스 (typer)
│   ├── 프로젝트 관리 명령어
│   ├── 파이프라인 실행 명령어  
│   └── 컴포넌트 탐색 명령어
│
├── 🏭 팩토리 레이어
│   ├── 컴포넌트 생성 및 캐싱
│   ├── 레지스트리 통합
│   └── 의존성 해결
│
├── ⚙️ 컴포넌트 아키텍처
│   ├── 🔌 어댑터 (SQL, Storage, BigQuery, Feast)
│   ├── 🔄 프로세서 (Scalers, Encoders, Imputers)
│   ├── 📊 평가기 (Classification, Regression, Timeseries)
│   ├── 🎯 훈련기 (Default, Hyperparameter Tuning)
│   ├── 📡 페처 (Feature Store, Pass-through)
│   └── 🗂️ 데이터 핸들러 (Tabular, Timeseries)
│
├── ⚡ 파이프라인 레이어  
│   ├── 학습 파이프라인
│   ├── 추론 파이프라인
│   └── 서빙 파이프라인
│
├── ⚙️ 설정 시스템
│   ├── 환경 설정 (local, dev, prod)
│   ├── 모델 레시피 (YAML 기반)
│   └── 런타임 설정
│
└── 🎨 콘솔 통합
    ├── 리치 콘솔 매니저
    ├── 진행 상황 추적
    └── 계층적 출력
```

## 🔧 컴포넌트 시스템

### 데이터 어댑터
- **SQL 어댑터** - 직접 데이터베이스 연결
- **스토리지 어댑터** - 로컬 파일, S3, GCS
- **BigQuery 어댑터** - Google BigQuery 통합
- **Feast 어댑터** - 피처 스토어 통합

### 전처리기
- **스케일러** - StandardScaler, MinMaxScaler, RobustScaler
- **인코더** - OneHotEncoder, LabelEncoder, TargetEncoder
- **임퓨터** - SimpleImputer, KNNImputer
- **피처 생성기** - Polynomial, Interaction features

### 평가기
- **분류** - Accuracy, Precision, Recall, F1, ROC-AUC
- **회귀** - MSE, MAE, R², MAPE
- **시계열** - SMAPE, MASE, seasonal metrics

### 사용자 정의 모델
- **FT Transformer** - 테이블형 데이터를 위한 Feature Tokenizer Transformer
- **LSTM 시계열** - 시계열을 위한 순환 신경망
- **Scikit-learn 모델** - 모든 sklearn 추정기 지원
- **XGBoost, CatBoost, LightGBM** - 그래디언트 부스팅 프레임워크

## 🎨 리치 콘솔 출력

프레임워크는 아름다운 계층적 콘솔 인터페이스를 제공합니다:

```
🚀 학습 파이프라인
환경: local | 작업: classification | 모델: RandomForestClassifier

📊 데이터 로딩
✅ 데이터 로드 완료 (1000 행, 15 열)

⚡ 전처리 파이프라인 구축
🔍 1단계: scaler on ['age', 'income']
   🎯 전체 적용: 2개 열
📊 데이터 변환 완료 (1000 행, 15 열)

🏭 팩토리 초기화: customer_churn_model
✅ 데이터 어댑터 생성: storage
✅ 모델 생성: RandomForestClassifier
✅ 평가기 생성: classification

🤖 모델 학습
████████████████████████████████████████ 100%

📊 모델 평가
┏━━━━━━━━━━━┳━━━━━━━━━┓
┃ 메트릭     ┃ 값      ┃
┡━━━━━━━━━━━╇━━━━━━━━━┩
│ Accuracy  │ 0.8542  │
│ Precision │ 0.8123  │
│ Recall    │ 0.8756  │
│ F1-Score  │ 0.8428  │
└───────────┴─────────┘

🏁 학습 파이프라인 완료
```

## ⚙️ 설정

### 환경 설정 (`config/local.yaml`)

```yaml
environment:
  name: "local"
  
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "my_experiment"

data_source:
  adapter_type: "storage"
  config:
    base_path: "./data"

feature_store:
  provider: "feast"
  config:
    feature_server_url: "localhost:6566"
```

### 모델 레시피 (`recipes/classification.yaml`)

```yaml
name: "분류_모델"
task_choice: "classification"

data:
  loader:
    source_uri: "data/train.csv"
  data_interface:
    target_column: "target"
    entity_columns: ["user_id"]
    feature_columns: ["feature_1", "feature_2"]

model:
  class_path: "sklearn.ensemble.RandomForestClassifier"
  hyperparameters:
    tuning_enabled: true
    n_trials: 50
    optimization_metric: "f1_weighted"
    
preprocessor:
  steps:
    - type: "imputer"
      strategy: "median"
      columns: ["numerical_features"]
    - type: "encoder"  
      columns: ["categorical_features"]
```

## ⚡ 고급 사용법

### 사용자 정의 모델 통합

```python
# 사용자 정의 모델 등록
from src.components.trainer import TrainerRegistry
from sklearn.base import BaseEstimator

class CustomModel(BaseEstimator):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def fit(self, X, y):
        # 사용자 정의 학습 로직
        return self
    
    def predict(self, X):
        # 사용자 정의 예측 로직
        return predictions

# 레시피에서 사용
model:
  class_path: "my_models.CustomModel"
  hyperparameters:
    param1: 2.5
```

### Timeseries 규약 및 Feature Store 가이드

- **Timeseries 필수 항목**: `task_choice: timeseries`인 레시피는 `data.data_interface.timestamp_column`을 반드시 지정해야 합니다. 누락 시 Validator가 PR에서 실패합니다.
- **카탈로그-데이터핸들러 매칭**: 모델 카탈로그의 `data_handler` 선언을 우선합니다. 예: LSTM 시계열은 `data_handler: deeplearning`이 정상입니다.
- **Feature Store 사용 시**: `data.fetcher.type: feature_store`일 때 `data.fetcher.timestamp_column` 지정을 권장합니다(PIT join 기준). 레시피 빌더와 템플릿에서 해당 항목을 안내합니다.

### 피처 스토어 통합

```python
# 피처 스토어 페처 설정
data:
  fetcher:
    type: "feature_store"
    timestamp_column: "event_timestamp"
    features:
      - "user_features:age"
      - "user_features:income"
      - "transaction_features:amount"
```

### 하이퍼파라미터 튜닝

```python
# Optuna 기반 튜닝 활성화
model:
  hyperparameters:
    tuning_enabled: true
    n_trials: 100
    timeout: 3600  # 1시간
    optimization_metric: "f1_weighted"
    direction: "maximize"
    
    # 고정 파라미터
    fixed:
      random_state: 42
    
    # 튜닝 가능한 파라미터  
    tunable:
      n_estimators: [50, 100, 200]
      max_depth: [3, 5, 7, 10]
      min_samples_split: [2, 5, 10]
```

## 📊 MLflow 통합

프레임워크는 완전한 MLflow 통합을 제공합니다:

- **자동 실험 추적** - 모든 실행이 자동으로 기록됨
- **모델 레지스트리** - 전체 메타데이터와 함께 모델 저장
- **아티팩트 관리** - 코드, 데이터, 모델 버전 관리
- **모델 서빙** - MLflow 레지스트리에서 직접 로딩
- **실험 비교** - 풍부한 메트릭 및 파라미터 로깅

```python
# MLflow 아티팩트 접근
import mlflow

# 학습된 모델 로드
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

# 실행 정보 가져오기
run = mlflow.get_run(run_id)
print(f"정확도: {run.data.metrics['accuracy']}")
```

## 🎯 API 서빙

프로덕션 레디 FastAPI 서버를 시작하세요:

```bash
# 서버 시작
mmp serve-api --host 0.0.0.0 --port 8000

# 사용 가능한 API 엔드포인트:
# POST /predict - 단일 예측
# POST /batch-predict - 배치 예측  
# GET /health - 헬스 체크
# GET /model-info - 모델 메타데이터
```

### API 사용법

```python
import httpx

# 단일 예측
response = httpx.post("http://localhost:8000/predict", 
    json={"features": {"age": 35, "income": 50000}})

# 배치 예측
response = httpx.post("http://localhost:8000/batch-predict",
    json={"instances": [
        {"age": 35, "income": 50000},
        {"age": 42, "income": 75000}
    ]})
```

## 🚀 배포

### Docker 배포

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8000
CMD ["mmp", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
```

### 환경 변수

```bash
# MLflow 설정
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
export MLFLOW_EXPERIMENT_NAME=production

# 피처 스토어
export FEAST_FEATURE_SERVER_URL=your-feast-server:6566

# 모델 설정  
export MODEL_RUN_ID=your-production-run-id
```

## 🧪 개발

### 테스트 실행

```bash
# 개발 의존성 설치
uv add --group dev

# 모든 테스트 실행
uv run pytest

# 커버리지와 함께 실행
uv run pytest --cov=src --cov-report=html

# 특정 테스트 카테고리 실행
uv run pytest tests/unit/           # 단위 테스트
uv run pytest tests/integration/   # 통합 테스트
```

### 코드 품질

```bash
# 코드 포맷팅
uv run black src/
uv run isort src/

# 코드 린팅  
uv run ruff check src/
uv run mypy src/

# 보안 스캔
uv run bandit -r src/
uv run safety check
```

## 🤝 기여하기

1. 저장소를 포크하세요
2. 기능 브랜치를 만드세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m '놀라운 기능 추가'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 여세요

### 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/modern-ml-pipeline.git
cd modern-ml-pipeline

# 개발 모드로 설치
uv add --group dev
uv run pre-commit install

# 커밋 전 테스트 실행
uv run pytest
```

## 📚 문서

- **[아키텍처 가이드](docs/architecture.md)** - 상세한 시스템 아키텍처
- **[컴포넌트 개발](docs/components.md)** - 사용자 정의 컴포넌트 생성  
- **[레시피 레퍼런스](docs/recipes.md)** - 완전한 레시피 설정
- **[배포 가이드](docs/deployment.md)** - 프로덕션 배포 전략
- **[API 레퍼런스](docs/api.md)** - 완전한 API 문서

## 🛠️ 문제 해결

### 일반적인 문제들

**MLflow 연결 오류**
```bash
# MLflow 서버 상태 확인
mmp system-check

# 설정 확인
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**누락된 의존성**
```bash
# 모든 extras 설치
uv add modern-ml-pipeline[all]

# 또는 특정 extras
uv add modern-ml-pipeline[ml-extras,cloud-extras]
```

**Python 버전 문제**
```bash
# Python 3.11 확인
uv python install 3.11
uv init --python 3.11
```

## ⚡ 성능

- **학습 속도** - 최적화된 데이터 로딩 및 전처리 파이프라인
- **메모리 효율성** - 지연 로딩 및 캐싱 전략
- **확장성** - 단일 머신 및 분산 설정 모두 지원
- **모니터링** - 내장된 성능 메트릭 및 로깅

## 🔒 보안

- **입력 유효성 검사** - Pydantic 스키마로 모든 입력 검증
- **SQL 인젝션 방지** - 매개변수화된 쿼리 및 ORM 사용
- **의존성 스캔** - `bandit` 및 `safety`를 이용한 정기적인 보안 감사
- **환경 격리** - 환경 간 명확한 분리

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 라이센스되어 있습니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **MLflow** - 실험 추적 및 모델 관리
- **FastAPI** - 고성능 API 프레임워크
- **Rich** - 아름다운 콘솔 출력
- **Typer** - 직관적인 CLI 개발
- **Pydantic** - 견고한 데이터 검증

---

**모던 ML 파이프라인 팀이 ❤️로 만들었습니다**

*ML 워크플로우에 혁명을 일으킬 준비가 되셨나요? 지금 바로 모던 ML 파이프라인을 시작하세요!*