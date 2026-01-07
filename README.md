# Modern ML Pipeline

YAML 설정 기반의 머신러닝 파이프라인 CLI 도구입니다.

코드를 수정하지 않고 **YAML 설정 파일**만으로 모델 학습부터 API 서빙까지 처리합니다.


## 주요 특징

- **설정 기반 (Config-driven)**: YAML만으로 실험을 정의
- **피처 스토어 연동**: Feature Store/Data Lake에 사전 정의된 피처 직접 활용
- **자동 실험 추적**: MLflow와 연동되어 모든 실험 결과와 모델이 자동 저장
- **Data Leakage 방지**: Train/Validation/Test/Calibration 4단계 분할 자동 처리
- **즉시 서빙**: 학습 완료 후 명령어 한 줄로 REST API 서버 기동


## 빠른 시작

### 1. 설치

**요구사항**: Python 3.11 또는 3.12

```bash
# Homebrew Python 사용 시
pipx install --python python3.11 modern-ml-pipeline

# pyenv 사용 시
pipx install --python ~/.pyenv/versions/3.11.10/bin/python modern-ml-pipeline
```

> **설치 전 준비**
> - Python 3.11: `brew install python@3.11` (Homebrew) 또는 `pyenv install 3.11.10` (pyenv)
> - pipx: `brew install pipx && pipx ensurepath` (macOS) 또는 `pip install pipx && pipx ensurepath`

상세 설치 옵션은 [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md)를 참고하세요.


### 2. 프로젝트 생성

```bash
mmp init my-project
```

생성되는 디렉토리 구조:

```text
my-project/
├── data/           # 학습/추론 데이터 파일
├── configs/        # 환경 설정 파일 (dev.yaml, prod.yaml 등)
├── recipes/        # 실험 레시피 파일
├── sql/            # SQL 쿼리 파일
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── .gitignore
```


### 3. 데이터 준비

학습에 사용할 데이터를 준비합니다. CSV 파일 또는 SQL 쿼리 파일을 사용할 수 있습니다.

**경로 규칙**

모든 파일 경로는 **프로젝트 루트 기준 상대 경로**로 지정합니다:

```bash
# 로컬 파일 (프로젝트 루트 기준)
mmp train ... -d data/train.csv        # data/ 디렉토리의 파일
mmp train ... -d sql/query.sql         # sql/ 디렉토리의 파일

# 클라우드 스토리지 (전체 경로 지정)
mmp train ... -d s3://bucket/data/train.csv
mmp train ... -d gs://bucket/data/train.csv
```

클라우드 스토리지 상세 설정은 [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md#클라우드-스토리지-설정)를 참고하세요.

**데이터 형식 요구사항**

- 각 행은 하나의 샘플을 나타냄
- `entity_columns`: 행을 식별하는 ID 컬럼 (학습에서 자동 제외)
- `target_column`: 예측 대상 컬럼
- 나머지 컬럼: 피처로 사용

Task별 상세 데이터 형식은 [Task 가이드](./docs/user/TASK_GUIDE.md)를 참고하세요.


### 4. 설정 파일 생성

#### Config 파일 (인프라 설정)

```bash
mmp get-config
```

**대화형 인터페이스**를 통해 MLflow, 스토리지, DB 연결 등 인프라 설정을 선택하고 `configs/{env}.yaml` 파일을 생성합니다.

상세 옵션은 [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md)를 참고하세요.


#### Recipe 파일 (실험 설정)

```bash
mmp get-recipe
```

**대화형 인터페이스**를 통해 Task, 모델, 전처리 등을 선택하고 `recipes/{name}.yaml` 파일을 생성합니다. 생성된 Recipe 파일에서 **데이터 컬럼 정보만 직접 수정**하면 됩니다:

```yaml
# recipes/my-recipe.yaml
task_choice: classification

data:
  data_interface:
    entity_columns: [user_id]      # [필수] 사용자가 직접 지정
    target_column: is_fraud        # [필수] 사용자가 직접 지정
    feature_columns: null          # [선택] null이면 자동 선택

model:
  class_path: xgboost.XGBClassifier  # 대화형에서 선택됨

# preprocessor:                    # [선택] 필요시 추가
#   steps:
#     - type: standard_scaler
```

**필수 수정 항목**

- `entity_columns`: 데이터의 ID 컬럼명
- `target_column`: 예측 대상 컬럼명

**선택 항목** (기본값으로 작동)

- `feature_columns`: 미지정 시 자동 선택
- `preprocessor`: 미지정 시 전처리 없음
- `model.hyperparameters`: 미지정 시 모델 기본값

**참고 문서**

- [Task 가이드](./docs/user/TASK_GUIDE.md): Task별 data_interface 설정, 지원 모델
- [전처리 레퍼런스](./docs/user/PREPROCESSOR_REFERENCE.md): 전처리 옵션 (스케일링, 결측치 처리 등)
- [설정 스키마](./docs/user/SETTINGS_SCHEMA.md): Config/Recipe YAML 전체 스키마


### 5. 학습 실행

```bash
# CSV 파일로 학습
mmp train --config configs/dev.yaml --recipe recipes/my-recipe.yaml --data data/train.csv

# SQL 파일로 학습 (DB에서 직접 데이터 로드)
mmp train --config configs/dev.yaml --recipe recipes/my-recipe.yaml --data sql/train_data.sql
```

SQL 파일 사용 시 Jinja2 템플릿을 지원합니다:

```sql
-- sql/train_data.sql.j2
SELECT user_id, feature_1, feature_2, target
FROM my_table
WHERE created_at BETWEEN '{{ data_interval_start }}' AND '{{ data_interval_end }}'
```

```bash
# 템플릿 파라미터 전달
mmp train -c configs/dev.yaml -r recipes/model.yaml -d sql/train_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'
```

**로그 파일**

학습 실행 시 상세 로그가 `logs/` 디렉토리에 자동 저장됩니다:

```text
logs/dev_my-recipe_20250107_123456.log
```

- 파일명 형식: `{환경}_{레시피명}_{타임스탬프}.log`
- 30일 이상 된 로그는 자동 삭제됩니다

명령어 상세 옵션은 [CLI 레퍼런스](./docs/user/CLI_REFERENCE.md)를 참고하세요.


### 6. 추론

학습된 모델로 예측을 수행합니다. 배치 추론과 실시간 API 서빙 두 가지 방식을 지원합니다.

#### 배치 추론

대량의 데이터를 한 번에 예측할 때 사용합니다.

```bash
# CSV 파일로 배치 추론
mmp batch-inference -c configs/dev.yaml --run-id <mlflow_run_id> -d data/test.csv

# SQL 파일로 배치 추론 (Jinja2 템플릿 파라미터 전달)
mmp batch-inference -c configs/dev.yaml --run-id <mlflow_run_id> -d sql/inference_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'
```

#### 실시간 API 서빙

REST API 서버를 기동하여 실시간 예측 요청을 처리합니다.

```bash
# API 서버 시작
mmp serve-api --config configs/dev.yaml --run-id <mlflow_run_id>
```

```bash
# API 호출 예시
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 0.5, "feature_2": 100}'
```

#### MLflow 서버 없이 배포

MLflow 서버 연결 없이 로컬 artifact만으로 Docker 배포가 가능합니다.

Config 파일에서 로컬 저장소를 설정합니다:

```yaml
# configs/local.yaml
mlflow:
  tracking_uri: "./mlruns"
  experiment_name: "my-experiment"
```

```bash
# 학습 및 서빙 (Config 설정 사용)
mmp train -c configs/local.yaml -r recipes/model.yaml -d data/train.csv
mmp serve-api -c configs/local.yaml --run-id <run_id>

# Docker 배포 시 mlruns/ 디렉토리 포함
```

API 서버 상세 사용법은 [API 서빙 가이드](./docs/user/API_SERVING_GUIDE.md)를 참고하세요.


## 지원 Task

| Task | 설명 | 활용 사례 |
|------|------|----------|
| Classification | 범주형 분류 | 사기 탐지, 이탈 예측 |
| Regression | 연속값 예측 | 집값 예측, 매출 예측 |
| Timeseries | 시계열 예측 | 일별 매출, 트래픽 예측 |
| Clustering | 비지도 군집화 | 고객 세분화 |
| Causal | 인과 추론 | 프로모션 효과 분석 |

각 Task별 데이터 형식과 모델 설정은 [Task 가이드](./docs/user/TASK_GUIDE.md)를 참고하세요.


## 지원 모델

| 라이브러리 | 모델 |
|------------|------|
| Scikit-learn | RandomForest, LogisticRegression, KMeans 등 |
| XGBoost | XGBClassifier, XGBRegressor |
| LightGBM | LGBMClassifier, LGBMRegressor |
| CatBoost | CatBoostClassifier, CatBoostRegressor |
| PyTorch | LSTM, TabNet |
| statsmodels | ARIMA, ExponentialSmoothing |
| CausalML | T-Learner, S-Learner |

```bash
# 사용 가능한 모델 목록 조회
mmp list models

# 사용 가능한 메트릭 목록 조회
mmp list metrics
```


## 문서

### 사용자 문서

| 순서 | 문서 | 설명 |
|------|------|------|
| 1 | [환경 설정 가이드](./docs/user/ENVIRONMENT_SETUP.md) | 설치, DB 연결, Cloud 설정 |
| 2 | [Task 가이드](./docs/user/TASK_GUIDE.md) | Task별 데이터 형식, 모델, Recipe 설정 |
| 3 | [설정 스키마](./docs/user/SETTINGS_SCHEMA.md) | Config/Recipe YAML 작성법 |
| 4 | [CLI 레퍼런스](./docs/user/CLI_REFERENCE.md) | 명령어 상세 옵션 |
| 5 | [API 서빙 가이드](./docs/user/API_SERVING_GUIDE.md) | REST API 서버 배포 |
| 6 | [전처리 레퍼런스](./docs/user/PREPROCESSOR_REFERENCE.md) | 전처리 상세 (선택) |
| 7 | [로컬 개발 환경](./docs/user/LOCAL_DEV_ENVIRONMENT.md) | Docker 기반 로컬 개발 (선택) |

### 개발자 문서

시스템 확장이나 기여를 원하시면 [개발자 문서](./docs/developer/)를 참고하세요.


## 도움말

```bash
# 전체 명령어 도움말
mmp --help

# 특정 명령어 사용법
mmp train --help

# 간략 출력 (진행 상태만)
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv -q
```

---

**Version**: 1.0.0 | **License**: Apache 2.0 | **Python**: 3.11+
