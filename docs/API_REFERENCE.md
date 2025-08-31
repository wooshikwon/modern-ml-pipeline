# CLI 명령어 레퍼런스

## 전역 옵션

### --version
현재 설치된 Modern ML Pipeline 버전을 표시합니다.

```bash
mmp --version
```

### --help
명령어 도움말을 표시합니다.

```bash
mmp --help
mmp <command> --help
```

---

## 명령어 목록

### mmp init

프로젝트 구조를 초기화합니다.

**사용법:**
```bash
mmp init [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--project-name` | STRING | 프로젝트 이름 | 대화형 입력 |
| `--with-mmp-dev` | BOOL | mmp-local-dev 환경 설치 | False |

**예제:**
```bash
# 대화형 모드
mmp init

# 옵션 지정
mmp init --project-name my-project --with-mmp-dev
```

**생성 구조:**
```
project-name/
├── configs/          # 환경별 설정 파일
├── recipes/          # ML Recipe 파일
├── sql/             # SQL 쿼리 파일
├── data/            # 데이터 디렉토리
├── .gitignore       # Git 제외 파일
└── README.md        # 프로젝트 설명
```

---

### mmp get-config

환경별 설정 파일을 생성합니다.

**사용법:**
```bash
mmp get-config [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--env-name`, `-e` | STRING | 환경 이름 | 대화형 입력 |
| `--non-interactive` | BOOL | 비대화형 모드 | False |
| `--template`, `-t` | STRING | 템플릿 (local/dev/prod) | None |

**예제:**
```bash
# 대화형 모드
mmp get-config --env-name dev

# 템플릿 사용
mmp get-config --env-name prod --template prod --non-interactive

# 단축 옵션
mmp get-config -e local -t local
```

**템플릿 설명:**
- `local`: 로컬 개발 환경 (PostgreSQL + 로컬 스토리지)
- `dev`: 개발 서버 환경 (PostgreSQL + MLflow + Redis)
- `prod`: 프로덕션 환경 (BigQuery + GCS + MLflow)

**생성 파일:**
- `configs/{env_name}.yaml`: 환경별 설정 파일
- `.env.{env_name}.template`: 환경변수 템플릿

---

### mmp system-check

시스템 연결 상태를 검사합니다.

**사용법:**
```bash
mmp system-check [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--env-name`, `-e` | STRING | 검사할 환경 이름 | ENV_NAME 환경변수 |
| `--actionable`, `-a` | BOOL | 실행 가능한 해결책 제시 | False |

**예제:**
```bash
# 기본 검사
mmp system-check --env-name dev

# 해결책 포함
mmp system-check -e prod --actionable
```

**검사 항목:**
- MLflow 서버 연결
- 데이터베이스 연결 (PostgreSQL/BigQuery)
- Redis 연결 (Feature Store)
- 스토리지 접근 권한
- Feature Store 설정

---

### mmp get-recipe

대화형으로 ML Recipe를 생성합니다.

**사용법:**
```bash
mmp get-recipe [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--output`, `-o` | PATH | 출력 파일 경로 | recipes/{name}.yaml |
| `--force` | BOOL | 기존 파일 덮어쓰기 | False |

**예제:**
```bash
# 대화형 Recipe 생성
mmp get-recipe

# 특정 경로에 저장
mmp get-recipe --output recipes/custom_model.yaml
```

**생성 과정:**
1. 모델 카테고리 선택 (Tree-based, Linear, Deep Learning)
2. 구체적 모델 선택 (XGBoost, LightGBM 등)
3. 데이터 소스 설정 (SQL, CSV, Parquet)
4. 특성 처리 파이프라인 구성
5. 하이퍼파라미터 설정

---

### mmp train

학습 파이프라인을 실행합니다.

**사용법:**
```bash
mmp train [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--recipe-file`, `-r` | PATH | Recipe 파일 경로 | 필수 |
| `--env-name`, `-e` | STRING | 사용할 환경 이름 | ENV_NAME 환경변수 |
| `--params`, `-p` | JSON | Jinja 템플릿 파라미터 | None |

**예제:**
```bash
# 기본 실행
mmp train --recipe-file recipes/model.yaml --env-name dev

# 파라미터 전달
mmp train -r recipes/model.yaml -e prod --params '{"date": "2024-01-01", "sample_rate": 0.1}'

# 환경변수 사용
export ENV_NAME=staging
mmp train --recipe-file recipes/model.yaml
```

**실행 과정:**
1. 환경별 설정 로드
2. Recipe 파일 파싱
3. 데이터 로드
4. 전처리 파이프라인 실행
5. 모델 학습
6. MLflow에 결과 기록
7. 모델 저장

---

### mmp batch-inference

배치 추론을 실행합니다.

**사용법:**
```bash
mmp batch-inference [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--run-id` | STRING | MLflow Run ID | 필수 |
| `--env-name`, `-e` | STRING | 사용할 환경 이름 | ENV_NAME 환경변수 |
| `--params`, `-p` | JSON | Jinja 템플릿 파라미터 | None |

**예제:**
```bash
# 기본 추론
mmp batch-inference --run-id abc123def456 --env-name prod

# 파라미터 전달
mmp batch-inference --run-id abc123def456 -e dev --params '{"batch_size": 1000}'
```

---

### mmp serve-api

모델 서빙 API를 실행합니다.

**사용법:**
```bash
mmp serve-api [OPTIONS]
```

**옵션:**
| 옵션 | 타입 | 설명 | 기본값 |
|------|------|------|--------|
| `--run-id` | STRING | MLflow Run ID | 필수 |
| `--env-name`, `-e` | STRING | 사용할 환경 이름 | ENV_NAME 환경변수 |
| `--host` | STRING | 바인딩할 호스트 | 0.0.0.0 |
| `--port` | INT | 바인딩할 포트 | 8000 |

**예제:**
```bash
# 기본 서빙
mmp serve-api --run-id abc123def456 --env-name dev

# 커스텀 호스트/포트
mmp serve-api --run-id abc123def456 -e prod --host localhost --port 8080
```

**API 엔드포인트:**
- `GET /health`: 헬스 체크
- `POST /predict`: 예측 요청
- `GET /metrics`: 모델 메트릭

---

### mmp list

사용 가능한 컴포넌트를 나열합니다.

**사용법:**
```bash
mmp list <SUBCOMMAND>
```

**서브커맨드:**

#### mmp list models
사용 가능한 모델 목록을 표시합니다.

```bash
mmp list models
```

출력 예:
```
✅ Available Models from Catalog:

--- Tree-based Models ---
- xgboost.XGBClassifier (xgboost)
- xgboost.XGBRegressor (xgboost)
- lightgbm.LGBMClassifier (lightgbm)
- lightgbm.LGBMRegressor (lightgbm)

--- Linear Models ---
- sklearn.linear_model.LogisticRegression (scikit-learn)
- sklearn.linear_model.LinearRegression (scikit-learn)
```

#### mmp list adapters
사용 가능한 데이터 어댑터를 표시합니다.

```bash
mmp list adapters
```

출력 예:
```
✅ Available Adapters:
- sql
- csv
- parquet
- storage
- feast
```

#### mmp list evaluators
사용 가능한 평가자를 표시합니다.

```bash
mmp list evaluators
```

출력 예:
```
✅ Available Evaluators:
- classification
- regression
- clustering
- ranking
```

#### mmp list preprocessors
사용 가능한 전처리기를 표시합니다.

```bash
mmp list preprocessors
```

출력 예:
```
✅ Available Preprocessor Steps:
- standard_scaler
- min_max_scaler
- one_hot_encoder
- label_encoder
- imputer
```

---

## 환경변수

### 필수 환경변수

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `ENV_NAME` | 현재 환경 이름 | dev, staging, prod |
| `DB_HOST` | 데이터베이스 호스트 | localhost |
| `DB_PORT` | 데이터베이스 포트 | 5432 |
| `DB_USER` | 데이터베이스 사용자 | postgres |
| `DB_PASSWORD` | 데이터베이스 비밀번호 | secret123 |
| `DB_NAME` | 데이터베이스 이름 | ml_database |

### 선택적 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | ./mlruns |
| `GCP_PROJECT` | GCP 프로젝트 ID | None |
| `GCS_BUCKET` | GCS 버킷 이름 | None |
| `REDIS_HOST` | Redis 호스트 | localhost |
| `REDIS_PORT` | Redis 포트 | 6379 |
| `LOG_LEVEL` | 로그 레벨 | INFO |

---

## 설정 파일 형식

### Config 파일 (configs/{env_name}.yaml)

```yaml
environment:
  app_env: "${ENV_NAME}"
  gcp_project_id: "${GCP_PROJECT:}"

mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:./mlruns}"
  experiment_name: "${ENV_NAME}-experiment"

data_adapters:
  default_loader: "sql"
  adapters:
    sql:
      class_name: "SqlAdapter"
      config:
        connection_uri: "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

artifact_stores:
  local:
    enabled: true
    base_uri: "./data"
  gcs:
    enabled: "${GCS_ENABLED:false}"
    bucket: "${GCS_BUCKET:}"
```

### Recipe 파일 (recipes/{name}.yaml)

```yaml
name: "model_name"
description: "Model description"

model:
  class_path: "xgboost.XGBClassifier"
  
  loader:
    adapter: "sql"
    source_uri: "sql/train_query.sql"
    entity_schema:
      entity_columns: ["user_id"]
      timestamp_column: "created_at"
  
  data_interface:
    task_type: "classification"
    target_column: "target"
    feature_columns: ["feature_1", "feature_2"]
  
  preprocessor:
    steps:
      - type: "standard_scaler"
        columns: ["feature_1", "feature_2"]
      - type: "one_hot_encoder"
        columns: ["categorical_feature"]
  
  hyperparameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.3
  
  evaluation:
    metrics: ["accuracy", "precision", "recall", "f1"]
    cross_validation:
      enabled: true
      folds: 5
```

---

## 에러 코드

| 코드 | 설명 | 해결 방법 |
|------|------|-----------|
| 1 | 일반 오류 | 로그 확인 |
| 2 | 파일을 찾을 수 없음 | 파일 경로 확인 |
| 3 | 연결 실패 | system-check 실행 |
| 4 | 권한 오류 | 파일/디렉토리 권한 확인 |
| 5 | 설정 오류 | Config/Recipe 파일 검증 |

---

## 디버깅

### 상세 로그 활성화

```bash
export LOG_LEVEL=DEBUG
mmp train --recipe-file recipes/model.yaml --env-name dev
```

### 설정 검증

```python
from src.settings import load_settings_by_file
settings = load_settings_by_file("recipes/model.yaml", env_name="dev")
print(settings.dict())
```

### 연결 테스트

```bash
# 모든 서비스 테스트
mmp system-check --env-name dev --actionable

# Python에서 직접 테스트
python -c "
from src.cli.utils.env_loader import load_config_with_env
config = load_config_with_env('dev')
print(config)
"
```