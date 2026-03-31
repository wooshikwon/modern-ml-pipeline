# 설치 후 첫 학습까지

> README를 읽고 `pip install modern-ml-pipeline` (또는 pipx)로 설치를 마쳤다면, 이 가이드를 따라 첫 학습을 실행하세요.

---

## 1. 프로젝트 초기화

```bash
mmp init my-project
cd my-project
```

생성되는 구조:

```text
my-project/
├── configs/             # 환경별 설정
├── recipes/             # 실험 레시피
├── data/                # 데이터 (CSV, SQL)
├── Dockerfile
└── docker-compose.yml
```

학습 데이터를 `data/` 디렉토리에 준비합니다 (CSV 또는 SQL 파일).

---

## 2. Config 설정 (인프라)

### 2-1. Config 파일 생성

```bash
mmp get-config
```

대화형 인터페이스에서 MLflow, 데이터소스, 스토리지 등을 선택하면 두 파일이 생성됩니다:

| 생성 파일 | 용도 |
|----------|------|
| `configs/{env}.yaml` | 인프라 설정 (MLflow, DB, Storage) |
| `.env.{env}.template` | 민감 정보 템플릿 (인증키, 비밀번호) |

### 2-2. 환경변수 설정

```bash
cp .env.local.template .env.local
vi .env.local   # 실제 값 입력
```

```bash
# .env.local 예시
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
DB_USER=mluser
DB_PASSWORD=secretpassword
```

> `.env.*` 파일은 `.gitignore`에 추가하세요. 템플릿(`.env.*.template`)만 커밋합니다.

CLI 명령어는 Config 파일명에서 환경 이름을 추출하여 `.env.{env}` 파일을 **자동 로드**합니다:

| Config 파일 | 자동 로드 |
|-------------|----------|
| `configs/local.yaml` | `.env.local` |
| `configs/dev.yaml` | `.env.dev` |
| `configs/prod.yaml` | `.env.prod` |

### 2-3. 데이터소스별 Config 예시

**로컬 파일 (가장 간단)**

```yaml
mlflow:
  tracking_uri: ./mlruns
  experiment_name: my-first-experiment
```

데이터를 CSV로 직접 전달하면 `data_source` 설정이 불필요합니다.

**BigQuery** (`cloud-extras` 설치 필요)

```yaml
data_source:
  name: BigQuery
  adapter_type: sql
  config:
    connection_uri: bigquery://my-project
    project_id: my-project
    credentials_path: "${GOOGLE_APPLICATION_CREDENTIALS}"
    location: US
    use_pandas_gbq: true
```

**PostgreSQL**

```yaml
data_source:
  adapter_type: sql
  config:
    connection_uri: "postgresql://${DB_USER:postgres}:${DB_PASSWORD:}@localhost:5432/mydb"
```

Config YAML에서 환경변수는 `${VAR_NAME}` 또는 `${VAR_NAME:기본값}` 문법으로 참조합니다.

---

## 3. Recipe 설정 (실험)

```bash
mmp get-recipe
```

대화형 인터페이스에서 Task, 모델, 전처리를 선택하면 `recipes/{name}.yaml`이 생성됩니다. 생성 후 **데이터 컬럼 정보만 수정**하면 됩니다:

```yaml
# recipes/my-recipe.yaml
task_choice: classification

data:
  data_interface:
    entity_columns: [user_id]      # ID 컬럼
    target_column: is_fraud        # 예측 대상

model:
  class_path: xgboost.XGBClassifier
```

> Task별 데이터 형식과 모델 옵션은 [Task 가이드](./user/TASK_GUIDE.md)를, 전체 스키마는 [설정 스키마](./user/SETTINGS_SCHEMA.md)를 참고하세요.

---

## 4. 사전 검증

### 4-1. Config + Recipe 검증

```bash
mmp validate -c configs/dev.yaml -r recipes/my-recipe.yaml
```

Config와 Recipe의 스키마 오류, 누락 필드, 타입 불일치 등을 사전에 잡아줍니다. 학습 전에 반드시 실행하세요.

### 4-2. 시스템 연결 점검

```bash
mmp system-check -c configs/dev.yaml --actionable
```

MLflow, DB, Storage 등 인프라 연결 상태를 점검합니다. `--actionable` 플래그를 추가하면 문제 해결 방법도 함께 출력됩니다.

```text
시스템 연결 검사 결과:
  PackageDependencies: 패키지 설치 완료
  MLflow: 연결됨 (./mlruns)
  Database: 연결됨
```

Recipe를 포함한 전체 검증:

```bash
mmp system-check -c configs/dev.yaml -r recipes/my-recipe.yaml --actionable
```

---

## 5. 첫 학습 실행

```bash
# CSV 데이터로 학습
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml -d data/train.csv

# SQL 파일로 학습 (BigQuery/PostgreSQL)
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml -d data/query.sql

# SQL에 Jinja 변수 전달
mmp train -c configs/dev.yaml -r recipes/my-recipe.yaml \
  -d data/query.sql -p '{"start_date": "2024-01-01", "end_date": "2024-12-31"}'
```

학습 완료 시 출력되는 `run_id`를 기억하세요. 추론과 서빙에 사용됩니다.

---

## 6. MLflow 결과 확인

### UI 실행

```bash
mlflow ui --port 5000
```

브라우저에서 `http://localhost:5000` 접속

### 자동 기록 항목

| 항목 | 내용 |
|------|------|
| 파라미터 | 모델 하이퍼파라미터, 전처리 설정 |
| 메트릭 | 학습/검증 성능 지표 (accuracy, f1, rmse 등) |
| 아티팩트 | 학습된 모델, 전처리기, SHAP 분석 결과 |
| 태그 | Task 유형, 모델 클래스, 실행 환경 정보 |

### 실험 비교 워크플로우

Recipe를 여러 개 만들어 다양한 모델을 실험하고, MLflow UI에서 비교합니다:

```bash
mmp train -c configs/dev.yaml -r recipes/xgb-baseline.yaml -d data/train.csv
mmp train -c configs/dev.yaml -r recipes/lgbm-tuned.yaml -d data/train.csv
```

MLflow UI에서 비교할 run들을 체크박스로 선택 > "Compare" 버튼으로 메트릭을 시각 비교하고, 최적 모델의 `run_id`를 확인합니다.

### 최적 모델 서빙

```bash
mmp serve-api -c configs/dev.yaml --run-id <best_run_id>
```

---

## 다음 단계

- [Task 가이드](./user/TASK_GUIDE.md) - Task별 데이터 형식과 모델 설정
- [설정 스키마](./user/SETTINGS_SCHEMA.md) - Config/Recipe YAML 상세 옵션
- [CLI 레퍼런스](./user/CLI_REFERENCE.md) - 명령어 상세 옵션
- [MLflow 가이드](./user/MLFLOW_GUIDE.md) - 실험 추적 심화
- [API 서빙 가이드](./user/API_SERVING_GUIDE.md) - REST API 서버 사용법
- 로컬 개발 환경(Docker 기반 MLflow, PostgreSQL, Redis)이 필요하면 [mmp-local-dev](https://github.com/wooshikwon/mmp-local-dev) 참조
