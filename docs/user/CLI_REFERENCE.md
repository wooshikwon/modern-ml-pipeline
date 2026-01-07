# CLI 레퍼런스 (CLI Reference)

`mmp` (Modern ML Pipeline) 명령어의 상세 사용법을 안내합니다.

> **Tip**
>
> 모든 명령어는 `--help` 옵션으로 도움말을 확인할 수 있습니다.


## 1. 프로젝트 관리 (Setup)

### `mmp init`

새로운 ML 프로젝트 폴더 구조를 생성합니다.

```bash
mmp init [PROJECT_NAME]
```

| 인자 | 설명 |
|------|------|
| `PROJECT_NAME` | 프로젝트 이름 (미지정 시 대화형으로 입력) |

**생성되는 구조:**

```text
my-project/
├── configs/        # Config 파일
├── recipes/        # Recipe 파일
├── data/           # 데이터 파일
├── sql/            # SQL 쿼리 파일
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```


### `mmp get-config`

인프라 설정 파일(`configs/*.yaml`)을 대화형으로 생성합니다.

```bash
mmp get-config [OPTIONS]
```

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--output` | `-o` | 출력 파일 경로 (기본: `configs/{env}.yaml`) |

**대화형 질문:**
- 환경 이름 (local, dev, prod)
- MLflow 연결 정보
- 데이터 소스 유형 및 연결 정보


### `mmp get-recipe`

실험 레시피 파일(`recipes/*.yaml`)을 대화형으로 생성합니다.

```bash
mmp get-recipe [OPTIONS]
```

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--output` | `-o` | 출력 파일 경로 (기본: `recipes/{name}.yaml`) |

**대화형 질문:**
- Task 유형 (classification, regression, timeseries, clustering, causal)
- 모델 선택
- 전처리 옵션


### `mmp system-check`

시스템 연결 상태와 패키지 의존성을 진단합니다.

```bash
mmp system-check [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--config` | `-c` | O | Config 파일 경로 |
| `--recipe` | `-r` | - | Recipe 파일 경로 (모델 의존성 점검) |
| `--actionable` | `-a` | - | 해결 방법(설치 명령어) 함께 출력 |

**예시:**

```bash
# Config만 점검
mmp system-check -c configs/dev.yaml

# Recipe 포함 전체 점검 + 해결책 출력
mmp system-check -c configs/dev.yaml -r recipes/model.yaml -a
```


## 2. 파이프라인 실행 (Execution)

### `mmp train`

모델 학습을 실행합니다.

```bash
mmp train [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--config` | `-c` | O | Config 파일 경로 |
| `--recipe` | `-r` | O | Recipe 파일 경로 |
| `--data` | `-d` | - | 데이터 파일 경로 (CSV, Parquet, SQL) |
| `--params` | `-p` | - | Jinja 템플릿 파라미터 (JSON 형식) |
| `--record-reqs` | - | - | 패키지 요구사항을 MLflow artifact에 기록 |
| `-q` | `--quiet` | - | 요약 출력 모드 (진행 상태만 표시) |

**예시:**

```bash
# CSV 파일로 학습 (기본: 상세 로그 출력)
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv

# SQL 템플릿으로 학습 (파라미터 전달)
mmp train -c configs/dev.yaml -r recipes/model.yaml \
  -d sql/train_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'

# Quiet 모드 (진행 상태만)
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv -q
```

**출력:**

```text
mmp v1.0.0

[1/6] Loading config          done
[2/6] Checking dependencies   done
[3/6] Loading data            done  10,000 rows
[4/6] Training model          done
[5/6] Evaluating              done  accuracy: 0.92
[6/6] Saving artifacts        done

Run ID: 859b24da15c8...
```


### `mmp batch-inference`

학습된 모델로 배치 추론을 실행합니다.

```bash
mmp batch-inference [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | MLflow Run ID (학습된 모델) |
| `--config` | `-c` | - | Config 파일 경로 (미지정 시 artifact 사용) |
| `--recipe` | `-r` | - | Recipe 파일 경로 (미지정 시 artifact 사용) |
| `--data` | `-d` | - | 추론 데이터 경로 (미지정 시 artifact SQL 사용) |
| `--params` | `-p` | - | Jinja 템플릿 파라미터 (JSON 형식) |
| `-q` | `--quiet` | - | 요약 출력 모드 |

**예시:**

```bash
# 학습 시 저장된 artifact 설정 그대로 사용
mmp batch-inference --run-id abc123def456

# Config/Recipe override
mmp batch-inference --run-id abc123 -c configs/prod.yaml

# 데이터 소스 override + 파라미터 전달
mmp batch-inference --run-id abc123 \
  -d sql/inference_data.sql.j2 \
  --params '{"data_interval_start": "2025-02-01", "data_interval_end": "2025-02-28"}'
```


### `mmp serve-api`

학습된 모델을 REST API 서버로 서빙합니다.

```bash
mmp serve-api [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | MLflow Run ID (서빙할 모델) |
| `--config` | `-c` | O | Config 파일 경로 |
| `--host` | - | - | 바인딩 호스트 (기본: `0.0.0.0`) |
| `--port` | - | - | 바인딩 포트 (기본: `8000`) |

**예시:**

```bash
# 기본 서버 시작
mmp serve-api --run-id abc123 -c configs/prod.yaml

# 호스트/포트 지정
mmp serve-api --run-id abc123 -c configs/prod.yaml --host localhost --port 8080
```

**출력:**

```text
mmp v1.0.0

[1/3] Loading settings        done  run_id: abc123...
[2/3] Checking package deps   done  verified
[3/3] Starting API server     done

      API Server:   http://0.0.0.0:8000
      API Docs:     http://0.0.0.0:8000/docs
      Health Check: http://0.0.0.0:8000/health
```

> **Note**
>
> API 서버 상세 사용법은 [API 서빙 가이드](./API_SERVING_GUIDE.md)를 참고하세요.


## 3. 정보 조회 (Utilities)

### `mmp list`

등록된 컴포넌트 목록을 조회합니다.

```bash
mmp list <COMPONENT>
```

| 컴포넌트 | 설명 |
|----------|------|
| `models` | 사용 가능한 모델 목록 (class_path 포함) |
| `preprocessors` | 전처리기 목록 |
| `metrics` | Task별 평가 메트릭 목록 |
| `adapters` | 데이터 어댑터 목록 |

**예시:**

```bash
# 모델 목록 (class_path 복사해서 recipe에 사용)
mmp list models

# 출력 예시:
# Classification:
#   - xgboost.XGBClassifier                         (xgboost)
#   - sklearn.ensemble.RandomForestClassifier       (scikit-learn)
#
# Tip: class_path 값을 recipe의 model.class_path에 그대로 사용하세요.

# 전처리기 목록
mmp list preprocessors

# Task별 메트릭 목록
mmp list metrics
```


## 4. 터미널 출력

### 기본 출력 (상세 모드)

기본적으로 모든 명령어는 상세 로그를 출력합니다:

- `[DATA:...]` - 데이터 로딩/저장 상세
- `[MODEL:...]` - 모델 컴포넌트 동작
- `[PREPROCESS:...]` - 전처리 단계별 변환

이는 K8s 로그, CI/CD 파이프라인 등에서 디버깅에 유용합니다.

### Quiet 모드 (`-q`)

진행 상태만 간략히 확인하려면 `-q` 옵션을 사용합니다:

```bash
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv -q
```

Quiet 모드 출력 예시:

```text
[1/6] Loading config          done
[2/6] Checking dependencies   done
[3/6] Loading data            done  10,000 rows
```

### 로그 파일

모든 실행 로그는 파일에 자동 저장됩니다:

- 위치: `logs/{env}_{recipe}_{timestamp}.log`
- 로그 레벨: DEBUG (상세 로그 포함)
- 보관: 30일 후 자동 삭제


## 5. 자주 쓰는 패턴

### SQL 템플릿 파라미터 전달

```bash
# 학습
mmp train -c configs/dev.yaml -r recipes/model.yaml \
  -d sql/train_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'

# 추론
mmp batch-inference --run-id abc123 \
  -d sql/inference_data.sql.j2 \
  --params '{"data_interval_start": "2025-02-01", "data_interval_end": "2025-02-28"}'
```

### 하이퍼파라미터 실험

```bash
for lr in 0.01 0.05 0.1; do
  mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv \
    --params "{\"model.hyperparameters.values.learning_rate\": $lr}"
done
```

### 환경별 설정 분리

```bash
# 개발 환경
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv

# 운영 환경
mmp train -c configs/prod.yaml -r recipes/model.yaml -d sql/prod_data.sql
```
