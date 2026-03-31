# CLI 레퍼런스

`mmp` (Modern ML Pipeline) v1.4.5 명령어 레퍼런스.

> 모든 명령어는 `--help`로 도움말을 확인할 수 있다.
> 모든 명령어는 **프로젝트 루트 디렉토리에서 실행**해야 한다. 설정 파일(`configs/`, `recipes/`)과 데이터 파일(`data/`) 경로는 현재 작업 디렉토리 기준 상대 경로로 해석된다.


## 글로벌 옵션

| 옵션 | 설명 |
|------|------|
| `--version` | 버전 출력 후 종료 |
| `-q` / `--quiet` | 요약 출력 모드 (진행 상태만 표시) |


## 1. 프로젝트 관리

### `mmp init`

새 ML 프로젝트 폴더 구조를 생성한다.

```bash
mmp init [PROJECT_NAME]
```

| 인자 | 설명 |
|------|------|
| `PROJECT_NAME` | 프로젝트 이름 (미지정 시 대화형 입력) |

생성 구조: `configs/`, `recipes/`, `data/`, `docker-compose.yml`, `Dockerfile`, `pyproject.toml`, `README.md`


### `mmp get-config`

인프라 설정 파일(`configs/*.yaml`)을 대화형으로 생성한다.

```bash
mmp get-config [-o OUTPUT_PATH]
```

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--output` | `-o` | 출력 파일 경로 (기본: `configs/{env}.yaml`) |


### `mmp get-recipe`

실험 레시피 파일(`recipes/*.yaml`)을 대화형으로 생성한다.

```bash
mmp get-recipe [-o OUTPUT_PATH]
```

| 옵션 | 단축 | 설명 |
|------|------|------|
| `--output` | `-o` | 출력 파일 경로 (기본: `recipes/{name}.yaml`) |


### `mmp system-check`

시스템 연결 상태와 패키지 의존성을 진단한다.

```bash
mmp system-check -c CONFIG [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--config` | `-c` | O | Config 파일 경로 |
| `--recipe` | `-r` | - | Recipe 파일 경로 (모델 의존성 점검) |
| `--actionable` | `-a` | - | 해결 방법(설치 명령어) 함께 출력 |

```bash
mmp system-check -c configs/dev.yaml -r recipes/model.yaml -a
```


### `mmp validate`

Recipe + Config YAML을 사전 검증한다. 학습 없이 Pydantic 파싱, Jinja 렌더링, Catalog/Business/Compatibility 검증을 수행한다. `--data` 지정 시 데이터 접근 및 feature_columns 존재 여부까지 확인한다.

```bash
mmp validate -r RECIPE -c CONFIG [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--recipe` | `-r` | O | Recipe 파일 경로 |
| `--config` | `-c` | O | Config 파일 경로 |
| `--data` | `-d` | - | 데이터 파일 경로 (SQL 사용 시 생략 가능) |
| `--params` | `-p` | - | Jinja 템플릿 파라미터 (JSON 형식) |

```bash
# 설정만 검증
mmp validate -r recipes/model.yaml -c configs/local.yaml

# 데이터 포함 검증
mmp validate -r recipes/model.yaml -c configs/dev.yaml -d data/train.csv

# Jinja 파라미터 포함
mmp validate -r recipes/model.yaml -c configs/prod.yaml \
  --params '{"date": "2025-01-01"}'
```


## 2. ML 파이프라인

### `mmp train`

모델 학습을 실행한다.

```bash
mmp train -c CONFIG -r RECIPE [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--config` | `-c` | O | Config 파일 경로 |
| `--recipe` | `-r` | O | Recipe 파일 경로 |
| `--data` | `-d` | - | 데이터 파일 경로 (CSV, Parquet, SQL) |
| `--params` | `-p` | - | Jinja 템플릿 파라미터 (JSON 형식) |
| `--record-reqs` | - | - | 패키지 요구사항을 MLflow artifact에 기록 |

```bash
# CSV 학습
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv

# SQL 템플릿 + 파라미터
mmp train -c configs/dev.yaml -r recipes/model.yaml \
  -d data/train_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'
```

출력 예시:

```text
mmp v1.4.5

[1/6] Loading config          done
[2/6] Checking dependencies   done
[3/6] Loading data            done  10,000 rows
[4/6] Training model          done
[5/6] Evaluating              done  accuracy: 0.92
[6/6] Saving artifacts        done

Run ID: 859b24da15c8...
```


### `mmp batch-inference`

학습된 모델로 배치 추론을 실행한다.

```bash
mmp batch-inference --run-id RUN_ID [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | MLflow Run ID |
| `--config` | `-c` | - | Config 파일 경로 (미지정 시 artifact 사용) |
| `--recipe` | `-r` | - | Recipe 파일 경로 (미지정 시 artifact 사용) |
| `--data` | `-d` | - | 추론 데이터 경로 (미지정 시 artifact SQL 사용) |
| `--params` | `-p` | - | Jinja 템플릿 파라미터 (JSON 형식) |
| `--output-path` | `-o` | - | 결과 저장 경로 (Config override) |

```bash
# artifact 설정 그대로 사용
mmp batch-inference --run-id abc123def456

# 결과 경로 직접 지정
mmp batch-inference --run-id abc123 -o gs://bucket/2025/01/09/result.parquet
```

`--output-path` 지원 포맷: `parquet`, `csv`, `json` (확장자 자동 감지). 우선순위: CLI 옵션 > Config YAML > 기본값(`predictions_{run_id}.parquet`).


### `mmp serve-api`

학습된 모델을 REST API 서버로 서빙한다.

```bash
mmp serve-api --run-id RUN_ID -c CONFIG [OPTIONS]
```

| 옵션 | 단축 | 필수 | 설명 |
|------|------|------|------|
| `--run-id` | - | O | MLflow Run ID |
| `--config` | `-c` | O | Config 파일 경로 |
| `--host` | - | - | 바인딩 호스트 (기본: `0.0.0.0`) |
| `--port` | - | - | 바인딩 포트 (기본: `8000`) |

```bash
mmp serve-api --run-id abc123 -c configs/prod.yaml --port 8080
```

출력 예시:

```text
mmp v1.4.5

[1/3] Loading settings        done  run_id: abc123...
[2/3] Checking package deps   done  verified
[3/3] Starting API server     done

      API Server:   http://0.0.0.0:8000
      API Docs:     http://0.0.0.0:8000/docs
      Health Check: http://0.0.0.0:8000/health
```


## 3. 조회

### `mmp list`

등록된 컴포넌트 목록을 조회한다.

```bash
mmp list <COMPONENT>
```

| 컴포넌트 | 설명 |
|----------|------|
| `models` | 사용 가능한 모델 목록 (class_path 포함) |
| `adapters` | 데이터 어댑터 목록 |
| `evaluators` | 평가자 목록 |
| `metrics` | Task별 평가 메트릭 목록 |
| `preprocessors` | 전처리기 목록 |

```bash
mmp list models         # 모델 목록 (class_path를 recipe에 사용)
mmp list metrics        # Task별 메트릭
mmp list preprocessors  # 전처리기
```


## 4. 자주 쓰는 패턴

### SQL 템플릿 파라미터 전달

```bash
mmp train -c configs/dev.yaml -r recipes/model.yaml \
  -d data/train_data.sql.j2 \
  --params '{"data_interval_start": "2025-01-01", "data_interval_end": "2025-01-31"}'
```

### 학습 전 사전 검증

```bash
# validate로 설정 오류 조기 발견 → 통과 후 학습 실행
mmp validate -r recipes/model.yaml -c configs/dev.yaml -d data/train.csv
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv
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
mmp train -c configs/dev.yaml -r recipes/model.yaml -d data/train.csv   # 개발
mmp train -c configs/prod.yaml -r recipes/model.yaml -d data/prod.sql   # 운영
```

### Airflow 배치 추론

```bash
mmp batch-inference --run-id {{ model_run_id }} \
  -o "gs://bucket/predictions/{{ ds }}/batch_{{ ds_nodash }}.parquet"
```
