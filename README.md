# 🚀 Modern ML Pipeline v2.0

**차세대 MLOps 플랫폼 - 학습부터 배포까지 자동화된 머신러닝 파이프라인**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/dependency-uv-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Breaking Changes in v2.0**: 모든 CLI 명령어가 변경되었습니다. [마이그레이션 가이드](#migration-from-v1)를 참조하세요.

## 📋 프로젝트 소개

Modern ML Pipeline은 **YAML 설정만으로 머신러닝 모델을 학습하고 배포**할 수 있는 통합 MLOps 플랫폼입니다.

### 🎯 핵심 특징

- **🔧 Zero-Code ML**: YAML 레시피만으로 모든 ML 모델 실험 가능
- **⚡ 자동 최적화**: Optuna 기반 하이퍼파라미터 자동 튜닝
- **🏗️ 완전한 재현성**: 동일한 결과 100% 보장
- **🌍 멀티 환경**: LOCAL → DEV → PROD 단계적 확장
- **🚀 즉시 배포**: 학습된 모델 바로 API 서빙
- **🧪 견고한 테스트**: 100% 단위 테스트 안정화 달성 (77% 성능 향상)

### 🆕 v2.0 새로운 기능

- **5단계 워크플로우**: 체계적인 프로젝트 설정 프로세스
- **환경별 설정 분리**: Recipe(논리)와 Config(물리) 완전 분리
- **스마트 DB 엔진 선택**: URI 스키마 기반 자동 데이터베이스 엔진 최적화
- **향상된 CLI**: 직관적인 명령어 체계

---

## 🚀 빠른 시작 (5분 설정)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline

# Python 환경 설정 (uv 권장)
uv venv && uv sync

# CLI 도구 설치 (전역 사용 가능)
uv pip install -e .
```

### 2. 5단계 워크플로우 (v2.0)

```bash
# Step 1: 프로젝트 초기화
mmp init --project-name my-ml-project

# Step 2: 환경 설정 생성 (대화형)
mmp get-config --env-name dev
# .env.dev.template 생성됨 → .env.dev로 복사하여 실제 값 입력

# Step 3: 시스템 연결 확인
mmp system-check --env-name dev

# Step 4: 레시피 생성 (대화형)
mmp get-recipe

# Step 5: 모델 학습
mmp train --recipe-file recipes/my_model.yaml --env-name dev
```

### 3. 모델 배포 및 추론

```bash
# 학습에서 나온 run-id 사용 (예: abc123def456)
RUN_ID="your-run-id-here"

# 배치 추론
mmp batch-inference --run-id $RUN_ID --env-name dev

# 실시간 API 서빙
mmp serve-api --run-id $RUN_ID --env-name dev

# API 테스트
curl http://localhost:8000/predict -X POST \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 1, "event_ts": "2024-01-01T00:00:00"}'
```

---

## 📖 기본 사용법

### CLI 명령어 전체 목록 (v2.0)

```bash
# 🔄 5단계 워크플로우
mmp init --project-name <name>              # 1. 프로젝트 초기화
mmp get-config --env-name <env>             # 2. 환경 설정 생성
mmp system-check --env-name <env>           # 3. 시스템 연결 확인
mmp get-recipe                              # 4. 레시피 생성 (대화형)
mmp train --recipe-file <path> --env-name <env>  # 5. 모델 학습

# 모델 추론
mmp batch-inference --run-id <id> --env-name <env>  # 배치 추론
mmp serve-api --run-id <id> --env-name <env>        # 실시간 API

# 도움말
mmp --help                                   # 전체 도움말
mmp <command> --help                         # 명령어별 도움말
```

### 환경별 설정 (v2.0)

```yaml
# configs/dev.yaml - 개발 환경 설정
environment:
  env_name: dev  # 환경 이름

adapters:
  sql:
    connection_uri: "${DB_CONNECTION_URI:postgresql://localhost/dev_db}"
    
mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI:./mlruns}"
```

```bash
# .env.dev - 환경변수 파일
DB_CONNECTION_URI=postgresql://user:pass@localhost/dev_db
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Recipe 파일 작성법

Recipe는 모델의 모든 논리를 정의하는 YAML 파일입니다:

```yaml
# recipes/my_model.yaml
name: "my_first_model"
description: "RandomForest 분류 모델"

model:
  # 모델 클래스 (sklearn, xgboost, lightgbm 등)
  class_path: "sklearn.ensemble.RandomForestClassifier"
  
  # 하이퍼파라미터
  hyperparameters:
    n_estimators: 100
    max_depth: {type: "int", low: 3, high: 10}  # 자동 최적화
  
data:
  # 데이터 로딩
  loader:
    name: "default_loader"
    adapter: sql  # config에 정의된 어댑터 사용
    source_uri: "SELECT * FROM train_data LIMIT 10000"
  
  # 데이터 인터페이스
  data_interface:
    task_type: "classification"
    target_column: "target"

# 평가 설정
evaluation:
  metrics: ["accuracy", "roc_auc", "f1"]
  validation:
    method: "split"
    test_size: 0.2
```

---

## 🔄 스마트 데이터베이스 연결 (v2.0 신기능)

SqlAdapter가 URI 스키마를 자동으로 인식하여 최적의 엔진 설정을 적용합니다:

### 지원 데이터베이스

```yaml
# configs/prod.yaml
adapters:
  sql:
    # BigQuery (자동 인식 및 최적화)
    connection_uri: "bigquery://my-project/my-dataset"
    credentials_path: "/path/to/credentials.json"  # BigQuery 인증
    
    # PostgreSQL (연결 풀링 자동 설정)
    connection_uri: "postgresql://user:pass@host/dbname"
    
    # MySQL (연결 재활용 자동 설정)
    connection_uri: "mysql://user:pass@host/dbname"
    
    # SQLite (로컬 개발용)
    connection_uri: "sqlite:///path/to/database.db"
```

각 데이터베이스별로 자동 적용되는 최적화:
- **BigQuery**: 인증 처리, 적절한 풀 크기
- **PostgreSQL**: 연결 풀링, 타임아웃 설정
- **MySQL**: 연결 재활용, 풀 관리
- **SQLite**: 스레드 안전 설정

---

## <a name="migration-from-v1"></a>🔄 v1.x에서 v2.0 마이그레이션 가이드

### 주요 변경사항

#### 1. CLI 명령어 변경
```bash
# v1.x (이전)
uv run python main.py train --recipe recipes/model.yaml
uv run python main.py batch-inference --run-id abc123

# v2.0 (현재) - 환경 이름 필수
mmp train --recipe-file recipes/model.yaml --env-name dev
mmp batch-inference --run-id abc123 --env-name dev
```

#### 2. 디렉토리 구조 변경
```bash
# v1.x
config/           # ❌ 더 이상 지원 안함
models/recipes/   # ❌ 더 이상 지원 안함
.env             # ❌ 더 이상 지원 안함

# v2.0
configs/         # ✅ 환경별 설정
recipes/         # ✅ 레시피 파일
.env.dev         # ✅ 환경별 환경변수 파일
.env.prod
```

#### 3. Settings API 변경
```python
# v1.x
from src.settings import load_settings_by_file
settings = load_settings_by_file(recipe_file)  # env_name 선택적

# v2.0
from src.settings import load_settings
settings = load_settings(recipe_file, env_name)  # env_name 필수
```

### 마이그레이션 단계

1. **프로젝트 구조 업데이트**
   ```bash
   # 디렉토리 이름 변경
   mv config configs
   mv models/recipes recipes
   
   # 환경별 .env 파일 생성
   mv .env .env.dev
   cp .env.dev .env.prod
   ```

2. **CLI 명령어 업데이트**
   - 모든 스크립트에서 `uv run python main.py` → `mmp`로 변경
   - 모든 실행 명령어에 `--env-name` 추가

3. **코드 업데이트**
   - `load_settings_by_file()` → `load_settings()`로 변경
   - `environment.app_env` → `environment.env_name` 사용

4. **테스트 실행**
   ```bash
   # 설정 검증
   mmp system-check --env-name dev
   
   # 학습 테스트
   mmp train --recipe-file recipes/test.yaml --env-name dev
   ```

---

## 🧪 개발자 테스트 가이드

### ⚡ 테스트 실행 전략

```bash
# 빠른 개발용 (핵심만 - 3.00초)
uv run pytest -m "core and unit" -v

# 표준 CI (기본 스위트)
uv run pytest -q -m "not slow and not integration"

# 성능 최적화 (병렬 실행)
uv run pytest -n auto tests/unit/ -v

# 전체 커버리지 측정
uv run pytest --cov=src --cov-report=term-missing --fail-under=90 -q
```

### 📊 테스트 품질 지표

- **테스트 안정화**: 100% 단위 테스트 통과
- **성능 최적화**: 77% 실행 시간 단축
- **커버리지**: 90%+ (Settings 모듈)
- **Factory 패턴**: 완전 적용

---

## 🔧 고급 기능

### 1. 동적 SQL 템플릿 (Jinja2)

```sql
-- sql/dynamic_query.sql.j2
SELECT user_id, features, target
FROM my_table 
WHERE date = '{{ target_date }}'
LIMIT {{ limit | default(1000) }}
```

```bash
# 템플릿 파라미터와 함께 실행
mmp train --recipe-file recipes/model.yaml --env-name dev \
  --params '{"target_date": "2024-01-01", "limit": 5000}'
```

### 2. Feature Store 연동

```yaml
# recipes/feature_store_model.yaml
preprocessor:
  augmenter:
    type: "feature_store"
    features:
      - namespace: "user_demographics"
        features: ["age", "country"]
```

### 3. Docker 배포

```bash
# 이미지 빌드
docker build -t mmp-api --target serve .

# 모델 서빙
docker run --rm -p 8000:8000 \
  -e ENV_NAME=prod \
  mmp-api --run-id $RUN_ID
```

---

## 📚 문서

- [상세 가이드](.claude/CLI_REDEVELOPMENT_PLAN_INDEX.md)
- [API 레퍼런스](docs/api.md)
- [레시피 작성 가이드](docs/recipe_guide.md)
- [문제 해결](docs/troubleshooting.md)

---

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📞 지원

- **버그 리포트**: [GitHub Issues](https://github.com/wooshikwon/modern-ml-pipeline/issues)
- **이메일**: wooshik.kwon@example.com
- **문서**: [Wiki](https://github.com/wooshikwon/modern-ml-pipeline/wiki)

---

*Modern ML Pipeline v2.0 - Build with ❤️ for ML Engineers*