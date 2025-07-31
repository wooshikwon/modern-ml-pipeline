# �� Modern ML Pipeline

**차세대 MLOps 플랫폼 - 학습부터 배포까지 자동화된 머신러닝 파이프라인**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/dependency-uv-green.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 프로젝트 소개

Modern ML Pipeline은 **YAML 설정만으로 머신러닝 모델을 학습하고 배포**할 수 있는 통합 MLOps 플랫폼입니다.

### 🎯 핵심 특징

- **🔧 Zero-Code ML**: YAML 레시피만으로 모든 ML 모델 실험 가능
- **⚡ 자동 최적화**: Optuna 기반 하이퍼파라미터 자동 튜닝
- **🏗️ 완전한 재현성**: 동일한 결과 100% 보장
- **🌍 멀티 환경**: LOCAL → DEV → PROD 단계적 확장
- **🚀 즉시 배포**: 학습된 모델 바로 API 서빙

---

## 🚀 빠른 시작 (5분 설정)

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline

# Python 환경 설정 (uv 권장)
uv venv && uv sync
# 또는 pip 사용시: pip install -r requirements.txt
```

### 2. 프로젝트 초기화

```bash
# 새 프로젝트 구조 생성
uv run python main.py init

# 생성된 파일 확인
ls config/    # base.yaml, data_adapters.yaml
ls recipes/   # example_recipe.yaml
```

### 3. 첫 번째 레시피 생성 (`guide` 명령어)

```bash
# sklearn의 RandomForestClassifier에 대한 레시피 템플릿을 생성합니다.
uv run python main.py guide sklearn.ensemble.RandomForestClassifier > recipes/my_first_model.yaml

# 생성된 파일을 열어 source_uri, target_column 등을 당신의 데이터에 맞게 수정하세요.
```

### 4. 모델 검증 및 학습

```bash
# 수정된 레시피 파일의 유효성을 검증합니다.
uv run python main.py validate --recipe-file recipes/my_first_model.yaml

# 모델 학습을 실행합니다.
uv run python main.py train --recipe-file recipes/my_first_model.yaml

# 학습 결과 확인 (MLflow UI)
# (mmp-local-dev 환경의 docker-compose up -d가 실행되어 있어야 합니다)
open http://127.0.0.1:5002
```

### 5. 모델 배포 및 추론

```bash
# 학습에서 나온 run-id 사용 (예: abc123def456)
RUN_ID="your-run-id-here"

# 배치 추론
uv run python main.py batch-inference --run-id $RUN_ID

# 실시간 API 서빙
uv run python main.py serve-api --run-id $RUN_ID
# API 테스트: curl http://localhost:8000/predict -X POST -d '{"feature1": 1.0}'
```

---

## 📖 기본 사용법

### CLI 명령어 전체 목록

```bash
# 프로젝트 관리
uv run python main.py init [--dir ./my-project]     # 새 프로젝트 초기화
uv run python main.py validate --recipe-file <path> # 설정 파일 검증

# 레시피 가이드
uv run python main.py guide <model_class_path>       # 모델에 맞는 레시피 템플릿 생성

# 모델 개발
uv run python main.py train --recipe-file <path>    # 모델 학습
uv run python main.py train --recipe-file <path> --context-params '{"date":"2024-01-01"}'  # 동적 파라미터

# 모델 추론
uv run python main.py batch-inference --run-id <id> # 배치 추론
uv run python main.py serve-api --run-id <id>       # 실시간 API

# 시스템 검증
uv run python main.py test-contract                 # 인프라 연결 테스트
```

### Recipe 파일 작성법

Recipe는 모델의 모든 논리를 정의하는 YAML 파일입니다:

```yaml
# recipes/my_model.yaml
model:
  # 모델 클래스 (sklearn, xgboost, lightgbm 등 모든 Python 패키지)
  class_path: "sklearn.ensemble.RandomForestClassifier"
  
  # 하이퍼파라미터 (고정값 또는 최적화 범위)
  hyperparameters:
    n_estimators: 100              # 고정값
    max_depth: {type: "int", low: 3, high: 10}  # 자동 최적화 범위
  
  # 데이터 로딩
  loader:
    name: "default_loader"
    source_uri: "data/my_dataset.parquet"  # 파일 경로 또는 SQL
    adapter: storage
  
  # 데이터 전처리
  preprocessor:
    name: "default_preprocessor"
    params:
      exclude_cols: ["id", "timestamp"]
  
  # 모델 설정
  data_interface:
    task_type: "classification"    # classification, regression, causal
    target_col: "target"

# 자동 하이퍼파라미터 최적화 (선택사항)
hyperparameter_tuning:
  enabled: true
  n_trials: 50
  metric: "roc_auc"
  direction: "maximize"
```

---

## 🔧 고급 기능

### 1. 동적 SQL 템플릿 (Jinja2)

```sql
-- recipes/sql/dynamic_query.sql.j2
SELECT user_id, feature1, feature2, target
FROM my_table 
WHERE date = '{{ target_date }}'
LIMIT {{ limit | default(1000) }}
```

```bash
# 템플릿 파라미터와 함께 실행
uv run python main.py train \
  --recipe-file recipes/templated_model.yaml \
  --context-params '{"target_date": "2024-01-01", "limit": 5000}'
```

### 2. Feature Store 연동

```yaml
# recipes/feature_store_model.yaml
model:
  augmenter:
    type: "feature_store"
    features:
      - feature_namespace: "user_demographics"
        features: ["age", "country"]
      - feature_namespace: "user_behavior"
        features: ["click_rate", "conversion_rate"]
```

### 3. 환경별 설정 관리

```bash
# 환경 변수로 설정 전환
APP_ENV=local   uv run python main.py train ...  # 로컬 파일 기반
APP_ENV=dev     uv run python main.py train ...  # PostgreSQL + Redis  
APP_ENV=prod    uv run python main.py train ...  # BigQuery + Redis Labs
```

---

## 🌍 환경별 설정

### LOCAL 환경 (기본)
- **데이터**: 로컬 파일 (Parquet, CSV)
- **Feature Store**: 비활성화 (Pass-through)
- **MLflow**: 로컬 디렉토리 (`./mlruns`)
- **특징**: 빠른 실험, 외부 의존성 없음

### DEV 환경 
```bash
# mmp-local-dev 인프라 필요 (별도 설치)
git clone https://github.com/wooshikwon/mmp-local-dev.git ../mmp-local-dev
cd ../mmp-local-dev && docker-compose up -d

# DEV 환경에서 실행
APP_ENV=dev uv run python main.py train --recipe-file recipes/my_model.yaml
```

- **데이터**: PostgreSQL
- **Feature Store**: PostgreSQL + Redis
- **MLflow**: 공유 서버
- **특징**: 완전한 기능, 팀 협업

### PROD 환경
- **데이터**: BigQuery, Snowflake
- **Feature Store**: BigQuery + Redis Labs
- **MLflow**: 클라우드 스토리지
- **특징**: 확장성, 안정성

---

## 📊 지원하는 ML 프레임워크

### 분류 (Classification)
```yaml
# scikit-learn
class_path: "sklearn.ensemble.RandomForestClassifier"
class_path: "sklearn.linear_model.LogisticRegression"

# XGBoost
class_path: "xgboost.XGBClassifier"

# LightGBM  
class_path: "lightgbm.LGBMClassifier"
```

### 회귀 (Regression)
```yaml
class_path: "sklearn.ensemble.RandomForestRegressor"
class_path: "sklearn.linear_model.LinearRegression"
class_path: "xgboost.XGBRegressor"
class_path: "lightgbm.LGBMRegressor"
```

### 인과추론 (Causal Inference)
```yaml
# CausalML
class_path: "causalml.inference.meta.XGBTRegressor"
class_path: "causalml.inference.meta.TRegressor"
```

---

## 🐛 트러블슈팅

### 자주 발생하는 문제

**1. MLflow 연결 오류**
```bash
# MLflow 서버 확인
curl http://localhost:5002/health
# 환경변수 확인
echo $MLFLOW_TRACKING_URI
```

**2. 데이터 파일을 찾을 수 없음**
```bash
# 현재 경로 확인
pwd
# 데이터 파일 경로 확인 (프로젝트 루트 기준)
ls data/my_dataset.parquet
```

**3. 패키지 의존성 오류**
```bash
# 필요한 패키지 추가 설치
uv add scikit-learn xgboost lightgbm
# 또는: pip install scikit-learn xgboost lightgbm
```

**4. Feature Store 연결 오류**
```bash
# Redis 연결 확인
redis-cli ping
# PostgreSQL 연결 확인  
psql -h localhost -p 5432 -U mlpipeline_user -d mlpipeline_db
```

### 로그 확인

```bash
# 상세 로그 출력
export LOG_LEVEL=DEBUG
uv run python main.py train --recipe-file recipes/my_model.yaml

# 로그 파일 위치
tail -f logs/modern_ml_pipeline.log
```

---

## 📚 추가 문서

- **[개발자 가이드](docs/DEVELOPER_GUIDE.md)**: 심화 사용법 및 커스터마이징
- **[인프라 가이드](docs/INFRASTRUCTURE_STACKS.md)**: 환경별 인프라 설정
- **[Blueprint](blueprint.md)**: 시스템의 핵심 설계 원칙과 실제 코드 구현을 연결한 기술 청사진

---

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📞 지원 및 문의

- **이슈 제보**: [GitHub Issues](https://github.com/wooshikwon/modern-ml-pipeline/issues)
- **문서**: [Wiki](https://github.com/wooshikwon/modern-ml-pipeline/wiki)
- **이메일**: [your-email@example.com](mailto:your-email@example.com)
