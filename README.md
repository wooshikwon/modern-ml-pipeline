# 🚀 Modern ML Pipeline (Blueprint v17.0)

**"Automated Excellence Vision" - "코드로서의 계약"으로 구현된 차세대 MLOps 플랫폼**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Blueprint v17.0](https://img.shields.io/badge/blueprint-v17.0-green.svg)](blueprint.md)
[![Contract v1.0](https://img.shields.io/badge/contract-v1.0-purple.svg)](tests/integration/expected-dev-contract.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 프로젝트 개요

Modern ML Pipeline은 **무제한적인 실험 자유도**와 **완전히 일관된 재현성**을 동시에 보장하는 혁신적인 MLOps 플랫폼입니다. Blueprint v17.0의 10대 핵심 설계 원칙과 **"코드로서의 계약(Contract as Code)"** 아키텍처를 통해 **자동화된 하이퍼파라미터 최적화**, **환경별 차등적 기능 분리**, **완전한 Data Leakage 방지**를 구현했습니다.

### 🎯 Blueprint v17.0 핵심 철학

```yaml
LOCAL 환경: "제약은 단순함을 낳고, 단순함은 집중을 낳는다"
  → 빠른 실험과 디버깅의 성지

DEV 환경: "모든 기능이 완전히 작동하는 안전한 실험실"  
  → `mmp-local-dev`와 연동되는 통합 개발 허브

PROD 환경: "성능, 안정성, 관측 가능성의 완벽한 삼위일체"
  → 확장성과 안정성의 정점
```

---

## 🚀 빠른 시작 (5분 개발 환경 설정)

이 프로젝트는 ML 로직을 담당하는 `modern-ml-pipeline`과, 인프라를 담당하는 `mmp-local-dev` 두 개의 저장소로 구성됩니다.

### 1단계: 저장소 클론

```bash
# 이 저장소 (ML 파이프라인)
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
# 인프라 저장소
git clone https://github.com/wooshikwon/mmp-local-dev.git ../mmp-local-dev

cd modern-ml-pipeline
```

### 2단계: 개발 환경 시작

새로 만든 `setup-dev-environment.sh` 관리자 스크립트를 사용하여 `mmp-local-dev` 인프라(PostgreSQL, Redis, MLflow, Feast)를 시작합니다.

```bash
# DEV 환경 시작 (../mmp-local-dev/setup.sh를 자동으로 실행)
./setup-dev-environment.sh start
```

### 3단계: 첫 번째 실험 실행

```bash
# 가상환경 활성화 및 의존성 설치
uv venv && uv sync

# DEV 환경에서 학습 실행
APP_ENV=dev uv run python main.py train --recipe-file recipes/models/classification/local_test.yaml

# 결과 확인
open http://localhost:5000  # MLflow UI
```

---

## 🏗️ 아키텍처 하이라이트

### 10대 핵심 설계 원칙

| 원칙 | 내용 | 혜택 |
|------|------|------|
| **1. 레시피는 논리, 설정은 인프라** | 모델 로직과 인프라 완전 분리 | 환경 무관한 재현성 |
| **2. 통합 데이터 어댑터** | 모든 데이터 소스 표준화 | BigQuery↔S3↔Local 즉시 전환 |
| **3. URI 기반 동적 팩토리** | 선언적 설정으로 자동 구성 | 코드 수정 없는 확장성 |
| **4. 순수 로직 아티팩트** | 환경 독립적 Wrapped Model | 100% 동일 실행 보장 |
| **5. 컨텍스트 주입 Augmenter** | 배치/실시간 동일 로직 | Feature Store 완벽 활용 |
| **6. 자기 기술 API** | SQL 파싱으로 API 자동 생성 | 스키마 변경 무관한 서빙 |
| **7. 하이브리드 통합 인터페이스** | SQL 자유도 + Feature Store | 최고의 유연성과 일관성 |
| **8. 자동 HPO + Data Leakage 방지** | Optuna 통합 + Train-only Fit | 최고 성능 + 완전한 안전성 |
| **9. 환경별 차등적 기능 분리** | LOCAL/DEV/PROD 맞춤 기능 | 점진적 복잡성 증가 |
| **10. "코드로서의 계약"** | `dev-contract.yml` 기반 자동 검증 | 견고한 양방향 호환성 보장 |

### 혁신적인 기능들

#### 🤖 자동화된 하이퍼파라미터 최적화
```yaml
# Recipe에서 범위만 정의하면 자동 최적화
hyperparameters:
  learning_rate: {type: "float", low: 0.01, high: 0.3, log: true}
  n_estimators: {type: "int", low: 50, high: 1000}

hyperparameter_tuning:
  enabled: true
  n_trials: 50
  metric: "roc_auc"
```

#### 🏪 완전한 Feature Store 통합
```yaml
# 환경별 Feature Store 자동 연결
augmenter:
  type: "feature_store"
  features:
    - feature_namespace: "user_demographics"
      features: ["age", "country_code"]
    - feature_namespace: "product_details"
      features: ["price", "category"]
```

#### 🔄 환경별 원활한 전환
```bash
# 동일한 Recipe, 다른 환경
APP_ENV=local python main.py train --recipe-file my_experiment    # 빠른 프로토타입
APP_ENV=dev python main.py train --recipe-file my_experiment      # 완전한 기능
APP_ENV=prod python main.py train --recipe-file my_experiment     # 운영 환경
```

---

## 🎮 사용법

### 기본 워크플로우

```bash
# 1. 학습 (자동 HPO + Feature Store)
APP_ENV=dev python main.py train --recipe-file models/classification/xgboost_classifier

# 2. 배치 추론 (동일한 Wrapped Artifact)
APP_ENV=dev python main.py batch-inference --run-id <run_id> --input-file data/test.parquet

# 3. API 서빙 (자기 기술 API)
APP_ENV=dev python main.py serve-api --run-id <run_id>

# 4. 모델 평가
APP_ENV=dev python main.py evaluate --run-id <run_id> --input-file data/test.parquet
```

### 지원하는 모델 생태계

**분류 (Classification)**
- Scikit-learn: RandomForest, LogisticRegression, SVM
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- 딥러닝: Neural Networks (Scikit-learn MLPClassifier)

**회귀 (Regression)**  
- 선형: LinearRegression, Ridge, Lasso, ElasticNet
- 트리: RandomForest, XGBoost, LightGBM
- 커널: SVR

**인과추론/업리프트 (Causal Inference)**
- CausalML: XGBTRegressor, S-Learner, T-Learner

**클러스터링 (Clustering)**
- K-Means, DBSCAN, Hierarchical Clustering

모든 모델은 **Recipe YAML 파일 하나로 즉시 실험 가능**하며, **자동 하이퍼파라미터 최적화**를 지원합니다.

---

## 📊 개발환경 관리

`setup-dev-environment.sh` 스크립트를 통해 `modern-ml-pipeline` 디렉토리를 벗어나지 않고 DEV 환경을 편리하게 관리할 수 있습니다.

```bash
# DEV 환경 상태 확인
./setup-dev-environment.sh status

# DEV 환경 중지
./setup-dev-environment.sh stop

# DEV 환경 완전 삭제 (볼륨 포함)
./setup-dev-environment.sh clean

# DEV 환경 재시작
./setup-dev-environment.sh start

# DEV 환경이 계약을 준수하는지 테스트
./setup-dev-environment.sh test
```

---

## 📁 프로젝트 구조

```
modern-ml-pipeline/
├── 📊 config/                  # 환경별 인프라 설정
│   ├── base.yaml              # 공통 기본 설정
│   ├── dev.yaml               # DEV 환경 (Feature Store 포함)
│   └── prod.yaml              # PROD 환경 (BigQuery + Redis Labs)
├── 🧪 recipes/                # 모델 실험 정의 (논리)
│   ├── models/                # 카테고리별 모델 Recipe
│   │   ├── classification/    # 분류 모델들
│   │   ├── regression/        # 회귀 모델들
│   │   └── causal/           # 인과추론 모델들
│   └── sql/                   # Spine 생성용 SQL
├── 🔧 src/                    # 핵심 엔진
│   ├── core/                  # Factory, Trainer, Augmenter
│   ├── interface/             # 추상 기본 클래스 (ABC)
│   ├── pipelines/             # Train/Inference 파이프라인
│   ├── settings/              # 설정 관리 (Pydantic)
│   └── utils/                 # 어댑터 & 시스템 유틸리티
├── 🚀 serving/                # API 서빙
├── 🧪 tests/                  # 전체 테스트 스위트
│   └── integration/
│       └── expected-dev-contract.yml # 소비자 측 기대 계약서
├── 📋 main.py                 # 단일 CLI 진입점
├── 🛠️ setup-dev-environment.sh # DEV 환경 관리 스크립트
└── 📖 blueprint.md            # 전체 아키텍처 설계 문서
```

---

## 🔬 고급 사용법

### 커스텀 모델 추가

```yaml
# recipes/my_custom_model.yaml
model:
  class_path: "your_package.YourCustomModel"  # 동적 import
  hyperparameters:
    param1: {type: "float", low: 0.1, high: 1.0}
    param2: {type: "int", low: 10, high: 100}

# pandas DataFrame 기반 fit/predict 인터페이스만 구현하면 즉시 사용 가능
```

### 환경별 설정 커스터마이징

```yaml
# config/my_env.yaml
database:
  host: "my-custom-db.com"
  
feature_store:
  feast_config:
    offline_store:
      type: "snowflake"
      # Snowflake 설정...

# 사용법
APP_ENV=my_env python main.py train --recipe-file my_model
```

### API 서빙 고급 활용

```bash
# 자동 생성된 API 스키마 확인
curl http://localhost:8000/docs

# 실시간 예측 (Feature Store 자동 조회)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "123", "product_id": "456"}'
```

---

## 🧪 테스트

```bash
# 전체 테스트 스위트
python -m pytest tests/ -v

# 특정 컴포넌트 테스트
python -m pytest tests/core/test_factory.py -v

# 통합 테스트 (소비자 측 계약 검증 포함)
pytest tests/integration/ -v

# 인프라 자체 테스트 (공급자 측 계약 검증)
(cd ../mmp-local-dev && uv run python test-integration.py)
```

---

## 📚 문서

- **[Blueprint v17.0 전체 문서](blueprint.md)** - 10대 설계 원칙과 철학
- **[개발 환경 계약서 (원본)](../mmp-local-dev/dev-contract.yml)** - `mmp-local-dev`가 제공하는 서비스 명세
- **[개발자 가이드](docs/DEVELOPER_GUIDE.md)** - 상세 개발 가이드
- **[API 문서](http://localhost:8000/docs)** - FastAPI 자동 생성 문서 (서빙 시)

---

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 개발 환경 설정

```bash
# 개발용 의존성 설치
pip install -r requirements-dev.lock

# Pre-commit hooks 설정
pre-commit install

# 개발환경 실행
./setup-dev-environment.sh start
```

---

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Blueprint v17.0의 혁신

이 프로젝트는 다음과 같은 MLOps 분야의 혁신을 달성했습니다:

- 🚀 **완전한 재현성**: 어떤 환경에서도 100% 동일한 실행 결과
- 🤖 **자동화된 최적화**: 수동 튜닝의 한계를 뛰어넘는 Optuna 통합
- 🏪 **오픈소스 Feature Store**: 벤더 종속성 없는 Feast 기반 아키텍처
- 🔄 **환경별 최적화**: LOCAL/DEV/PROD 각각의 목적에 맞춘 차별화
- 🛡️ **완전한 안전성**: Data Leakage 원천 차단 및 투명한 검증
- 🌐 **무제한 확장성**: 로컬부터 글로벌 엔터프라이즈까지
- **"코드로서의 계약"**: `dev-contract.yml` 기반 자동 검증으로 견고한 호환성

**Modern ML Pipeline으로 MLOps의 새로운 표준을 경험하세요!** ✨
