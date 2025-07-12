# Modern ML Pipeline: A Blueprint for Production-Ready MLOps

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

이 프로젝트는 단순한 코드의 집합이 아니라, 현대적인 MLOps 환경에서 머신러닝 파이프라인이 갖춰야 할 핵심 원칙과 구조를 담은 **살아있는 청사진(Living Blueprint)** 입니다.

## 的核心哲学 (Core Philosophy)

우리는 다음과 같은 원칙 위에 이 시스템을 구축했습니다.

1.  **모델 레시피 중심 (Recipe-Centric):** 하나의 모델을 정의하는 모든 것(데이터, 피처링, 전처리, 하이퍼파라미터)은 단일 `recipe/*.yaml` 파일 안에서 완결됩니다.
2.  **계층화된 설정 관리 (Hierarchical Configuration):** `config/base.yaml`을 기본으로, `APP_ENV` 환경 변수에 따라 `config/dev.yaml` 또는 `config/prod.yaml`을 덧씌워 환경을 명시적이고 안전하게 관리합니다.
3.  **환경 인지 컴포넌트 (Environment-Aware):** 파이프라인은 현재 실행 환경을 인지하여, 코드 변경 없이 로컬과 클라우드 환경 간의 전환을 자동으로 수행합니다.
4.  **강건한 스키마 계약 (Robust Schema Contract):** 모델 실행 직전, 데이터 스키마를 검증하여 데이터 불일치로 인한 오류를 사전에 방지하고 "빠른 실패(Fail-fast)"를 유도합니다.
5.  **컨텍스트에 따른 아티팩트 전략 분리 (Context-Aware Artifacts):** 실시간 서빙은 속도를 위해 통합된 모델(`PyfuncWrapper`)을, 배치 추론은 투명성을 위해 개별 아티팩트를 재조립하여 사용합니다.

## 아키텍처 다이어그램 (Architecture Diagram)

```
/
├── config/                     # 1. 운영 환경 설정 (DevOps/MLOps)
│   ├── base.yaml               #    - 모든 환경의 공통 기반이자, 로컬 개발의 기본값
│   ├── dev.yaml                #    - 개발(dev) 환경에서 덮어쓸 설정
│   └── prod.yaml               #    - 운영(prod) 환경에서 덮어쓸 설정
├── recipes/                    # 2. 모델 레시피 디렉토리 (DS/MLE의 핵심 작업 공간)
│   ├── example_recipe.yaml     #    - 새로운 레시피 작성을 위한 마스터 템플릿
│   ├── xgboost_x_learner.yaml
│   └── sql/                    #    - 레시피들이 참조하는 SQL 스크립트
│       ├── loaders/
│       └── features/
├── src/                        # 3. 파이프라인 실행 코드 (Python)
│   ├── core/                   #    - Loader, Augmenter, Trainer 등 핵심 로직
│   ├── pipelines/              #    - train, inference 등 파이프라인 흐름 제어
│   ├── utils/                  #    - artifact, mlflow 등 공통 유틸리티
│   └── ...
├── .env.example                # 환경 변수 설정을 위한 템플릿
├── pyproject.toml              # 프로젝트 메타데이터 및 의존성 정의
├── requirements.lock           # uv로 생성된 프로덕션용 의존성 잠금 파일
└── requirements-dev.lock       # uv로 생성된 개발용 의존성 잠금 파일
```

## 설치 및 개발 환경 설정 (Installation & Setup)

이 프로젝트는 `uv`를 사용하여 매우 빠르고 재현 가능한 개발 환경을 구축합니다.

1.  **저장소 복제 (Clone Repository):**
    ```bash
    git clone <repository_url>
    cd modern-ml-pipeline
    ```

2.  **가상 환경 생성 및 활성화:**
    Python 3.10 이상을 권장합니다.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **`uv` 설치:**
    Python 패키지 관리자인 `pip`을 사용하여 `uv`를 설치합니다.
    ```bash
    pip install uv
    ```

4.  **의존성 동기화:**
    `uv`를 사용하여 `requirements-dev.lock` 파일에 명시된 모든 개발 및 프로덕션 의존성을 가상 환경에 설치합니다.
    ```bash
    uv pip sync requirements-dev.lock
    ```

5.  **`.env` 파일 설정:**
    `.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 로컬 개발 환경에 필요한 환경 변수를 설정합니다.
    ```bash
    cp .env.example .env
    # nano .env 또는 다른 편집기로 .env 파일 수정
    ```

## 파이프라인 실행 방법 (Usage)

모든 파이프라인은 `main.py`를 통해 실행됩니다.

### 모델 학습 (Training)

```bash
# 'xgboost_x_learner' 레시피로 모델 학습 실행
python main.py train --model-name "xgboost_x_learner"

# 특정 컨텍스트 파라미터(e.g., 캠페인 ID)를 주입하여 학습 실행
python main.py train \
    --model-name "xgboost_x_learner" \
    --context-params '{"campaign_id": "special_promo_2024_q2"}'
```

### 배치 추론 (Batch Inference)

```bash
# 특정 run_id의 아티팩트를 사용하여 배치 추론 실행
python main.py batch-inference \
    --model-name "xgboost_x_learner" \
    --run-id "abcdef1234567890" \
    --context-params '{"campaign_id": "retention_campaign_2024_q3"}'
```

### API 서버 실행 (API Serving)

```bash
# 기본값(xgboost_x_learner, Production 스테이지)으로 API 서버 실행
python main.py serve-api

# dev 환경에서 다른 모델로 API 서버 실행
APP_ENV=dev python main.py serve-api --model-name "causal_forest"
```