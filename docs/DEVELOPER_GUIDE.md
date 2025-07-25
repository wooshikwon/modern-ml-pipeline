# Modern ML Pipeline: 개발자 가이드

이 문서는 우리 팀의 개발자가 이 파이프라인을 사용하여 **새로운 모델을 추가하고, 실험하며, 배포하는 전체 과정**을 안내하는 실용적인 가이드입니다. `README.md`를 통해 개발 환경 설정을 마쳤다고 가정합니다.

## 1. 황금률: "당신은 레시피 파일의 주인입니다"

우리 파이프라인의 가장 중요한 설계 철학은 **"모델의 모든 논리적 정의는 YAML 레시피 파일 안에서 완결된다"**는 것입니다. 이 파일은 프로젝트 내 `recipes/`에 위치하거나, **원하는 어떤 경로에서든 지정하여 로드**할 수 있습니다.

*   **당신의 핵심 작업 공간:** 대부분의 경우, 당신은 레시피 YAML 파일과 관련 SQL 파일만 생성하거나 수정하게 될 것입니다.
*   **수정할 필요 없는 엔진:** `src/` 디렉토리의 코드는 파이프라인의 "엔진"입니다. 우리 파이프라인은 모델 클래스를 동적으로 로드하므로, 새로운 모델을 추가하기 위해 **`src/` 디렉토리의 코드를 수정하거나 팩토리에 무언가를 등록할 필요가 전혀 없습니다.**

## 2. 새로운 모델 추가 워크플로우: `my_uplift_model` 만들기

Pandas DataFrame 기반의 `fit/predict` 인터페이스를 가진 어떤 모델이든 레시피 파일 하나만으로 즉시 파이프라인에 통합할 수 있습니다. 새로운 업리프트 모델을 추가하는 과정을 예시로 설명합니다.

### 1단계: "올인원" 레시피 작성 (`recipes/`)

모든 것은 YAML 파일 하나를 작성하는 것에서 시작됩니다. `recipes/models/causal/my_uplift_model.yaml` 파일을 생성하고, 모델의 모든 것을 선언적으로 정의합니다.

```yaml
# recipes/models/causal/my_uplift_model.yaml

# 1. 모델 클래스와 하이퍼파라미터 정의
model:
  # Python 경로를 통해 어떤 모델이든 동적으로 직접 임포트합니다.
  class_path: "causalml.inference.meta.XGBTRegressor"
  
  # Optuna를 통한 자동 최적화를 위해 하이퍼파라미터 범위를 지정합니다.
  hyperparameters:
    learning_rate: {type: "float", low: 0.01, high: 0.2, log: true}
    n_estimators: {type: "int", low: 100, high: 800}
    max_depth: {type: "int", low: 3, high: 8}

# 2. 하이퍼파라미터 튜닝 설정
hyperparameter_tuning:
  enabled: true
  n_trials: 30
  metric: "qini_score" # 이 모델에 맞는 평가 지표
  direction: "maximize"

# 3. 데이터 로더 정의 (예측 대상 정의)
loader:
  # DEV/PROD 환경에서는 이 SQL을 실행하여 예측 대상을 가져옵니다.
  source_uri: "recipes/sql/loaders/user_with_recent_session.sql"

# 4. 피처 증강 정의 (예측 대상에 살 붙이기)
augmenter:
  # DEV/PROD 환경에서는 Feature Store를 사용합니다.
  type: "feature_store" 
  features:
    - feature_namespace: "user_demographics"
      features: ["age", "country_code"]
    - feature_namespace: "user_purchase_summary"
      features: ["ltv", "total_purchase_count"]
    - feature_namespace: "session_summary"
      features: ["time_on_page_seconds", "click_count"]

# 5. 데이터 명세 정의
data_interface:
  task_type: "causal"
  target_col: "outcome"
  treatment_col: "grp" 
  treatment_value: "treatment"

# ... (필요하다면 전처리기(preprocessor) 섹션 추가) ...
```

이것으로 새로운 모델 추가 작업은 모두 끝났습니다. **`src/` 코드는 단 한 줄도 수정할 필요가 없습니다.**

## 3. 개발 생명주기: Local -> Dev

1.  **로컬 개발 (`APP_ENV=local`):**
    *   `config/local.yaml`에 정의된 대로, `mmp-local-dev` 같은 외부 인프라 없이 독립적으로 실행됩니다.
    *   `loader`와 `augmenter`는 실제 동작 대신, `data/` 디렉토리의 샘플 데이터를 그대로 통과(pass-through)시킵니다.
    *   이를 통해 모델의 핵심 로직, 데이터 명세, 전처리 과정 등을 아주 빠르게 디버깅하고 검증할 수 있습니다.
    ```bash
    # LOCAL 환경에서 학습 실행 (빠른 검증)
    uv run python main.py train --recipe-file recipes/models/causal/my_uplift_model.yaml
    ```

2.  **개발 서버 테스트 (`APP_ENV=dev`):**
    *   로컬 개발이 완료되면, `dev` 환경에서 실제 인프라와의 연동을 테스트합니다.
    *   `setup-dev-environment.sh`를 사용하여 `mmp-local-dev` 스택(PostgreSQL, Redis, MLflow)을 시작합니다.
    *   `dev` 환경에서는 `loader`가 실제 SQL을 실행하고, `augmenter`가 Feature Store에 연결되어 동작합니다.
    ```bash
    # DEV 환경 시작
    ./setup-dev-environment.sh start

    # DEV 환경에서 학습 실행 (실제 인프라 연동)
    APP_ENV=dev uv run python main.py train --recipe-file recipes/models/causal/my_uplift_model.yaml
    ```

이러한 **"코드는 동일, 환경 설정만 변경"**하는 방식을 통해, 우리는 매우 안정적이고 예측 가능한 방식으로 모델을 개발하고 검증할 수 있습니다.
