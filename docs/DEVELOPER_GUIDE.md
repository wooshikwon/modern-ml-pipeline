# Modern ML Pipeline: 개발자 가이드

이 문서는 우리 팀의 개발자가 이 파이프라인을 사용하여 **새로운 모델을 추가하고, 실험하며, 배포하는 전체 과정**을 안내하는 실용적인 가이드입니다. `README.md`를 통해 개발 환경 설정을 마쳤다고 가정합니다.

## 1. 황금률: "당신은 `recipes/` 디렉토리의 주인입니다"

우리 파이프라인의 가장 중요한 설계 철학은 **"모델의 모든 논리적 정의는 `recipes/` 디렉토리 안에서 완결된다"**는 것입니다.

*   **수정해야 할 파일:** 대부분의 경우, 당신은 `recipes/` 디렉토리 안의 파일들만 생성하거나 수정하게 될 것입니다.
    *   `recipes/my_new_model.yaml` (새로운 레시피 작성)
    *   `recipes/sql/**/*.sql` (레시피에 필요한 SQL 작성)
*   **건드리지 말아야 할 파일:** `src/` 디렉토리의 코드는 파이프라인의 "엔진"입니다. 특별한 기능 개선이 아닌 이상, 이 디렉토리의 코드를 수정할 필요는 거의 없습니다.

## 2. 새로운 모델 추가 워크플로우: `my_causal_model` 만��기

새로운 인과추론 모델 `my_causal_model`을 추가하는 과정을 단계별로 따라가 보겠습니다.

### 1단계: 모델 구현체 작성 (`src/models/`)

먼저, 모델의 핵심 알고리즘을 담은 Python 클래스를 작성합니다.

1.  `src/models/my_causal_model.py` 파일을 생성합니다.
2.  `src/interface/base_model.py`의 `BaseModel`을 상속받아, `fit`과 `predict` 메서드를 구현합니다.

    ```python
    # src/models/my_causal_model.py
    from src.interface.base_model import BaseModel
    # ... (필요한 라이브러리 임포트) ...

    class MyCausalModel(BaseModel):
        def __init__(self, settings: Settings):
            # 레시피의 hyperparameters를 사용하여 모델 초기화
            self.params = settings.model.hyperparameters.__root__
            self.model = SomeCausalLibrary(**self.params)

        def fit(self, X, y, treatment):
            self.model.fit(X, y, treatment)
            return self

        def predict(self, X):
            return self.model.predict(X)
    ```

3.  `src/core/factory.py`의 `create_model` 함수에 새로운 모델을 등록합니다.

    ```python
    # src/core/factory.py
    # ...
    from src.models.my_causal_model import MyCausalModel

    class Factory:
        def create_model(self) -> BaseModel:
            # ...
            if model_name == "my_causal_model":
                return MyCausalModel(settings=self.settings)
            # ...
    ```

### 2단계: SQL 스크립트 작성 (`recipes/sql/`)

다음으로, 이 모델이 사용할 데이터와 피처를 정의하는 SQL 파일을 작성합니다.

1.  **로더 SQL 작성:** `recipes/sql/loaders/my_model_loader.sql` 파일을 생성합니다. 이 쿼리는 학습/추론의 대상이 될 `member_id`와 `event_timestamp` 등을 정의합니다.
2.  **Augmenter SQL 작성:** `recipes/sql/features/my_model_features.sql` 파일을 생성합니다. 이 쿼리는 `{{ temp_target_table_id }}`를 사용하여 로더의 결과와 조인하고, 필요한 피처를 증강하는 로직을 담습니다.

### 3단계: "올인원" 레시피 작성 (`recipes/`)

이제 모든 재료를 조립할 시간입니다. `recipes/my_causal_model.yaml` 파일을 생성하고, 모델의 모든 것을 정의합니다.

```yaml
# recipes/my_causal_model.yaml
name: "my_causal_model"

loader:
  name: "my_data_source"
  source_sql_path: "recipes/sql/loaders/my_model_loader.sql"
  local_override_path: "local/data/my_sample_data.csv"

augmenter:
  name: "my_feature_set"
  source_template_path: "recipes/sql/features/my_model_features.sql"
  local_override_path: "local/data/my_sample_features.parquet"

preprocessor: # 전처리가 필요 없다면 이 섹션을 통째로 삭제 (Pass-through)
  name: "my_preprocessor"
  params:
    exclude_cols: ["member_id", "event_timestamp"]

data_interface:
  features:
    # 최종적으로 모델에 들어갈 피처와 타입을 모두 명시
    feature_a: "numeric"
    feature_b: "category"
  target_col: "outcome"
  treatment_col: "grp"
  treatment_value: "treatment"

hyperparameters:
  # ... (MyCausalModel이 사용할 하이퍼파라미터) ...
```

이것으로 새로운 모델 추가 작업은 모두 끝났습니다.

## 3. 개발 생명주기: Local -> Dev -> Prod

1.  **로컬 개발 (`APP_ENV=local`):**
    *   `local_override_path`에 지정된 작은 규모의 샘플 데이터(`my_sample_data.csv`, `my_sample_features.parquet`)를 사용하여, 코드 변경이 있을 때마다 로컬에서 빠르게 파이프라인을 실행하고 디버깅합니다.
    *   이 단계에서는 클라우드 리소스에 전혀 접근하지 않으므로, 비용 걱정 없이 안전하게 개발할 수 있습니다.
    ```bash
    # 로컬에서 학습 실행
    python main.py train --model-name "my_causal_model"
    ```

2.  **개발 서버 테스트 (`APP_ENV=dev`):**
    *   로컬 개발이 완료되면, 코드를 Git에 푸시하고 CI/CD 파이프라인을 통해 `dev` 환경에 배포합니다.
    *   `dev` 환경에서는 `source_sql_path`와 `source_template_path`가 사용되어, 실제 개발용 데이터베이스(e.g., BigQuery dev project)를 대상으로 전체 파이프라인이 정상적으로 동작하는지 검증합니다.
    ```bash
    # dev 환경에서 배치 추론 실행 (CI/CD 스크립트 예시)
    export APP_ENV=dev
    export GCP_PROJECT_ID="your-dev-gcp-project-id"
    # ... (기타 환경 변수 설정) ...
    python main.py batch-inference --model-name "my_causal_model" --run-id "..."
    ```

3.  **운영 배포 (`APP_ENV=prod`):**
    *   `dev` 환경에서 안정성이 검증되면, 동일한 코드와 아티팩트를 `prod` 환경으로 승격시킵니다.
    *   `prod` 환경에서는 `config/prod.yaml`에 정의된 실제 운영 DB와 MLflow 서버를 사용하여, 최종적으로 파이프라인이 실행됩니다.

이러한 "코드와 레시피는 동일, 환경 설정만 변경"하는 방식을 통해, 우리는 매우 안정적이고 예측 가능한 방식으로 모델을 개발하고 배포할 수 있습니다.
