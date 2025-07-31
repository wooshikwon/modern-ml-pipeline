알겠습니다. `next_step.md`에 정리된 개선안들을 실제 코드로 구현하기 위한 구체적이고 상세한 개발 계획 보고서를 작성해 드리겠습니다. 이 보고서는 각 개선 항목의 목표, 변경이 필요한 파일, 그리고 단계별 개발 절차를 명확히 제시하여 즉시 개발에 착수할 수 있는 가이드 역할을 할 것입니다.

---

### **MMP 시스템 개선을 위한 상세 개발 계획 보고서**

#### **1. 개요**

본 문서는 `next_step.md`에서 합의된 시스템 개선안을 실제 개발로 이행하기 위한 구체적인 실행 계획을 기술합니다. 각 과제는 **(1)핵심 로직 개선 → (2)사용자 경험 향상 → (3)견고성 및 재현성 강화**의 순서로, 시스템의 근본적인 부분부터 점진적으로 안정화하고 완성도를 높이는 방향으로 진행합니다.

---

#### **Phase 1: 핵심 로직 개선 및 명확성 강화**

시스템의 철학과 구현 사이의 잠재적 충돌을 해결하고, 동작의 명확성을 높이는 데 집중합니다.

##### **과제 1.1: 지능형 Augmenter 구현**

*   **목표**: `config`의 환경 설정이 `recipe`의 논리적 요구사항을 안전하게 덮어쓸 수 있도록 하여, 레시피 수정 없이 모든 환경에서 파이프라인이 정상 동작하도록 보장합니다.
*   **주요 변경 파일**:
    *   `src/engine/factory.py`
    *   `src/components/_augmenter/_augmenter.py`
    *   `src/components/_augmenter/_pass_through.py`

*   **상세 개발 절차**:
    1.  **`Factory` 로직 수정 (`factory.py`)**:
        *   `create_augmenter` 메서드 내부의 로직을 변경합니다.
        *   기존에는 레시피의 `augmenter.type`을 기준으로 분기했다면, 이제는 **`settings.feature_store.provider` 값을 최우선으로 확인**합니다.
        *   만약 `settings.feature_store.provider`가 `"passthrough"`이면, 레시피의 `augmenter` 설정과 관계없이 강제로 `PassThroughAugmenter`를 생성하여 반환하도록 수정합니다.
    2.  **`PassThroughAugmenter` 기능 강화 (`_pass_through.py`)**:
        *   `_augment` 메서드에 `logger`를 추가합니다.
        *   메서드 실행 시, `"INFO: 'passthrough' 모드가 활성화되어 피처 증강을 건너뜁니다."` 와 같은 로그를 출력하여, 사용자에게 현재 동작 상태를 명확히 알립니다.
    3.  **기존 `_augmenter.py` 역할 유지**: `FeastAugmenter`와 같은 실제 증강 로직은 그대로 유지합니다. `Factory`의 변경으로 인해 이 로직은 `passthrough`가 아닐 때만 호출될 것입니다.

---

##### **과제 1.2: 데이터 어댑터 타입 명시성 강화**

*   **목표**: 데이터 어댑터 선택의 모호함을 제거하고, 레시피만으로 데이터 로딩 방식을 명확히 알 수 있도록 합니다.
*   **주요 변경 파일**:
    *   `src/settings/_recipe_schema.py`
    *   `src/engine/factory.py`
    *   `config/local.yaml`
    *   모든 `recipes/**/*.yaml` 및 `tests/fixtures/recipes/**/*.yaml` 파일

*   **상세 개발 절차**:
    1.  **설정 파일 정리 (`config/local.yaml`)**:
        *   `data_adapters` 섹션의 `default_loader` 키와 값을 완전히 삭제합니다.
    2.  **레시피 스키마 수정 (`_recipe_schema.py`)**:
        *   `LoaderSettings` Pydantic 모델에 `adapter: str` 필드를 새롭게 추가합니다. 이 필드는 "sql" 또는 "storage"와 같은 값을 갖게 될 것입니다.
    3.  **`Factory` 로직 단순화 (`factory.py`)**:
        *   `create_data_adapter` 메서드를 수정합니다.
        *   더 이상 `source_uri`의 확장자나 `default_loader`를 확인하지 않습니다.
        *   오직 `settings.recipe.model.loader.adapter` 값을 기준으로, 등록된 어댑터 레지스트리에서 해당 어댑터를 찾아 생성하도록 로직을 단순화합니다.
    4.  **전체 레시피 파일 업데이트**:
        *   `recipes/` 및 `tests/fixtures/recipes/` 내의 모든 `.yaml` 파일을 열어 `loader` 섹션에 `adapter: storage` 또는 `adapter: sql` 필드를 추가합니다. (`source_uri`가 `.sql`이면 `sql`, 아니면 `storage`로 설정)

---

#### **Phase 2: 사용자 경험(UX) 향상**

시스템의 동작 과정을 사용자에게 명확하게 전달하여 "블랙박스"처럼 느껴지는 구간을 해소합니다.

##### **과제 2.1: 하이퍼파라미터 튜닝 우선순위 로깅**

*   **목표**: 하이퍼파라미터 튜닝 실행 여부가 어떤 설정에 의해 결정되었는지 명확한 로그를 남겨 사용자의 혼란을 방지합니다.
*   **주요 변경 파일**: `src/components/_trainer/_trainer.py`

*   **상세 개발 절차**:
    1.  **`Trainer` 클래스의 `train` 메서드 초입부를 수정합니다.**
    2.  실제 학습 로직에 들어가기 전, `settings.hyperparameter_tuning.enabled` 값과 `settings.recipe.model.hyperparameter_tuning.enabled` 값을 비교하는 조건문을 추가합니다.
    3.  만약 전역 설정(`config`)에 의해 튜닝이 비활성화되었다면, `logger.info("전역 설정(config)에 따라 하이퍼파라미터 튜닝이 비활성화되었습니다.")` 와 같은 로그를 출력합니다.
    4.  반대로 레시피 설정에 의해 비활성화되었다면, 그에 맞는 로그를 남깁니다.

---

##### **과제 2.2: Optuna 학습 과정 실시간 로깅**

*   **목표**: 장시간 소요될 수 있는 하이퍼파라미터 튜닝 과정의 진행 상태를 실시간으로 보여주어 사용자 경험을 개선합니다.
*   **주요 변경 파일**:
    *   `src/components/_trainer/_optimizer.py` (또는 `_trainer.py` 내 로직)
    *   `src/utils/integrations/optuna_integration.py` (신규 생성 또는 수정)

*   **상세 개발 절차**:
    1.  **콜백 함수 정의 (`optuna_integration.py`)**:
        *   `study`와 `trial` 객체를 인자로 받는 콜백 함수 `logging_callback`을 정의합니다.
        *   함수 내부에서는 `trial.number`, `trial.value`, `study.best_value` 등의 정보를 조합하여 `"Trial {}/{} 완료. 현재 점수: {:.4f}, 최고 점수: {:.4f}"` 와 같은 로그를 `logger`를 통해 출력합니다.
    2.  **`Trainer`와 콜백 연동 (`_optimizer.py`)**:
        *   `OptunaOptimizer`의 `optimize` 메서드 (또는 `Trainer`의 튜닝 실행 로직)를 수정합니다.
        *   `study.optimize()` 메서드를 호출할 때, `callbacks=[logging_callback]` 인자를 전달하여 위에서 정의한 콜백 함수를 등록합니다.

---

#### **Phase 3: 시스템 견고성 및 완전한 재현성 확보**

##### **과제 3.1: 레시피 사전 유효성 검증 강화**

*   **목표**: 데이터 로딩 등 무거운 작업을 시작하기 전에 레시피의 논리적 모순을 미리 발견하여 시간과 자원을 절약합니다.
*   **주요 변경 파일**: `src/settings/_recipe_schema.py`

*   **상세 개발 절차**:
    1.  **`RecipeSettings` 모델 수정 (`_recipe_schema.py`)**:
        *   Pydantic의 `@root_validator` 또는 최신 버전의 `@model_validator` 데코레이터를 사용하여 모델 전체를 검증하는 메서드 `validate_recipe_consistency`를 추가하거나 강화합니다.
        *   이 메서드 내부에, `task_type`과 `evaluation.metrics` 간의 호환성을 검증하는 로직을 추가합니다. (예: `if values.get('task_type') == 'classification' and 'mse' in values.get('evaluation_metrics'): raise ValueError(...)`)
        *   이 검증 로직은 `load_settings_by_file` 함수에서 Pydantic 모델이 생성될 때 자동으로 호출되어, `train` 명령어 실행 초기에 오류를 발생시킵니다.

---

##### **과제 3.2: 아티팩트에 패키지 의존성 내장**

*   **목표**: 모델 아티팩트가 자체적으로 실행 환경의 패키지 버전 정보를 포함하게 하여, 시간이 지나도 100% 동일한 환경에서 실행될 수 있도록 완전한 재현성을 확보합니다.
*   **주요 변경 파일**:
    *   `src/pipelines/train_pipeline.py`
    *   `src/utils/system/environment_check.py` (신규 생성 가능)

*   **상세 개발 절차**:
    1.  **의존성 추출 함수 구현 (`environment_check.py`)**:
        *   `subprocess` 모듈을 사용하여 `["uv", "pip", "freeze"]` 명령어를 실행하고, 그 결과를 문자열 리스트로 반환하는 `get_pip_requirements()` 함수를 구현합니다.
    2.  **`run_training` 파이프라인 수정 (`train_pipeline.py`)**:
        *   `mlflow.pyfunc.log_model` 함수를 호출하기 직전에, `pip_reqs = get_pip_requirements()`를 호출하여 현재 환경의 의존성 목록을 가져옵니다.
        *   `mlflow.pyfunc.log_model`을 호출할 때 `pip_requirements=pip_reqs` 인자를 추가하여 의존성 목록을 아티팩트와 함께 저장합니다.
    3.  이제 MLflow는 이 모델을 로드할 때, 저장된 패키지 버전 정보를 사용하여 가상 환경을 구성하거나 사용자에게 경고를 표시하여 재현성을 극대화합니다.

---

#### **Phase 4: 최종 시스템 완전성 강화**

이전 단계에서 완료된 개발 사항들을 최종 검토하고, 발견된 미구현 기능 및 설계 불일치를 해결하여 시스템의 완전성을 달성하는 데 집중합니다.

##### **과제 4.1: [최우선] 동적 하이퍼파라미터 유효성 검증 구현**

*   **목표**: 레시피에 정의된 하이퍼파라미터가 실제 모델 클래스에서 유효한지 설정 로딩 시점에 동적으로 검증하여, "fail fast" 원칙을 구현하고 시스템의 견고성을 극대화합니다.
*   **주요 변경 파일**:
    *   `src/settings/_recipe_schema.py`

*   **상세 개발 절차**:
    1.  **`ModelSettings` 스키마 수정 (`_recipe_schema.py`)**:
        *   `ModelSettings` Pydantic 모델 내에 `@model_validator(mode='after')` 데코레이터를 사용한 `validate_hyperparameters` 메서드를 추가합니다.
        *   이 메서드는 `model.class_path`를 동적으로 `import`하고, 파이썬의 `inspect.signature`를 사용하여 모델 클래스의 `__init__` 메서드가 허용하는 파라미터 목록을 추출합니다.
        *   레시피의 `hyperparameters` 딕셔너리에 있는 모든 키가 추출된 유효 파라미터 목록에 포함되어 있는지 검사합니다.
        *   만약 유효하지 않은 파라미터가 발견되면, 사용 가능한 파라미터 목록과 함께 명확한 `ValueError`를 발생시킵니다.

---

##### **과제 4.2: [차선] Evaluator 생성 로직 리팩토링**

*   **목표**: `Evaluator` 생성 책임을 `Trainer`에서 `train_pipeline.py`로 이동시켜, 시스템의 모든 핵심 컴포넌트가 동일한 의존성 주입(DI) 패턴을 따르도록 설계 일관성을 확보합니다.
*   **주요 변경 파일**:
    *   `src/pipelines/train_pipeline.py`
    *   `src/components/_trainer/_trainer.py`

*   **상세 개발 절차**:
    1.  **`Trainer` 클래스 수정 (`_trainer.py`)**:
        *   `train` 메서드 내의 `self._create_evaluator()` 호출 부분을 제거합니다.
        *   대신 `evaluator: BaseEvaluator`를 `train` 메서드의 새로운 인자로 추가합니다.
        *   `evaluate`를 호출할 때, `self.evaluator` 대신 인자로 받은 `evaluator`를 사용하도록 수정합니다.
    2.  **`train_pipeline` 수정 (`train_pipeline.py`)**:
        *   `Trainer`를 생성하기 전에, `factory.create_evaluator()`를 호출하여 `Evaluator` 인스턴스를 생성합니다.
        *   `trainer.train(...)` 메서드를 호출할 때, 새로 생성한 `evaluator` 인스턴스를 인자로 전달합니다.

---

#### **Phase 5: 문서 최신화 및 사용자 가이드 강화**

최근 완료된 모든 기능 개선 및 리팩토링 사항을 공식 문서에 반영하여, 사용자가 시스템의 현재 상태와 모든 기능을 정확히 이해하고 활용할 수 있도록 합니다.

##### **과제 5.1: `README.md` 최신화**

*   **목표**: 프로젝트의 첫인상인 `README.md`를 최신 기능 중심으로 업데이트하여, 신규 사용자가 시스템의 강력함을 즉시 인지하고 모범적인 사용 흐름을 따르도록 유도합니다.
*   **주요 변경 파일**:
    *   `README.md`

*   **상세 개발 절차**:
    1.  **"빠른 시작" 섹션 개편**:
        *   "3. 첫 번째 모델 학습" 단계의 첫 번째 명령어로 `uv run python main.py guide sklearn.ensemble.RandomForestClassifier > recipes/my_first_model.yaml`을 제시합니다.
        *   `guide` 명령어를 통해 사용자가 레시피의 구조와 유효한 하이퍼파라미터를 학습하고, 이를 기반으로 파일을 수정하는 과정을 자연스럽게 안내합니다. (예: "생성된 `my_first_model.yaml` 파일을 열어 `target_column` 등을 수정하세요.")
        *   수정된 파일을 `validate`하고 `train`하는 후속 단계를 명확히 연결합니다.
    2.  **"기본 사용법" 섹션 업데이트**:
        *   "CLI 명령어 전체 목록"에 `guide` 명령어에 대한 설명을 추가합니다. (`레시피 가이드` 소제목 사용)
        *   "Recipe 파일 작성법"의 YAML 예시 코드 블록 내 `loader` 섹션에 `adapter: storage` 필드를 추가하여 최신 스키마를 반영합니다.
    3.  **"추가 문서" 섹션 설명 개선**:
        *   `Blueprint` 링크의 설명을 "시스템의 핵심 설계 원칙과 실제 코드 구현을 연결한 기술 청사진"으로 수정하여 문서의 가치를 정확히 전달합니다.

---

##### **과제 5.2: `docs/DEVELOPER_GUIDE.md` 심화 내용 보강**

*   **목표**: 고급 사용자와 기여자를 위해, `README.md`에서 깊게 다루지 않는 신규 기능의 상세한 원리와 사용법을 제공하여 시스템 활용도를 극대화합니다.
*   **주요 변경 파일**:
    *   `docs/DEVELOPER_GUIDE.md`

*   **상세 개발 절차**:
    1.  **"핵심 컨셉: 동적 레시피 가이드 및 검증" 섹션 신설**:
        *   **가이드 (`guide`)**: `guide` 명령어의 상세한 사용법, 다양한 모델 클래스 경로(`xgboost`, `lightgbm` 등)에 대한 출력 예시, 그리고 "왜 이 기능이 강력한가"(모델 인트로스펙션 원리)에 대해 간략히 설명합니다.
        *   **자동 검증 (Validation)**: 시스템이 설정 로딩 시점에 수행하는 두 가지 핵심 유효성 검증(태스크-지표 호환성, 모델-하이퍼파라미터 호환성)에 대해 상세히 기술합니다. 각 검증 실패 시 발생하는 오류 메시지 예시와 해결 방법을 제시하여 사용자가 문제를 쉽게 해결할 수 있도록 돕습니다.
    2.  **문서 전반의 레시피 예시 최신화**:
        *   `grep` 또는 유사 도구를 사용하여 문서 내 모든 `loader:` YAML 코드 블록을 찾습니다.
        *   각 `loader` 섹션에 `source_uri`의 종류에 맞춰 `adapter: sql` 또는 `adapter: storage` 필드를 빠짐없이 추가합니다.