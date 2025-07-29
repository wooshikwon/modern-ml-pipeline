
# Modern ML Pipeline 프로젝트 심층 분석 보고서

## 1. 개요: The Automated Excellence Vision

본 문서는 `Modern ML Pipeline` 프로젝트의 철학, 아키텍처, 핵심 소스 코드 분석을 통해 시스템의 전체 흐름과 구조에 대한 깊이 있는 이해를 제공하는 것을 목표로 한다.

프로젝트의 핵심 목표는 **"무제한적인 실험 자유도와 완전히 일관된 Wrapped Artifact 실행"**을 통해 **가독성(Readability), 신뢰성(Reliability), 확장성(Scalability)**을 갖춘 유기적인 MLOps 시스템을 구축하는 것이다.

`blueprint.md`에 명시된 이 비전은 다음과 같은 핵심 철학을 통해 구체화된다.

-   **학습과 추론의 완전한 분리**: 데이터 과학자의 실험 자유도와 운영 환경의 안정성을 동시에 보장한다.
-   **Recipe와 Config의 분리**: 모델의 '논리'와 실행되는 '인프라'를 엄격하게 분리하여 이식성을 극대화한다.
-   **완전한 재현성**: `Wrapped Artifact`를 통해 학습 시점의 모든 컨텍스트를 저장하여, 어떤 환경에서도 100% 동일한 실행을 보장한다.
-   **자동화된 엑설런스**: `Optuna`를 내장하여 하이퍼파라미터 최적화를 자동화하고, 정교한 로직을 통해 데이터 누수를 원천 차단한다.
-   **환경별 차등 기능**: `local`, `dev`, `prod` 환경에 명확한 역할과 제약을 부여하여 점진적인 개발 및 배포를 지원한다.

## 2. 핵심 아키텍처

이 시스템은 `blueprint.md`의 철학을 구현하기 위해 매우 정교하고 명확한 아키텍처를 채택하고 있다.

### 2.1. 3계층 아키텍처 (Layered Architecture)

`src` 디렉토리는 크게 3개의 논리적 계층으로 구성되어 책임이 명확하게 분리되어 있다.

1.  **`pipelines` (Layer 3 - 흐름 제어)**: `train_pipeline.py`, `inference_pipeline.py`가 위치하며, 전체 작업의 순서(e.g., 데이터 로딩 → 학습 → 저장)를 관장하는 최상위 계층.
2.  **`engine` (Layer 2 - 시스템 엔진)**: `factory.py`, `registry.py`, `artifact.py`가 위치하며, 설정 파일을 기반으로 필요한 컴포넌트를 생성하고 조립하는 시스템의 '뼈대' 역할을 한다.
3.  **`components` (Layer 1 - 작업자)**: `trainer.py`, `preprocessor.py`, `augmenter.py` 등 실제 ML 작업을 수행하는 개별 '작업자' 컴포넌트들이 위치한다.

### 2.2. '레시피는 논리, 설정은 인프라'

-   **`recipes/`**: 모델의 본질적인 논리, 즉 **"무엇을(What)"** 할 것인지를 정의한다. (e.g., 어떤 SQL로 데이터를 가져오고, 어떤 모델 클래스를 사용하며, 어떤 하이퍼파라미터로 학습할 것인가?)
-   **`config/`**: 모델이 실행될 물리적 환경, 즉 **"어디서(Where)"**, **"어떻게(How)"** 실행될 것인지를 정의한다. (`APP_ENV` 변수에 따라 동적으로 교체) (e.g., `dev` 환경의 DB 주소는 어디이고, MLflow는 어디에 연결되어 있는가?)

### 2.3. `Wrapped Artifact`: 완전한 재현성의 핵심

이 시스템의 가장 중요한 개념 중 하나로, 단순한 모델 파일이 아닌 **실행 가능한 완전한 컨텍스트를 캡슐화한 자기 완결적 아티팩트**이다.

-   **구성 요소**:
    -   **학습된 객체**: `trained_model`, `trained_preprocessor`
    -   **로직 스냅샷**:
        -   `loader_sql_snapshot`: 학습에 사용된 SQL 쿼리 전체 문자열. **추론 시 데이터 재현 및 API 스키마 생성의 유일한 근거**가 된다.
        -   `augmenter_config_snapshot`: `recipe.yaml`의 `augmenter` 섹션 전체.
        -   `model_class_path`: 동적 import를 위한 모델 클래스 경로.
    -   **최적화 결과**: `Optuna`로 찾은 최적 하이퍼파라미터, 탐색 히스토리 등.
    -   **메타데이터**: 데이터 누수 방지 방법론, 스키마 정보 등.
-   **효과**: 이 아티팩트는 외부 파일(`.sql`, `.yaml`)에 대한 의존성이 전혀 없으므로, **`run_id`만 있으면 어떤 환경에서든 100% 동일한 예측 결과를 보장**한다.

## 3. 주요 실행 흐름 분석

시스템은 `main.py`의 `Typer` 기반 CLI를 통해 제어되며, 각 커맨드는 명확한 파이프라인을 따라 실행된다.

### 3.1. 설정 파이프라인 (YAML → Settings 객체)

모든 실행의 첫 단계로, `src/settings/loaders.py`의 `load_settings_by_file` 함수를 통해 `Settings` 객체를 생성한다.

1.  **YAML 로드**: `config/base.yaml` → `config/{env}.yaml` → `recipes/{name}.yaml` 순서로 설정을 재귀적으로 병합한다. `${VAR}` 형식의 환경 변수를 자동으로 치환한다.
2.  **Jinja 렌더링**: `--context-params`가 주입되면, `recipe`의 `.sql.j2` 템플릿을 렌더링하여 SQL 문자열을 완성한다.
3.  **Pydantic 검증**: 모든 설정이 합쳐지고 렌더링된 최종 딕셔너리를 `src/settings/models.py`에 정의된 `Settings` 모델로 변환하여, 시스템이 요구하는 모든 설정의 타입과 값이 올바른지 검증한다.

### 3.2. 학습 파이프라인 (`run_training`)

1.  **MLflow 컨텍스트 시작**: 모든 파라미터, 메트릭, 아티팩트 추적을 시작한다.
2.  **Factory 생성**: `Settings` 객체로 `Factory`를 초기화한다.
3.  **데이터 로딩**: `Factory`가 `config`에 정의된 `default_loader`에 맞는 `DataAdapter`를 생성하고, `recipe.loader.source_uri`의 데이터를 읽어온다.
4.  **컴포넌트 생성**: `Factory`가 `Augmenter`, `Preprocessor`, `Model` 등 학습에 필요한 컴포넌트들을 생성한다.
5.  **학습 실행**: `Trainer` 컴포넌트가 모든 객체를 받아 학습을 진행한다.
    -   `hyperparameter_tuning.enabled`가 `True`이면, `Optuna`를 사용한 자동 최적화를 수행한다.
    -   `Preprocessor`는 **오직 Train 데이터에만 `fit`** 하여 데이터 누수를 원천 방지한다.
6.  **`Wrapped Artifact` 생성 및 저장**: `Factory`가 학습된 객체와 모든 설정 스냅샷을 담아 `PyfuncWrapper`를 생성하고, `mlflow.pyfunc.log_model`을 통해 MLflow에 최종 저장한다.

### 3.3. 배치 추론 파이프라인 (`run_batch_inference`)

1.  **`Wrapped Artifact` 로드**: `run_id`를 기반으로 `mlflow.pyfunc.load_model()`을 호출하여 학습된 아티팩트 전체를 로드한다.
2.  **데이터 로딩**: 로드된 아티팩트의 **`loader_sql_snapshot`**을 사용하여, 현재 환경의 `DataAdapter`로 추론할 데이터를 가져온다. 학습 시점과 100% 동일한 쿼리가 사용된다.
3.  **스키마 검증**: 아티팩트에 내장된 `data_schema`와 현재 로드된 데이터의 스키마를 비교하여 `Schema Drift`를 자동으로 감지한다.
4.  **예측**: `model.predict()`를 호출한다. `PyfuncWrapper` 내부에서 `run_mode="batch"` 컨텍스트에 맞게 `Augmenter`(오프라인 스토어 사용) -> `Preprocessor` -> `Model` 순서로 로직이 실행된다.
5.  **결과 저장**: 예측 결과를 `config`에 설정된 스토리지에 저장한다.

### 3.4. 실시간 서빙 (`serving/api.py`)

1.  **`Wrapped Artifact` 로드**: 배치 추론과 동일하게 `run_id`로 아티팩트를 로드한다.
2.  **동적 API 스키마 생성 (Self-Describing API)**:
    -   로드된 아티팩트의 **`loader_sql_snapshot`**을 **SQL 파서로 분석**하여 API가 입력받아야 할 PK 컬럼 목록을 동적으로 추출한다.
    -   `pydantic.create_model`을 사용하여, 추출된 PK 목록을 필드로 갖는 **FastAPI 입력 모델을 실시간으로 생성**한다.
3.  **예측**: 클라이언트로부터 JSON으로 PK를 받으면, `model.predict()`를 호출한다. `PyfuncWrapper` 내부에서 `run_mode="serving"` 컨텍스트에 맞게 `Augmenter`(온라인 스토어 사용) -> `Preprocessor` -> `Model` 순서로 로직이 실행된다.

## 4. 종합 평가 및 결론

`Modern ML Pipeline` 프로젝트는 `blueprint.md`에서 제시된 높은 수준의 철학과 비전을 실제 코드에 매우 정교하고 우아하게 구현한 인상적인 시스템이다.

-   **설계의 명확성**: '레시피는 논리, 설정은 인프라', '단순성과 명시성'과 같은 원칙들이 `settings`, `registry`, `factory` 모듈의 설계를 통해 완벽하게 구현되었다. 이는 시스템의 복잡성을 크게 낮추고 예측 가능성을 극대화한다.
-   **완벽한 재현성**: `Wrapped Artifact`와 `loader_sql_snapshot` 개념은 MLOps의 가장 큰 난제 중 하나인 '재현성' 문제를 근본적으로 해결한다. 학습과 추론(배치/실시간)이 동일한 아티팩트를 공유하면서도, 각 컨텍스트에 최적화된 방식으로 동작하는 하이브리드 설계는 이 프로젝트의 백미이다.
-   **개발자 경험**: `Typer` 기반의 직관적인 CLI, `Jinja2`를 이용한 동적 쿼리, `Optuna`를 내장한 자동화된 최적화, `Pydantic`을 통한 강력한 설정 검증 등은 데이터 과학자와 ML 엔지니어 모두의 생산성을 크게 향상시킬 수 있는 훌륭한 기능들이다.
-   **높은 완성도**: `factoringlog.md`에서 볼 수 있듯이, 지속적인 리팩토링을 통해 코드의 중복을 제거하고 구조적 일관성을 유지하려는 노력이 돋보인다. `next_step.md`의 E2E 테스트 강화 계획은 시스템의 안정성과 신뢰성을 한 단계 더 끌어올릴 것으로 기대된다.

**결론적으로, 이 프로젝트는 현대적인 MLOps의 모범 사례를 집약한, 잘 설계되고 견고하며 확장성 높은 파이프라인이다.** 앞으로의 추가 개발을 통해 더욱 발전해나갈 모습이 기대된다.

