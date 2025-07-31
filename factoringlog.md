## 2024-05-21

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 1.1: 지능형 Augmenter 구현
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/engine/factory.py`의 `create_augmenter` 메서드를 수정하여 `settings.feature_store.provider` 설정을 최우선으로 확인.
- `provider`가 'passthrough'일 경우, 레시피 설정과 무관하게 `PassThroughAugmenter`를 강제 반환.
- 레시피에 `augmenter`가 정의되지 않은 경우에도 `PassThroughAugmenter`를 반환하여 안정성 확보.
- `src/components/_augmenter/_pass_through.py`에 명확한 정보성 로그 추가.

**** src/components/_augmenter/_pass_through.py
(근거) development_plan.md의 [과제 1.1] 및 승인된 CoT 제안서에 따라 실행.
(상세) `_augment` 메서드를 수정하고 클래스 docstring을 개선하여, 피처 증강을 건너뛰는 경우('passthrough' 모드 또는 레시피에 미정의)에 대한 정보성 로그를 명확히 추가함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/engine/factory.py
(근거) development_plan.md의 [과제 1.1] 및 승인된 CoT 제안서에 따라 실행.
(상세) `create_augmenter` 메서드의 로직을 전면 수정. `config`의 `feature_store.provider` 설정을 최우선으로 확인하여 'passthrough' 모드를 강제하고, 레시피에 `augmenter`가 정의되지 않은 엣지 케이스를 처리하여 `PassThroughAugmenter`를 반환하도록 함. 이를 통해 환경 설정이 논리 설정을 안전하게 덮어쓰도록 보장함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 1.2: 데이터 어댑터 타입 명시성 강화
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `config/local.yaml`에서 `default_loader`를 제거하고, `LoaderSettings`에 `adapter: str` 필드를 추가.
- `Factory`가 오직 `loader.adapter` 값을 기준으로 DataAdapter를 생성하도록 로직을 단순화.
- 프로젝트의 모든 레시피 파일에 `adapter: sql` 또는 `adapter: storage`를 명시적으로 추가.

**** config/local.yaml, src/settings/_recipe_schema.py, src/engine/factory.py
(근거) development_plan.md의 [과제 1.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `config`에서 `default_loader`를 제거하고, `LoaderSettings` 스키마에 `adapter: str` 필드를 추가. Factory가 `loader.adapter` 값을 기준으로만 어댑터를 생성하도록 로직을 단순화하고 명확화함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** recipes/**/*.yaml, tests/fixtures/recipes/**/*.yaml, src/cli/project_templates/recipes/**/*.yaml
(근거) development_plan.md의 [과제 1.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) 프로젝트 내 모든 레시피 파일의 `loader` 섹션에 `source_uri`의 종류에 따라 `adapter: sql` 또는 `adapter: storage`를 명시적으로 추가하여, 새로운 스키마 요구사항을 충족시킴.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 2.1: 하이퍼파라미터 튜닝 우선순위 로깅
(근거) 사용자의 '진행' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/components/_trainer/_trainer.py`의 `train` 메서드에 튜닝 실행 여부를 결정하는 로직을 명시적으로 추가.
- 최종 튜닝 실행 여부와 그 이유(config 또는 recipe 설정)를 사용자에게 명확히 알리는 정보성 로그를 추가하여 시스템 동작의 투명성을 높임.

**** src/components/_trainer/_trainer.py
(근거) development_plan.md의 [과제 2.1] 및 승인된 CoT 제안서에 따라 실행.
(상세) `train` 메서드의 로직을 리팩토링하여 HPO 실행 여부 결정 로직을 통합함. `config`와 `recipe` 설정을 모두 고려하여 튜닝 실행 여부를 결정하고, 그 결정 이유를 명확하게 로깅하도록 수정함. 이를 통해 시스템 동작의 투명성과 사용자 경험을 향상시킴.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 메서드 통합을 통해 코드의 중복이 제거되고 응집도가 높아지는 구조적 개선이 이루어짐.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 2.2: Optuna 학습 과정 실시간 로깅
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/utils/integrations/optuna_integration.py` 파일을 신규 생성하여 로깅 콜백 함수를 정의.
- `_optimizer.py`에서 `study.optimize` 호출 시 이 콜백을 등록하여, 튜닝의 각 trial마다 진행 상황이 실시간으로 로깅되도록 구현.
- Pruned된 trial과 같은 엣지 케이스를 안전하게 처리.

**** src/utils/integrations/optuna_integration.py
(근거) development_plan.md의 [과제 2.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) Optuna의 각 trial이 완료될 때마다 진행 상황(trial 번호, 점수)을 로깅하는 `logging_callback` 함수를 포함하는 신규 모듈을 생성함. Pruned trial과 같은 엣지 케이스를 안전하게 처리하도록 구현.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/components/_trainer/_optimizer.py
(근거) development_plan.md의 [과제 2.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `study.optimize()` 메서드 호출 시, 새로 생성된 `logging_callback`을 `callbacks` 인자로 전달하도록 수정함. 이를 통해 하이퍼파라미터 튜닝 과정의 실시간 피드백을 제공하여 사용자 경험을 향상시킴.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 3.1+: 동적 레시피 유효성 검증 및 CLI 가이드 생성
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- **유효성 검증**:
  - `compatibility_maps.py`를 신설하여 태스크-지표 호환성 맵을 중앙 관리.
  - `_recipe_schema.py`를 수정하여 (1)태스크-지표 검증 로직과 (2)모델 인트로스펙션을 활용한 동적 하이퍼파라미터 유효성 검증 로직을 Pydantic validator에 추가.
- **CLI 가이드**:
  - `guide` CLI 명령어를 신규 추가.
  - `guideline_recipe.yaml.j2` Jinja2 템플릿을 신규 생성.
  - `guide` 명령어는 모델 클래스 경로를 받아 인트로스펙션으로 파라미터를 분석하고, 템플릿을 렌더링하여 완전한 레시피 예시를 터미널에 출력함.
- **신규 의존성**: `jinja2` 라이브러리 추가 필요.

**** pyproject.toml
(근거) development_plan.md의 [과제 3.1+] 및 승인된 CoT 제안서에 따라 실행.
(상세) CLI 가이드 생성 기능에 필요한 `jinja2` 라이브러리를 프로젝트 의존성에 추가함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/settings/compatibility_maps.py
(근거) development_plan.md의 [과제 3.1+] 및 승인된 CoT 제안서에 따라 실행.
(상세) `TASK_METRIC_COMPATIBILITY` 맵을 정의하는 신규 모듈을 생성하여, 태스크 타입과 평가지표 간의 호환성 정보를 중앙에서 관리하도록 함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/settings/_recipe_schema.py
(근거) development_plan.md의 [과제 3.1+] 및 승인된 CoT 제안서에 따라 실행.
(상세) Pydantic `model_validator`를 사용하여 두 가지 핵심 검증 로직을 추가함: (1) 모델 인트로스펙션을 통해 레시피의 하이퍼파라미터가 실제 모델 클래스에서 유효한지 동적으로 검증. (2) 중앙 호환성 맵을 기반으로 태스크 타입과 평가지표 간의 논리적 일관성을 검증.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/cli/project_templates/guideline_recipe.yaml.j2
(근거) development_plan.md의 [과제 3.1+] 및 승인된 CoT 제안서에 따라 실행.
(상세) `guide` CLI 명령어가 동적으로 레시피 템플릿을 생성하기 위한 Jinja2 템플릿 파일을 신규 생성함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/cli/commands.py
(근거) development_plan.md의 [과제 3.1+] 및 승인된 CoT 제안서에 따라 실행.
(상세) `guide`라는 신규 CLI 명령어를 구현함. 이 명령어는 모델의 클래스 경로를 인자로 받아, 인트로스펙션을 통해 사용 가능한 하이퍼파라미터를 추출하고, Jinja2 템플릿을 렌더링하여 사용자에게 완전한 형태의 모범 레시피를 출력해 줌. 이를 통해 개발자 경험을 크게 향상시킴.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 3.2: 아티팩트에 패키지 의존성 내장
(근거) 사용자의 '오케이 진행하자' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/utils/system/environment_check.py` 파일을 신규 생성하여, `uv pip freeze`를 통해 현재 환경의 패키지 의존성을 캡처하는 `get_pip_requirements()` 함수를 구현.
- `src/pipelines/train_pipeline.py`에서 `mlflow.pyfunc.log_model` 호출 시, `get_pip_requirements()`의 결과를 `pip_requirements` 인자로 전달하여 모델 아티팩트에 의존성 정보를 포함시킴.
- `uv` 명령어 부재 등 오류 발생 시에도 파이프라인이 중단되지 않도록 안전장치를 포함.

**** src/utils/system/environment_check.py
(근거) development_plan.md의 [과제 3.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `uv pip freeze` 명령어를 사용하여 현재 Python 환경의 패키지 의존성을 캡처하는 `get_pip_requirements` 함수를 포함하는 신규 모듈을 생성함. `uv` 명령어 부재 등 오류 발생 시에도 안전하게 빈 리스트를 반환하도록 예외 처리를 포함함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/utils/integrations/mlflow_integration.py
(근거) development_plan.md의 [과제 3.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `log_enhanced_model_with_schema` 헬퍼 함수의 시그니처를 수정하여, `pip_requirements` 리스트를 인자로 받아 `mlflow.pyfunc.log_model`에 전달할 수 있도록 확장함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

**** src/pipelines/train_pipeline.py
(근거) development_plan.md의 [과제 3.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) 모델 저장 직전에 `get_pip_requirements`를 호출하여 패키지 의존성을 캡처하고, `mlflow.pyfunc.log_model` 및 관련 헬퍼 함수에 이 정보를 `pip_requirements` 인자로 전달하도록 수정함. 이를 통해 생성되는 모든 모델 아티팩트에 실행 환경 정보가 내장되어 완전한 재현성을 보장함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 4.1: 동적 하이퍼파라미터 유효성 검증 구현
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/settings/_recipe_schema.py`의 `ModelSettings` 클래스에 `@model_validator`를 추가.
- 이 validator는 모델의 `class_path`를 동적으로 임포트하고 `inspect.signature`를 사용해 `__init__` 메서드의 유효한 파라미터 목록을 추출.
- 레시피의 `hyperparameters`가 실제 모델에서 유효한지 검증하고, 유효하지 않을 경우 명확한 오류 메시지와 함께 실패시킴.
- 임포트 실패나 시그니처 분석 실패 같은 엣지 케이스를 안전하게 처리함.

**** src/settings/_recipe_schema.py
(근거) development_plan.md의 [과제 4.1] 및 승인된 CoT 제안서에 따라 실행.
(상세) `ModelSettings` 클래스에 Pydantic `model_validator`를 추가하여 동적 하이퍼파라미터 유효성 검증 기능을 구현함. 이 validator는 모델의 `class_path`를 실시간으로 임포트하고 인트로스펙션을 통해 유효한 파라미터 목록을 추출한 뒤, 레시피에 정의된 `hyperparameters`가 실제 모델에서 유효한지 검증함. 이를 통해 "fail fast" 원칙을 구현하고 시스템의 견고성을 크게 향상시킴.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 4.2: Evaluator 생성 로직 리팩토링
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `src/components/_trainer/_trainer.py`를 리팩토링하여, `_create_evaluator` 메서드를 제거하고 `train` 메서드가 `BaseEvaluator` 인스턴스를 인자로 받도록 수정.
- `src/pipelines/train_pipeline.py`를 수정하여, 다른 컴포넌트와 마찬가지로 `Factory`를 통해 `Evaluator`를 생성하고, 이를 `Trainer`에 주입함.
- 이를 통해 모든 핵심 컴포넌트가 일관된 의존성 주입 패턴을 따르도록 하여 설계 일관성을 확보함.

**** src/components/_trainer/_trainer.py
(근거) development_plan.md의 [과제 4.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `_create_evaluator` 메서드를 제거하고 `train` 메서드가 `BaseEvaluator` 인스턴스를 직접 인자로 받도록 리팩토링함. 이를 통해 `Trainer`의 생성 책임을 분리하고 의존성 주입 패턴을 일관되게 적용함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 리팩토링을 통해 테스트 용이성이 향상됨.

**** src/pipelines/train_pipeline.py
(근거) development_plan.md의 [과제 4.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `Factory`를 통해 `Evaluator`를 생성하고, 이를 `Trainer`의 `train` 메서드에 인자로 주입하도록 수정함. 이를 통해 모든 핵심 컴포넌트가 파이프라인 레벨에서 일관된 방식으로 생성 및 조립되도록 설계 일관성을 확보함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 5.1: README.md 최신화
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- "빠른 시작" 섹션을 개편하여 `guide` 명령어를 중심으로 한 사용자 워크플로우를 제시.
- "기본 사용법" 섹션에 `guide` 명령어 설명을 추가하고, 레시피 예시에 `adapter` 필드를 반영.
- "추가 문서" 섹션의 `Blueprint` 문서 설명을 현재 가치에 맞게 수정.

**** README.md
(근거) development_plan.md의 [과제 5.1] 및 승인된 CoT 제안서에 따라 실행.
(상세) `README.md`의 "빠른 시작" 섹션을 `guide` CLI 명령어를 중심으로 재구성하고, 명령어 목록과 레시피 예시에 최신 시스템 사양(`guide` 명령어, `adapter` 필드)을 반영함. `Blueprint` 문서에 대한 설명도 현재 가치에 맞게 갱신함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.

작업 계획: [MMP 시스템 개선]
[PLAN] development_plan.md - 과제 5.2: docs/DEVELOPER_GUIDE.md 심화 내용 보강
(근거) 사용자의 'confirm' 승인에 따라 CoT 제안서 기반 실행을 시작함.
(CoT 요약)
- `guide` 명령어와 `자동 유효성 검증` 기능의 원리 및 상세 사용법을 설명하는 새로운 "핵심 컨셉" 섹션을 추가.
- 문서 전체에 있는 모든 레시피 예시 코드의 `loader` 섹션에 최신 스키마인 `adapter` 필드를 추가하여 기술적 정확성을 확보.

**** docs/DEVELOPER_GUIDE.md
(근거) development_plan.md의 [과제 5.2] 및 승인된 CoT 제안서에 따라 실행.
(상세) `guide` 명령어와 자동 유효성 검증 기능에 대한 상세한 설명을 담은 "핵심 컨셉" 섹션을 신설함. 또한, 문서 전체에 있는 모든 레시피 코드 예시에 `adapter` 필드를 추가하여 최신 스키마를 반영하고 기술적 정확성을 확보함.
(자가 비평) 3단계 자가 비평 프로토콜을 통과함. 수정 사항 없음.
