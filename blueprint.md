# Modern ML Pipeline Blueprint

이 문서는 Modern ML Pipeline(MMP)의 핵심 설계 철학, 아키텍처, 그리고 실행 흐름을 기술하는 기술 청사진입니다. 이 시스템이 '왜' 이렇게 설계되었고, 그 철학이 '어떻게' 코드로 구현되었는지 명확하게 연결하여 설명하는 것을 목표로 합니다.

---

## 1. 핵심 설계 철학 (Core Design Philosophy)

MMP는 세 가지 핵심 원칙 위에 세워졌으며, 이 원칙들은 시스템 아키텍처 전반에 걸쳐 일관되게 적용됩니다.

### 1.1. 원칙 1: 설정과 논리의 분리 (Recipe is Logic, Config is Infra)

모델의 본질적인 로직과 그것이 실행될 환경의 인프라 설정을 엄격하게 분리합니다.

-   **철학**: `recipes/*.yaml`은 모델의 **논리(Logic)**, 즉 '무엇을' 할 것인지를 정의합니다. 반면 `config/*.yaml`은 모델이 실행될 **인프라(Infrastructure)**, 즉 '어디서', '어떤 제약으로' 실행될지를 정의합니다.

-   **구현 상세**:
    -   이 철학은 `recipes/`와 `config/`라는 물리적인 디렉토리 구조로 명확히 구현됩니다.
    -   `src/settings/loaders.py`의 `load_settings_by_file` 함수는 양쪽의 YAML 파일을 함께 로드하여 하나의 `Settings` 객체로 병합합니다. 예를 들어, 모델의 종류(`class_path`)는 `recipe`에서, MLflow 서버 주소(`tracking_uri`)는 `config`에서 가져와 조립함으로써 철학을 코드로 실현합니다.

### 1.2. 원칙 2: 환경별 역할 분담 (Environment-Driven Philosophy)

각 실행 환경(`local`, `dev`, `prod`)은 뚜렷한 목적과 그에 맞는 기능 제약을 가집니다.

-   **철학**: `local`은 빠른 실험과 디버깅, `dev`는 완전한 기능의 통합 테스트, `prod`는 안정성과 확장성에 집중합니다. 동일한 코드가 환경에 따라 차등적으로 동작해야 합니다.

-   **구현 상세**:
    -   `config/base.yaml`을 `config/local.yaml`이 덮어쓰는 계층적 설정 구조를 통해 구현됩니다.
    -   예를 들어, `config/local.yaml`에서 `feature_store.provider: passthrough`로 설정하면 피처 스토어 기능을 비활성화하고, `hyperparameter_tuning.timeout`을 짧게 설정하여 `local` 환경의 "빠른 피드백"이라는 목적을 달성합니다. `APP_ENV` 환경 변수가 이 모든 것을 제어합니다.

### 1.3. 원칙 3: 선언적 파이프라인 (Declarative Pipeline)

사용자는 명령형 코드가 아닌 선언적 YAML 파일을 통해 파이프라인을 정의합니다.

-   **철학**: 개발자는 '어떻게'의 복잡성에서 벗어나 '무엇을'에만 집중할 수 있어야 합니다. 시스템은 이 선언을 해석하여 실행을 책임집니다.

-   **구현 상세**:
    -   사용자는 `recipes/*.yaml` 파일에 원하는 모델, 전처리기 타입, 파라미터 등을 선언하기만 하면 됩니다.
    -   그러면 `src/engine/Factory`가 이 명세를 읽어 필요한 컴포넌트들을 동적으로 조립하여 파이프라인을 구성합니다. 복잡한 파이프라인 구성 로직이 사용자로부터 완벽하게 추상화됩니다.

---

## 2. 아키텍처 및 실행 흐름 (Architecture & Execution Flow)

### 2.1. 3계층 아키텍처 (3-Layer Architecture)

MMP는 명확한 역할 분리를 위해 3계층 아키텍처를 채택했습니다.

```
/
├── config/         # [인프라] 환경별 설정
├── recipes/        # [모델 논리] 모델 실험의 청사진
└── src/            # [핵심 로직]
    ├── pipelines/  #  - Layer 3: Pipelines (전체 흐름을 조율하는 오케스트레이터)
    ├── engine/     #  - Layer 2: Engine (팩토리, 레지스트리 등 시스템의 뼈대)
    └── components/ #  - Layer 1: Components (Trainer, Preprocessor 등 실제 ML 작업자)
```

-   **Layer 3 (Pipelines)**: `run_training` 등 엔드투엔드 흐름을 관장합니다.
-   **Layer 2 (Engine)**: `Factory` 등 시스템의 핵심 동작 원리를 제공합니다.
-   **Layer 1 (Components)**: `Trainer` 등 실제 ML 연산을 수행하는 구체적인 도구들입니다.

### 2.2. 학습 실행 흐름 (`train` 기준)

1.  **명령 실행**: 사용자가 `python main.py train --recipe-file <...>`을 실행합니다.
2.  **설정 로딩**: `src/settings/loaders.py`가 `config`와 `recipe`를 병합하여 유효성이 검증된 `Settings` 객체를 생성합니다.
3.  **파이프라인 시작**: `src/pipelines/train_pipeline.py`의 `run_training` 함수가 `Settings` 객체를 전달받아 MLflow 추적을 시작합니다.
4.  **동적 조립**: `src/engine/Factory`가 `Settings` 명세에 따라 `DataAdapter`, `Augmenter`, `Preprocessor`, `Model` 등 모든 컴포넌트를 동적으로 생성합니다.
5.  **학습 오케스트레이션**: `src/components/Trainer`가 모든 컴포넌트와 데이터를 받아 **[증강 → 전처리 → (튜닝) → 학습 → 평가]** 과정을 책임지고 수행합니다.
6.  **아티팩트 패키징**: 학습된 모델과 모든 변환 로직이 `PyfuncWrapper`라는 단일 객체로 패키징됩니다.
7.  **최종 저장**: 이 `PyfuncWrapper`는 모든 메타데이터와 함께 MLflow 서버에 저장되어 완전한 재현성을 보장합니다.

---

## 3. 핵심 컴포넌트와 디자인 패턴 (Core Components & Design Patterns)

MMP의 설계 철학은 다음 네 가지 핵심 컴포넌트에 의해 구현됩니다.

### 3.1. Factory & Registry: 동적 조립의 엔진

-   **역할**: `Settings` 객체에 정의된 `class_path`나 `type` 같은 명세를 보고, 필요한 컴포넌트의 구체적인 구현체를 동적으로 생성하고 주입(DI)하는 역할을 합니다.
-   **설계 의도**: "명시성에 기반한 지능형 팩토리" 원칙의 구현체입니다. `Factory` 덕분에 파이프라인 코드는 특정 컴포넌트에 종속되지 않으며, 새로운 컴포넌트가 추가되어도 기존 코드 수정 없이 레시피 수정만으로 확장이 가능합니다. (`src/engine/factory.py`)

### 3.2. Settings: 검증된 실행 계획

-   **역할**: 단순한 설정값이 아닌, Pydantic으로 모든 필드의 타입과 값이 검증된 '실행 계획 객체'입니다.
-   **설계 의도**: 파이프라인의 모든 단계에 일관된 설정을 전파하는 단일 진실 공급원(Single Source of Truth)입니다. 이를 통해 파이프라인 실행 도중 발생할 수 있는 설정 오류를 원천 차단합니다. (`src/settings/schema.py`)

### 3.3. Trainer: 학습 로직의 캡슐화

-   **역할**: 데이터 분할, 하이퍼파라미터 튜닝(Optuna), 모델 피팅, 평가 지표 계산 등 복잡한 학습 과정을 책임지는 '전문 작업자'입니다.
-   **설계 의도**: 핵심 학습 로직을 한 곳에 집중시켜 파이프라인의 다른 부분과 분리합니다. 이를 통해 학습 로직의 일관성을 유지하고 테스트 용이성을 확보합니다. (`src/components/_trainer/_trainer.py`)

### 3.4. PyfuncWrapper: 완전한 재현성의 보증

-   **역할**: 학습의 최종 결과물입니다. 학습된 모델뿐만 아니라, 예측에 필요한 모든 전처리/증강 로직을 포함하는 자기 완결적(self-contained) 아티팩트입니다.
-   **설계 의도**: "실행 시점에 조립되는 순수 로직 아티팩트" 원칙의 구현체입니다. `PyfuncWrapper`는 DB 접속 정보와 같은 인프라 설정을 포함하지 않는 '순수 로직의 캡슐'이므로, 어떤 환경에서든 100% 동일한 예측 결과를 보장하여 완전한 재현성을 실현합니다. (`src/engine/_artifact.py`)