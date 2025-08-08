# Modern ML Pipeline Blueprint

이 문서는 Modern ML Pipeline(MMP)의 핵심 설계 철학, 아키텍처, 개발 철학 및 실행 흐름을 기술하는 **최종 청사진**입니다. 이 청사진은 MMP를 현대적인 ML/DL 파이프라인 라이브러리로 완성하기 위한 방향성을 제시하며, 앞으로의 개발 과정에서 일관된 원칙으로 작용할 것입니다. 시스템이 '왜' 이렇게 설계되었고, 그 철학이 '어떻게' 코드로 구현되어야 하는지를 명확히 연결하여 설명하며, 개발 시 준수해야 할 가이드라인을 함께 담고 있습니다.

---

## 1. 핵심 설계 철학 (Core Design Philosophy)

MMP는 **네 가지** 핵심 원칙 위에 세워져 있으며, 이 원칙들은 시스템 아키텍처와 개발 전반에 일관되게 적용됩니다. 모든 신규 기능과 코드는 이 철학을 준수하도록 설계됩니다.

### 1.1. 원칙 1: 설정과 논리의 분리 (Recipe is Logic, Config is Infra)

모델의 본질적인 **논리**와 그것이 실행될 **환경 설정**을 엄격하게 분리합니다.

* **철학**: `recipes/*.yaml` 파일들은 파이프라인의 **논리** – 즉 '무엇을 할 것인가'를 선언합니다. 반면 `config/*.yaml` 파일들은 파이프라인의 실행 **인프라** – 즉 '어떤 환경에서, 어떤 제약으로 실행할 것인가'를 정의합니다. 한마디로 *Recipe는 ML 실험의 논리를, Config는 그 논리를 담는 기반 환경을 나타냅니다.*

* **구현 상세**:

  * 이 철학은 디렉토리 구조로 명확히 드러납니다 (`recipes/` vs `config/`). 예를 들어, 사용자 레시피 YAML에서는 사용할 모델 종류나 전처리 방법 등의 **클래스 경로**(`class_path`)와 하이퍼파라미터 등 **논리적 설정**을 기술합니다. 한편 `config/base.yaml` 및 환경별 `config/<env>.yaml`에서는 **실행 인프라 설정**(예: MLflow 서버 주소 `tracking_uri`, 데이터 경로, 자원 한계 등)을 정의합니다.
  * `src/settings/loaders.py`의 `load_settings_by_file` 함수는 지정된 recipe YAML과 config YAML을 함께 로드하여 하나의 통합된 `Settings` 객체로 병합합니다. 이 때 Pydantic 기반의 스키마 검증을 통해 논리와 인프라 설정이 올바르게 조합됩니다. 예를 들어 **레시피**에서 모델 클래스 경로(`model.class_path`)를, **환경 설정**에서 MLflow 추적 주소(`tracking_uri`)를 가져와 **동일한 Settings 객체**로 합치는 식입니다. 이렇게 함으로써 논리와 환경이 분리되어 있어도 최종 실행 단계에서는 필요한 정보가 완전하게 결합됩니다.

### 1.2. 원칙 2: 환경별 역할 분담 (Environment-Driven Philosophy)

각 실행 환경(`local`, `local-dev`, `prod`)은 뚜렷한 목적과 그에 맞는 동작상의 차별점을 가집니다. **동일한 코드베이스**가 환경에 따라 다르게 동작하되, 궁극적으로 일관된 결과를 내도록 합니다.

* **철학**: 개발 초기에는 빠른 실험과 디버깅을, 검증 단계(`local-dev`)에서는 프로덕션에 근접한 통합 테스트를, 운영 단계(`prod`)에서는 **안정성과 확장성**을 최우선으로 합니다. 이를 위해 환경별로 일부 기능을 켜거나 끄고, 파이프라인의 동작을 조정함으로써 **환경 특화 동작**을 구현합니다. 코드에는 환경 분기에 대한 논리가 최소화되고, 설정을 통해 환경 차이를 유발하도록 합니다.

* **구현 상세**:

  * 계층적 설정 파일 구조를 사용합니다. `config/base.yaml`에 기본값을 정의하고, 예를 들어 `config/local.yaml` 또는 `config/local-dev.yaml`이 이를 **오버라이드**하도록 합니다. 실행 시 설정 로더는 `APP_ENV` 환경 변수에 따라 해당 환경의 설정을 불러와 base 설정에 덮어씁니다.
  * **Local**: 빠른 피드백 중심. `feature_store.provider: none`, `serving.enabled: false`. 필요 시 전체 피처가 포함된 샘플 테이블을 로드하고 `PassThroughAugmenter`로 학습합니다.
  * **Local-Dev**: 프로덕션 유사 통합 검증. mmp-local-dev로 띄운 로컬/개발용 Feature Store에 연결(`feature_store.provider: feast`). Augmenter는 Feature Store 기반으로 동작하며, 서빙도 활성화하여 온라인 조회 경로를 검증합니다.
  * **Prod**: 실 Feature Store/대용량 데이터/보안 최적화. 전체 데이터셋 대상으로 파이프라인을 실행하고, 타임아웃/리소스 한계를 운영 목적에 맞게 설정합니다. 모든 환경 차이는 설정으로 제어되며, 코드 변경 없이 환경 교체가 가능합니다.

### 1.3. 원칙 3: 선언적 파이프라인 (Declarative Pipeline)

사용자는 **명령형 코드**를 작성하는 대신, **선언적 YAML** 파일 편집만으로 파이프라인을 구성합니다. 파이프라인의 상세 구현보다는 원하는 구성요소와 목표만을 기술하면, 시스템이 해당 선언을 해석하여 동작합니다.

* **철학**: 개발자는 '어떻게' 수행할지에 대한 복잡한 코드 작성에서 벗어나, '무엇을 할 것인지'에만 집중해야 합니다. 모델 구조나 전처리 방법 등의 선택지를 YAML로 선언하면, 구체적인 실행 절차는 프레임워크가 알아서 수행합니다. 이를 통해 반복적인 파이프라인 코딩 작업을 줄이고 **생산성을 향상**시킵니다.

* **구현 상세**:

  * 사용자는 `recipes/*.yaml` 파일에 **모델 종류**, **전처리기/증강기 종류**, **평가 지표**, **튜닝 파라미터** 등 실험에 필요한 요소들을 선언적으로 나열합니다. 예를 들어 아래와 같은 YAML 스니펫이 있다고 가정합니다.

    ```yaml
    # recipes/example.yaml
    model:
      class_path: myproject.models.SklearnRandomForest
      params:
        n_estimators: 100
        max_depth: 5
    preprocessor:
      class_path: myproject.preprocessing.StandardScaler
    augmentor:
      class_path: myproject.augmentation.NoOpAugmentor
    metrics: [accuracy, f1]
    ```

    이렇게 YAML에 **하고자 하는 바**를 모두 명시하면,
  * `src/engine/factory.py`의 `Factory` 클래스가 이 명세(`Settings` 객체로 파싱된)를 읽어들여 필요한 컴포넌트들을 동적으로 **조립**합니다. 개발자는 `RandomForestClassifier`를 어떻게 학습하고 전처리와 연결하는지 직접 코드로 짤 필요 없이, MMP가 해당 선언을 바탕으로 적절한 `DataAdapter`, `Preprocessor`, `Augmentor`, `Model` 객체 등을 생성하여 파이프라인을 구성해 줍니다. 이처럼 **파이프라인 구성 로직을 완전히 추상화**함으로써, 사용자는 ML 실험의 목표 설정에만 집중할 수 있습니다.

### 1.4. 원칙 4: 모듈화와 확장성 (Modularity & Extensibility)

시스템의 구성 요소들은 높은 **모듈화**를 갖추고, 새로운 요구사항에 대해 **유연한 확장**이 가능해야 합니다.

* **철학**: MMP의 모든 컴포넌트(데이터 어댑터, 전처리기, 모델 등)는 **명확한 인터페이스**와 역할을 가지며 상호 의존성을 최소화합니다. 이를 통해 특정 컴포넌트를 교체하거나 새로운 기능을 추가하더라도 기존 코드에 영향을 주지 않고 쉽게 확장할 수 있습니다. 프레임워크 자체가 **Open-Closed Principle**(확장에는 열려 있고 변경에는 닫혀 있음)을 지향하여, 개발자는 시스템을 포크하거나 크게 수정하지 않고도 설정만으로 새로운 알고리즘이나 리소스를 활용할 수 있습니다.

* **구현 상세**:

  * 이러한 모듈화는 **Factory & Registry 패턴**을 통해 구현됩니다. `Factory`는 설정에 명시된 `class_path`를 활용해 해당 클래스를 동적으로 불러오므로, 새로운 클래스(예: 새로운 모델 아키텍처나 커스텀 전처리기)를 추가하려면 해당 클래스를 코드로 작성하고 레시피에 경로만 적어주면 됩니다. 기존 파이프라인 코드(오케스트레이션 로직)는 전혀 수정할 필요가 없습니다.
  * 컴포넌트 간에는 **계약된 인터페이스**를 유지합니다. 예를 들어 모든 모델 클래스는 `fit()`과 `predict()` (또는 `transform()`) 메서드를 가져야 한다든지, 전처리기는 `fit_transform()`/`transform()` 메서드를 가진다는 식의 암묵적 규약이 있습니다. Trainer 등 파이프라인 코드에서는 이 공통 인터페이스만을 상대하므로, 새로운 모델/전처리기가 추가되어도 호환만 된다면 정상 동작합니다.
  * 또한 MMP는 구성 요소 추가 시 **Registry**에 해당 클래스를 등록하거나, 경로만 알면 Factory가 찾아가도록 설계되어 있습니다. 이를 통해 *플러그인 형태로 새로운 기능을 주입*할 수 있으며, MMP를 장기적으로 발전시켜 나가도 일관된 구조를 유지할 수 있습니다.

---

## 2. 아키텍처 및 실행 흐름 (Architecture & Execution Flow)

MMP의 전체 구조는 **레이어드 아키텍처**와 **파이프라인 오케스트레이션** 개념으로 설명할 수 있습니다. 핵심 디렉토리 구조와 학습/추론 파이프라인의 실행 흐름은 다음과 같습니다.

### 2.1. 3계층 아키텍처 (3-Layer Architecture)

MMP는 명확한 역할 분리를 위해 **3계층 아키텍처**를 채택하였습니다. 각 계층은 서로 다른 책임을 가지며, 아래와 같은 디렉토리 구조로 구현됩니다:

```
/ (프로젝트 루트)
├── config/         # [인프라] 환경별 설정파일 모음 (base.yaml, local.yaml, dev.yaml 등)
├── recipes/        # [실험 논리] 파이프라인/모델 설정 모음 (*.yaml 레시피 파일들)
└── src/            # [핵심 로직 구현]
    ├── pipelines/  #  - **Layer 3:** 파이프라인 오케스트레이션 (엔드투엔드 흐름 제어)
    ├── engine/     #  - **Layer 2:** 엔진/프레임워크 (팩토리, 레지스트리, 아티팩트 관리 등)
    └── components/ #  - **Layer 1:** 개별 컴포넌트 (Trainer, Preprocessor 등 실제 ML 작업 단위들)
```

* **Layer 1 – Components**: 개별 머신러닝 작업 수행자들의 모음입니다. 데이터 입수, 변환, 모델 훈련 등 각 작업을 담당하는 모듈들이 여기에 속합니다. 예를 들어 `DataAdapter` (데이터 로딩), `Augmenter` (데이터 증강), `Preprocessor` (데이터 전처리), `Trainer` (학습 제어), 그리고 다양한 **Model** 구현체들이 모두 `src/components/` 아래에 위치합니다. 각 컴포넌트는 독립적인 기능 단위를 가지며, 상위 레이어의 지시에 따라 자신의 작업을 수행합니다.
* **Layer 2 – Engine**: 파이프라인의 **엔진**에 해당하는 계층으로, 컴포넌트들을 적절히 생성하고 연결하는 역할을 합니다. `Factory`, `Registry`와 같은 핵심 인프라 로직, 설정(`Settings`) 관리, 최종 학습 아티팩트(`PyfuncWrapper`) 구성 등이 이 레이어에서 이루어집니다. 한마디로 컴포넌트들이 유기적으로 동작하도록 하는 **중추 신경**입니다.
* **Layer 3 – Pipelines**: 최상위 계층으로, **엔드투엔드 실행 흐름**을 관리합니다. 예를 들어 `train_pipeline.py`에서는 데이터 불러오기부터 모델 훈련, 평가, 아티팩트 저장까지 일련의 과정을 orchestration(오케스트레이션)합니다. 각 파이프라인은 Engine 레이어를 활용하여 필요한 컴포넌트를 생성하고, 정해진 순서에 따라 호출합니다. Pipeline 계층의 코드는 "비즈니스 로직(학습/추론 시나리오)"에 집중하며, 실제 동작 구현은 하위 레이어에 위임합니다.

이러한 3계층 구조 덕분에, MMP는 **관심사의 분리**를 달성합니다. 새 컴포넌트를 추가하거나 기존 로직을 개선할 때, 해당 계층의 코드만 수정하면 되고 다른 계층은 영향을 적게 받습니다.

### 2.2. 학습 파이프라인 실행 흐름 (Training Pipeline Flow)

MMP의 학습 파이프라인은 `train` 명령을 통해 실행되며, **데이터 준비 → 모델 학습 → 평가 및 저장**의 단계를 거칩니다. 주요 흐름은 아래와 같습니다 (MLflow Tracking 및 Optuna HPO 통합 기준):

1. **명령 시작**: 사용자가 명령줄에서 `python main.py train --recipe-file <recipe.yaml>` 을 실행합니다. (`main.py`는 커맨드 엔트리 포인트로, subcommand로 `train` 등을 제공한다고 가정합니다.)
2. **설정 로딩**: `src/settings/loaders.py`에서 `load_settings_by_file`가 호출되어, 지정된 레시피 파일과 현재 `APP_ENV`에 대응하는 config 파일을 모두 읽어들입니다. 이때 Pydantic `Settings` 스키마에 따라 필드들이 검증되고, 최종적으로 통합된 **`Settings` 객체**가 생성됩니다. 이 객체에는 모델/데이터 관련 논리 설정부터 리소스/인프라 설정까지 실행에 필요한 모든 정보가 담겨 있습니다.
3. **파이프라인 초기화**: `src/pipelines/train_pipeline.py`의 `run_training(settings: Settings)` 함수가 호출됩니다. 이 함수는 우선 MLflow로 실험 **추적을 시작**하여 (예: `mlflow.start_run()`), 해당 실행에 대한 로그(메타데이터, 매트릭 등)를 남길 준비를 합니다. 이어서 설정 객체를 인자로 받아 본격적인 파이프라인 절차를 밟습니다.
4. **동적 컴포넌트 조립**: `Engine` 레이어의 `Factory`가 `Settings`에 명시된 내용대로 필요한 **컴포넌트 인스턴스들을 생성**합니다. 예를 들어 Settings에 데이터 소스가 CSV로 지정되어 있으면 `DataAdapter`는 CSV 로더로, 전처리기로 StandardScaler를 쓴다고 명시되면 해당 클래스 인스턴스로, 모델이 XGBoost라면 그 모델 클래스 인스턴스로 만들어줍니다. 이때 `Factory`는 내부적으로 `Registry`나 파이썬의 dynamic import를 이용하여 `class_path`에 해당하는 클래스를 찾아 생성합니다. 또한 모든 컴포넌트가 `Settings`나 상호 필요한 의존 객체들을 주입받도록 설계되어 있습니다 (예: `Trainer` 생성 시 앞서 만든 모델, 전처리기 인스턴스를 전달). **이 단계까지** 파이프라인에 필요한 준비물(데이터 처리기, 모델, 트레이너 등)이 모두 준비됩니다.
5. **학습 오케스트레이션**: `src/components/Trainer` 객체가 중심이 되어 학습 과정을 진행합니다. Trainer는 이전 단계에서 생성된 `DataAdapter`, `Augmenter`, `Preprocessor`, `Model` 등을 받아 일련의 **학습 파이프라인 단계**를 실행합니다:

   * **데이터 적재**: DataAdapter로 엔터티 식별자와 타임스탬프(학습 시 레이블 포함)를 최소 컬럼으로 적재합니다.
   * **피처 조회(Augmenter)**: Augmenter가 Feature Store 기반 Point-in-Time 조인으로 피처를 조회·결합합니다. (학습/배치: 오프라인 조회, 서빙: 온라인 조회)
   * **데이터 전처리**: Preprocessor를 이용해 결측치 처리, 스케일링, 인코딩 등의 변환을 적용합니다. 전처리기는 훈련 데이터에서 학습(`fit_transform`)되고, 동일한 변환이 검증/테스트 데이터에도 적용됩니다.
   * **하이퍼파라미터 튜닝 (선택)**: Settings에 튜닝 설정이 있는 경우 Optuna 등을 활용하여 모델의 최적 하이퍼파라미터를 탐색합니다. (예: 지정된 평가 지표를 최대화하는 파라미터 조합 탐색) 로컬 환경에서는 이 과정이 제한되거나 매우 짧게 수행되고, dev/prod에서는 지정한 시간(`hyperparameter_tuning.timeout`)이나 trial 횟수만큼 수행합니다.
   * **모델 학습**: 최종 설정된 파라미터로 `Model`의 `fit()`을 호출하여 훈련 데이터를 학습시킵니다. Trainer는 `Settings`에 정의된 에폭 수, 배치 크기 등의 파라미터를 모델에 전달하거나 내부에서 반복문을 돌며 학습을 관리할 수도 있습니다 (모델 종류에 따라 다름).
   * **모델 평가**: 학습된 모델에 대해 검증 데이터셋으로 예측을 수행하고, `Settings`에 지정된 평가 지표(예: accuracy, F1, AUC 등)를 계산합니다. 이러한 결과는 MLflow 등으로 **로그**되어 추적 가능합니다.
6. **아티팩트 패키징**: 학습이 완료되면, `src/engine/_artifact.py`의 `PyfuncWrapper`를 생성하여 **훈련된 모델과 모든 전처리/증강 로직을 하나로 캡슐화**합니다. `PyfuncWrapper`는 MLflow의 PyFunc 모델 형식을 활용하며, 내부에 모델 가중치와 함께 **데이터 전처리 파이프라인 객체들을 포함**합니다. 이를 통해 나중에 예측시 동일한 전처리-예측 흐름을 재현할 수 있습니다. Trainer는 이 wrapper 객체를 생성한 후, 필요하면 추가 메타데이터(예: 모델 이름, 버전, 평가 결과)를 설정합니다.
7. **모델 등록 및 저장**: 완성된 `PyfuncWrapper` 객체를 MLflow에 \*\*모델 아티팩트로 저장(registry)\*\*합니다. `mlflow.log_model()`을 통해 원본 데이터에 대한 변환 로직, 모델 가중치, 모델 클래스 등 **재현에 필요한 모든 정보**가 기록됩니다. 이때 Settings 자체도 yaml이나 dict 형태로 함께 저장하여, 나중에 어떤 설정으로 학습되었는지 완전히 추적 가능합니다. 최종적으로 MLflow 상에 등록된 모델에는 유니크한 버전이 부여되고, 필요시 배포 단계에서 이 모델을 불러와 활용할 수 있습니다.

위의 절차를 통해 **한 번의 `train` 실행으로 완벽히 재현가능한 결과물**이 만들어집니다. 모든 구성 요소와 설정이 명시적으로 관리되고 MLflow에 기록되므로, 추후 같은 레시피와 config로 `train`을 실행하면 동일한 결과를 얻을 수 있습니다.

### 2.3. 추론 파이프라인 실행 흐름 (Inference Pipeline Flow)

훈련된 모델을 활용하여 새로운 데이터에 예측을 수행하는 과정도 MMP 철학에 따라 **일관된 절차**로 관리됩니다. 추론 파이프라인의 예상 흐름은 다음과 같습니다:

1. **모델 로딩**: `run_id`로 MLflow에서 `PyfuncWrapper`를 로드합니다(`runs:/{run_id}/model`).
2. **입력 스키마**: API/배치 입력은 `entity_schema` 기반으로 정의되며, 엔터티 식별자와 타임스탬프만 포함합니다(레이블/피처 제외).
3. **데이터 적재(배치)**: 배치 추론의 경우, Loader SQL(스냅샷 또는 템플릿+컨텍스트)로 엔터티+타임스탬프만 적재합니다.
4. **예측 호출**: `model.predict(df, params={run_mode: "serving"|"batch"})`를 호출하면 내부에서 `augment(피처 조회) → preprocess → predict` 순으로 실행됩니다.
   - serving: 온라인 Feature Store 조회로 PIT-join 수행(폴백/패스스루 금지)
   - batch: 오프라인 Feature Store 조회가 기본, 레시피에 **명시된 경우에 한해** SQL 폴백 허용
5. **결과 처리**: 예측 결과를 응답/저장하고, 사용된 run_id, 실행 run_id, 타임스탬프 등의 메타데이터를 함께 기록합니다.

훈련-서빙 간 일관성은 `PyfuncWrapper`에 캡슐화된 Augmenter/Preprocessor/Model 체인을 통해 보장됩니다. 어떤 환경에서도 동일한 입력 계약(엔터티+타임스탬프)과 동일한 피처 조회·변환·예측 경로가 재현됩니다.
서버 시작 시에는 Augmenter 타입을 검증하며, `pass_through`/`sql_fallback`인 경우 서빙 진입을 차단합니다.

---

## 3. 핵심 컴포넌트와 디자인 패턴 (Core Components & Design Patterns)

MMP의 설계 철학은 다음과 같은 핵심 컴포넌트들과 디자인 패턴들을 통해具現化(구현)됩니다. 각 컴포넌트는 앞서 소개한 레이어 구조의 **Layer 1**에 해당하며, Engine 및 Pipeline 레이어와 상호작용하면서 동작합니다. 또한 이들 컴포넌트의 구현에는 디자인 패턴상의 의도가 반영되어 있어, 철학과 코드 구현을 이어주는 역할을 합니다.

### 3.1. Factory & Registry: 동적 조립의 엔진

* **역할**: Factory는 `Settings` 객체에 정의된 명세를 참고하여 필요한 컴포넌트 인스턴스를 **동적으로 생성**하고 의존성을 주입하는 역할을 맡습니다. Registry는 등록된 클래스나 함수 레퍼런스를 관리하여 Factory가 문자열로 주어진 `class_path`를 실제 객체로 변환할 수 있도록 도와줍니다. 쉽게 말해, **사용자가 선언한 구성 요소들을 실제 실행 객체로 바꾸는 엔진**입니다.

* **설계 의도**: "명시성에 기반한 지능형 팩토리" 원칙의 구현체입니다. Factory 덕분에 파이프라인 오케스트레이션 코드 자체는 구체적인 클래스에 의존하지 않고, Settings의 명세만 따르면 됩니다. 새로운 컴포넌트(예: 새로운 모델 클래스나 데이터 로더)를 추가하더라도 **기존 코드를 수정할 필요 없이** 레지스트리에 등록하고 레시피에 경로만 지정하면 통합될 수 있습니다. 이는 시스템이 **확장에는 열려 있고 변경에는 닫혀 있게** 해주며, 결과적으로 개발 생산성과 유지보수성을 높여줍니다. (`src/engine/factory.py`)
  
* **Augmenter 선택 정책(요약)**
  - 입력 신호: `environment.app_env`, `feature_store.provider`, `serving.enabled`, `run_mode(train|batch|serving)`, `recipe.model.augmenter(및 fallback)`
  - 규칙:
    1) `local` 또는 `provider=none` 또는 `augmenter.type=pass_through` → `PassThroughAugmenter`
    2) `augmenter.type=feature_store` ∧ `provider in {feast, mock}` ∧ 헬스체크 성공 → `FeatureStoreAugmenter`
    3) (train/batch 전용) 2 실패 ∧ 레시피에 `fallback.sql` 명시 → `SqlJoinAugmenter`
    4) serving에서는 1·3 금지(진입 차단). 실패 시 명확한 에러 반환

### 3.2. Settings: 검증된 실행 계획

* **역할**: Settings는 단순한 설정값 모음이 아니라, Pydantic 모델을 통해 **모든 필드의 타입과 값이 검증된 실행 계획 객체**입니다. Recipes와 Config에서 불러온 원시 YAML 데이터는 Settings 객체로 파싱됨으로써, 타입 불일치나 필수 필드 누락 등의 문제가 사전에 차단됩니다.

* **설계 의도**: Settings는 파이프라인 전 단계에 걸쳐 **단일 진실 공급원(Single Source of Truth)** 역할을 합니다. 한 번 생성된 Settings 객체는 파이프라인의 모든 컴포넌트에 전달되어 동일한 설정값에 기반해 움직이게 합니다. 이를 통해 사람이나 코드의 실수로 인한 설정 불일치를 방지하고, 중간 단계에서 설정을 별도로 관리해야 하는 복잡성을 제거합니다. 결과적으로 파이프라인 실행 중 발생할 수 있는 환경/설정 오류를 원천 봉쇄하고 **신뢰성 있는 실행**을 보장합니다. (`src/settings/schema.py`, `src/settings/loaders.py`)

### 3.3. Trainer: 학습 로직의 캡슐화

* **역할**: Trainer는 모델 학습 과정을 고수준으로 캡슐화한 **전문 작업자**입니다. 데이터 분할, 모델 훈련, 검증, 하이퍼파라미터 튜닝, 평가 지표 계산 등 학습과 관련된 모든 세부 로직을 내부에서 처리합니다. Pipeline 계층에서는 Trainer에게 필요한 자원(DataAdapter가 불러온 데이터, 구성된 Model 등)을 건네주고 `train()` 또는 `run()`을 호출하기만 하면, Trainer가 알아서 일련의 과정을 **오케스트레이션**합니다.

* **설계 의도**: 핵심 학습 로직을 한 곳에 모아두어 **응집도**를 높이고, 다른 부분과의 **결합도**를 낮추기 위함입니다. Trainer는 다른 컴포넌트 (전처리기, 모델 등)와 상호작용하지만, 그 구체적인 방식은 Trainer 내부에 숨겨져 있습니다. 이는 유지보수와 테스트에 유리한 구조로, 예를 들어 학습 과정에 버그가 있더라도 Trainer 모듈만 집중해서 조사하면 됩니다. 또한 Trainer는 일정한 인터페이스를 제공하므로(예: `Trainer.train()`), Pipeline 코드가 단순해지고 읽기 쉬워집니다. (`src/components/_trainer/_trainer.py`)

### 3.4. PyfuncWrapper: 완전한 재현성의 보증

* **역할**: PyfuncWrapper는 학습의 최종 결과물을 담은 **자기 완결적(self-contained) 모델 아티팩트**입니다. 단순히 모델 가중치만 담는 것이 아니라, 예측에 필요한 모든 전처리/증강 로직과 메타정보를 함께 포함합니다. 이 객체 하나만 있으면 별도의 환경 설정이나 추가 코드 없이 **동일한 예측**을 재현할 수 있습니다. MLflow의 PyFunc 모델 규격을 따르며, 주로 모델 서빙 및 배포 단계에서 활용됩니다.

* **설계 의도**: "실행 시점에 조립되는 순수 로직 아티팩트" 원칙의 구현체입니다. PyfuncWrapper에는 데이터베이스 접속 정보나 파일 경로 같은 **환경 의존적 설정은 배제하고**, 순수한 **모델+로직**만을 담았습니다. 이렇게 함으로써 어떤 시스템에서 이 아티팩트를 불러오더라도 **100% 동일한 예측 결과**를 보장합니다. 예를 들어 로컬에서 학습한 모델을 PyfuncWrapper로 패키징해 놓으면, 이를 가져다가 AWS 서버나 기타 환경에서 로드하여 예측해도 로컬과 완전히 동일한 동작을 합니다. 이는 곧 **완전한 재현성**과 **환경 간 이식성**을 의미하며, MMP의 궁극적인 목적 중 하나인 "한 번 학습하고 어디서든 활용 가능"을 실현해줍니다. (`src/engine/_artifact.py`)

* **스냅샷/메타데이터**: 재현성과 추적성을 위해 다음 정보를 함께 보관합니다.
  - `loader_sql_snapshot`: 학습 시 사용한 로더 SQL/URI 스냅샷
  - `augmenter_config_snapshot`: Augmenter 설정 스냅샷(타입/피처 참조 등)
  - `entity_schema_snapshot`: 엔터티/타임스탬프 스키마 스냅샷
  - `feature_refs_snapshot`: 조회된 피처 네임스페이스/피처 목록(버전/태그 포함 가능)
  - `signature`/`data_schema`: 입력/출력 스키마 및 시그니처

### 3.5. DataAdapter: 데이터 입수의 추상화

* **역할**: DataAdapter는 파이프라인의 **데이터 입구**입니다. 원본 데이터 소스에서 훈련에 필요한 데이터를 읽어오는 작업을 추상화하며, 소스가 무엇이든 간에 동일한 인터페이스로 데이터를 제공합니다. 예를 들어 CSV 파일, 데이터베이스 쿼리, 데이터 레이크, 또는 Feature Store 등 **다양한 데이터 소스에 대한 통합된 로딩 인터페이스**를 제공합니다.

 * **설계 의도**: 데이터 입수 과정을 추상화하여 논리와 인프라를 분리하려는 설계 철학을 구현한 컴포넌트입니다. DataAdapter를 통해, 파이프라인 코드는 데이터가 어디에서 오는지 몰라도 상관없도록 만들고자 했습니다. 환경별 설정에 따라 DataAdapter는 다른 방식으로 동작할 수 있습니다 (`Settings`에 지정된 `data_source` 종류에 따라 다른 DataAdapter 클래스를 Factory가 생성). 예를 들어 local 환경에서는 로컬 파일에서 데이터를 읽고, prod 환경에서는 데이터 웨어하우스/데이터레이크/스토리지에서 읽습니다. Feature Store 연동은 Augmenter의 책임입니다. 이렇게 함으로써 **데이터 소스 변경에 대한 유연성**을 확보하고, 코드의 다른 부분에 영향을 주지 않으면서 데이터 인프라를 교체하거나 확장할 수 있습니다. 또 한 가지 의도는, DataAdapter 단계에서 가져온 원본 데이터에 대해 일관된 포맷(예: Pandas DataFrame 등)을 반환함으로써 이후 단계(전처리, 학습)가 소스에 무관하게 동일하게 처리되도록 하는 것입니다.

### 3.6. Preprocessor & Augmenter: 데이터 변환과 피처 조회 파이프라인

* **역할 분리**
  - Preprocessor: 결측/스케일링/인코딩 등 피처 변환 수행(fit/transform)
  - Augmenter: Feature Store 기반 **Point-in-Time(PIT) 조인**으로 피처를 조회·결합(데이터 증강(SMOTE)과는 다른 개념)

* **Augmenter 데이터 계약**
  - 입력: `entity_schema.entity_columns + [timestamp_column]`(학습 시 `target_column`, causal은 `treatment_column` 포함 가능)
  - 처리: run_mode에 따라 offline(학습/배치) PIT 조회 또는 online(서빙) 조회 수행
  - 출력: 입력과 동일 키(엔터티+타임스탬프) 기준 Left Join으로 피처 추가

* **구현 전략**
  - FeatureStoreAugmenter: offline=`get_historical_features_with_validation`, serving=`get_online_features`
  - SqlJoinAugmenter(학습/배치 전용): 안전 템플릿 SQL 렌더링→조회→Left Join
  - PassThroughAugmenter: 입력 그대로 반환(로컬/외부에서 이미 조인된 테이블 사용 시)

* **정합성/정책**
  - 서빙은 Feature Store 기반 Augmenter만 허용(폴백/패스스루 금지)
  - 학습/배치는 레시피에 **명시된 경우에 한해** SQL 폴백 허용(자동 폴백 없음)
  - Preprocessor는 Augmenter 이후에 적용되어 훈련-서빙 일관성 유지

### 3.7. Model 컴포넌트: 플러그형 모델 구현

* **역할**: Model 컴포넌트는 실제 ML/DL **알고리즘**을 구현하는 부분입니다. 사용자 요구에 따라 선형 회귀부터 복잡한 신경망까지 다양할 수 있으며, MMP에서는 이를 추상화하여 **플러그형 모듈**로 다룹니다. 즉, 모델 훈련과 예측에 필요한 인터페이스(`fit()`, `predict()` 등)를 갖춘 어떠한 클래스도 Model 컴포넌트로 사용할 수 있습니다. MMP 자체에서 sklearn, XGBoost, PyTorch 등 여러 프레임워크의 모델을 포용할 수 있도록 설계되었습니다.

* **설계 의도**: 다양한 모델을 수용하기 위해 **유연한 추상화**를 의도했습니다. Factory는 `Settings`에 명시된 `model.class_path`를 통해 해당 클래스를 불러오므로, 새로운 알고리즘 구현체를 추가하려면 적절한 패키지 경로로 클래스를 정의하기만 하면 됩니다. 모델 컴포넌트는 Trainer와 상호작용하면서 학습 및 예측을 수행하는데, Trainer는 모델이 해당 인터페이스를 제대로 구현했다고 가정하고 호출합니다. 이를 활용하면, 예를 들어 **딥러닝 모델**의 경우도 미리 규약된 `Trainer <-> Model` 인터페이스만 지켜지면 MMP 파이프라인에 쉽게 통합될 수 있습니다 (필요하다면 전용 Trainer나 Adapter를 작성하여 특수 처리도 가능하게 여유를 두었음). 모델 컴포넌트를 이처럼 플러그인화함으로써, MMP 사용자는 알고리즘 선택의 폭이 넓어지고 새 모델 등장 시 빠르게 실험해볼 수 있습니다. 또한 Model 컴포넌트는 내부적으로 MLflow 등과 연계되어 모델의 상태를 추적하거나 저장할 수 있으므로, **실험 추적**과 **모델 버저닝**에도 자연스럽게 녹아듭니다.

---

## 4. 개발 철학과 향후 방향 (Development Philosophy & Future Direction)

Modern ML Pipeline 프로젝트의 개발은 위에 제시된 청사진을 **단일 진실 소스**로 삼아 이루어집니다. 다음은 본 청사진이 개발 과정에서 가지는 의미와, 미래에 계획된 확장 방향에 대한 설명입니다.

* **청사진 준수**: 이 문서는 MMP의 최종적인 지향점을 나타내며, 모든 새로운 코드 작성과 기능 추가는 본 청사진의 원칙과 아키텍처를 엄격히 준수해야 합니다. 개발자는 기능을 구현할 때 "설정-논리 분리", "환경 특화 설정", "선언적 구성", "모듈화/확장성" 네 가지 원칙을 항상 고려해야 합니다. 예를 들어 새로운 전처리 알고리즘을 추가할 때, 해당 로직은 `recipes`에서 선언되고 `components`에 모듈로 구현되어 Factory가 조립할 수 있어야 합니다. 만약 개발 도중 청사진과 다른 결정을 해야 한다면, 반드시 설계 수준에서 논의되고 철학과 부합하는 방향으로 조정되어야 합니다.

* **테스트와 검증**: 청사진의 철학에 따라, 설정과 로직이 분리되고 컴포넌트가 모듈화되어 있기 때문에 **단위 테스트** 작성이 용이합니다. 개발자는 각 컴포넌트를 개별적으로 테스트함으로써, 통합 전에 품질을 확보할 수 있습니다. 예컨대 Settings 스키마 검증 테스트, Factory의 동적 로딩 테스트, Trainer의 학습/평가 루틴 테스트 등을 통해 문제를 조기에 발견합니다. 이러한 테스트 철학 역시 "개발 과정에서의 신속한 피드백(local 환경 활용)"이라는 원칙 2와 맥락을 같이합니다.

* **향후 확장 방향**: MMP는 **현대적인 ML/DL 파이프라인 전반**을 아우르는 라이브러리를 목표로 합니다. 향후에는 현재의 학습 파이프라인 외에 **데이터 파이프라인**(예: 대규모 데이터 준비 및 Feature Engineering 단계 분리), **모델 서빙/배포 통합**(예: PyfuncWrapper를 활용한 즉시 배포 기능), **온라인 추적 및 모니터링**(배포 후 모델 성능 저하 감지 등) 기능들이 추가될 가능성이 있습니다. 이러한 기능을 추가할 때도, 본 청사진의 철학을 따르는 방향으로 구현할 것입니다. 예를 들어 데이터 파이프라인을 추가한다면 declarative YAML로 DAG를 정의하고, 새로운 components (FeatureGenerator 등)를 만들어 Factory가 조립하는 식으로 일관성을 유지할 것입니다.

* **협업 및 자동화**: 이 청사진은 개발자들뿐만 아니라, 코드 생성 도구나 AI 코-파일럿과 같은 **개발 보조 시스템**에도 제공되어 일관된 개발을 도울 수 있습니다. 설계 원칙이 사람과 도구 모두에게 공유됨으로써, 프로젝트의 비전이 흐려지지 않고 유지됩니다. MMP 프로젝트의 모든 기여자는 이 문서를 숙지하고, 기능 추가나 수정 시 이에 부합하는지 검토하는 것을 원칙으로 합니다.

마지막으로, Modern ML Pipeline은 \*\*"한 번 개발하면, 설정만 바꾸어 다양한 환경과 요구에 대응한다"\*\*는 목표를 지향합니다. 이 청사진은 그 목표를 이루기 위한 나침반 역할을 할 것이며, 여기 담긴 원칙과 구조를 기반으로 MMP를 현대적인 ML/DL 파이프라인의 모범 사례로 발전시켜 나갈 것입니다.

---

## (부록) 설정 스키마 개요(요약)

- `config/*.yaml`
  - `environment.app_env`: `local | local-dev | prod`
  - `feature_store.provider`: `none | feast | mock`
  - `serving.enabled`: `true | false`
  - `mlflow`, `artifact_stores`, 그 외 인프라 제약

- `recipes/*.yaml`
  - `model.class_path`
  - `model.loader`: `{ adapter, source_uri, entity_schema{ entity_columns[], timestamp_column } }`
  - `model.augmenter`
    - `type`: `feature_store | pass_through | sql_fallback`
    - `features`: `[{ feature_namespace, features[] }]` (feature_store 시 필수)
    - `fallback?`: `{ type: "sql", sql: { template_path, params_allowlist[] } }` (train/batch 한정)
  - `model.preprocessor`, `evaluation`, `hyperparameter_tuning`

- 정책 요약
  - Serving은 `feature_store` Augmenter만 허용(폴백/패스스루 금지)
  - Train/Batch는 레시피에 명시된 경우에 한해 `sql` 폴백 허용(자동 폴백 없음)
  - Local: `provider=none`, `augmenter=pass_through`, `serving.enabled=false`
  - Local-Dev/Prod: `provider=feast`, `augmenter=feature_store`, 서빙 허용

---

## 5. 보안/신뢰성/성능(요약)

- 안전한 템플릿/쿼리: allowlist 기반 파라미터, 금칙어(DDL/DML) 차단, 타임아웃, LIMIT 가드
- FS 헬스체크: 연결/권한/레포 구성 검증, 실패 시 명확한 메시지와 정책적 대응
- 정합성: 오프라인 조회에서 PIT 시점·스키마 검증 사용(미래 데이터 누출 차단)
- 운영 로그: 행/컬럼 수, 조인율/Null 비율, 지표, 경과시간 등 표준 로그 채택
- 직렬화 안정성: Augmenter 외부 의존성은 lazy init로 직렬화 안전성 확보

---

## 6. 테스트 원칙(요약)

- Settings 로더/스키마: recipe+config 병합 및 필수 필드 검증
- Factory/Registry: 컴포넌트 생성과 Augmenter 선택 정책(run_mode/환경/provider/헬스체크/폴백) 검증
- Augmenter 단위:
  - FeatureStore: offline `get_historical_features_with_validation`(PIT 검증), online `get_online_features` 조회 성공
  - SqlJoin: 안전 템플릿 렌더링(allowlist/금칙어/타임아웃/LIMIT)과 키 기반 Left Join 검증
  - PassThrough: 입력=출력 스키마 동일성
- 파이프라인 E2E(local, local-dev): Loader 최소 컬럼→Augmenter PIT→Preprocessor→Model 학습/추론
- Serving API: `entity_schema` 기반 입력 모델 생성, `PassThrough/SqlFallback` 서빙 차단
- MLflow 통합: 시그니처/데이터 스키마/스냅샷(Loader SQL, Augmenter config, Entity schema, Feature refs) 저장 확인

---

## 7. 레시피/설정 예시(요약)

### 11.1 Local

```yaml
# config/local.yaml
environment:
  app_env: local
feature_store:
  provider: none
serving:
  enabled: false
```

```yaml
# recipes/local_example.yaml
model:
  class_path: sklearn.linear_model.LogisticRegression
  loader:
    adapter: sql
    source_uri: recipes/sql/local_full_table.sql    # 전체 피처 포함 허용(로컬 편의)
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
  augmenter:
    type: pass_through
  preprocessor:
    column_transforms: {}
```

### 11.2 Local-Dev (mmp-local-dev 기반)

```yaml
# config/local-dev.yaml
environment:
  app_env: local-dev
feature_store:
  provider: feast
  feast_config:
    repo_path: ./local_fs_repo
serving:
  enabled: true
```

```yaml
# recipes/local_dev_example.yaml
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: sql
    source_uri: recipes/sql/load_entity_ts_and_label.sql.j2
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
  augmenter:
    type: feature_store
    features:
      - feature_namespace: user_profile
        features: [age, region, loyalty_score]
    fallback:
      type: sql
      sql:
        template_path: recipes/sql/pit_join_fallback.sql.j2
        params_allowlist: [start_ts, end_ts]
  preprocessor:
    column_transforms: {}
```

### 11.3 Prod

```yaml
# config/prod.yaml
environment:
  app_env: prod
feature_store:
  provider: feast
  feast_config:
    repo_path: s3://prod/feast/repo
serving:
  enabled: true
```

```yaml
# recipes/prod_example.yaml
model:
  class_path: xgboost.XGBClassifier
  loader:
    adapter: sql
    source_uri: recipes/sql/load_entity_ts_and_label_prod.sql.j2
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
  augmenter:
    type: feature_store
    features:
      - feature_namespace: user_profile
        features: [age, region, loyalty_score]
  preprocessor:
    column_transforms: {}
```

---

## 8. 실행 플로우 다이어그램(요약)

### 8.1 Train

```mermaid
flowchart TD
  A[Load Settings (recipe+config)] --> B[Factory Init]
  B --> C[Data Loader: entity + timestamp (+label)]
  C --> D[Augmenter: PIT feature retrieval (offline)]
  D --> E[Preprocessor: fit/transform]
  E --> F[Model: fit (HPO optional)]
  F --> G[Evaluate + Log Metrics]
  G --> H[PyfuncWrapper Package]
  H --> I[MLflow Log/Registry]
```

### 8.2 Batch Inference

```mermaid
flowchart TD
  A[Load PyfuncWrapper (run_id)] --> B[Render/Use Loader SQL]
  B --> C[Data Loader: entity + timestamp]
  C --> D[model.predict(params={run_mode: "batch"})]
  D --> E[Augmenter: PIT feature retrieval (offline)]
  E --> F[Preprocessor: transform]
  F --> G[Model: predict]
  G --> H[Save Results + Meta]
```

### 8.3 API Serving

```mermaid
flowchart TD
  A[Load PyfuncWrapper (run_id)] --> B[Dynamic Request Model (entity_schema)]
  B --> C[Request: entity + timestamp]
  C --> D[model.predict(params={run_mode: "serving"})]
  D --> E[Augmenter: online feature retrieval]
  E --> F[Preprocessor: transform]
  F --> G[Model: predict]
  G --> H[Response]
```
