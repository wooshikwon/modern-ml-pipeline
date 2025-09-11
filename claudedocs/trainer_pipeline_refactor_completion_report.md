### 목적과 범위

- **목적**: 테스트 리팩토링(컨텍스트 아키텍처) 이후, 런타임 코드베이스에서 진행된 컴포넌트/팩토리/트레이너/파이프라인 관련 구조 개편의 완료 결과를 정리합니다.
- **범위**: `components/`(Trainer 레이어 포함), `factory/`, `pipelines/`, 관련 통합 지점(Registry/DI/MLflow 로깅/Fetcher 경로).

### 최종 아키텍처 정렬(핵심 합의)

- **레이어 역할**
  - **Pipeline**: 전반 오케스트레이션(로딩 → 증강 → 분할/전처리 → Trainer 호출 → 평가/로깅 → 패키징). HPO는 호출하지 않음.
  - **Trainer**: 모델 학습의 단일 책임. Settings 기반으로 고정 학습/튜닝 분기를 수행하고, 튜닝 시 내부에서 Optimizer를 사용하여 최적화 후 최종 재학습.
  - **Factory**: 모든 컴포넌트 생성 책임. Registry 기반 생성 일원화. Trainer에는 `factory_provider` DI를 주입해 내부 의존(예: Evaluator, OptunaIntegration)을 필요 시 획득.
  - **Registry**: 각 레이어의 구현을 등록/생성. Trainer 레이어는 단일 `TrainerRegistry.register`로 트레이너/옵티마이저 모두 등록.

- **핵심 디자인 원칙**
  - **Registry 패턴**: 모든 생성은 Registry→Factory 일관 흐름. Trainer 레이어에서도 동일 패턴 유지.
  - **DI(Dependency Injection)**: Trainer가 엔진/인프라에 직접 의존하지 않도록 `factory_provider`를 통해 외부에서 의존 획득.
  - **분리된 책임**: 데이터 준비/전처리는 Pipeline·DataHandler·Preprocessor에서, 학습/HPO는 Trainer에서, 로깅/패키징은 전용 유틸/Factory에서 담당.

### 변경 사항 상세

- **Trainer 레이어 (`src/components/trainer/trainer.py`)**
  - 반환형을 `(trained_model, trainer_info)`로 표준화하여 파이프라인/로깅과의 인터페이스 명확화.
  - `factory_provider`(Callable)을 DI로 받아 내부에서 `factory.create_evaluator()` 등을 호출 가능.
  - HPO 복원: `TrainerRegistry.create_optimizer("optuna", settings, factory_provider)`로 옵티마이저 조회→Objective에서 evaluator로 검증 점수 산출→최적 파라미터 `set_params` 적용 후 재학습.
  - 작업별 학습 루틴 통일: classification/regression/clustering/causal/timeseries에 맞게 `_fit_model` 분기. causal은 `additional_data['treatment']` 사용.
  - 자체 메타 수집(`training_results`)에 HPO 결과 병합.

- **Optimizer(Optuna) (`src/components/trainer/modules/optimizer.py`)**
  - `TrainerRegistry.register("optuna", OptunaOptimizer)`로 트레이너 레이어 내에서 self-register.
  - `factory.create_optuna_integration()` 사용으로 엔진 의존을 DI 경유.
  - metric→direction 자동 매핑, trial 진행 상황 주기 출력, best params/score/시간·이력 수집 반환.

- **TrainerRegistry (`src/components/trainer/registry.py`)**
  - 단일 `register(name, klass)`로 트레이너/옵티마이저 모두 등록 가능.
  - `create(trainer_type, ...)`, `create_optimizer(name, ...)` API 제공.
  - 기존 테스트/팩토리 호출과의 호환성 유지.

- **Factory (`src/factory/factory.py`)**
  - `_ensure_components_registered()`에서 `src.components.trainer` 임포트 → Trainer/Optimizer의 self-registration 보장.
  - `create_trainer()`가 `TrainerRegistry.create(..., settings=self.settings, factory_provider=lambda: self)`로 DI 주입.
  - `create_optuna_integration()` 제공(Trainer 내부 Optimizer가 사용).
  - Fetcher 생성 경로 정비: pass_through/feature_store 분기, 캐싱/로그 일관화.

- **Pipeline (`src/pipelines/train_pipeline.py`)**
  - HPO 로직 제거 및 **Trainer에 위임**. 파이프라인은 학습 호출만 수행.
  - 순서 정리: Data Loading → Feature Augmentation(`fetcher.fetch`) → split_and_prepare(DataHandler) → Preprocessing → `trainer.train` → Evaluator → 결과 로깅 → PyfuncWrapper 패키징.
  - MLflow 로깅 단순화: `log_training_results(settings, metrics, training_results)`로 일원화.

- **Fetcher (`src/components/fetcher/modules/pass_through_fetcher.py`)**
  - 사용자가 증강 비활성화를 원할 때 자동 선택. 입력 DataFrame을 변경 없이 반환.
  - Factory가 로컬/미구성/명시적 `pass_through` 조건에서 자동 선택.

### 호환성 및 이전성

- **외부 API**: 파이프라인 엔트리포인트/CLI 영향 없음. 내부적으로 Trainer 반환형이 `(model, info)`로 표준화되었고 파이프라인이 이를 사용.
- **자동 등록**: Factory 인스턴스화 시 컴포넌트 모듈 self-registration 보장 → 레지스트리 일관 동작.
- **폴백 경로**: 옵티마이저 미등록/Optuna 미설치 시 Trainer는 고정 파라미터 학습으로 안전 폴백.

### 설정(Recipe) 가이드

- **HPO 활성화 예시**
```yaml
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  hyperparameters:
    tuning_enabled: true
    optimization_metric: accuracy  # or f1, roc_auc, r2, mae, mse, ...
    n_trials: 20
    timeout: 600
    tunable:
      n_estimators: {low: 50, high: 300}
      max_depth: {low: 3, high: 12}
    fixed:
      random_state: 42
```

- **Fetcher 비활성화(패스스루)**
  - 레시피에 fetcher 미정의 또는 `type: pass_through` 설정, 혹은 로컬 환경/feature_store 미구성 시 자동 패스스루 동작.

### Timeseries 규약 및 카탈로그 매칭

- **timestamp_column 필수**: `task_choice: timeseries`인 레시피는 `data.data_interface.timestamp_column`을 반드시 지정해야 하며, Validator에서 이를 강제한다.
- **카탈로그의 data_handler 우선**: DataHandler 선택은 카탈로그의 `data_handler` 선언이 우선한다. Timeseries라도 시퀀스 텐서 전처리가 필요한 LSTM은 `deeplearning` 핸들러가 맞다.
- **Feature Store와의 연계**: Feature Store 사용 시 `data.fetcher.timestamp_column` 지정이 권장된다. 이는 PIT join 기준 컬럼으로 사용되며, 핸들러에서 제외 컬럼 판단에도 활용된다.

### 리스크/주의 및 후속 과제

- **경고 항목**: 일부 외부 라이브러리 경고(예: MLflow deprecation, pandas FutureWarning)는 기능 영향이 없으며 추후 버전 정합성 점검 필요.
- **평가 지표 스키마**: 사용자가 `optimization_metric`을 비표준 키로 지정 시 evaluator가 해당 키를 반환하는지 확인 필요.
- **딥러닝/시계열 고도화**: LSTM 등 시퀀스 모델의 원형 입력 유지/스키마 확장 여부는 추후 설계.

### 변경 파일 개요(핵심)

- `src/components/trainer/trainer.py`: DI·튜닝 복원·반환형 표준화·작업별 `_fit_model` 정리.
- `src/components/trainer/registry.py`: 단일 `register` + `create_optimizer` 추가.
- `src/components/trainer/modules/optimizer.py`: Optuna Optimizer 구현 및 self-register.
- `src/pipelines/train_pipeline.py`: Trainer 위임형으로 간소화, 로깅/패키징 일관화.
- `src/factory/factory.py`: 컴포넌트 auto-register 보장, Trainer DI 주입, Fetcher/OptunaIntegration 생성 경로 정리.
- `src/components/fetcher/modules/pass_through_fetcher.py`: 패스스루 증강 경로 명시화.

### 결론

- 테스트 리팩토링 철학(컨텍스트/표준화/점진적 무리 없는 변경)을 런타임 코드에도 일관 적용했습니다. Trainer가 학습의 단일 책임을 명확히 갖고, Registry·Factory·Pipeline의 역할이 분리되어 유지보수성과 확장성이 향상되었습니다. HPO는 Trainer 내부로 안전하게 복원되었으며, DI와 Registry로 결합도를 낮추면서도 사용성은 단순화되었습니다.

