### 목적

- **목표**: `trainer_pipeline_refactor_completion_report.md`와 `tests_structure_refactoring_analysis.md`의 합의에 따라, Trainer/Pipeline 리팩토링에 대응하는 테스트 코드 수정 전략을 수립합니다.
- **대상**: `tests/` 전반의 컨텍스트 기반 구조를 유지하며, 새 학습 책임 분리(HPO 포함)와 Fetcher/Preprocessor/DataHandler/MLflow 로깅/패키징 경로를 검증합니다.

### 테스트 철학(상위 원칙)

- **컨텍스트 우선**: 퍼블릭 API 호출만, 오케스트레이션 재구현 금지.
- **관찰 중심**: 결과(MLflow 메트릭/파라미터/시그니처/스키마, 반환값)를 관찰로 검증.
- **격리와 결정론**: UUID 명명, 고정 시드, 임시 디렉토리, `file://.../mlruns` 고정.
- **점진적 마이그레이션**: v1과 v2 공존 가능. 회귀 없을 때만 대체.

### 변경 핵심에 대한 테스트 영향 분석

- **Trainer 내부 HPO 복원**
  - 파이프라인은 `trainer.train(...)`만 호출. 테스트는 HPO 수행 여부/결과가 `trainer_info['hyperparameter_optimization']`에 반영되는지 확인.
  - 튜닝 메트릭/방향/베스트 파라미터/베스트 스코어/트라이얼 카운트 등 핵심 필드 검증.

- **TrainerRegistry 단일 register + Optimizer 등록**
  - 컨텍스트 초기화 시점에 `src.components.trainer` 임포트로 self-registration 보장. 테스트는 `TrainerRegistry.get_available_optimizer_types()`에 `optuna` 존재 검증.

- **Factory DI**
  - `Factory.create_trainer()`가 `factory_provider`를 주입. Trainer 내부 objective에서 evaluator 사용 가능함을 통합 테스트로 보증.

- **Fetcher 경로**
  - 기본/로컬/미구성/명시적 `pass_through`에서 증강 스킵 확인.
  - feature_store가 구성된 경우 `feature_store_fetcher`가 호출되어 열/행 변화(또는 로그) 검증.

- **Causal 처리**
  - `TabularDataHandler._prepare_causal_data`가 `additional_data['treatment']`를 채우고 Trainer가 이를 사용해 학습하는지 확인.

- **MLflow 로깅 단일화**
  - `log_training_results`로 메트릭/파라미터(HPO 포함)/콘솔 출력 요약이 기록되는지 검증.
 
- **Timeseries 규약 준수**
  - timeseries 작업에서 `data_interface.timestamp_column` 누락 시 Validator가 실패하는지 유닛 테스트로 보장.
  - Feature Store 사용 시 `fetcher.timestamp_column` 지정 권장 문구를 레시피 템플릿과 빌더에서 노출하고, 존재 시 핸들러 제외 컬럼 로직에 반영되는지 통합 테스트로 점검.

### 테스트 수정 로드맵(컨텍스트 기반)

1) MLflow 통합 컨텍스트 강화
- 기존 v2 케이스 유지 + 추가 검증:
  - HPO 활성/비활성 두 케이스에서 run 하나 생성, 메트릭 존재, 파라미터(HPO best_params 병합) 기록 확인.
  - Catalog/Signature/Schema의 존재 확인 유지.

2) Component 상호작용 컨텍스트 업데이트
- Trainer가 DI로 evaluator를 사용해 objective를 평가하는 경로를 시나리오로 점검.
- Fetcher가 pass_through/feature_store에서 각각 올바르게 동작하는지 데이터 변형/로그 기반 검증.

3) DataHandler 컨텍스트 점검
- `split_and_prepare` 반환의 `additional_data` 스키마 검증(특히 causal: `treatment`).
- timeseries에서 timestamp 기반 분할/특성 생성 검증 지속.

4) Pipeline 시나리오 컨텍스트
- end-to-end: 로딩→증강→분할/전처리→Trainer 위임→평가/로깅→패키징 흐름을 하나의 시나리오로 검증.
- 파이프라인이 HPO를 호출하지 않음을 보조 단언(트레이너 info에 HPO 결과가 있고, 파이프라인 자체에는 튜닝 루프 부재).

### 구체 테스트 케이스 제안(v2 추가)

- **tests/integration/test_mlflow_integration.py**:
  - `test_training_with_hpo_logs_hyperopt_results_v2`
    - tuning_enabled: true → `trainer_info['hyperparameter_optimization']` 필드들 검증.
  - `test_training_without_hpo_records_fixed_params_v2`
    - tuning_enabled: false → HPO disabled 메타 확인.

- **tests/integration/test_pipeline_orchestration.py**:
  - `test_pipeline_delegates_hpo_to_trainer_v2`
    - 파이프라인 호출 후, HPO 루프가 파이프라인에 없고(trainer만 수행), 결과는 MLflow에 기록됨 검증.

- **tests/integration/test_component_interactions.py**:
  - `test_trainer_uses_evaluator_via_di_in_hpo_objective_v2`
    - Factory를 통한 evaluator 사용 경로(목표 메트릭 키 일치) 검증.

- **tests/integration/test_settings_integration.py**:
  - `test_optimizer_registry_self_registration_v2`
    - Factory 초기화 후 `TrainerRegistry.get_available_optimizer_types()`에 `optuna` 포함 검증.

- **tests/integration/test_component_interactions.py (fetcher)**:
  - `test_pass_through_fetcher_skips_augmentation_v2`
  - `test_feature_store_fetcher_augmentation_happens_v2`

- **tests/integration/test_error_propagation.py**:
  - `test_hpo_missing_optuna_falls_back_to_fixed_training_v2` (옵션: optuna 미설치/비등록 시 폴백 경로)

- **tests/unit/components/test_trainer_interfaces.py**:
  - `test_trainer_returns_model_and_info_tuple_v2`
  - `test_causal_training_consumes_treatment_from_additional_data_v2`
 
- **tests/unit/settings/test_settings_validation.py**:
  - `test_recipe_validation_timeseries_requires_timestamp` 유지/강화.
  - (선택) `test_feature_store_timestamp_is_propagated_when_present_v2` — fetcher가 feature_store일 때 timestamp_column이 설정되어 핸들러 제외 컬럼 계산에 반영되는지 확인.

### 구현 지침(요약)

- 컨텍스트 픽스처 재사용: MLflow/Component/Scenario 컨텍스트에 새 단언만 추가.
- UUID 기반 명명과 고정 시드 유지. MLflow `file://.../mlruns` 준수.
- 테스트는 퍼블릭 API 호출만 사용(Factory/Trainer/Run pipeline 등).
- 신규 v2 케이스는 기존 v1과 나란히 두고, 성공/성능 검증 후 대체.

### 완료 기준

- 새 v2 케이스 모두 통과, 기존 v1과 결과 동등.
- MLflow 아티팩트(메트릭/파라미터/시그니처/스키마) 일관성 유지.
- 컨텍스트 최소 규약 및 표준 정책(경로, 명명, 격리, 시드) 준수.

