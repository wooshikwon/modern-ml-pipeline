## Comprehensive Testing Strategy & Architecture (2025 Edition)

## Executive Summary

- 목적: 현재 리팩토링된 테스트 구조(Context-based)와 런타임 아키텍처(Trainer/Factory/Pipeline/Registry/DI/HPO 복원)에 정합적인, 구현 지향형 테스트 전략의 단일 기준 문서.
- 철학: No Mock Hell, 퍼블릭 API 계약 검증, 결정론적 실행, CI 게이트로 품질 보장.
- 상태(현재): 컨텍스트 아키텍처 도입, MLflow/UUID 표준화, Validator 게이트 도입, Fetcher/Timeseries/Causal/LSTM 경로 테스트 강화 완료.
- 목표(단기): 필수 시나리오의 회귀 방지 + 문서-코드 일치성 유지 + CI 안정화.

---

## Principles & Policies

### Core Principles
- No Mock Hell: 가능하면 실제 컴포넌트/실제(작은) 데이터로 검증.
- Public API 우선: CLI/파이프라인/팩토리/컴포넌트의 공개 경계만 호출.
- Deterministic: 고정 시드, UUID 명명, 테스트별 격리 리소스, 고정 MLflow 파일 스토어.
- Zero-Risk Incremental: v1/v2 공존 가능, 회귀 없을 때만 교체.

### Repository-wide Policies (Tests)
- 데이터 경로 주입: 데이터 경로는 항상 CLI `--data-path` 또는 테스트 컨텍스트 빌더의 `with_data_path()`로만 주입. 레시피에는 `loader.source_uri`를 저장하지 않음.
- MLflow 표준: `file://{temp_dir}/mlruns` 고정, 실험/런명은 `uuid4().hex[:8]` 기반.
- Pydantic v2: v1 패턴 금지(`class Config`, `@validator`, `.dict()` 등).
- Timeseries 규약(필수): `recipe.data.data_interface.timestamp_column` 필수. 누락 시 Validator/CI에서 실패.
- Feature Store 권장: `data.fetcher.timestamp_column`을 지정(POINT-IN-TIME join 기준 컬럼).
- 카탈로그-핸들러 매칭: DataHandler 선택은 카탈로그의 `data_handler` 선언이 우선(LSTM은 timeseries이더라도 `deeplearning`).

---

## Current Architecture Alignment

### Runtime (요약)
- Pipeline: 오케스트레이션(로딩→증강→분할/전처리→Trainer 호출→평가/로깅→패키징)
- Trainer: 학습 단일 책임, 내부 HPO 복원(Optimizer registry), DI(`factory_provider`)로 Evaluator/Optuna 접근
- Factory: 모든 컴포넌트 생성, self-registration 트리거, PyfuncWrapper 생성
- Registry: 트레이너/옵티마이저/핸들러/평가자/어댑터/페처/프로세서 등록·조회 일원화

### Tests (요약)
- Context-based Test Architecture: `tests/fixtures/contexts/*` 중심. MLflow/Component/Scenario 컨텍스트.
- 표준화: MLflow 파일 스토어 고정, UUID 명명, 고정 시드, 격리 리소스, 결정론적 실행.
- 헬퍼: `SettingsBuilder.with_timestamp_column()`, `.with_treatment_column()`으로 가독성/일관성 확보.

---

## Tests Directory Layout (현행)

```
tests/
├── conftest.py                  # 전역 픽스처, SettingsBuilder, 데이터 생성기, 컨텍스트 등록
├── fixtures/
│   ├── contexts/                # MLflow/Component/DB 등 컨텍스트
│   ├── data/                    # 샘플 데이터(csv 등)
│   ├── templates/               # 템플릿(YAML 등)
│   └── expected/                # 기준 메트릭/응답(선택)
├── unit/
│   ├── cli/                     # CLI 명령 테스트
│   ├── components/              # 컴포넌트별 단위 테스트
│   ├── factory/                 # 팩토리/레지스트리/계약
│   ├── settings/                # 로딩/검증/카탈로그
│   └── utils/                   # 유틸리티/로깅 등
└── integration/
    ├── test_mlflow_integration.py
    ├── test_pipeline_orchestration.py
    ├── test_component_interactions.py
    ├── test_settings_integration.py
    ├── test_database_integration.py
    └── test_error_propagation.py
```

---

## CI Strategy

- 워크플로 분리: unit / integration / v2_pilot
- 게이트(중요):
  - Validator 크리티컬 체크: timeseries에서 `timestamp_column` 누락 시 PR 실패
  - v2 시나리오 최소 검증: MLflow/컴포넌트 흐름 핵심 경로 그린 유지
- 권장 실행: `pytest -n auto --dist=loadscope --durations=15`

---

## Coverage Map (핵심 시나리오)

### Covered (완료)
- MLflow 통합: 실험/런 생성, 메트릭·파라미터(HPO 포함) 로깅, 아티팩트 등록
- Fetcher 분기: `pass_through` 스킵, feature_store 선택(Feast 미설치 시 합리적 skip)
- Causal 경로: `datahandler.split_and_prepare()`가 `additional_data['treatment']` 채우고 Trainer가 소비
- Timeseries 핸들러: 시간 기준 분할/특성 처리, `timestamp_column` 필수 검증
- LSTM(TimeSeries+DeepLearning): 시퀀스(flatten→3D 복원) 학습 end-to-end 경로
- HPO(Trainer 내부): Optimizer registry, DI로 evaluator 사용하여 objective 측정
- Settings Validation: Timeseries `timestamp_column` 필수 유닛 및 CI 게이트

### Recommended Adds (경량·고효율)
- Unit: PyfuncWrapper 시그니처 생성 시 FS `timestamp_column`이 datetime 캐스팅·스키마에 반영되는지 확인
- Unit: TimeseriesDataHandler의 exclude 컬럼 계산에서 `fetcher.type == feature_store` 분기 반영 확인(가짜 설정 주입)
- CLI(E2E-lite): `get-recipe`에서 timeseries 선택 시 `timestamp_column` 미입력 불가(프롬프트 validator) 검증(Typer CliRunner)
- E2E 최소 1건: `mmp train` 단일 경로가 30초 내 그린, 모델/아티팩트 생성 확인
- 템플릿 스냅샷: `recipe.yaml.j2`의 시계열/FS 주석 및 필수 키 존재 스냅샷 1건

---

## Test Writing Standards

### Naming
- 함수: `test_<area>_<behavior>[_v2]`
- 컨텍스트: `*TestContext`, 매니저 `*ContextManager`
- 픽스처: `<area>_test_context`, `settings_builder`, `isolated_temp_directory`

### Patterns
- Given/When/Then 명료화, 퍼블릭 API만 호출
- 실패는 “합리적 문구/키워드” 포함 여부로 관대하게 검증
- 결정론: 고정 시드, UUID 명명, 격리 경로

### Performance Budgets (권장)
- Unit: < 100ms 평균
- Integration: < 5s 평균
- E2E: < 30s 단건

---

## Developer Workflow

### How to Run
```bash
# 전체
pytest -q

# 유닛
pytest -q tests/unit

# 통합
pytest -q tests/integration -m "not slow"

# 특정 크리티컬 게이트
pytest -q tests/unit/settings/test_settings_validation.py::TestRecipeValidation::test_recipe_validation_timeseries_requires_timestamp
```

### Parallel & Isolation
- `-n auto` 권장, MLflow 파일스토어는 테스트별 고정 디렉토리 사용(컨텍스트에서 보장)
- optuna/feast 등 외부 의존 미설치 시 해당 테스트는 합리적으로 skip

---

## Roadmap (Now → Next)

### Now (즉시)
1) Unit: PyfuncWrapper FS timestamp 반영 테스트 추가
2) Unit: TimeseriesDataHandler exclude 컬럼 분기 테스트 추가
3) CLI(E2E-lite): `get-recipe` timeseries 프롬프트 강제 검증 추가

### Next (단기)
4) E2E 1건: `mmp train` happy path(MLflow run, 아티팩트 존재) 검증
5) 템플릿 스냅샷 1건 보강(시계열/FS 주석 & 키)

### Later (선택)
- 성능 벤치마크 태그 정비, 긴 테스트는 `-m not slow` 제외
- MLflow 아티팩트 동등성 비교 리포트 자동화(선택)

---

## Success Criteria
- CI: Validator 게이트 + v2 핵심 시나리오 그린 유지
- 커버리지: 필수 흐름(Fetcher 분기, Causal, Timeseries, LSTM, HPO, MLflow) 최소 1건 이상
- 문서-코드 일치: Timeseries/FS 정책, 데이터 경로 주입 정책이 테스트·템플릿·CLI에 일관 반영

---

## References
- Tests Structure & Policies: `claudedocs/tests_structure_refactoring_analysis.md`
- Trainer/Pipeline Refactor Report: `claudedocs/trainer_pipeline_refactor_completion_report.md`
- Test Strategy (Trainer/Pipeline): `claudedocs/trainer_pipeline_test_strategy.md`

---

## Appendix

### Quick Context Example (MLflow)
```python
def test_mlflow_quickstart(mlflow_test_context):
    with mlflow_test_context.for_classification(experiment="quickstart") as ctx:
        result = run_train_pipeline(ctx.settings)
        assert result is not None
        assert ctx.experiment_exists()
```

### SettingsBuilder Helpers
```python
settings = settings_builder \
  .with_task("timeseries") \
  .with_timestamp_column("timestamp") \
  .with_treatment_column("treatment") \
  .build()
```