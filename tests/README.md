# Tests Architecture & Authoring Guide

## 목적
이 문서는 현재 리팩토링된 테스트 구조의 전체 스켈레톤, 각 구성요소의 역할, 파일별 규칙과 패턴, 그리고 새로운 테스트를 작성할 때 일관성과 품질을 유지하기 위한 구체적인 가이드를 제공합니다. 큰 관점 → 세부 관점 순으로 설명합니다.

---

## 1) 전체 스켈레톤(High-level Skeleton)
```
tests/
├── conftest.py                     # 공용 픽스처 등록 (context fixtures 포함)
├── fixtures/
│   ├── contexts/                   # Context 클래스(설정/자원 준비 + 관찰 헬퍼)
│   │   ├── mlflow_context.py       # MLflowTestContext (uuid 실험명, file://mlruns, search_runs)
│   │   ├── database_context.py     # DatabaseTestContext (sqlite 격리)
│   │   └── component_context.py    # ComponentTestContext (Factory 스택)
│   ├── templates/                  # 테스트용 YAML 템플릿(필요 시)
│   └── expected/                   # 기대 산출물(지속 유지 필요 항목만)
│       └── metrics/
│           └── classification_baseline.json
├── integration/                    # 통합/시나리오 테스트 (퍼블릭 API 중심)
│   ├── test_mlflow_integration.py
│   ├── test_component_interactions.py
│   ├── test_database_integration.py
│   ├── test_pipeline_orchestration.py
│   ├── test_settings_integration.py
│   └── test_error_propagation.py
└── unit/                           # 단위 테스트(인터페이스 계약, 경계 조건)
    ├── settings/ ...
    ├── cli/ ...
    └── factory/ ...
```

### 스위트 구분과 철학
- Unit: 최소 격리, 빠른 피드백, 계약 검증(인터페이스/타입/경계값).
- Integration: 실제 퍼블릭 API 호출 중심, 컨텍스트로 세팅을 중앙화, 비즈니스 의도 검증.
- E2E(필요 시): 전체 파이프라인 흐름 정상 동작 검증(현재는 Integration 내에서 커버).

---

## 2) 핵심 역할(Responsibilities)
- Contexts (tests/fixtures/contexts/*):
  - 설정/데이터/자원 준비를 캡슐화하고, 퍼블릭 API 호출 후 결과를 관찰하는 헬퍼 제공.
  - “오케스트레이션을 호출하고 결과를 관찰”만 담당. 비즈니스 로직 재구현 금지.
- Templates (tests/fixtures/templates/*):
  - 테스트 전용 YAML 템플릿(최소화). 중복/중복된 설정 방지 목적.
- Expected (tests/fixtures/expected/*):
  - 장기 유지가 필요한 기준 산출물만 저장(예: metrics baseline). 빈 디렉터리/플레이스홀더 금지.
- Integration tests:
  - 컨텍스트를 사용해 설정/자원 준비 코드를 제거하고, 검증 로직에 집중.
- Unit tests:
  - 빠르고 결정론적이며, 외부 자원 의존 없이 동작.

---

## 3) 표준 정책(Policies)
- MLflow 파일 스토어: `file://{temp_dir}/mlruns` 고정 (외부 경로 금지).
- 실험명/모델명: `uuid4().hex[:8]` 접미사 사용(시간기반 명명 금지).
- 데이터 경로 주입: CLI `--data-path`(또는 테스트 컨텍스트 빌더의 `with_data_path`)로만 주입. 레시피에 `loader.source_uri`를 저장하지 않음.
- 상태 격리: 테스트마다 새 `Settings`/`Factory` 생성. 전역 캐시/공유 금지.
- 결정론성: 컨텍스트 데이터 생성은 고정 시드(`seed=42` 기본) 사용.
- Pydantic v2: `@field_validator`, `model_config=ConfigDict(...)`, `.model_dump()` 사용.
- 병렬 실행: 워커 간 MLflow 파일 스토어 충돌 방지를 위해 temp 디렉토리별 고유 경로 사용.

---

## 4) 컨텍스트 상세(Contexts Detail)

### 4.1 MLflowTestContext (`tests/fixtures/contexts/mlflow_context.py`)
- 제공 속성: `settings`, `data_path`, `tracking_uri`, `experiment_name`
- 필수 헬퍼: `experiment_exists()`, `get_experiment_run_count()`, `get_run_metrics()`
- 구현 요점:
  - `tracking_uri = f"file://{temp_dir}/mlruns"`
  - `experiment_name = f"{suffix}-{uuid4().hex[:8]}"`
  - `MlflowClient.search_runs([...])` 사용(MLflow 3.x 호환)

예시:
```python
with mlflow_test_context.for_classification(experiment="experiment_creation") as ctx:
    mlflow.set_tracking_uri(ctx.mlflow_uri)
    result = run_train_pipeline(ctx.settings)
    assert result is not None
    assert ctx.experiment_exists()
    assert ctx.get_experiment_run_count() == 1
    assert isinstance(ctx.get_run_metrics(), dict)
```

### 4.2 ComponentTestContext (`tests/fixtures/contexts/component_context.py`)
- 역할: Factory 스택(adapter/model/evaluator/preprocessor) 준비와 데이터 흐름 관찰.
- 규칙: 퍼블릭 API(`Factory.create_*`)만 호출, 내부 엔진 재현 금지.

예시:
```python
with component_test_context.classification_stack() as ctx:
    raw_df = ctx.adapter.read(ctx.data_path)
    processed_df = ctx.prepare_model_input(raw_df)
    assert ctx.validate_data_flow(raw_df, processed_df)
```

### 4.3 DatabaseTestContext (`tests/fixtures/contexts/database_context.py`)
- 역할: SQLite 기반 임시 DB 준비 및 테이블 적재(테스트 격리 보장).
- 규칙: DB는 temp 디렉토리 하위 파일로만 생성, 외부 의존 금지.

---

## 5) 파일/디렉토리별 세부 규칙(Per-file Guidance)
- `tests/conftest.py`:
  - 컨텍스트 픽스처 등록(`mlflow_test_context`, `component_test_context`, `database_test_context`).
  - 데이터 생성기, 임시 디렉토리, 성능 벤치마크 픽스처 등 공용 제공.
- `tests/fixtures/templates/*`:
  - 꼭 필요한 최소 템플릿만 유지. 중복 템플릿 금지.
- `tests/fixtures/expected/*`:
  - 현재 유지 대상: `metrics/classification_baseline.json`.
  - 빈 디렉토리(예: predictions/responses)는 제거. 필요해지면 실제 산출물과 함께 추가.
- `tests/integration/*`:
  - MLflow/컴포넌트/DB/파이프라인/설정/에러 전파 등 기능별 그룹화.
  - MLflow 관련 테스트는 반드시 컨텍스트 또는 정책을 준수(URI, uuid, search_runs).
- `tests/unit/*`:
  - 인터페이스 계약 준수, 에지 케이스 검증, 외부 I/O 없음.

---

## 6) 테스트 작성 절차(Authoring Flow)
1. 목적 정의: 무엇을 검증할 것인가? (행동/계약/시나리오)
2. 컨텍스트 선택/확장: MLflow/Component/Database 중 선택, 필요하면 최소 규약 준수로 신규 추가.
3. 데이터 준비: 컨텍스트가 제공하는 생성기/헬퍼 사용(고정 시드). 파일은 temp 디렉토리에 저장.
4. 퍼블릭 API 호출: `run_train_pipeline`, `Factory.create_*`, `MlflowClient` 등.
5. 관찰/검증: 컨텍스트 헬퍼로 결과를 관찰하고 단언. 비즈니스 로직 재구현 금지.
6. 표준 정책 확인: URI/file-store, uuid 명명, data-path 주입, 결정론성.
7. 성능/격리 점검: 한 테스트-한 run, 상태 공유 금지, 필요 시 `performance_benchmark` 사용.
8. 실행/병렬: `pytest -n auto` 권장. MLflow 파일 스토어 경로 충돌 방지 확인.

---

## 7) 안티 패턴(Anti-Patterns)
- 컨텍스트 내부에서 엔진/파이프라인 로직 재현
- 전역 상태/캐시 공유
- 시간기반 명명(`int(time.time())`) 사용
- 외부 경로 MLflow store 사용(`sqlite:///...` or 원격) – 테스트는 `file://.../mlruns` 고정
- 레시피에 `loader.source_uri` 영구 저장(주입만 허용)
- 불필요한 모킹(레지스트리/팩토리 경로)

---

## 8) 마이그레이션/유지 전략
- A/B 공존: 기존(v1)과 새로운(context v2) 테스트를 같은 파일에서 공존 가능.
- 동등성 검증: 결과/아티팩트(메트릭/파라미터/시그니처/스키마) 비교로 동등성 보장.
- 성공 후 정리: v2가 안정화되면 v1 중복 세팅/헬퍼 제거.

---

## 9) CI/실행 가이드
- 스위트 분리 실행: `unit`/`integration`/`e2e` 잡 분리 권장.
- 병렬: `pytest -n auto --dist=loadscope --durations=15`.
- 게이팅: A/B 동등성, 컨텍스트 init 시간 상한(0.12s) 체크.

---

## 10) 새 테스트 리뷰 체크리스트
- [ ] 컨텍스트 사용으로 세팅 코드 최소화
- [ ] MLflow file store/uuid 명명/seed 정책 준수
- [ ] 퍼블릭 API만 호출(엔진 재현 금지)
- [ ] 상태 공유 없음(Factory/Settings 새로 생성)
- [ ] 환경/경로는 temp 디렉토리 내부만 사용
- [ ] 필요 시 기대 산출물은 실제 값으로 추가(빈 디렉토리 금지)
- [ ] 실행 시간/격리/플레이키 방지 점검

---

## 부록: 스니펫
- MLflow 실험 검증 스니펫
```python
with mlflow_test_context.for_classification(experiment="exp") as ctx:
    mlflow.set_tracking_uri(ctx.mlflow_uri)
    result = run_train_pipeline(ctx.settings)
    client = MlflowClient(tracking_uri=ctx.mlflow_uri)
    exp = client.get_experiment_by_name(ctx.experiment_name)
    assert exp is not None
    assert client.get_run(result.run_id) is not None
```

- Component 데이터 흐름 검증 스니펫
```python
with component_test_context.classification_stack() as ctx:
    raw_df = ctx.adapter.read(ctx.data_path)
    processed_df = ctx.prepare_model_input(raw_df)
    assert ctx.validate_data_flow(raw_df, processed_df)
```

- Database 컨텍스트 스니펫
```python
with database_test_context.sqlite_db({"users": df}) as db:
    assert db.connection_uri.startswith("sqlite:///")
```

---

본 가이드는 리팩토링 철학(책임 분리, 표준화, 퍼블릭 API 중심, 제로리스크 전환)을 유지하면서 테스트를 확장/개선하기 위한 실천적 기준입니다. 새로운 테스트 추가 시 위 가이드를 준수하면, 리팩토링 결과를 안정적으로 유지하고 품질을 지속적으로 향상시킬 수 있습니다.