# Test Architecture & Development Guide

## 목적
이 문서는 Modern ML Pipeline 프로젝트의 테스트 아키텍처, 개발 철학, 그리고 테스트 작성 시 준수해야 할 핵심 원칙을 정의합니다.

---

## 1. 테스트 철학

### 핵심 원칙
- **Real Object Testing**: Mock 사용을 최소화하고 실제 객체 사용
- **Public API Focus**: 내부 구현이 아닌 퍼블릭 인터페이스 테스트
- **Deterministic Execution**: 모든 테스트는 결정론적이고 재현 가능
- **Test Isolation**: 각 테스트는 독립적으로 실행 가능
- **Clear Boundaries**: 테스트 계층별 명확한 책임 분리

### No Mock Hell 원칙
Mock 사용은 계층별로 엄격히 제한:
- **금지**: 내부 컴포넌트, 비즈니스 로직
- **허용**: 외부 서비스(MLflow server, DB), 네트워크 I/O
- **예외**: CLI 단위 테스트에서 파이프라인 실행 부분

---

## 실행/메트릭 수집 표준

- 단일 실행(그룹 분리 + 커버리지 + 메트릭 집계):
  ```bash
  python3 scripts/run_tests_split.py
  ```

- 병렬 워커 제어(기본은 안정성 우선으로 각 그룹 1 워커):
  ```bash
  UNIT_WORKERS=1 INTEGRATION_WORKERS=1 E2E_WORKERS=1 python3 scripts/run_tests_split.py
  ```

- 산출물(`reports/`):
  - `pytest.unit.json`, `pytest.integration.json`, `pytest.e2e.json`
  - `coverage.unit.xml`, `coverage.integration.xml`, `coverage.e2e.xml`
  - `metrics.summary.json`

- 사용 플러그인/옵션:
  - `pytest-xdist`, `pytest-cov`, `pytest-json-report`, `pytest-timeout`
  - 기본 타임아웃: `--timeout=60` (테스트 단위)

- 서버 테스트 직렬화:
  - 서버/프로세스 의존 테스트는 `@pytest.mark.server`를 사용
  - `server_serial_execution` 고정(fixture)과 파일락으로 병렬 경합 방지

- 실행 소음/프로세스 관리:
  - `MMP_QUIET_PROMPTS=1`로 비대화형 프롬프트 메시지 억제
  - 전역 프로세스 강제 종료는 기본 비활성화: `MMP_ENABLE_GLOBAL_KILL=0` (필요 시 `1`)

---

## 2. 디렉토리 구조

```
tests/
├── conftest.py                # 공용 fixtures, Context 클래스
├── fixtures/                  # 테스트 자원
│   ├── contexts/              # Test Context 클래스들
│   │   ├── mlflow_context.py
│   │   ├── component_context.py
│   │   └── database_context.py
│   ├── data/                  # 테스트 데이터 파일
│   ├── expected/              # 기대 결과 (실제 값만, 빈 디렉토리 금지)
│   └── templates/             # 테스트용 YAML 템플릿
├── unit/                      # 단위 테스트
│   ├── cli/                   # CLI 명령어 파싱
│   ├── components/            # 개별 컴포넌트
│   ├── factory/               # Factory 패턴
│   ├── models/                # ML 모델
│   ├── pipelines/             # 파이프라인 로직
│   ├── serving/               # API 서빙
│   └── settings/              # 설정 시스템
├── integration/               # 통합 테스트
│   ├── test_mlflow_integration.py
│   ├── test_component_interactions.py
│   └── test_pipeline_orchestration.py
└── e2e/                       # End-to-End 테스트
```

---

## 3. 테스트 계층별 가이드

### Unit Tests (`tests/unit/`)

**목적**: 개별 컴포넌트의 인터페이스 계약 검증

**원칙**:
- 단일 클래스/함수 테스트
- 외부 의존성 격리
- 빠른 실행 (< 100ms per test)
- 결정론적 결과

**Mock 정책**:
```python
# ✅ 허용: 외부 의존성
with patch('mlflow.start_run'):
    ...

# ❌ 금지: 내부 컴포넌트
# Bad: Mock(spec=DataAdapter)
# Good: 실제 DataAdapter 인스턴스 사용
```

**특수 케이스 - CLI 단위 테스트**:
```python
# CLI는 인터페이스만 테스트
def test_train_command_parsing():
    # Pipeline 실행은 Mock 허용 (CLI 범위 밖)
    with patch('src.pipelines.run_train_pipeline'):
        result = runner.invoke(app, ['--recipe', 'r.yaml'])
        # CLI 인자 파싱만 검증
```

### Integration Tests (`tests/integration/`)

**목적**: 컴포넌트 간 상호작용 검증

**원칙**:
- 2개 이상 컴포넌트 통합
- Context 클래스로 환경 구성
- 실제 데이터 플로우 검증
- Mock 최소화

**필수 사용**:
```python
# Context 클래스로 환경 구성
with mlflow_test_context.for_classification() as ctx:
    # 실제 컴포넌트 상호작용
    result = factory.create_adapter().read(data_path)
```

### E2E Tests (`tests/e2e/`)

**목적**: 전체 워크플로우 검증

**원칙**:
- 실제 사용 시나리오 재현
- CLI → Pipeline → Output 전체 플로우
- Mock 완전 금지
- 성능 기준 검증 (< 10분)

---

## 4. Context 클래스 시스템

### 역할
Context 클래스는 테스트 환경 설정과 결과 관찰을 캡슐화:

```python
# MLflowTestContext: MLflow 실험 환경
with mlflow_test_context.for_classification(experiment="test") as ctx:
    assert ctx.experiment_exists()
    assert ctx.get_experiment_run_count() > 0

# ComponentTestContext: Factory 스택 환경
with component_test_context.classification_stack() as ctx:
    raw_df = ctx.adapter.read(ctx.data_path)
    assert ctx.validate_data_flow(raw_df, processed_df)

# DatabaseTestContext: 임시 DB 환경
with database_test_context.sqlite_db({"users": df}) as db:
    assert db.connection_uri.startswith("sqlite:///")
```

### 규칙
- 설정 코드 캡슐화
- 퍼블릭 API만 사용
- 비즈니스 로직 재구현 금지
- 헬퍼 메서드로 관찰 지원

---

## 5. 표준 정책

### 명명 규칙
```python
# MLflow 실험명: UUID 접미사 사용
experiment_name = f"test_exp_{uuid4().hex[:8]}"

# 파일 경로: 임시 디렉토리 사용
file_path = isolated_temp_directory / "test.csv"

# 시간 기반 명명 금지
# ❌ Bad: f"run_{int(time.time())}"
# ✅ Good: f"run_{uuid4().hex[:8]}"
```

### MLflow 설정
```python
# 파일 스토어만 사용 (원격 금지)
tracking_uri = f"file://{temp_dir}/mlruns"

# 실험 격리
experiment_name = f"test_{test_name}_{uuid4().hex[:8]}"
```

### 데이터 격리
```python
# 데이터는 temp 디렉토리에
data_path = isolated_temp_directory / "data.csv"

# loader.source_uri는 런타임 주입
settings.recipe.data.loader.source_uri = str(data_path)
```

### 결정론적 실행
```python
# conftest.py의 autouse fixture가 자동 적용
# - random.seed(42)
# - np.random.seed(42)
# - torch.manual_seed(42)
# - PYTHONHASHSEED='42'
```

---

## 6. Fixture 가이드

### 기본 Fixtures (`conftest.py`)

```python
@pytest.fixture
def settings_builder():
    """Settings 객체 빌더"""
    return SettingsBuilder()

@pytest.fixture
def isolated_temp_directory():
    """격리된 임시 디렉토리"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_data_generator():
    """테스트 데이터 생성기"""
    return TestDataGenerator()

@pytest.fixture
def add_model_computed():
    """Model.computed 필드 추가 헬퍼"""
    # SettingsFactory의 동적 필드 추가 시뮬레이션
```

### Context Fixtures

```python
@pytest.fixture
def mlflow_test_context(isolated_temp_directory):
    """MLflow 테스트 컨텍스트"""
    return MLflowTestContext(isolated_temp_directory)

@pytest.fixture
def component_test_context(isolated_temp_directory, settings_builder):
    """컴포넌트 테스트 컨텍스트"""
    return ComponentTestContext(isolated_temp_directory, settings_builder)
```

---

## 7. 안티 패턴

### 피해야 할 패턴

**❌ Context 내부에서 비즈니스 로직 재구현**
```python
# Bad: Context가 직접 계산
def validate_metrics(self):
    return self.calculate_accuracy()  # 비즈니스 로직

# Good: 실제 컴포넌트 사용
def validate_metrics(self):
    return self.evaluator.evaluate()  # 퍼블릭 API
```

**❌ 전역 상태 공유**
```python
# Bad: 전역 변수
GLOBAL_SETTINGS = None

# Good: 각 테스트마다 새로 생성
settings = SettingsBuilder().build()
```

**❌ 시간 기반 명명**
```python
# Bad: 타임스탬프
run_name = f"run_{datetime.now()}"

# Good: UUID
run_name = f"run_{uuid4().hex[:8]}"
```

**❌ 외부 경로 사용**
```python
# Bad: 고정 경로
data_path = "/tmp/test_data.csv"

# Good: 임시 디렉토리
data_path = isolated_temp_directory / "test_data.csv"
```

---

## 8. 테스트 작성 체크리스트

### 새 테스트 작성 시

- [ ] **적절한 계층 선택**: unit/integration/e2e 중 선택
- [ ] **Context 사용**: 환경 설정은 Context 클래스로
- [ ] **실제 객체 우선**: Mock 대신 실제 컴포넌트 사용
- [ ] **격리 보장**: 임시 디렉토리, UUID 명명
- [ ] **결정론적**: 시드 고정, 랜덤 요소 제거
- [ ] **퍼블릭 API**: 내부 구현 의존 금지
- [ ] **빠른 실행**: Unit < 100ms, Integration < 1s
- [ ] **명확한 검증**: Given-When-Then 구조

### 코드 리뷰 체크리스트

- [ ] Mock 사용이 정당한가? (외부 서비스만?)
- [ ] Context 클래스를 적절히 활용했는가?
- [ ] 테스트가 독립적으로 실행 가능한가?
- [ ] 임시 자원이 정리되는가?
- [ ] 실행 시간이 적절한가?
- [ ] 에러 메시지가 명확한가?

---

## 9. CI/CD 통합

### 테스트 실행 전략

```bash
# 단위 테스트 (빠른 피드백)
pytest tests/unit -n auto --maxfail=3

# 통합 테스트 (병렬 실행)
pytest tests/integration -n 4

# E2E 테스트 (순차 실행)
pytest tests/e2e

# 전체 테스트 with 커버리지
pytest tests/ --cov=src --cov-report=html --cov-fail-under=90
```

### 성능 기준

- Unit tests: < 2분
- Integration tests: < 5분
- E2E tests: < 10분
- Total: < 15분

---

## 10. 특수 케이스 가이드

### Model.computed 필드 처리

Settings의 Model 객체에 동적으로 추가되는 computed 필드 처리:

```python
# Unit Test: add_model_computed fixture 사용
def test_something(settings_builder, add_model_computed):
    settings = settings_builder.build()
    settings = add_model_computed(settings)  # computed 추가

# Integration Test: SettingsFactory 직접 사용
settings = SettingsFactory.for_training(...)  # 자동 추가
```

### Async 테스트

```python
# pytest-asyncio 사용
@pytest.mark.asyncio
async def test_async_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get("/health")
```

### 병렬 실행 안전성

```python
# 각 테스트는 고유 식별자 사용
experiment_name = f"test_{test_name}_{uuid4().hex[:8]}"

# 파일 충돌 방지
temp_dir = isolated_temp_directory / f"test_{uuid4().hex[:8]}"
```

---

## 부록: 테스트 패턴 예시

### Unit Test 패턴
```python
class TestDataAdapter:
    def test_read_csv(self, isolated_temp_directory):
        # Given: 테스트 데이터
        data_path = isolated_temp_directory / "test.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(data_path)

        # When: 실제 adapter 사용
        adapter = StorageAdapter(settings)
        result = adapter.read(str(data_path))

        # Then: 결과 검증
        assert len(result) == 2
```

### Integration Test 패턴
```python
def test_pipeline_flow(component_test_context):
    with component_test_context.classification_stack() as ctx:
        # Given: 컴포넌트 스택
        raw_data = ctx.adapter.read(ctx.data_path)

        # When: 컴포넌트 상호작용
        processed = ctx.preprocessor.transform(raw_data)
        model = ctx.trainer.train(processed)

        # Then: 데이터 플로우 검증
        assert ctx.validate_data_flow(raw_data, processed)
```

### E2E Test 패턴
```python
def test_train_to_serve_workflow(mlflow_test_context):
    with mlflow_test_context.for_classification() as ctx:
        # Given: 전체 환경
        runner = CliRunner()

        # When: 전체 워크플로우
        train_result = runner.invoke(app, ['train', ...])
        serve_result = runner.invoke(app, ['serve', ...])

        # Then: 종단 검증
        assert ctx.experiment_exists()
        assert ctx.get_model_version() > 0
```

---

이 가이드는 테스트 품질과 유지보수성을 보장하면서도 실용적인 접근을 유지하는 것을 목표로 합니다.
각 테스트는 명확한 목적을 가지고, 적절한 계층에서, 올바른 도구를 사용하여 작성되어야 합니다.