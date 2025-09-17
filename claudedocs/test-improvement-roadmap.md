# 테스트 개선 로드맵: 0 에러 및 90%+ 커버리지 달성

## 현황 분석 요약

### 테스트 철학 (tests/README.md)
- **핵심 원칙**: Real Object Testing, Public API Focus, Deterministic Execution, Test Isolation
- **No Mock Hell**: Mock 사용 최소화, 실제 객체 우선
- **계층 구조**: Unit (< 100ms) → Integration (< 1s) → E2E (< 10분)
- **Context 시스템**: 테스트 환경 캡슐화 및 격리

### 기존 인프라 (conftest.py)
- **SettingsBuilder**: 유연한 테스트 설정 생성 패턴
- **Context Classes**: MLflowTestContext, ComponentTestContext, DatabaseTestContext
- **Real Component Fixtures**: real_dataset_files, factory_with_real_storage_adapter
- **Performance Tracking**: real_component_performance_tracker
- **Isolation Fixtures**: isolated_temp_directory, isolated_working_directory

### 현재 상태 (2025-09-17 기준)
| 지표 | 현재 | 목표 | 격차 | 근본 원인 |
|------|------|------|------|-----------|
| **CLI/Pipeline 커버리지** | 43% | 90% | -47% | 일부 CLI 명령어, API Serving 미완성 |
| **테스트 실패** | 4+ | 0 | -4 | Pipeline integration 이슈 |
| **완료된 영역** | CLI 핵심 | 전체 | - | train/init 명령어 90%+ 달성 |

### 완료된 작업 (Phase 0-2 부분)
- ✅ **train_command.py**: 91% 커버리지 달성
- ✅ **init_command.py**: 90% 커버리지 달성
- ✅ **main_commands.py**: 89% 커버리지 달성
- ✅ **StorageAdapter**: Path 호환성 수정 완료
- ✅ **Inference Pipeline**: format_predictions multiclass 처리 개선

---

## Phase 1: 긴급 수정 (미완료 항목)
**목표**: 실패 테스트 수정 및 기본 안정성 확보

### 1.1 Scaler Registration 수정 (미완료)
```python
# 실제 컴포넌트 테스트 필요
def test_scaler_registration(component_test_context):
    with component_test_context.classification_stack() as ctx:
        # Import to trigger registration
        import src.components.preprocessor.modules.scaler

        # Registry에서 실제 생성 테스트
        scaler = PreprocessorStepRegistry.create('standard_scaler')
        assert isinstance(scaler, StandardScalerWrapper)
```

### 1.2 Factory Calibration 수정 (미완료)
```python
# Real Factory 테스트 필요
def test_calibrator_creation(settings_builder, add_model_computed):
    settings = settings_builder.with_calibration(True).build()
    settings = add_model_computed(settings)  # computed 필드 추가
    factory = Factory(settings)
    calibrator = factory.create_calibrator()
    assert calibrator is not None
```

**남은 작업**:
- Scaler Registration 에러 수정
- Factory Calibration 로직 수정
- 테스트 실패 0건 달성

---

## Phase 2: 핵심 경로 커버리지 (미완료 항목)
**목표**: Context 패턴 활용한 실제 코드 커버리지 80%+

### 2.1 CLI 커맨드 테스트 리팩토링 (부분 완료)

#### 남은 작업
```python
# tests/unit/cli/commands/test_serve_command.py (현재 58%)
def test_serve_command_with_real_components(
    cli_test_environment,
    mlflow_test_context
):
    """MLflow run 없이도 동작하는 실제 컴포넌트 테스트"""
    with mlflow_test_context.for_classification("cli_serve") as ctx:
        # Only mock run_api_server for unit test
        with patch('src.cli.commands.serve_command.run_api_server') as mock_server:
            result = runner.invoke(app, [
                'serve',
                '--run-id', 'dummy_run_id',
                '--config-path', str(cli_test_environment['config_path'])
            ])
            # SettingsFactory 호출 및 파라미터 검증에 집중
```

```python
# tests/unit/cli/commands/test_inference_command.py (현재 56%)
def test_inference_command_with_real_components(
    cli_test_environment,
    mlflow_test_context
):
    """실제 컴포넌트로 inference 커맨드 테스트"""
    with mlflow_test_context.for_classification("cli_inference") as ctx:
        with patch('src.cli.commands.inference_command.run_inference_pipeline') as mock_pipeline:
            result = runner.invoke(app, [
                'inference',
                '--run-id', 'dummy_run_id',
                '--config-path', str(cli_test_environment['config_path']),
                '--data-path', str(cli_test_environment['data_path'])
            ])
            # SettingsFactory 및 파라미터 파싱 검증
```

#### Integration Tests - 미생성
```python
# tests/integration/test_cli_pipeline_integration.py (새로 생성 필요)
def test_train_end_to_end_with_contexts(
    mlflow_test_context,
    component_test_context
):
    """Context 활용한 전체 플로우 테스트"""
    with mlflow_test_context.for_classification("e2e_train") as mlflow_ctx:
        with component_test_context.classification_stack() as comp_ctx:
            # Real CLI → Real Pipeline → Real MLflow
            runner = CliRunner()
            result = runner.invoke(app, [
                'train',
                '--recipe-path', str(comp_ctx.settings.recipe),
                '--config-path', str(comp_ctx.settings.config)
            ])

            assert result.exit_code == 0
            assert mlflow_ctx.get_experiment_run_count() > 0
            assert mlflow_ctx.verify_mlflow_artifacts()
```

### 2.2 Pipeline 테스트 - Real Components (부분 완료)

#### 남은 작업
```python
# Performance Tracking 활용 (미활용)
def test_train_pipeline_with_performance_tracking(
    mlflow_test_context,
    real_component_performance_tracker
):
    """성능 추적을 포함한 실제 파이프라인 테스트"""
    with mlflow_test_context.for_classification("pipeline_train") as ctx:
        with real_component_performance_tracker.measure_time("complete_workflow"):
            # 실제 파이프라인 실행
            from src.pipelines.train_pipeline import run_train_pipeline
            result = run_train_pipeline(ctx.settings)

        # 검증
        assert ctx.get_experiment_run_count() > 0
        metrics = ctx.get_run_metrics()
        assert 'accuracy' in metrics

        # 성능 검증
        real_component_performance_tracker.assert_time_under("complete_workflow", 2.0)
```

### 2.3 API Serving 테스트 - TestClient 활용 (완전 미완료)

#### ServingTestContext 생성 필요
```python
# tests/fixtures/contexts/serving_context.py (새로 생성 필요)
class ServingTestContext:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir

    @contextmanager
    def with_trained_model(self):
        """훈련된 모델과 함께 서빙 환경 구성"""
        # MLflow에 모델 저장
        # FastAPI 앱 구성
        # TestClient 준비
        yield ServingContextManager(...)
```

#### FastAPI TestClient 테스트
```python
# tests/unit/serving/test_endpoints.py (새로 생성 필요)
from fastapi.testclient import TestClient

def test_predict_endpoint_with_real_model(
    serving_test_context
):
    """실제 모델과 TestClient 사용"""
    with serving_test_context.with_trained_model() as ctx:
        # FastAPI TestClient - 실제 서버 시뮬레이션
        from src.serving.router import create_app
        app = create_app(ctx.settings)
        client = TestClient(app)

        # 실제 예측 테스트
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0, 4.0]}
        )
        assert response.status_code == 200
        assert "prediction" in response.json()
```

#### Integration Tests with Real Server
```python
# tests/integration/test_serving_integration.py (새로 생성 필요)
def test_serving_lifecycle_with_contexts(
    mlflow_test_context,
    serving_test_context
):
    """서버 생명주기 전체 테스트"""
    with mlflow_test_context.for_classification("serve") as mlflow_ctx:
        with serving_test_context.with_trained_model() as serve_ctx:
            # Start server
            client = TestClient(serve_ctx.app)

            # Health check
            assert client.get("/health").status_code == 200

            # Model info
            info = client.get("/model/info").json()
            assert info["model_name"] == serve_ctx.model_name

            # Batch prediction
            batch_response = client.post(
                "/predict/batch",
                json={"instances": [[1,2,3,4], [5,6,7,8]]}
            )
            assert len(batch_response.json()["predictions"]) == 2
```

### 2.4 Core Components 강화 (부분 완료)

#### 남은 작업
```python
# SQL, Feast 어댑터 테스트 강화 필요
- SQL 어댑터: 트랜잭션 처리, 에러 복구
- Feast 어댑터: 온라인/오프라인 서빙

# 모든 전처리 모듈 테스트 필요
- Scaler, Encoder, Transformer
- 데이터 타입 처리
- Null 값 처리
```

**목표 달성 상황**:
- CLI 핵심 명령어: ✅ 90%+ 달성 (train, init)
- CLI 나머지 명령어: ❌ 56-58% (serve, inference)
- API Serving: ❌ 0% (완전 미착수)
- Core Components: ⚠️ 부분 달성 (Storage만)

---

## Phase 3: 모델 & 통합 커버리지 (Week 4)
**목표**: 고급 기능 및 통합 시나리오 커버리지

### 3.1 Custom Models (0% → 80%)

```python
# tests/unit/models/custom/
test_lstm_timeseries.py     # LSTM 시계열 모델
test_pytorch_utils.py        # PyTorch 유틸리티
test_ft_transformer.py       # FT-Transformer
test_timeseries_wrappers.py  # 래퍼 클래스
```

### 3.2 Feature Store Integration

```python
# tests/integration/test_feature_store.py
- Feast 어댑터 통합 (선택적)
- 피처 파이프라인
- 온라인/오프라인 서빙
```

### 3.3 Database Integration 강화

```python
# tests/integration/test_database_integration.py
- 연결 풀링 최적화
- 동시성 처리
- 트랜잭션 격리
```

### 3.4 E2E 최적화

```python
# tests/e2e/
- 메모리 사용량 프로파일링
- 병렬 실행 가능 영역 식별
- 테스트 데이터 크기 최적화
```

**예상 결과**:
- 커버리지: 65% → 85%
- 고급 기능 검증 완료

---

## Phase 4: 성능 최적화 & 완성 (Week 5)
**목표**: 90%+ 커버리지 달성 및 성능 최적화

### 4.1 테스트 성능 최적화

#### M1 MacBook 최적화
```ini
# pytest.ini 조정
addopts =
    -n 3  # 병렬 워커 3개로 제한
    --dist loadscope  # 스코프별 분산
    --maxprocesses 3  # 프로세스 제한
```

#### 테스트 격리 개선
```python
# conftest.py
@pytest.fixture(scope="module")
def shared_test_data():
    """모듈 레벨 공유 데이터"""

@pytest.fixture
def isolated_mlflow():
    """완전 격리된 MLflow 환경"""
```

### 4.2 Edge Case 커버리지

```python
# 경계값 테스트
- 빈 데이터셋 처리
- 대용량 데이터 처리
- 비정상 입력 처리
- 동시성 엣지 케이스
```

### 4.3 문서화 및 유지보수

```python
# 테스트 문서화
- 각 테스트의 목적 명시
- Context 사용 가이드
- 디버깅 가이드
```

### 4.4 CI/CD 파이프라인 최적화

```yaml
# .github/workflows/test.yml
- 단계별 테스트 실행
- 캐싱 전략
- 병렬 Job 구성
- 커버리지 리포트 자동화
```

**최종 결과**:
- ✅ 커버리지: 90%+
- ✅ 테스트 실패: 0
- ✅ 실행 시간: < 15분
- ✅ 메모리 안정성

---

## 구현 우선순위 매트릭스 (현재 상태 반영)

| 작업 | 영향도 | 난이도 | 우선순위 | 현재 상태 | 핵심 방법 |
|------|--------|--------|----------|----------|-----------|
| **CLI serve/inference 완성** | 매우높음 | 낮음 | P0 | 56-58% | MLflow mock 개선 |
| **API Serving 테스트** | 매우높음 | 중간 | P0 | 0% | ServingTestContext + TestClient |
| **Performance Tracking** | 높음 | 낮음 | P1 | 미사용 | real_component_performance_tracker |
| **Integration Tests** | 높음 | 중간 | P1 | 미생성 | CLI-Pipeline 통합 |
| **Scaler Registration** | 중간 | 낮음 | P1 | 미수정 | Real component 사용 |
| Model 테스트 | 중간 | 높음 | P2 | 0% | ComponentTestContext |
| 성능 최적화 | 중간 | 중간 | P3 | 미적용 | pytest.ini 조정 |
| Edge cases | 낮음 | 낮음 | P3 | 0% | Context variations |

---

## 성공 지표 (KPIs) - 현재 상태 업데이트

### 주간 체크포인트
- **Week 1**: ✅ CLI 핵심 명령어 90%+ 달성
- **Week 2**: ⚠️ 커버리지 43% 달성 (목표 40%+ 달성)
- **Week 3**: 🎯 커버리지 65%+ 달성 목표
- **Week 4**: 🎯 커버리지 85%+ 달성 목표
- **Week 5**: 🎯 커버리지 90%+ 및 성능 목표 달성

### 품질 지표
- 테스트 실행 시간: < 15분
- 메모리 사용량: < 8GB
- 테스트 안정성: Flaky test 0%
- 코드 리뷰 통과율: 100%

---

## 리스크 및 완화 방안

### 리스크
1. **API Serving 테스트 지연**: ServingTestContext 생성 복잡도
   - 완화: 기존 MLflowTestContext 패턴 활용

2. **MLflow Integration 이슈**: CLI 테스트에서 실제 run_id 필요
   - 완화: Dummy run_id로 설정 검증에 집중

3. **Performance Tracking 미활용**: 성능 검증 부재
   - 완화: 단계적 적용, 핵심 경로부터

4. **테스트 복잡도**: Context 시스템 학습 곡선
   - 완화: 상세 문서화, 예제 코드 제공

---

## 테스트 철학 준수 가이드

### ✅ 올바른 패턴
```python
# 1. Context 사용
def test_with_context(mlflow_test_context):
    with mlflow_test_context.for_classification("test") as ctx:
        # ctx provides everything needed

# 2. Real Components
def test_with_real_adapter(factory_with_real_storage_adapter):
    factory, data = factory_with_real_storage_adapter
    df = factory.create_data_adapter().read(data["path"])

# 3. Performance Tracking
def test_with_performance(real_component_performance_tracker):
    with real_component_performance_tracker.measure_time("operation"):
        # perform operation
    real_component_performance_tracker.assert_time_under("operation", 0.1)
```

### ❌ 안티패턴 (피해야 할 것)
```python
# 1. 내부 컴포넌트 Mock
@patch('src.cli.utils.InteractiveUI')  # ❌ Wrong

# 2. Context 미사용
settings = Settings(...)  # ❌ Wrong
settings = settings_builder.build()  # ✅ Right

# 3. 시간 기반 명명
f"test_{datetime.now()}"  # ❌ Wrong
f"test_{uuid4().hex[:8]}"  # ✅ Right
```

## 실행 체크리스트 (현재 상태 반영)

### 즉시 우선순위 (P0)
- [ ] **CLI serve/inference 명령어**: 58% → 80% 달성
- [ ] **ServingTestContext 생성**: API 서빙 테스트 기반 구축
- [ ] **FastAPI TestClient 테스트**: 실제 서빙 엔드포인트 검증

### 단기 우선순위 (P1)
- [ ] **Performance Tracking 적용**: real_component_performance_tracker 활용
- [ ] **Integration Tests 생성**: CLI-Pipeline 통합 테스트
- [ ] **Scaler Registration 수정**: 실패 테스트 해결

### 완료 기준 (업데이트)
- [ ] **CLI 커버리지 80%+**: serve, inference 명령어 포함
- [ ] **API Serving 커버리지 85%+**: TestClient 기반 실제 테스트
- [ ] **Performance 검증**: 모든 핵심 경로 성능 추적
- [ ] **테스트 실패 0건**: 모든 테스트 통과
- [ ] **전체 커버리지 90%+**: 실제 코드 실행 커버리지

---

## 참고 자료
- [Test Philosophy](../tests/README.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://testdriven.io/blog/testing-best-practices/)

---

## 진행 상황 요약

### 완료된 성과 (2025-09-17)
1. **CLI 핵심 명령어 초과 달성**
   - train_command.py: 91% 커버리지
   - init_command.py: 90% 커버리지
   - main_commands.py: 89% 커버리지

2. **중요한 버그 수정**
   - StorageAdapter Path 호환성 해결
   - Inference pipeline format_predictions 개선

3. **테스트 안정성 개선**
   - MLflow mocking 구조 개선
   - 전체 커버리지 22% → 43% 달성

### 남은 핵심 작업
1. **API Serving 테스트 (최우선)**
   - ServingTestContext 생성
   - FastAPI TestClient 활용
   - 서빙 엔드포인트 검증

2. **CLI 나머지 명령어 완성**
   - serve_command: 58% → 80%
   - inference_command: 56% → 80%

3. **Performance & Integration**
   - real_component_performance_tracker 활용
   - CLI-Pipeline 통합 테스트 생성

**현재 위치**: Phase 2 중반 (43% 커버리지)
**다음 목표**: Phase 2 완료 (65% 커버리지)

---

*초판 작성일: 2025-09-17*
*현재 상태 반영 개정일: 2025-09-17*
*작성자: ML Pipeline Test Team*
*버전: 3.0.0 (Phase 1-2 실행 결과 반영판)*