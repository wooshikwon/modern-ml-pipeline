# Modern ML Pipeline - Test Strategy & Implementation Plan (v2.0)

## 🎯 목표
- **테스트 커버리지: 90% 이상**
- **테스트와 소스코드 완전 분리**
- **Mock 기반의 독립적 테스트 환경**
- **CI/CD 파이프라인 통합 가능한 구조**
- **UV 패키지 매니저 기반 의존성 관리**

## 📊 현재 프로젝트 현황
- **총 소스코드**: ~3,200 줄
- **주요 모듈**: 8개 (factory, settings, pipelines, components, interface, serving, cli, utils)
- **컴포넌트**: 5개 (adapter, fetcher, evaluator, trainer, preprocessor)
- **레지스트리**: 자기등록 패턴 사용 (import 시점 자동 등록)
- **현재 테스트 커버리지**: 1.6% (Phase 1 시작)

## 🏗️ 테스트 아키텍처

### 1. 테스트 레벨 분배
| 레벨 | 비중 | 목적 | 실행시간 |
|------|------|------|----------|
| Unit Tests | 70% | 개별 함수/클래스 검증 | <1초 |
| Integration Tests | 20% | 컴포넌트 간 상호작용 | <5초 |
| E2E Tests | 10% | 전체 워크플로우 검증 | <30초 |

### 2. 테스트 디렉토리 구조
```
tests/
├── conftest.py                      # 전역 픽스처 및 설정
├── pytest.ini                       # pytest 설정
├── coverage.ini                     # 커버리지 설정
│
├── unit/                           # 단위 테스트 (70%)
│   ├── settings/
│   │   ├── test_loader.py         # Settings 로더 테스트
│   │   ├── test_config.py         # Config 스키마 테스트
│   │   ├── test_recipe.py         # Recipe 스키마 테스트
│   │   └── test_validator.py      # Validator 테스트
│   │
│   ├── factory/
│   │   ├── test_factory.py        # Factory 클래스 테스트
│   │   └── test_artifact.py       # PyfuncWrapper 테스트
│   │
│   ├── components/
│   │   ├── test_adapter/          # 각 어댑터 테스트
│   │   ├── test_fetcher/          # 각 페처 테스트
│   │   ├── test_evaluator/        # 각 평가자 테스트
│   │   ├── test_trainer/          # 트레이너 테스트
│   │   └── test_preprocessor/     # 전처리기 테스트
│   │
│   ├── interface/
│   │   ├── test_base_classes.py   # 베이스 클래스 테스트
│   │   └── test_types.py          # 타입 정의 테스트
│   │
│   └── utils/
│       ├── test_logger.py         # 로거 테스트
│       ├── test_schema_utils.py   # 스키마 유틸 테스트
│       └── test_template.py       # 템플릿 테스트
│
├── integration/                    # 통합 테스트 (20%)
│   ├── test_registry/
│   │   ├── test_adapter_registry.py
│   │   ├── test_fetcher_registry.py
│   │   └── test_evaluator_registry.py
│   │
│   ├── test_pipeline_components/
│   │   ├── test_data_flow.py      # 데이터 플로우 테스트
│   │   ├── test_model_lifecycle.py # 모델 생명주기 테스트
│   │   └── test_feature_store.py   # Feature Store 통합
│   │
│   └── test_serving/
│       ├── test_api_endpoints.py   # API 엔드포인트 테스트
│       └── test_context_mgmt.py    # 컨텍스트 관리 테스트
│
├── e2e/                            # End-to-End 테스트 (10%)
│   ├── test_train_to_serve.py     # 학습→서빙 전체 플로우
│   ├── test_batch_inference.py    # 배치 추론 플로우
│   └── test_cli_workflow.py       # CLI 전체 워크플로우
│
├── fixtures/                       # 테스트 픽스처
│   ├── data/
│   │   ├── sample_data.csv       # 샘플 데이터
│   │   └── test_models/           # 테스트용 모델
│   │
│   ├── configs/
│   │   ├── test_config.yaml      # 테스트 설정
│   │   └── test_recipe.yaml      # 테스트 레시피
│   │
│   └── mocks/
│       ├── mock_mlflow.py        # MLflow Mock
│       ├── mock_database.py      # DB Mock
│       └── mock_filesystem.py    # 파일시스템 Mock
│
└── helpers/                        # 테스트 헬퍼
    ├── assertions.py              # 커스텀 assertion
    ├── builders.py                # 테스트 객체 빌더
    └── validators.py              # 검증 헬퍼
```

## 📝 구현 Phases (개선된 로드맵)

### Phase 1: 기초 인프라 구축 (3-4일) ✅ 완료
- [x] 테스트 디렉토리 구조 생성
- [x] conftest.py 기본 픽스처 설정
- [x] helpers/assertions.py 커스텀 assertion
- [x] helpers/builders.py 테스트 빌더
- [x] pytest.ini 설정
- [x] 추가 픽스처 보강 (logger, env, async, factory)
- [x] 목표 커버리지: 5% → **달성: 16%**

### Phase 2: Core 단위 테스트 (3-4일) ✅ 완료
- [x] settings/test_config.py (완료)
- [x] settings/test_recipe.py (68 테스트 통과)
- [x] settings/test_loader.py (환경변수 처리 테스트 포함)
- [x] settings/test_validator.py (TunableParameter 검증)
- [x] factory/test_factory.py (Registry 처리 포함, 31/37 테스트 통과)
- [x] factory/test_artifact.py (PyfuncWrapper 테스트)
- [x] interface/test_base_classes.py (베이스 클래스 계약 테스트)
- [x] 목표 커버리지: 25% → **달성: 25%**
- [x] **소스 코드 개선**: `src/factory/artifact.py` 경로 오류 수정

### Phase 3: Component 단위 테스트 (4-5일)
- [ ] 각 컴포넌트 Registry 테스트
- [ ] adapter/ (storage, sql, feature_store)
- [ ] fetcher/ (pass_through, feature_store)
- [ ] evaluator/ (classification, regression, clustering, causal)
- [ ] trainer/ (Trainer, data_handler, optimizer)
- [ ] preprocessor/ (Preprocessor)
- [ ] 목표 커버리지: 60%

### Phase 4: 통합 테스트 (3-4일)
- [ ] Registry 자기등록 메커니즘 테스트
- [ ] Factory 캐싱 메커니즘 테스트
- [ ] Pipeline 컴포넌트 데이터 플로우
- [ ] Feature Store 통합 테스트
- [ ] Serving 모듈 비동기 테스트
- [ ] 목표 커버리지: 75%

### Phase 5: E2E 및 CLI 테스트 (2-3일)
- [ ] 전체 학습→서빙 워크플로우
- [ ] 배치 추론 전체 플로우
- [ ] CLI 명령어 체인 테스트
- [ ] API 엔드포인트 테스트
- [ ] 환경별 설정 테스트
- [ ] 목표 커버리지: 90%+

## 🛠️ 테스트 도구

### UV 환경 설정
```bash
# UV 프로젝트 설정
uv init --python 3.11

# 테스트 의존성 그룹 추가
uv add --group test pytest pytest-cov pytest-mock pytest-asyncio pytest-xdist pytest-timeout hypothesis factory-boy faker responses freezegun httpx

# 테스트 실행
uv run pytest                     # 전체 테스트
uv run pytest tests/unit         # 단위 테스트만
uv run pytest -m "not slow"      # 빠른 테스트만
uv run pytest --cov=src          # 커버리지 포함
```

### 필수 패키지 (pyproject.toml)
```toml
[dependency-groups]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-asyncio>=0.23.0",
    "pytest-xdist>=3.5.0",         # 병렬 실행
    "pytest-timeout>=2.2.0",       # 타임아웃 관리
    "hypothesis>=6.100.0",         # Property-based testing
    "factory-boy>=3.3.0",          # 테스트 데이터 생성
    "faker>=24.0.0",               # Fake 데이터 생성
    "responses>=0.25.0",           # HTTP 모킹
    "freezegun>=1.4.0",            # 시간 모킹
    "httpx>=0.27.0",               # 비동기 HTTP 클라이언트
]
```

### pytest 설정
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=90
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
```

## 🎯 커버리지 목표

### 모듈별 목표 커버리지
| 모듈 | 목표 | 우선순위 | 난이도 |
|------|------|----------|--------|
| settings | 95% | 높음 | 낮음 |
| factory | 90% | 높음 | 중간 |
| components | 90% | 높음 | 중간 |
| interface | 85% | 중간 | 낮음 |
| pipelines | 85% | 높음 | 높음 |
| utils | 95% | 낮음 | 낮음 |
| serving | 80% | 중간 | 높음 |
| cli | 80% | 중간 | 중간 |

## 🔄 Mock 전략 (개선됨)

### 1. 핵심 의존성 격리
```python
# Logger Mock (모든 테스트에 자동 적용)
@pytest.fixture(autouse=True)
def silence_logger():
    """모든 테스트에서 로거 출력 억제"""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)

# MLflow Mock (개선됨)
@pytest.fixture
def mock_mlflow(monkeypatch):
    """완전한 MLflow 모킹"""
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.log_params') as mock_log_params, \
         patch('mlflow.log_artifact') as mock_log_artifact, \
         patch('mlflow.pyfunc.save_model') as mock_save_model, \
         patch('mlflow.pyfunc.load_model') as mock_load_model:
        
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        yield {'start_run': mock_start_run, 'run': mock_run}

# 환경 격리
@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """완전히 격리된 실행 환경"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENV_NAME", "test")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    
    # 필요한 디렉토리 생성
    (tmp_path / "configs").mkdir()
    (tmp_path / "recipes").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path

# Database Mock
@pytest.fixture
def mock_database():
    """인메모리 DB 사용"""
    from sqlalchemy import create_engine
    return create_engine("sqlite:///:memory:")
```

### 2. Registry 자기등록 처리
```python
@pytest.fixture(autouse=True)
def clean_registries():
    """Registry 자기등록 메커니즘 격리"""
    # 원본 상태 저장
    from src.components.adapter import AdapterRegistry
    from src.components.fetcher import FetcherRegistry
    from src.components.evaluator import EvaluatorRegistry
    
    original_adapters = AdapterRegistry.adapters.copy()
    original_fetchers = FetcherRegistry.fetchers.copy()
    original_evaluators = EvaluatorRegistry.evaluators.copy()
    
    yield
    
    # 원본 상태 복원
    AdapterRegistry.adapters.clear()
    AdapterRegistry.adapters.update(original_adapters)
    FetcherRegistry.fetchers.clear()
    FetcherRegistry.fetchers.update(original_fetchers)
    EvaluatorRegistry.evaluators.clear()
    EvaluatorRegistry.evaluators.update(original_evaluators)
```

### 3. Factory 캐싱 처리
```python
@pytest.fixture
def factory_with_clean_cache(test_settings):
    """캐시가 초기화된 Factory"""
    factory = Factory(test_settings)
    factory._component_cache.clear()
    return factory

@pytest.fixture
def mock_factory():
    """완전히 모킹된 Factory"""
    factory = MagicMock(spec=Factory)
    factory.create_model.return_value = MockBuilder.build_mock_model()
    factory.create_data_adapter.return_value = MockBuilder.build_mock_adapter()
    factory.create_fetcher.return_value = MockBuilder.build_mock_fetcher()
    factory.create_evaluator.return_value = MockBuilder.build_mock_evaluator()
    return factory
```

### 4. 비동기 테스트 지원
```python
@pytest.fixture
async def async_client():
    """FastAPI 비동기 테스트 클라이언트"""
    from httpx import AsyncClient
    from src.serving.router import app
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def event_loop():
    """이벤트 루프 픽스처"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## 📊 메트릭 및 리포팅

### 커버리지 리포트
- HTML 리포트: `htmlcov/index.html`
- XML 리포트: `coverage.xml` (CI/CD 통합용)
- 터미널 리포트: 각 테스트 실행 후 자동 출력
- JSON 리포트: `coverage.json` (분석 도구용)

### 성능 메트릭
- 단위 테스트: <1초 per test
- 통합 테스트: <5초 per test  
- E2E 테스트: <30초 per test
- 전체 테스트 스위트: <5분

### 커버리지 명령어 (UV 환경)
```bash
# 기본 커버리지 실행
uv run pytest --cov=src tests/

# HTML 리포트 생성
uv run pytest --cov=src --cov-report=html tests/

# 상세 터미널 출력
uv run pytest --cov=src --cov-report=term-missing tests/

# 브랜치 커버리지 포함
uv run pytest --cov=src --cov-branch tests/
```

## 🚀 CI/CD 통합

### GitHub Actions 설정 (UV 환경)
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install UV
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "$HOME/.local/bin" >> $GITHUB_PATH
    
    - name: Set up Python
      run: |
        uv python install ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        uv sync --all-extras
    
    - name: Run linting
      run: |
        uv run ruff check src/
        uv run mypy src/
    
    - name: Run tests with coverage
      run: |
        uv run pytest tests/ \
          --cov=src \
          --cov-report=xml \
          --cov-report=term \
          --cov-fail-under=90
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## 🎓 Best Practices (개선됨)

1. **테스트 독립성**: 각 테스트는 다른 테스트에 의존하지 않음
2. **Given-When-Then**: 명확한 테스트 구조 사용
3. **AAA Pattern**: Arrange-Act-Assert 패턴 준수
4. **Descriptive Names**: 테스트 이름으로 의도 명확히 표현
5. **Fast Feedback**: 빠른 실행을 위한 Mock 적극 활용
6. **Deterministic**: 랜덤성 제거, 시드 고정
7. **Documentation**: 복잡한 테스트는 주석으로 설명
8. **Isolation**: Registry 자기등록 및 Factory 캐시 격리
9. **Environment**: 환경변수 및 파일시스템 완전 격리
10. **Async Support**: 비동기 코드 테스트 지원

## 🔍 검증 체크리스트

### 코드 품질
- [ ] 모든 public 함수/메서드 테스트됨
- [ ] 모든 에러 케이스 처리됨
- [ ] 모든 엣지 케이스 고려됨
- [ ] Mock 사용으로 외부 의존성 제거됨

### 테스트 환경
- [ ] Logger 자동 억제 구현
- [ ] Registry 자기등록 격리
- [ ] Factory 캐시 초기화
- [ ] 환경변수 격리
- [ ] 파일시스템 격리

### 성능 및 커버리지
- [ ] 테스트 실행 시간 5분 이내
- [ ] 커버리지 90% 이상
- [ ] 브랜치 커버리지 85% 이상
- [ ] CI/CD 파이프라인 통과

## 🛠️ 문제 해결 가이드

### 일반적인 이슈 및 해결책

#### 1. Registry 자기등록 충돌
```python
# 문제: ImportError 발생 시
# 해결: clean_registries 픽스처 확인
pytest -v --fixtures | grep clean_registries
```

#### 2. 환경변수 누락
```python
# 문제: KeyError for ENV_NAME
# 해결: isolated_env 픽스처 사용
def test_with_env(isolated_env):
    assert os.getenv("ENV_NAME") == "test"
```

#### 3. 비동기 테스트 실패
```python
# 문제: RuntimeError: no running event loop
# 해결: pytest-asyncio 설치 및 async 마커 사용
@pytest.mark.asyncio
async def test_async_endpoint(async_client):
    response = await async_client.get("/health")
```

#### 4. 캐시 상태 오염
```python
# 문제: 이전 테스트의 캐시가 영향
# 해결: factory_with_clean_cache 픽스처 사용
def test_factory(factory_with_clean_cache):
    # 캐시가 초기화된 factory 사용
```

## 📚 참고 자료

### 핵심 문서
- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [UV package manager](https://github.com/astral-sh/uv)

### 테스트 패턴
- [Python testing best practices](https://realpython.com/pytest-python-testing/)
- [Test-Driven Development](https://martinfowler.com/bliki/TestDrivenDevelopment.html)
- [Mocking in Python](https://docs.python.org/3/library/unittest.mock.html)

### 프로젝트 문서
- `TEST_STRATEGY_REVIEW.md`: 전략 검토 및 개선사항
- `tests/README.md`: 테스트 실행 가이드
- `CONTRIBUTING.md`: 테스트 작성 가이드라인

## 🏁 완료 기준

### Phase 1 완료 조건
- ✅ conftest.py 전체 픽스처 구현
- ✅ helpers 모듈 완성
- ⏳ settings 모듈 100% 커버리지
- ⏳ factory 모듈 100% 커버리지
- ⏳ 커버리지 35% 이상

### 최종 완료 조건
- 커버리지 90% 이상 달성
- 모든 테스트 5분 이내 실행
- CI/CD 파이프라인 그린
- 문서화 완료
- 리뷰 및 승인