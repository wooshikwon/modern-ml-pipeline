# 테스트 코드 수정 종합 계획 (Test Fix Comprehensive Plan)

## 📊 현재 테스트 실행 결과 요약

### 전체 통계
- **총 테스트**: 1,478개
- **성공**: 1,105개 (74.8%)
- **실패**: 332개 (22.5%)
- **에러**: 16개 (1.1%)
- **스킵**: 25개 (1.7%)
- **수집 실패**: 2개 (missing dependencies)

### 계층별 실패 현황
- **Unit Tests**: 약 320개 실패 (주로 CLI, MLflow, DataHandler)
- **Integration Tests**: 9개 실패 (주로 MLflow workflows)
- **E2E Tests**: 1개 실패 (concurrent runs)
- **Collection Errors**: 2개 (missing dependencies)

## 🔍 주요 문제점 분석

### 1. 의존성 문제 (Dependency Issues)
**영향 범위**: 2개 테스트 파일이 실행조차 되지 않음

#### 문제점:
- `psutil` 모듈 없음 → `tests/e2e/test_cli-workflow.py` 수집 실패
- `category_encoders` 모듈 없음 → `tests/unit/components/test_preprocessor/test_encoder.py` 수집 실패

#### 근본 원인:
- `pyproject.toml`의 dependencies에 category_encoders는 있지만 psutil이 없음
- 테스트 전용 의존성이 별도로 관리되지 않음

### 2. MLflow Integration 문제
**영향 범위**: 12+ 단위 테스트 실패

#### 문제점:
```python
# 테스트 기대값
Expected: start_run(run_name='test_run')
# 실제 동작
Actual: start_run(run_name='test_run_20250908_094344_e915807c')
```

#### 근본 원인:
- 소스 코드에서 run_name에 timestamp와 UUID suffix를 자동 추가하도록 변경됨
- 테스트가 이 변경사항을 반영하지 못함

### 3. CLI Command Tests 문제
**영향 범위**: 약 40개 CLI 관련 테스트 실패

#### 문제점:
- `InteractiveUI` 클래스 인터페이스 변경
- `RecipeBuilder.build_recipe()` 메서드 없음
- `ConfigBuilder` 초기화 방식 변경

#### 근본 원인:
- CLI 명령어 구조가 리팩토링되었으나 테스트가 구버전 기준으로 작성됨
- Mock 객체가 새로운 인터페이스를 따르지 않음

### 4. Timeseries DataHandler 문제
**영향 범위**: 16개 timeseries 관련 테스트 에러

#### 문제점:
```python
AttributeError: type object 'RecipeBuilder' has no attribute 'build_recipe'
```

#### 근본 원인:
- `RecipeBuilder` 클래스 API가 변경됨
- 테스트에서 사용하는 mock/fixture가 구버전 API 사용

### 5. 동시성 문제 (Concurrency Issues)
**영향 범위**: MLflow concurrent runs 테스트

#### 문제점:
```python
KeyError('distutils.debug')
```

#### 근본 원인:
- setuptools가 distutils를 대체하면서 발생하는 import 순서 문제
- 멀티스레드 환경에서 모듈 임포트 충돌

### 6. 테스트 규약 위반
**영향 범위**: 2개 e2e 테스트

#### 문제점:
- 테스트 함수가 dict를 반환 (None을 반환해야 함)

#### 근본 원인:
- 테스트 작성 시 pytest 규약을 따르지 않음

## 🎯 우선순위별 수정 계획

### 🔴 Priority 1: 즉시 수정 (Critical - 테스트 실행 차단)

#### 1.1 의존성 추가
**작업 내용**:
```toml
# pyproject.toml
[project.optional-dependencies]
test = [
    "pytest>=8.4.1",
    "pytest-cov>=6.2.1",
    "pytest-xdist>=3.8.0",
    "psutil>=5.9.0",  # 추가
]
```

**수정 파일**:
- `pyproject.toml`

**예상 소요 시간**: 5분

#### 1.2 RecipeBuilder API 수정
**작업 내용**:
```python
# tests/helpers/builders.py 또는 해당 위치
class RecipeBuilder:
    @classmethod
    def build_recipe(cls, ...):  # 메서드 추가 또는 수정
        # 또는 테스트에서 새 API 사용하도록 변경
        pass
```

**수정 파일**:
- `tests/unit/components/test_datahandler/test_timeseries_handler.py`
- 관련 fixture 파일들

**예상 소요 시간**: 30분

### 🟡 Priority 2: 기능 테스트 복구 (High - 주요 기능 검증)

#### 2.1 MLflow Integration 테스트 수정
**작업 내용**:
```python
# 변경 전
mock_mlflow.start_run.assert_called_with(run_name='test_run')

# 변경 후
mock_mlflow.start_run.assert_called_once()
args, kwargs = mock_mlflow.start_run.call_args
assert kwargs['run_name'].startswith('test_run')
```

**수정 파일**:
- `tests/unit/utils/test_mlflow_integration.py` (12개 테스트)

**예상 소요 시간**: 1시간

#### 2.2 CLI Command 테스트 Mock 업데이트
**작업 내용**:
- `InteractiveUI` mock 객체 새 인터페이스에 맞게 수정
- `ConfigBuilder`, `RecipeBuilder` 초기화 방식 업데이트

**수정 파일**:
- `tests/unit/cli/test_commands/test_init_command.py`
- `tests/unit/cli/test_commands/test_get_config_command.py`
- `tests/unit/cli/test_commands/test_get_recipe_command.py`
- `tests/unit/cli/test_commands/test_list_commands.py`

**예상 소요 시간**: 2시간

### 🟢 Priority 3: 안정성 개선 (Medium - 테스트 품질)

#### 3.1 동시성 문제 해결
**작업 내용**:
```python
# conftest.py 또는 테스트 시작 부분에 추가
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

# 또는 pytest.ini에 추가
[pytest]
filterwarnings = 
    ignore::UserWarning:_distutils_hack
```

**수정 파일**:
- `tests/conftest.py`
- `tests/pytest.ini`

**예상 소요 시간**: 15분

#### 3.2 E2E 테스트 반환값 수정
**작업 내용**:
```python
# 변경 전
def test_complete_regression_pipeline_e2e():
    # ... 테스트 코드
    return result  # 제거

# 변경 후
def test_complete_regression_pipeline_e2e():
    # ... 테스트 코드
    # return 문 제거
```

**수정 파일**:
- `tests/e2e/test_regression-tabular.py`
- `tests/e2e/test_timeseries-basic.py`

**예상 소요 시간**: 10분

### 🔵 Priority 4: 장기 개선 (Low - 유지보수성)

#### 4.1 테스트 구조 개선
- 테스트용 fixture 중앙화
- Mock 객체 factory 패턴 도입
- 테스트 데이터 생성 유틸리티 강화

#### 4.2 테스트 문서화
- 각 테스트 모듈별 README 작성
- 테스트 실행 가이드 작성
- CI/CD 파이프라인 통합

## 📝 구현 체크리스트

### Phase 1: 즉시 수정 (1일)
- [ ] psutil 의존성 추가
- [ ] category_encoders 의존성 확인
- [ ] RecipeBuilder API 문제 해결
- [ ] 테스트 실행 가능 상태 확보

### Phase 2: 주요 테스트 복구 (2-3일)
- [ ] MLflow integration 테스트 12개 수정
- [ ] CLI command 테스트 40개 수정
- [ ] Timeseries handler 테스트 16개 수정
- [ ] Integration 테스트 9개 수정

### Phase 3: 안정화 (1일)
- [ ] 동시성 문제 해결
- [ ] E2E 테스트 반환값 수정
- [ ] 모든 테스트 실행 확인
- [ ] 테스트 커버리지 확인

### Phase 4: 문서화 (선택사항)
- [ ] 테스트 가이드 작성
- [ ] CI/CD 통합
- [ ] 테스트 best practices 문서화

## 🚀 실행 명령어

### 의존성 설치
```bash
pip install psutil
# 또는
pip install -e ".[test]"  # pyproject.toml 수정 후
```

### 테스트 실행 (단계별)
```bash
# 1. 수정된 테스트만 실행
pytest tests/unit/utils/test_mlflow_integration.py -v

# 2. Unit 테스트 실행
pytest tests/unit/ --ignore=tests/unit/components/test_preprocessor/test_encoder.py -v

# 3. Integration 테스트 실행
pytest tests/integration/ -v

# 4. E2E 테스트 실행
pytest tests/e2e/ --ignore=tests/e2e/test_cli-workflow.py -v

# 5. 전체 테스트 실행
pytest tests/ -v --tb=short
```

## 📈 예상 결과

### 수정 후 목표
- **성공률**: 95% 이상 (현재 74.8%)
- **실패**: 20개 미만 (현재 332개)
- **에러**: 0개 (현재 16개)
- **수집 실패**: 0개 (현재 2개)

### 개선 지표
- 테스트 실행 시간 단축
- 테스트 안정성 향상
- 유지보수성 개선
- CI/CD 통합 가능

## 💡 추가 권장사항

### 1. 테스트 전략 개선
- **Unit Test**: Mock 사용 최소화, 실제 객체 사용 선호
- **Integration Test**: 실제 환경과 유사한 설정 사용
- **E2E Test**: 시나리오 기반 테스트 강화

### 2. 코드 품질 도구 도입
```bash
# pre-commit hooks
pre-commit install

# 테스트 커버리지
pytest --cov=src --cov-report=html

# 코드 품질 검사
ruff check src/ tests/
mypy src/
```

### 3. 지속적 개선
- 테스트 실패 시 즉시 수정하는 문화 정착
- 새 기능 추가 시 테스트 먼저 작성 (TDD)
- 정기적인 테스트 리뷰 및 리팩토링

## 🎯 결론

현재 테스트 실패의 주요 원인은:
1. **소스 코드와 테스트 코드의 동기화 부족** (70% 이상의 실패 원인)
2. **의존성 관리 미흡** (수집 실패 원인)
3. **API 변경 시 테스트 업데이트 누락**

이 계획을 따라 단계적으로 수정하면 1주일 내에 테스트 성공률을 95% 이상으로 복구할 수 있을 것으로 예상됩니다.