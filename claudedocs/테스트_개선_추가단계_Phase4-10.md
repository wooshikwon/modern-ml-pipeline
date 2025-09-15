# 테스트 개선 추가 단계 (Phase 4-10)

## 현재 상태 분석 (Phase 1-3 완료 후)

### 완료된 개선사항
- ✅ Phase 1: 테스트 구조 표준화 (Context 클래스 도입)
- ✅ Phase 2: Coverage 향상 기본 작업
- ✅ Phase 3: 테스트 아키텍처 최적화 (병렬 실행, CI/CD)

### 발견된 주요 문제점

#### 1. Coverage 심각한 부족
- **Utils 모듈**: 29% (목표: 90%)
  - `data_io.py`: 47%
  - `mlflow_integration.py`: 65%
  - `console_manager.py`: 73%
- **Models 모듈**: 24% (목표: 90%)

#### 2. Mock 사용 문제
- 20개 이상의 테스트 파일이 여전히 `Mock()` 사용
- Context 클래스 미적용 파일:
  - CLI 테스트 전체 (7개 파일)
  - Factory 테스트 (2개 파일)
  - Serving 테스트 (2개 파일)

#### 3. 테스트 품질 문제
- **Parametrized 테스트**: 0개 (엣지 케이스 테스트 없음)
- **Error handling 테스트**: 137개만 존재 (전체 1189개 중)
- **Property-based 테스트**: 0개 (hypothesis 미사용)
- **Doctest**: 0개

#### 4. 테스트 유형 부재
- **E2E 테스트**: 완전 부재
- **Performance 테스트**: 완전 부재
- **Security 테스트**: 2개만 존재

#### 5. Integration 테스트 문제
- Context 클래스 사용: 3개 파일만 (전체 중 극소수)
- Async 테스트: 2개 (모두 실패 상태)

---

## Phase 4: Mock → Context 전환 완료
**목표**: 모든 Mock 사용을 Context 클래스로 교체

### 우선순위 1: CLI 테스트 (7개 파일)
```python
# 변환 대상
tests/unit/cli/test_config_commands.py
tests/unit/cli/test_get_recipe_command.py
tests/unit/cli/test_inference_command.py
tests/unit/cli/test_init_command.py
tests/unit/cli/test_serve_command.py
tests/unit/cli/test_system_checker.py
tests/unit/cli/test_train_command.py
```

**작업 내용**:
1. `ComponentTestContext` 도입
2. Mock 객체를 실제 컴포넌트로 교체
3. 격리된 테스트 환경 구성

### 우선순위 2: Serving 테스트 (2개 파일)
```python
tests/unit/serving/test_lifespan.py  # asyncio 에러 수정 필요
tests/unit/serving/test_router.py
```

**작업 내용**:
1. Async 테스트 프레임워크 정립
2. `pytest-asyncio` 설정
3. FastAPI TestClient 활용

### 우선순위 3: Factory 테스트 (2개 파일)
```python
tests/unit/factory/test_component_creation.py
tests/unit/factory/test_component_creation_old_mocks.py  # 삭제 고려
```

### 성공 기준
- [ ] Mock() 사용 0개
- [ ] 모든 테스트 Context 클래스 사용
- [ ] Async 테스트 정상 동작

---

## Phase 5: Parametrized & Edge Case Testing
**목표**: 체계적인 엣지 케이스 테스트 구축

### 작업 내용

#### 1. Parametrized 테스트 도입
```python
# 예시: Utils 모듈
@pytest.mark.parametrize("input_format,expected_output", [
    ("csv", pd.DataFrame),
    ("parquet", pd.DataFrame),
    ("json", pd.DataFrame),
    ("invalid", ValueError),
])
def test_data_io_formats(input_format, expected_output):
    ...
```

#### 2. 경계값 테스트
- 빈 데이터셋
- 단일 레코드
- 대용량 데이터 (>100MB)
- 특수 문자/인코딩
- NULL/NaN 처리

#### 3. Error Scenario 테스트
```python
# 모든 public API에 대해
@pytest.mark.parametrize("invalid_input,expected_error", [
    (None, TypeError),
    ({}, ValueError),
    ("", ValueError),
])
```

### 성공 기준
- [ ] 각 모듈당 최소 20개 parametrized 테스트
- [ ] Error handling 테스트 300개 이상
- [ ] 모든 public API 경계값 테스트 완료

---

## Phase 6: Coverage 90% 달성
**목표**: 핵심 모듈 커버리지 90% 달성

### 우선순위별 작업

#### 1. Utils 모듈 (현재 29% → 90%)
```
작업 대상:
- data_io.py (47% → 90%)
  - [ ] CSV/Parquet/JSON 읽기/쓰기
  - [ ] 대용량 파일 처리
  - [ ] 인코딩 에러 처리

- mlflow_integration.py (65% → 90%)
  - [ ] 실험 추적
  - [ ] 모델 로깅
  - [ ] 아티팩트 관리

- console_manager.py (73% → 90%)
  - [ ] 모든 출력 포맷
  - [ ] 인터랙티브 모드
  - [ ] 에러 표시
```

#### 2. Models 모듈 (현재 24% → 90%)
```
작업 대상:
- custom/__init__.py (69% → 90%)
- timeseries_wrappers.py (86% → 90%)
- lstm_timeseries.py (93% → 95%)
```

### 구현 전략
1. 미커버 라인 분석 (`--cov-report=term-missing`)
2. 핵심 경로 우선 테스트
3. 엣지 케이스 추가

### 성공 기준
- [ ] Utils 모듈 90% 이상
- [ ] Models 모듈 90% 이상
- [ ] 전체 프로젝트 85% 이상

---

## Phase 7: E2E 테스트 구축
**목표**: 전체 파이프라인 통합 테스트

### 테스트 시나리오

#### 1. 학습 파이프라인 E2E
```python
tests/e2e/test_train_pipeline_e2e.py
- 데이터 로드 → 전처리 → 학습 → 모델 저장 → MLflow 로깅
```

#### 2. 추론 파이프라인 E2E
```python
tests/e2e/test_inference_pipeline_e2e.py
- 모델 로드 → 데이터 준비 → 예측 → 결과 검증
```

#### 3. API 서빙 E2E
```python
tests/e2e/test_api_serving_e2e.py
- API 시작 → 헬스체크 → 예측 요청 → 배치 예측 → 종료
```

#### 4. CLI 워크플로우 E2E
```python
tests/e2e/test_cli_workflow_e2e.py
- init → train → serve → predict 전체 흐름
```

### 성공 기준
- [ ] 4개 핵심 시나리오 구현
- [ ] 실제 데이터/모델 사용
- [ ] 10분 내 실행 완료

---

## Phase 8: Performance 테스트
**목표**: 성능 벤치마크 및 회귀 방지

### 테스트 구성

#### 1. 학습 성능 벤치마크
```python
tests/performance/test_training_performance.py
@pytest.mark.performance
def test_training_speed():
    # 1000개 샘플 학습 시간 < 30초
    # 10000개 샘플 학습 시간 < 5분
```

#### 2. 추론 성능 벤치마크
```python
tests/performance/test_inference_performance.py
- 단일 예측: < 100ms
- 배치 예측 (1000개): < 1초
- 동시 요청 처리: 100 req/sec
```

#### 3. 메모리 사용량 모니터링
```python
tests/performance/test_memory_usage.py
- 학습 중 메모리: < 2GB
- 추론 중 메모리: < 500MB
- 메모리 누수 체크
```

### 구현 도구
- `pytest-benchmark`
- `memory_profiler`
- `psutil`

### 성공 기준
- [ ] 모든 성능 목표 달성
- [ ] CI에서 자동 실행
- [ ] 성능 회귀 자동 감지

---

## Phase 9: 고급 테스트 기법 도입
**목표**: 테스트 품질 극대화

### 1. Property-based Testing (Hypothesis)
```python
from hypothesis import given, strategies as st

@given(
    data=st.lists(st.floats(min_value=-1e6, max_value=1e6)),
    batch_size=st.integers(min_value=1, max_value=1000)
)
def test_preprocessing_properties(data, batch_size):
    # 속성: 전처리 후에도 데이터 크기 유지
    # 속성: NULL 값 없음
    # 속성: 스케일 범위 준수
```

### 2. Mutation Testing
```python
# mutmut 도구 사용
# 코드 변형 후 테스트 실패 확인
mutmut run --paths-to-mutate=src/
```

### 3. Contract Testing
```python
# API 계약 테스트
from pydantic import BaseModel

class PredictionContract(BaseModel):
    input: List[float]
    output: float
    confidence: Optional[float]
```

### 4. Fuzzing
```python
# 랜덤 입력으로 크래시 테스트
@given(st.text())
def test_no_crash_on_random_input(random_input):
    # 어떤 입력이든 크래시 없이 처리
```

### 성공 기준
- [ ] Hypothesis 테스트 50개 이상
- [ ] Mutation coverage 80% 이상
- [ ] 모든 API contract 정의

---

## Phase 10: 테스트 문서화 및 모니터링
**목표**: 테스트 투명성 및 유지보수성 확보

### 1. 테스트 문서화
```python
# docstring으로 테스트 의도 명확화
def test_model_training_with_small_dataset():
    """
    Given: 100개 미만의 작은 데이터셋
    When: 모델 학습 시도
    Then: MinDataError 발생하고 적절한 메시지 반환

    Rationale: 최소 데이터 요구사항 검증
    Related: Issue #123, Requirement 4.2.1
    """
```

### 2. 테스트 리포트 자동화
```yaml
# .github/workflows/test-report.yml
- name: Generate test report
  run: |
    pytest --html=report.html --self-contained-html
    pytest --cov-report=html:htmlcov
```

### 3. 테스트 메트릭 대시보드
- Coverage 트렌드
- 테스트 실행 시간 트렌드
- Flaky 테스트 추적
- 테스트 실패율 모니터링

### 4. 테스트 가이드라인
```markdown
tests/TESTING_GUIDE.md
- 테스트 작성 표준
- Context 클래스 사용법
- 네이밍 컨벤션
- 안티패턴 예시
```

### 성공 기준
- [ ] 모든 테스트 docstring 작성
- [ ] 자동 리포트 생성
- [ ] 테스트 가이드 완성
- [ ] 메트릭 대시보드 구축

---

## 실행 우선순위 및 일정

### 즉시 실행 (Critical)
1. **Phase 4**: Mock → Context 전환 (1주)
2. **Phase 6**: Coverage 90% 달성 (2주)

### 단기 실행 (High Priority)
3. **Phase 5**: Parametrized Testing (1주)
4. **Phase 7**: E2E 테스트 (1주)

### 중기 실행 (Medium Priority)
5. **Phase 8**: Performance 테스트 (1주)
6. **Phase 9**: 고급 테스트 기법 (2주)

### 장기 실행 (Low Priority)
7. **Phase 10**: 문서화 및 모니터링 (지속적)

---

## 기대 효과

### 정량적 목표
- 테스트 커버리지: 29% → 90%
- 테스트 실행 시간: <10분 유지
- 버그 감소율: 70% 이상
- 배포 신뢰도: 95% 이상

### 정성적 개선
- 코드 변경 시 자신감 향상
- 리팩토링 안전성 확보
- 신규 개발자 온보딩 용이
- 프로덕션 이슈 사전 방지

---

## 참고사항

### 원칙 준수
- tests/README.md의 철학 엄격히 준수
- Context 클래스 우선 사용
- Real object testing 원칙
- 결정론적 테스트 보장

### 점진적 개선
- 각 Phase는 독립적으로 가치 제공
- 우선순위에 따라 유연하게 실행
- 지속적인 피드백과 개선

### 협업 고려사항
- 코드 리뷰 시 테스트 포함 필수
- 테스트 작성 가이드라인 공유
- 팀 전체 테스트 문화 정착