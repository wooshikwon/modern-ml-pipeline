# Modern ML Pipeline — 리팩터링 후 코드 리뷰 보고서

**분석 일시**: 2026-03-24 (Phase 1/2/3 + 테스트 정리 완료 후)
**분석 방법**: 4개 end-to-end 시나리오별 코드 흐름 추적

---

## 1. 종합 평가

| 영역 | 리팩터링 전 | 현재 | 변화 |
|------|-----------|------|------|
| **아키텍처 설계** | A | A | Registry 8개 모두 타입 검증, Factory 파사드 분할 |
| **코드 가시성** | B+ | A- | Factory 625줄→324줄, predict 155줄→30줄, validators 추출 |
| **에러 처리** | C+ | B+ | bare except 제거, 한국어 메시지 일관, 로깅 추가 |
| **테스트 품질** | B- | B+ | 67% 코드 축소, CI continue-on-error 제거, conftest 분리 |
| **보안** | B- | B | AuthConfig 미구현 유지, lenient 모드 경고 필요 |
| **확장성** | B+ | A- | BaseOptimizer 도입, BaseCalibrator 통일 |

**종합: B+ → A-**

---

## 2. 잘 된 점 — 리팩터링 성과

### 2.1 Factory 파사드 전환

Factory가 625줄 god-class에서 324줄 파사드로 전환되었다. `ComponentCreator`와 `ModelCreator`가 실제 생성 로직을 담당하고, Factory는 `@cached` 데코레이터와 위임만 수행한다.

```
Factory (파사드, 324줄)
  ├─ ComponentCreator (Tier 1/2/3, ~200줄)
  └─ ModelCreator (모델/HPO, ~150줄)
```

캐싱이 파사드에 집중되어 있으므로 Creator 클래스는 순수 생성 로직만 담당한다. 공개 API(`factory.create_data_adapter()` 등)는 변경 없이 유지되었다.

### 2.2 PyfuncWrapper predict() 분리

155줄의 단일 메서드가 5개 private 메서드로 분리되었다. 각 단계의 fallback이 명확해졌고, 로깅이 추가되어 어떤 경로로 실행되었는지 추적 가능하다.

```python
predict()  # 30줄 오케스트레이터
  → _fetch_features()      # Fetcher 호출, 실패 시 원본 반환
  → _apply_datahandler()   # DataHandler 변환, 실패 시 target 제외
  → _apply_preprocessing() # Preprocessor, 실패 시 이전 결과 유지
  → _run_prediction()      # 모델 예측 + 캘리브레이션
  → _format_output()       # DataFrame/list 변환
```

### 2.3 8개 Registry 일관성 달성

OptimizerRegistry에 `BaseOptimizer` 추상 클래스가 도입되어, 이제 8개 Registry 모두 `_base_class`를 가진다. 등록 시점에 `issubclass` 검사가 실행되므로 잘못된 컴포넌트 등록이 즉시 감지된다.

### 2.4 검증 체계 안정화

`ValidationResult`가 `NamedTuple`에서 `frozen dataclass`로 전환되어 mutable default 문제가 해결되었다. `field(default_factory=list)`로 인스턴스 간 warnings 오염이 방지된다.

### 2.5 테스트 코드 67% 축소

6개 핵심 파일이 4,931줄에서 1,647줄로 줄었다. conftest.py가 `tests/fixtures/`로 분리되어 fixture 발견이 명확해졌고, 표준 Mock이 `tests/fixtures/mocks.py`에 통합되었다.

### 2.6 에러 메시지 한국어 일관성

14개 파일의 ~25개 영어 에러 메시지가 한국어로 통일되었다. 사용자 향 메시지(`raise`, `HTTPException`)는 모두 한국어이다.

---

## 3. 잔여 이슈 — 개선 필요

### HIGH 우선순위 — ✅ 모두 해결

#### H-01. AppContext 초기화가 initialize() 메서드를 사용하지 않음 — ✅ 해결

**파일**: `mmp/serving/_lifespan.py:26-60`

`AppContext`에 `threading.Lock` 기반의 `initialize()` 메서드가 추가되었지만, `setup_api_context()`는 여전히 필드를 직접 설정한다:

```python
# 현재 — Lock 없이 직접 설정
app_context.model = mlflow.pyfunc.load_model(model_uri)
app_context.model_uri = model_uri
app_context.settings = settings
...
app_context._initialized = True
```

`initialize()` 메서드를 사용하도록 변경하면 스레드 안전성이 보장된다.

#### H-02. Lifespan 이벤트 미활용 — ✅ 해결

`_lifespan.py`의 `lifespan()` 함수가 빈 상태(yield만)이고, 실제 초기화는 `router.py`의 `run_api_server()`에서 `setup_api_context()`를 직접 호출한다. FastAPI의 lifespan 이벤트 내에서 초기화하면 멀티 워커 환경에서 더 안전하다.

#### H-03. SettingsFactory lenient 모드의 프로덕션 위험 — ✅ 해결

**파일**: `mmp/settings/factory.py:114-120`

`MMP_FACTORY_LENIENT=1` 환경변수로 서빙 검증을 완화할 수 있다. 프로덕션 환경에서 실수로 이 변수가 설정되면 검증이 우회된다.

**권장**: 환경 확인(`MMP_ENVIRONMENT != production`) 조건 추가.

#### H-04. env_resolver의 환경변수 누락 시 동작 모호 — ✅ 해결

**파일**: `mmp/settings/env_resolver.py:65`

`${VAR}` 패턴에서 default가 없고 환경변수도 없으면, 원본 문자열 `"${VAR}"`이 그대로 반환된다. 이후 Pydantic 파싱에서 타입 에러가 발생하지만, 원인 추적이 어렵다.

**권장**: 환경변수 누락 시 명확한 경고 로깅 또는 예외 발생.

### MEDIUM 우선순위 — ✅ 모두 해결

#### M-01. predict/predict_batch 검증 순서 통일 — ✅ 해결

`predict()`는 `validate_required_columns` → `validate_scalar_values` → `validate_numeric_types` 순서이고, `predict_batch()`는 `validate_scalar_values` → `validate_numeric_types` → `validate_required_columns` 순서다. 통일 필요.

#### M-02. BaseTrainer/BaseFetcher에 `__init__(settings)` 추가 — ✅ 해결

8개 Base 클래스 중 BaseTrainer와 BaseFetcher만 `__init__`이 정의되어 있지 않다. 다른 6개는 모두 `settings`를 받는다. 인터페이스 일관성을 위해 추가 권장.

#### M-03. _ensure_components_registered() 전체 Registry 검증 — ✅ 해결

Factory 부트스트랩에서 8개 Registry 중 CalibrationRegistry만 등록 여부를 검사한다. 다른 Registry가 비어있어도 감지하지 못한다.

#### M-04. ValidationOrchestrator에 data_source 호환성 검증 추가 — ✅ 해결

`validate_for_training()`에서 `CompatibilityValidator.validate_data_source_compatibility()`가 호출되지 않는 것으로 보인다. Config의 adapter_type과 Recipe의 source_uri 패턴 매칭이 누락될 수 있다.

#### M-05. mlflow_integration.py 파일 분리 — ✅ 해결

719줄이었던 단일 파일을 3개로 분리:
- `mlflow_tracker.py` (268줄) — MLflowTracker 클래스
- `mlflow_signature.py` (343줄) — MLflowSignatureBuilder 클래스
- `mlflow_integration.py` (146줄) — Re-export 파사드 (하위 호환 100%)

### LOW 우선순위

| ID | 이슈 | 비고 |
|----|------|------|
| L-01 | `factory.py:209-232` Config 기본값 보강이 DEBUG 로그만 남김 | 프로덕션에서 기본값 주입 사실을 놓치기 쉬움 |
| L-02 | `pyfunc_wrapper.py` fallback 체인에서 Silent Failure 가능 | 각 단계 실패 시 warning만, 어떤 경로 실행됐는지 불명확 |
| L-03 | `calibration/base.py` settings 파라미터가 실제 미사용 | 인터페이스 통일용이지만 주석 필요 |
| L-04 | `__init__.py` import 순서가 카테고리 간 불일치 | adapter는 Registry 먼저, evaluator는 modules 먼저 |
| L-05 | `mlflow_integration.py` 영문 주석 소수 잔존 | 한국어 통일 기준에서 예외 |

---

## 4. 구조 건강도 — 현재 상태

### 소스 코드 파일 크기

| 파일 | 줄 수 | 등급 | 이전 대비 |
|------|------|------|----------|
| `mlflow_integration.py` | 719 | **C+** | 클래스화했지만 파일 미분리 |
| `settings/factory.py` | 484 | **B-** | 539→484 (env_resolver 추출) |
| `pyfunc_wrapper.py` | 392 | **B** | predict 분리로 가독성 향상 |
| `factory/factory.py` | 324 | **A-** | 625→324 (파사드 전환) |
| `serving/_endpoints.py` | ~280 | **A-** | 464→~280 (validators 추출) |
| `serving/validators.py` | ~120 | **A** | 🆕 검증 통합 |
| `settings/env_resolver.py` | 104 | **A** | 🆕 환경변수 standalone |
| `factory/component_factory.py` | ~200 | **A-** | 🆕 컴포넌트 생성 |
| `factory/model_factory.py` | ~150 | **A-** | 🆕 모델/HPO 생성 |
| `optimizer/base.py` | ~25 | **A** | 🆕 BaseOptimizer |

### 종합 점수

| 영역 | Training | Inference/Serving | Component | Settings/DX | **평균** |
|------|----------|-------------------|-----------|-------------|---------|
| 네이밍 | 8/10 | 8/10 | 9/10 | 8/10 | **8.3** |
| 모듈 경계 | 7/10 | 7/10 | 8/10 | 8/10 | **7.5** |
| 에러 처리 | 6/10 | 7/10 | 8/10 | 7/10 | **7.0** |
| 테스트 용이성 | 7/10 | 7/10 | 8/10 | 8/10 | **7.5** |
| 확장성 | 8/10 | 8/10 | 8/10 | 7/10 | **7.8** |
| **소계** | **7.2** | **7.4** | **8.2** | **7.6** | **7.6** |

---

## 5. 다음 단계 권장

### 즉시 (코드 수정)
1. `_lifespan.py`에서 `app_context.initialize()` 사용하도록 변경
2. predict/predict_batch 검증 순서 통일
3. `env_resolver.py`에 환경변수 누락 시 경고 로깅 추가

### 단기 (아키텍처)
4. `mlflow_integration.py`를 `mlflow_tracker.py` + `mlflow_signature.py`로 분리
5. BaseTrainer/BaseFetcher에 `__init__(settings)` 추가
6. `_ensure_components_registered()`에 전체 Registry 검증 추가

### 중기 (안전성)
7. lenient 모드에 프로덕션 환경 차단 조건 추가
8. ValidationOrchestrator에 data_source 호환성 검증 추가
9. E2E 테스트 추가 (Phase 4 과제)

---

*이 보고서는 Phase 1/2/3 리팩터링 및 테스트 정리가 완료된 후의 코드 상태를 기준으로, 4개 end-to-end 시나리오(Training, Inference/Serving, Component Architecture, Settings/DX)의 코드 흐름을 직접 추적하여 작성되었습니다.*
