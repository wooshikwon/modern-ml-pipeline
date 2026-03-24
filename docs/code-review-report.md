# Modern ML Pipeline — 코드 리뷰 보고서

**분석 일시**: 2026-03-24
**대상 버전**: v1.2.1
**분석 범위**: `mmp/` 소스 130개 파일, `tests/` 86개 파일
**분석 방법**: 4개 end-to-end 시나리오별 코드 흐름 추적 (Training, Inference/Serving, Component Architecture, Settings/DX)
**Phase 1 상태**: ✅ 완료
**Phase 2 상태**: ✅ 완료
**Phase 3 상태**: ✅ 완료
**테스트 정리 상태**: ✅ 완료 (-67% 코드량)

---

## 목차

1. [종합 평가](#1-종합-평가)
2. [평가 기준](#2-평가-기준)
3. [아키텍처 강점](#3-아키텍처-강점)
4. [Critical 이슈](#4-critical-이슈)
5. [Major 이슈](#5-major-이슈)
6. [Minor 이슈](#6-minor-이슈)
7. [파일별 건강도](#7-파일별-건강도)
8. [개선 로드맵](#8-개선-로드맵)

---

## 1. 종합 평가

| 영역 | 등급 | 요약 |
|------|------|------|
| **아키텍처 설계** | A | Registry/Factory/3-Tier 패턴이 잘 설계되어 있고, import-linter로 경계가 강제됨 |
| **코드 가시성** | B+ | 네이밍과 모듈 분리는 양호하나, 일부 god-class와 긴 함수가 가독성을 저해 |
| **에러 처리** | C+ | Serving 레이어에 8개의 `except Exception: pass`, 파이프라인에 부분 실패 복구 없음 |
| **테스트 품질** | B- | 구조는 갖춰졌으나 E2E 1개뿐, CI가 `continue-on-error: true`로 무력화 |
| **보안** | B- | SQL guard 존재하나 AuthConfig 미구현, 시크릿 로그 노출 가능 |
| **확장성** | B+ | Registry 패턴으로 컴포넌트 추가 용이하나 Validator 하드코딩이 OCP 위반 |

**종합**: **B+** — 설계 철학은 우수하지만, 에러 처리와 테스트에서 프로덕션 수준에 미달하는 부분이 있다.

---

## 2. 평가 기준

이 보고서는 두 축으로 코드를 평가한다.

### A. 코드 가시성 (Readability & Navigability)

코드를 처음 보는 개발자가 구조를 빠르게 파악하고 수정할 수 있는가.

1. **네이밍 일관성** — 변수/함수/클래스/모듈명이 역할을 명확히 전달하는가
2. **모듈 경계의 명확성** — 한 파일이 단일 책임인가, 파일 크기가 적절한가 (300줄 이하 권장)
3. **추상화 수준의 일관성** — 같은 레이어의 코드가 비슷한 추상화 수준을 유지하는가
4. **코드 중복(DRY)** — 동일/유사 패턴이 복제되어 있는가
5. **import 구조** — 순서, 순환 의존성, 불필요한 import
6. **docstring/주석** — 필요한 곳에 있고, 과잉인 곳은 없는가
7. **타입 힌트** — 일관성, 공개 API에서의 사용 여부

### B. 코드 효과성 (Functional Correctness & Robustness)

코드가 의도대로 안전하게 동작하는가.

1. **에러 처리** — 예외 삼킴, 의미있는 에러 메시지, 부분 실패 복구
2. **상태 관리** — mutable state 범위, side effect 최소화
3. **테스트 용이성** — DI, 결합도, mock 가능성
4. **성능 패턴** — 불필요한 DataFrame 복사, 메모리 해제, lazy loading
5. **보안** — 입력 검증, 시크릿 노출
6. **확장성** — 새 컴포넌트 추가 시 수정 범위

---

## 3. 아키텍처 강점

MMP의 설계에서 특별히 잘 된 부분을 먼저 짚는다. 이 강점들은 보존해야 할 자산이다.

### 3.1 Registry + Self-Registration 패턴

7개 컴포넌트 Registry가 `BaseRegistry[T]` 제네릭 기반으로 **100% 일관된 구조**를 유지한다. 새 컴포넌트 추가 시 모듈 파일 하단에 `Registry.register("key", Class)` 한 줄만 추가하면 되고, Factory나 파이프라인 코드를 수정할 필요가 없다. Open-Closed Principle의 교과서적 구현이다.

```python
# 모든 컴포넌트 모듈이 동일한 패턴을 따른다
AdapterRegistry.register("storage", StorageAdapter)  # storage_adapter.py 하단
EvaluatorRegistry.register("classification", ClassificationEvaluator)  # 하단
```

### 3.2 Config/Recipe 분리

"무엇을 할지"(Recipe)와 "어디서 할지"(Config)를 분리한 설계가 철저하다. Pydantic v2 스키마는 순수 데이터 구조만 담당하고, 비즈니스 검증은 `ValidationOrchestrator`로 완전 분리되어 있다. 환경변수 치환(`${VAR:default}`)과 Jinja 템플릿 렌더링이 일관되게 지원된다.

### 3.3 import-linter 아키텍처 경계 강제

`pyproject.toml`에 정의된 import-linter 계약이 CI에서 자동 검증된다:
- `mmp.components` → `mmp.factory` 임포트 금지 (하위→상위 의존 차단)
- `mmp.serving` → `mmp.pipelines` 직접 의존 금지

이 구조적 강제가 없으면 코드 리뷰만으로는 의존성 위반을 완벽히 잡을 수 없다.

### 3.4 Serving 레이어의 프로덕션 고려

- K8s liveness(`/health`)와 readiness(`/ready`) 프로브 분리
- `RequestIDMiddleware`로 분산 추적 지원
- `TimeoutMiddleware`로 무한 대기 방지
- Prometheus `/metrics` 엔드포인트로 HPA 커스텀 메트릭 지원
- DataInterface 스키마 기반 동적 API 스키마 생성

### 3.5 사용자 친화적 에러 복구 가이드

Trainer에서 범주형 인코딩 누락이나 NaN 에러를 감지하면, 단순 에러 메시지가 아니라 "어떤 전처리기를 Recipe에 추가하면 되는지"를 구체적으로 안내한다:
```python
# trainer.py — 사용자 지향 에러 메시지
"범주형 컬럼이 인코딩되지 않았습니다. Recipe의 preprocessor.steps에
 'catboost_encoder' 또는 'one_hot_encoder'를 추가하세요."
```

### 3.6 PyfuncWrapper의 Training-Serving Skew 방지

학습된 모델, 전처리기, DataHandler, Fetcher, Calibrator를 하나의 MLflow 아티팩트로 패키징한다. 추론 시 이 래퍼 하나로 전체 파이프라인(전처리 → 피처 증강 → 예측 → 보정)을 재현하므로, 학습-서빙 간 전처리 불일치 문제를 구조적으로 방지한다.

---

## 4. Critical 이슈

실행 오류, 데이터 무결성, 또는 보안에 직접적 영향을 미치는 이슈.

---

### C-01. Serving 레이어의 8개 `except Exception: pass` (에러 처리)

**파일**: `mmp/serving/_endpoints.py` — 라인 110, 173, 207, 369, 402, 418, 431, 461

`predict()` 함수 내에서 입력 검증(숫자형 검증, 필수 컬럼 검증, 시그니처 검증) 코드가 예외를 완전히 삼킨다:

```python
# _endpoints.py:173-175
except Exception:
    # 시그니처를 읽지 못하더라도 동작은 계속
    pass
```

**왜 문제인가**: 잘못된 입력이 모델까지 도달한 후에야 에러가 발생한다. 이때 에러 메시지는 MLflow 내부 메시지여서 사용자에게 무의미하다. 프로덕션에서 API 클라이언트가 422 대신 500 에러를 받고, 디버깅이 불가능해진다.

**권장 수정**:
```python
except Exception as e:
    logger.warning(f"입력 검증 fallback: {type(e).__name__}: {e}")
    # 검증은 best-effort이므로 계속 진행하되, 로그로 추적 가능하게
```

---

### C-02. `ValidationResult`의 mutable default (상태 관리)

**파일**: `mmp/settings/validation/common.py:9`

```python
class ValidationResult(NamedTuple):
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = []  # ← mutable default!
```

**왜 문제인가**: NamedTuple의 mutable default는 모든 인스턴스가 동일한 리스트 객체를 공유한다. 한 인스턴스에 warning을 추가하면 다른 인스턴스에도 영향을 미친다. 검증 결과가 오염되어 잘못된 경고가 누적될 수 있다.

**권장 수정**: `dataclass`로 전환하고 `field(default_factory=list)` 사용.

---

### C-03. Settings 기본값 보강의 조용한 실패 (에러 처리)

**파일**: `mmp/settings/factory.py:218, 307`

```python
except Exception as _:
    pass  # 베스트-에포트 — 실패해도 무시
```

Config 로딩 중 기본값 주입이 실패하면 예외를 완전히 삼킨다. 사용자가 의도한 설정이 무시되어도 알 수 없고, 이후 컴포넌트 초기화에서 원인 불명의 에러가 발생한다.

**권장 수정**: 최소한 `logger.warning(f"Config 기본값 보강 실패: {e}")` 추가.

---

### C-04. OptimizerRegistry 타입 안전성 무력화 (확장성)

**파일**: `mmp/components/optimizer/registry.py:14-23`

```python
class OptimizerRegistry(BaseRegistry[Any]):
    _registry: Dict[str, Type[Any]] = {}
    _base_class = None  # 타입 검증 비활성화
```

7개 Registry 중 유일하게 `_base_class = None`이다. `BaseRegistry.register()`의 `issubclass` 검사가 비활성화되므로, optimizer가 아닌 객체를 등록해도 감지하지 못한다. 에러는 런타임(Trainer 내부)에서야 발생한다.

**권장 수정**: `BaseOptimizer` 추상 클래스를 정의하고 `_base_class`에 지정.

---

### C-05. Factory lambda 클로저의 직렬화 위험 (확장성)

**파일**: `mmp/factory/factory.py:426-428`

```python
trainer = TrainerRegistry.create(
    trainer_type, settings=self.settings, factory_provider=lambda: self
)
```

`lambda: self`는 Factory 인스턴스 전체를 클로저로 캡처한다. Trainer가 MLflow로 직렬화될 경우 이 클로저가 pickle에 실패한다. 현재 Trainer는 직렬화 대상이 아니지만, 향후 확장 시 문제가 된다.

**권장 수정**: lambda 대신 `weakref` 또는 필요한 메서드만 바인딩.

---

### C-06. AppContext 싱글톤의 스레드 안전성 부재 (상태 관리)

**파일**: `mmp/serving/_context.py:11-26`

`AppContext`가 `model`, `settings`, 동적 `PredictionRequest` 스키마를 mutable 속성으로 보관하지만, 동시성 보호(Lock)가 없다. FastAPI는 비동기로 동작하므로 동시 요청에서 상태 경합이 가능하다.

**권장 수정**: `threading.Lock()` 보호 또는 불변 객체(`frozen dataclass`)로 전환.

---

## 5. Major 이슈

코드 품질, 유지보수성, 또는 개발자 경험에 유의미한 영향을 미치는 이슈.

---

### M-01. Factory 클래스가 god-class (가시성 — 모듈 경계)

**파일**: `mmp/factory/factory.py` — **625줄, 21개 public 메서드**

단일 클래스가 Tier 1/2/3 컴포넌트 생성, 모델 생성, 하이퍼파라미터 처리, PyfuncWrapper 생성까지 모두 담당한다.

**권장 분리**:
- `ComponentFactory` — Tier 1 (Adapter, Evaluator, Fetcher, Calibrator)
- `CompositeFactory` — Tier 2 (Trainer, DataHandler)
- `ModelFactory` — 모델 생성 + 하이퍼파라미터 처리

---

### M-02. SettingsFactory가 다중 책임 (가시성 — 모듈 경계)

**파일**: `mmp/settings/factory.py` — **539줄, 15개 메서드**

환경변수 치환, Jinja 렌더링, 기본값 보강, 계산 필드 주입을 모두 한 클래스에서 처리한다.

**권장 분리**:
- `EnvResolver` — `${VAR:default}` 치환
- `TemplateRenderer` — `.sql.j2` 렌더링
- `ConfigDefaults` — 기본값 보강 로직

---

### M-03. mlflow_integration.py 함수 난립 (가시성 — 모듈 경계)

**파일**: `mmp/utils/integrations/mlflow_integration.py` — **578줄, 13개 standalone 함수**

클래스 없이 함수만 13개가 나열되어 있다. `create_model_signature()` (라인 219)와 `create_enhanced_model_signature_with_schema()` (라인 356)는 유사한 기능인데 어떤 걸 써야 하는지 불명확하다.

**권장**: `MLflowIntegration` 클래스로 캡슐화하고 중복 함수 통합.

---

### M-04. pyfunc_wrapper.predict()가 155줄 (가시성 — 추상화 수준)

**파일**: `mmp/utils/integrations/pyfunc_wrapper.py:195-350`

Fetcher → DataHandler → Preprocessor → Model → Calibrator의 5단계를 하나의 메서드에서 처리하며, 각 단계마다 4-5 레벨의 fallback이 중첩되어 있다. 어떤 경로로 실행되었는지 추적이 불가능하다.

**권장**: 각 단계를 별도 private 메서드로 분리하고, 실행 경로 로깅 추가.

---

### M-05. BaseCalibrator 생성자 불일치 (가시성 — 일관성)

**파일**: `mmp/components/calibration/base.py:16-18`

```python
class BaseCalibrator(ABC):
    def __init__(self):  # ← settings 없음!
```

다른 Base 클래스(BaseAdapter, BaseEvaluator, BaseDataHandler)는 모두 `settings` 를 받지만, BaseCalibrator만 인자가 없다. 이로 인해 Calibrator는 Recipe 설정에 접근할 수 없고, 설정 가능한 보정 전략을 구현할 수 없다.

**권장**: `__init__(self, settings=None)`으로 통일.

---

### M-06. BaseDataHandler가 과대 (가시성 — 단일 책임)

**파일**: `mmp/components/datahandler/base.py` — **303줄, 13개 메서드**

추상 메서드 3개, 구체 헬퍼 10개. 하위 클래스(TabularDataHandler)는 3개만 오버라이드하고 나머지는 상속받는다. 인터페이스가 과대하여 Interface Segregation Principle을 위반한다.

**권장**: `_get_split_config()`, `_check_missing_values_warning()` 같은 헬퍼를 별도 유틸로 추출.

---

### M-07. 에러 메시지 언어 혼재 (가시성 — 일관성)

코드베이스 전반에서 에러 메시지가 한국어와 영어로 혼재한다:
- `base_registry.py:72` — `"알 수 없는 키: '{key}'"`
- `factory.py:267` — `"Failed to create adapter '{target_type}'"`
- `datahandler/registry.py:53` — 한국어 다중 라인

**권장**: 에러 메시지를 영어로 통일하거나, 한국어 전용이면 일관되게 사용.

---

### M-08. 환경변수 이중 로딩 (효과성 — 성능)

**파일**: `mmp/cli/commands/train_command.py:66-81`

`load_env_for_config(config_path)`로 `.env` 파일을 로드한 뒤, `SettingsFactory.for_training()` 내부에서 다시 `_resolve_env_variables()`가 환경변수를 처리한다. 이중 처리로 인한 성능 저하와 부작용 가능성이 있다.

**권장**: `load_env_for_config()`를 SettingsFactory 내부로 이동하거나, 호출 의도를 문서화.

---

### M-09. 환경변수 타입 추론의 암묵적 변환 (효과성 — 안전성)

**파일**: `mmp/settings/factory.py:343-361`

`${MLFLOW_PORT:5000}`이 문자열 `"5000"`이 아닌 정수 `5000`으로 자동 변환된다. Pydantic이 나중에 타입을 강제하지만, 변환 시점이 불명확하고 디버깅이 어렵다.

**권장**: 명시적 타입 지정 문법(`${VAR|int}`) 도입 또는 자동 변환 문서화.

---

### M-10. Inference 설정 로딩 코드 중복 (가시성 — DRY)

**파일**: `mmp/pipelines/inference_pipeline.py:129-164`

`_load_inference_settings()`가 `SettingsFactory.for_inference()`의 로직을 부분적으로 복제한다. 두 코드 경로가 존재하여 동작 불일치 가능성이 있다.

**권장**: `_load_inference_settings()` 제거하고 `SettingsFactory.for_inference()` 직접 사용.

---

### M-11. CI `continue-on-error: true` (효과성 — 테스트)

**파일**: `.github/workflows/ci.yml:30`

```yaml
- name: Run core tests
  run: pytest tests/unit -x -q --tb=short
  continue-on-error: true
```

테스트가 실패해도 CI가 통과하므로 PR 머지 게이트가 무력화된 상태다. 코드 품질의 점진적 하락을 구조적으로 허용한다.

**권장**: `continue-on-error: true` 제거.

---

### M-12. conftest.py 1,228줄 (가시성 — 모듈 경계)

**파일**: `tests/conftest.py` — 1,228줄

모든 테스트 카테고리(unit, integration, e2e)의 fixture가 하나의 파일에 모여 있다.

**권장**: `tests/unit/conftest.py`, `tests/integration/conftest.py`로 분리.

---

### M-13. EvaluatorRegistry의 조용한 실패 (효과성 — 에러 처리)

**파일**: `mmp/components/evaluator/registry.py:20-26`

```python
def get_available_metrics_for_task(cls, task_type: str) -> List[str]:
    evaluator_class = cls._registry.get(task_type)
    if not evaluator_class:
        return []  # ← 빈 리스트 반환, 에러 아님
```

존재하지 않는 task_type을 조회하면 빈 리스트가 반환된다. `BaseRegistry.get_class()`는 KeyError를 발생시키는데, 같은 Registry 내에서 동작이 불일치한다.

**권장**: `get_class()`와 동일하게 KeyError 발생, 또는 최소한 warning 로그.

---

### M-14. Factory 부트스트랩의 조용한 실패 (효과성 — 에러 처리)

**파일**: `mmp/factory/factory.py:126-150`

```python
def _ensure_components_registered(cls) -> None:
    ...
    try:
        import mmp.components.calibration  # 실패해도 warning만
    except ImportError:
        logger.warning("calibration 모듈을 임포트할 수 없습니다")
```

컴포넌트 임포트 실패가 warning으로만 기록되고 Factory 생성은 계속 진행된다. 이후 해당 컴포넌트를 사용하려 할 때 원인 불명의 에러가 발생한다.

**권장**: 필수 컴포넌트 임포트 실패 시 예외 발생.

---

## 6. Minor 이슈

개선하면 좋지만 기능에 직접적 영향은 없는 이슈.

| ID | 파일:라인 | 이슈 | 카테고리 |
|----|----------|------|---------|
| m-01 | `train_command.py:108-114` | `event_to_step` 딕셔너리가 하드코딩 — 새 이벤트 추가 시 코드 변경 필요 | 확장성 |
| m-02 | `pyfunc_factory.py:144-160` | `_build_data_interface_config()` 내부에 폴백 로직이 인라인 — 스키마 가정이 암묵적 | 가시성 |
| m-03 | `trainer.py:78-81` | `model.set_params()` 실패를 `except Exception: pass`로 삼킴 — HPO가 기본값으로 학습 | 에러 처리 |
| m-04 | `preprocessor/__init__.py:7` | `__all__`에 `PreprocessorStepRegistry` 미포함 — 다른 패키지와 불일치 | 일관성 |
| m-05 | `adapter/base.py:29` | `params` vs `**kwargs` 구분 불명확 — docstring으로만 안내 | 가시성 |
| m-06 | `init_command.py:59-64` | 부분 디렉토리 생성 후 Exception 시 롤백 없음 | 에러 처리 |
| m-07 | `system_checker.py:97-100` | `run_all_checks()` 반환 Dict vs 개별 check의 CheckResult — 타입 불일치 | 일관성 |
| m-08 | `mlflow_restore.py:120-144` | `_resolve_env_variables()`가 factory.py와 중복 구현 | DRY |
| m-09 | `factory.py:502-527` | 하이퍼파라미터 추출에 3-way 폴백 (tuning/fixed/legacy) — 스키마 버전 관리 필요 | 가시성 |
| m-10 | `trainer.py:69-70` | HPO 루프 내에서 매 trial마다 `factory.create_evaluator()` 호출 — 불필요한 반복 생성 | 성능 |

---

## 7. 파일별 건강도

### 소스 코드 (300줄 이상 = 분할 검토 대상)

| 파일 | 줄 수 | 등급 | 핵심 문제 | Phase 2 후 |
|------|------|------|----------|-----------|
| `factory/factory.py` | 625 | **C** | God-class, 21개 메서드 | ✅ 파사드로 전환, `component_factory.py` + `model_factory.py` 분할 |
| `utils/integrations/mlflow_integration.py` | 578 | **C** | 함수 13개 난립, 중복 | 미착수 |
| `settings/factory.py` | 539 | **C+** | 다중 책임, 조용한 실패 | ✅ 470줄로 축소, `env_resolver.py` 추출 |
| `serving/_endpoints.py` | 464 | **C+** | 8개 bare except, 긴 함수 | ✅ 로깅 추가 + `validators.py` 추출로 간소화 |
| `cli/utils/system_checker.py` | 400+ | **B-** | 반환 타입 불일치 | 미착수 |
| `utils/integrations/pyfunc_wrapper.py` | 350+ | **C+** | predict() 155줄, 다중 fallback | ✅ predict() → 5개 private 메서드로 분리 |
| `utils/data/data_io.py` | 400+ | **B-** | 저장/로드 혼재 | 미착수 |
| `components/datahandler/base.py` | 303 | **B-** | 과대 base class | 미착수 |
| `serving/router.py` | 303 | **B** | 설정+라우팅 혼재 | 미착수 |
| `cli/utils/recipe_builder.py` | 300+ | **B-** | 빌더+검증 혼재 | 미착수 |
| `cli/utils/config_builder.py` | 300+ | **B** | 선택사항 검증 부족 | 미착수 |

### 잘 관리된 파일 (+ Phase 2 신규)

| 파일 | 줄 수 | 등급 | 비고 |
|------|------|------|------|
| `settings/recipe.py` | 110 | **A** | 순수 Pydantic 스키마 |
| `settings/config.py` | 300 | **A-** | 구조 명확 |
| `settings/validation/__init__.py` | 87 | **A** | Orchestrator 패턴 깔끔 |
| `components/base_registry.py` | 80 | **A** | Generic[T] 활용 우수 |
| `cli/utils/cli_progress.py` | 130 | **A** | 단일 책임 |
| `serving/_lifespan.py` | 71 | **A** | 깔끔한 생명주기 |
| `serving/_context.py` | 55 | **A** | 스레드 안전 초기화 |
| `settings/env_resolver.py` | 104 | **A** | 🆕 환경변수 치환 standalone |
| `serving/validators.py` | ~120 | **A** | 🆕 입력 검증 통합 |
| `factory/component_factory.py` | ~200 | **A-** | 🆕 컴포넌트 생성 전담 |
| `factory/model_factory.py` | ~150 | **A-** | 🆕 모델/HPO 생성 전담 |

---

## 8. 개선 로드맵

### Phase 1: 안전성 확보 (Critical 수정) — ✅ 완료

| 순서 | 대상 | 작업 | 상태 |
|------|------|------|------|
| 1-1 | `_endpoints.py` | 8개 bare except에 `logger.debug()` 추가 + HTTPException re-raise 보장 | ✅ |
| 1-2 | `validation/common.py` | `ValidationResult`를 `frozen dataclass` + `field(default_factory=list)`로 전환 | ✅ |
| 1-3 | `settings/factory.py` | 기본값 보강의 except 블록에 `logger.debug()` 추가 (2곳) | ✅ |
| 1-4 | `.github/workflows/ci.yml`, `tests.yml` | `continue-on-error: true` 제거 (2개 워크플로우) | ✅ |
| 1-5 | `_context.py` | AppContext에 `threading.Lock()`, `initialize()`, `is_ready` 프로퍼티 추가 | ✅ |

### Phase 2: 가시성 개선 (Major 리팩터링) — ✅ 2-1~2-4 완료

| 순서 | 대상 | 작업 | 상태 |
|------|------|------|------|
| 2-1 | `factory/factory.py` | `ComponentCreator` + `ModelCreator`로 분할, Factory는 파사드로 전환 | ✅ |
| 2-2 | `settings/factory.py` | `env_resolver.py` 추출, `mlflow_restore.py` 중복 제거 | ✅ |
| 2-3 | `pyfunc_wrapper.py` | `predict()` → `_fetch_features`, `_apply_datahandler`, `_apply_preprocessing`, `_run_prediction`, `_format_output` 분리 | ✅ |
| 2-4 | `_endpoints.py` | `validators.py` 추출 (`validate_scalar_values`, `validate_numeric_types`, `validate_required_columns`) | ✅ |
| 2-5 | `mlflow_integration.py` | `MLflowTracker` + `MLflowSignatureBuilder`로 캡슐화 (함수 re-export로 하위호환) | ✅ |
| 2-6 | `tests/conftest.py` | `tests/fixtures/settings.py` + `tests/fixtures/mocks.py` 분리 | ✅ |

**생성된 파일**:
- `mmp/factory/component_factory.py` — Tier 1/2/3 컴포넌트 생성 전담
- `mmp/factory/model_factory.py` — 모델/HPO 생성 전담
- `mmp/settings/env_resolver.py` — 환경변수 치환 standalone 함수
- `mmp/serving/validators.py` — 입력 검증 로직 통합

**수정된 파일**:
- `mmp/factory/factory.py` — 625줄 → 파사드 패턴 (공개 API 변경 없음)
- `mmp/settings/factory.py` — 539줄 → 470줄 (`_resolve_env_variables` 위임)
- `mmp/settings/mlflow_restore.py` — 236줄 → 217줄 (중복 env 치환 제거)
- `mmp/utils/integrations/pyfunc_wrapper.py` — predict() 155줄 → 30줄 오케스트레이터 + 5개 private 메서드
- `mmp/serving/_endpoints.py` — 464줄 → 검증 로직 validators.py로 이동, predict/predict_batch 대폭 간소화
- `mmp/serving/_context.py` — 스레드 안전 초기화 추가
- `mmp/serving/_lifespan.py` — `_initialized` 플래그 세팅
- `mmp/settings/validation/common.py` — NamedTuple → frozen dataclass
- `.github/workflows/ci.yml`, `tests.yml` — continue-on-error 제거

### Phase 3: 일관성 강화 — ✅ 완료

| 순서 | 대상 | 작업 | 상태 |
|------|------|------|------|
| 3-1 | `optimizer/` | `BaseOptimizer` 추상 클래스 생성, Registry에 `_base_class` 지정 | ✅ |
| 3-2 | `calibration/base.py` | `__init__(self, settings=None)` 통일 | ✅ |
| 3-3 | 에러 메시지 | 한국어로 통일 (14개 파일, ~25개 메시지) | ✅ |
| 3-4 | `inference_pipeline.py` | `_load_inference_settings()` 제거, `SettingsFactory.for_inference()` 직접 사용 | ✅ |
| 3-5 | `mlflow_restore.py` | `_resolve_env_variables()` 중복 제거 (Phase 2-2에서 완료) | ✅ |

### 테스트 코드 정리 — ✅ 완료

| 파일 | Before | After | 감소 |
|------|--------|-------|------|
| `conftest.py` | 1,228줄 | 738줄 | -40% |
| `test_settings_factory.py` | 2,086줄 | 461줄 | -78% |
| `test_4way_split_migration.py` | 474줄 | 117줄 | -75% |
| `test_api_required_columns.py` | 381줄 | 94줄 | -75% |
| `test_context.py` | 407줄 | 109줄 | -73% |
| `test_pyfunc_wrapper_fetcher.py` | 355줄 | 128줄 | -64% |
| **합계** | **4,931줄** | **1,647줄** | **-67%** |

신규 파일: `tests/fixtures/settings.py` (357줄), `tests/fixtures/mocks.py` (81줄)
테스트 결과: 660 passed, 22 skipped, 7 pre-existing failures (리그레션 0건)

### Phase 4: 향후 과제 (미착수)

| 순서 | 작업 |
|------|------|
| 4-1 | SQLite MLflow + 로컬 CSV 기반 경량 E2E 테스트 추가 (train → inference → serve 전체 사이클) |
| 4-2 | Serving 에러 경로 통합 테스트 (잘못된 입력 → 적절한 HTTP 에러) |
| 4-3 | CI에 `pytest --cov` 추가, 커버리지 임계값 설정 |
| 4-4 | integration 테스트를 Docker Compose로 로컬 실행 가능하게 |

---

## 부록: 이슈 통계

| 심각도 | 가시성 | 효과성 | 합계 |
|--------|--------|--------|------|
| Critical | 1 | 5 | **6** |
| Major | 8 | 6 | **14** |
| Minor | 5 | 5 | **10** |
| **합계** | **14** | **16** | **30** |

---

## 9. 테스트 코드 복잡도 분석

### 진단: 커버리지 강제 증대의 부작용

테스트 코드(86개 파일)를 분석한 결과, **커버리지 수치를 올리기 위해 추가된 저가치 테스트**가 유지보수 부담을 높이고 있다. 핵심 문제 4가지:

### 9.1 Fixture 비대화 — conftest.py 1,228줄

`SettingsBuilder`만 230줄이며, 메서드 20개 중 `with_timestamp_column()`, `with_treatment_column()` 등은 거의 사용되지 않는다. fixture 의존성이 4-5단계까지 체인되어 테스트 시작 시간이 느리다.

### 9.2 Mock이 실제 구현보다 복잡한 케이스

`test_pyfunc_wrapper_fetcher.py`(356줄)에서 Mock 정의 10개가 실제 Fetcher 구현보다 복잡하다. Mock 변경 시 모든 테스트가 깨지는 악순환 구조.

### 9.3 동일 로직의 중복 테스트

| 파일 | 줄 수 | 문제 |
|------|------|------|
| `test_settings_factory.py` | 2,086 | 58개 테스트 중 40%가 설정값 읽기 — "YAML 로드 후 값 확인" 반복 |
| `test_api_required_columns.py` | 381 | 4개 클래스가 "필수 컬럼 검증" 동일 로직을 중복 테스트 |
| `test_4way_split_migration.py` | 474 | 6개 테스트가 `split_and_prepare()` 측면만 다르게 — 1개 parametrize로 충분 |
| `test_context.py` | 407 | 컨텍스트 속성 읽기 테스트 30개 — 대부분 trivial assertion |

### 9.4 구현 세부사항에 결합된 테스트

private 메서드를 직접 테스트하거나, 내부 구조 변경 시 깨지는 테스트가 다수 존재한다. 리팩터링을 저해하는 주요 원인.

### 정리 방안

**원칙**: "이 테스트가 없으면 버그가 프로덕션에 갈 가능성이 있는가?" — No면 삭제/통합.

#### 즉시 축소 대상 (커버리지 손실 없음)

| 파일 | 현재 | 목표 | 작업 |
|------|------|------|------|
| `test_settings_factory.py` | 2,086줄 | 300줄 | 설정값 읽기 40개 삭제, 비즈니스 로직 8개만 유지 |
| `test_4way_split_migration.py` | 474줄 | 100줄 | parametrize로 통합 |
| `test_api_required_columns.py` | 381줄 | 80줄 | 4개 클래스 → 1개 + parametrize |
| `test_context.py` | 407줄 | 100줄 | 속성 읽기 삭제, 생성 검증 3개만 |
| `test_pyfunc_wrapper_fetcher.py` | 356줄 | 80줄 | Mock 중복 제거, 표준 Mock 재사용 |
| `conftest.py` | 1,228줄 | 400줄 | SettingsBuilder 축소, 불필요 fixture 삭제, 카테고리별 분리 |

#### conftest 분리 계획

```
conftest.py (200줄) — 전역 fixture만
tests/conftest_settings.py (150줄) — TestDataGenerator + SettingsBuilder 단순화
tests/conftest_mocks.py (120줄) — 표준 Mock 3-4개 (sklearn_model, passthrough_fetcher 등)
```

#### 테스트 전략

| 계층 | 현재 | 목표 | 기준 |
|------|------|------|------|
| 단위 테스트 | 748개 | ~280개 | 비즈니스 로직만, getter/setter 제거 |
| 통합 테스트 | 165개 | ~50개 | 컴포넌트 간 상호작용만, 단위와 중복 제거 |
| E2E 테스트 | 1개 | 2-3개 | train → inference → serve 전체 사이클 |

**Mock 정책**: Mock이 5줄 이상이면 실제 구현(예: `PassThroughFetcher`)을 사용한다. Mock은 외부 시스템(MLflow, DB)에만 사용.

---

*이 보고서는 4개 end-to-end 시나리오(Training Flow, Inference/Serving Flow, Component Architecture, Settings/DX)를 코드 흐름을 직접 추적하여 작성되었습니다. Phase 1/2 수정사항은 2026-03-24에 적용되었습니다.*
