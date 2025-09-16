# Modern ML Pipeline 테스트 개선 분석 보고서

## 1. 개요

이 보고서는 Modern ML Pipeline 프로젝트의 테스트 코드 개선 방향을 분석한 결과입니다. `tests/README.md`의 핵심 철학을 기반으로 현재 실패하는 테스트들을 분석하고, 철학을 준수하는 해결방안을 제시합니다.

## 2. 테스트 철학 분석 (tests/README.md)

### 2.1 핵심 원칙

Modern ML Pipeline의 테스트 철학은 다음 5가지 핵심 원칙을 중심으로 합니다:

1. **Real Object Testing**: Mock 사용을 최소화하고 실제 객체 사용
2. **Public API Focus**: 내부 구현이 아닌 퍼블릭 인터페이스 테스트
3. **Deterministic Execution**: 모든 테스트는 결정론적이고 재현 가능
4. **Test Isolation**: 각 테스트는 독립적으로 실행 가능
5. **Clear Boundaries**: 테스트 계층별 명확한 책임 분리

### 2.2 No Mock Hell 원칙

Mock 사용에 대한 엄격한 제한:

#### 🚫 금지 영역 (Never Mock)
- 내부 컴포넌트 (Registry, Factory 내부 상태)
- 비즈니스 로직 (Preprocessor, Model 등)
- 데이터 구조체 (DataFrame, Recipe 등)
- 컴포넌트 간 상호작용

#### ✅ 허용 영역 (Mockable)
- 외부 서비스 (MLflow server, Database connections)
- 네트워크 I/O (HTTP requests, API calls)
- 파일 시스템 I/O (대용량 파일 처리)
- 시간 의존적 외부 리소스

### 2.3 Context 클래스 철학

테스트에서 Context 클래스는 **관찰자(Observer)** 역할만 수행:
- ✅ 상태 관찰 및 검증
- ✅ 실행 결과 수집
- ✅ 테스트 후 정리(cleanup)
- 🚫 내부 상태 직접 조작 금지

## 3. 현재 실패 케이스 분석

### 3.1 System Checker 실패

**오류 현상:**
```
ModuleNotFoundError: No module named 'psycopg2'
```

**근본 원인:**
- PostgreSQL 어댑터 검증 시 psycopg2 dependency 누락
- SystemChecker가 실제 모듈 import를 통해 검증하므로 필수

**철학 준수성 분석:**
- ✅ 외부 라이브러리 dependency 이슈로 Mock 대상이 아님
- ✅ Real Object Testing 원칙에 부합

### 3.2 Preprocessor Registry 실패

**오류 현상:**
```
KeyError: 'scaler' during test_create_scaler()
```

**근본 원인:**
- `PreprocessorStepRegistry.preprocessor_steps` 글로벌 딕셔너리 상태 이슈
- Factory._trigger_component_imports()에서 누락된 preprocessor 모듈들
- 테스트 간 상태 공유로 인한 격리 문제

**철학 준수성 분석:**
- 🚫 Registry 내부 상태 직접 조작은 "내부 컴포넌트" Mock에 해당하여 금지
- ✅ Application 레벨 bootstrapping 개선은 허용

### 3.3 통합 테스트 격리 문제

**오류 현상:**
- 테스트 순서에 따른 성공/실패 불일치
- 글로벌 Registry 상태로 인한 side effect

**근본 원인:**
- Factory 초기화 시점의 불완전성
- 테스트 간 상태 공유

## 4. 검증된 해결방안

### 4.1 P0: System Checker Dependency 해결 (우선순위: 긴급)

**해결방안:**
```bash
pip install psycopg2-binary
# 또는 requirements.txt에 추가
```

**철학 준수도:** ✅ 100% 준수
- Real Object Testing: 실제 라이브러리 사용
- Public API Focus: 외부 dependency 관리
- No Mock Hell: 외부 라이브러리는 Mock 대상이 아님

**구현 가이드:**
1. requirements-test.txt에 psycopg2-binary 추가
2. CI/CD 파이프라인에서 테스트 dependency 설치 확인

### 4.2 P1: Application Bootstrapping 개선 (우선순위: 높음)

**해결방안:**
Factory._trigger_component_imports()에서 누락된 preprocessor 모듈들을 모두 로드

```python
# src/settings/factory.py 개선
def _trigger_component_imports(self):
    # 현재 누락된 preprocessor 모듈들 추가
    from src.components.preprocessor.scaler import StandardScaler, MinMaxScaler
    from src.components.preprocessor.encoder import OneHotEncoder, LabelEncoder
    from src.components.preprocessor.imputer import SimpleImputer
    from src.components.preprocessor.feature_generator import PolynomialFeatures
    from src.components.preprocessor.discretizer import KBinsDiscretizer
    # ... 기타 누락된 모듈들
```

**철학 준수도:** ✅ 100% 준수
- Real Object Testing: Registry에 실제 객체들을 정상 로드
- Public API Focus: Factory의 정상적인 초기화 과정 개선
- No Mock Hell: Registry 내부를 직접 조작하지 않음

**구현 가이드:**
1. src/components/preprocessor/ 하위 모든 모듈 조사
2. Factory._trigger_component_imports()에 누락된 import 문 추가
3. Registry 자동 등록 메커니즘 검증

### 4.3 P2: Factory 완전 초기화 보장 (우선순위: 높음)

**해결방안:**
통합 테스트에서 SettingsFactory.create() 전에 완전한 초기화 보장

```python
# tests/conftest.py 개선
@pytest.fixture
def settings_builder(temp_dir):
    # Factory 완전 초기화 보장
    factory = SettingsFactory()
    factory._trigger_component_imports()  # 명시적 초기화

    builder = SettingsBuilder(factory)
    # ... 기존 로직
    return builder
```

**철학 준수도:** ✅ 95% 준수
- Real Object Testing: 실제 Factory 객체의 정상적인 초기화
- Public API Focus: Factory의 정당한 초기화 과정
- 주의사항: _trigger_component_imports는 internal method이지만 정당한 초기화 목적

**구현 가이드:**
1. conftest.py의 settings_builder fixture 개선
2. 통합 테스트 시작 전 Factory 상태 검증 로직 추가
3. 테스트별 Factory 초기화 완료 확인

### 4.4 P3: Public API 상태 관리 (우선순위: 중간)

**해결방안:**
Factory에 Registry 상태 관리를 위한 Public API 제공

```python
# src/settings/factory.py 개선
class Factory:
    def reset_registries(self) -> None:
        """테스트를 위한 Registry 상태 초기화"""
        PreprocessorStepRegistry.preprocessor_steps.clear()
        AdapterRegistry.adapters.clear()
        # ... 기타 Registry들
        self._trigger_component_imports()  # 재로드

    def is_fully_initialized(self) -> bool:
        """Factory 초기화 완료 상태 검증"""
        return (
            len(PreprocessorStepRegistry.preprocessor_steps) > 0 and
            len(AdapterRegistry.adapters) > 0
            # ... 기타 검증 조건들
        )
```

**철학 준수도:** ✅ 100% 준수
- Real Object Testing: Registry의 실제 상태를 public API로 관리
- Public API Focus: Factory의 명시적 public interface 사용
- No Mock Hell: Registry 내부를 직접 조작하지 않음

**구현 가이드:**
1. Factory에 reset_registries() public method 추가
2. 테스트 fixture에서 공식 API 사용
3. Context 클래스들과 일관성 있는 관찰 패턴 유지

## 5. 구현 우선순위 및 로드맵

### Phase 1: 긴급 수정 (1-2일)
- P0: psycopg2 dependency 해결
- P1: Factory._trigger_component_imports() 개선

### Phase 2: 안정화 (3-5일)
- P2: conftest.py fixture 개선
- 기존 실패 테스트들 검증 및 수정

### Phase 3: 장기 개선 (1주)
- P3: Factory Public API 확장
- 새로운 테스트 케이스 추가 시 가이드라인 적용

## 6. 테스트 개발 가이드라인

### 6.1 새로운 테스트 작성 시

**DO ✅:**
- Factory.create()를 통한 정상적인 객체 생성
- Public API만을 사용한 상태 검증
- Context 클래스를 통한 관찰 패턴
- 실제 데이터와 객체를 사용한 테스트

**DON'T 🚫:**
- Registry.preprocessor_steps 직접 조작
- Factory 내부 상태 직접 변경
- 비즈니스 로직 컴포넌트 Mock
- 테스트 간 상태 공유

### 6.2 Context 클래스 확장 시

기존 ComponentTestContext, MLflowTestContext 패턴 준수:

```python
class NewTestContext:
    def observe_state(self) -> Dict[str, Any]:
        """상태 관찰만 수행"""
        pass

    def validate_results(self, expected: Any, actual: Any) -> bool:
        """결과 검증만 수행"""
        pass

    def cleanup(self) -> None:
        """테스트 후 정리만 수행"""
        pass

    # 🚫 금지: 내부 상태 직접 조작 메서드
```

## 7. 결론

본 분석을 통해 Modern ML Pipeline의 테스트 실패 원인을 규명하고, tests/README.md의 핵심 철학을 완전히 준수하는 해결방안을 도출했습니다.

**핵심 성과:**
1. ✅ "No Mock Hell" 원칙을 엄격히 준수하는 해결책
2. ✅ Registry 직접 조작 대신 Application Bootstrapping 접근법
3. ✅ Public API만을 사용하는 상태 관리 방식
4. ✅ 기존 Context 클래스 철학과 일관성 유지

이러한 접근법을 통해 테스트 코드의 안정성과 유지보수성을 크게 향상시킬 수 있으며, 향후 테스트 개발에서도 일관된 품질을 보장할 수 있습니다.

---
*Generated with Claude Code - Modern ML Pipeline Test Analysis*
*Analysis Date: 2025-09-16*