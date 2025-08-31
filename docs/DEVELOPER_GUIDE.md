# 🚀 개발자 온보딩 가이드 - Modern ML Pipeline

**Phase 4-4.5 최적화 성과 반영 - 77% 성능 향상, 100% 테스트 안정화 달성**

---

## 📋 목차

1. [빠른 시작 (5분 설정)](#-빠른-시작-5분-설정)
2. [개발 환경 구성](#-개발-환경-구성) 
3. [TDD 워크플로](#-tdd-워크플로)
4. [Factory 패턴 마스터하기](#-factory-패턴-마스터하기)
5. [테스트 실행 전략](#-테스트-실행-전략-phase-4-최적화)
6. [코딩 가이드라인](#-코딩-가이드라인)
7. [첫 번째 기여](#-첫-번째-기여)
8. [트러블슈팅](#-트러블슈팅)

---

## 🚀 빠른 시작 (5분 설정)

### 1. 저장소 클론 및 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/wooshikwon/modern-ml-pipeline.git
cd modern-ml-pipeline

# 2. Python 환경 설정 (uv 권장)
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv 설치
uv sync  # 의존성 동기화

# 3. 환경변수 설정
cp .env.example .env
echo "APP_ENV=local" >> .env
```

### 2. 개발 환경 검증

```bash
# Phase 4-4.5 성과 검증 - 모든 테스트 통과 확인
uv run pytest -m "core and unit" -v  # 핵심 테스트 (3.00초)

# 전체 단위 테스트 안정화 확인 (79/79 테스트)
uv run pytest tests/unit/ -q

# 종합 성과 검증 스크립트 실행
./scripts/verify_test_coverage.sh
```

### 3. 첫 번째 실행

```bash
# 1. 프로젝트 초기화
uv run python main.py init

# 2. 예제 모델 학습
uv run python main.py train --recipe-file recipes/example_recipe.yaml

# 성공! MLflow UI가 자동으로 실행됩니다 🎉
```

---

## 🛠 개발 환경 구성

### 필수 도구 설치

```bash
# 1. uv (Python 패키지 매니저)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. pre-commit (코드 품질 관리)
uv run pre-commit install

# 3. 개발 의존성 확인
uv sync --all-extras
```

### IDE 설정 (VS Code 권장)

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["-v", "-s", "tests/unit/"],
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### 권장 확장프로그램

- `ms-python.python` - Python 지원
- `charliermarsh.ruff` - Linting 및 포매팅
- `ms-python.pytest` - 테스트 실행
- `ms-toolsai.jupyter` - Jupyter 노트북 지원

---

## 🧪 TDD 워크플로

**Ultra Think 원칙: RED → GREEN → REFACTOR**

### 1. RED (실패하는 테스트 작성)

```python
# tests/unit/components/test_new_feature.py
import pytest
from tests.factories.test_data_factory import TestDataFactory
from tests.factories.settings_factory import SettingsFactory

@pytest.mark.unit
@pytest.mark.core  # 핵심 기능은 core 마커 추가
class TestNewFeature:
    """새로운 기능 테스트 - Blueprint 원칙 준수"""
    
    def test_new_feature_should_process_data(self, test_factories):
        """Given/When/Then 패턴으로 테스트 작성"""
        # Given: Factory 패턴으로 테스트 데이터 생성
        data = test_factories['data'].create_classification_data(n_samples=10)
        settings = test_factories['settings'].create_minimal_settings()
        
        # When: 기능 실행
        # result = NewFeature().process(data, settings)  # 아직 구현 안됨
        
        # Then: 기대 결과 검증
        # assert result is not None
        pytest.fail("구현 필요 - RED 단계")
```

### 2. GREEN (최소한의 구현)

```python
# src/components/new_feature.py
class NewFeature:
    """새로운 기능 - 최소 구현"""
    
    def process(self, data, settings):
        """데이터 처리 - 테스트를 통과시키는 최소 구현"""
        return {"processed": True}  # 최소 구현
```

### 3. REFACTOR (코드 개선)

```python
# src/components/new_feature.py
from typing import Dict, Any
import pandas as pd

class NewFeature:
    """새로운 기능 - 리팩토링된 버전"""
    
    def process(self, data: pd.DataFrame, settings: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 처리 - Blueprint 원칙 준수
        
        Args:
            data: 입력 데이터프레임
            settings: 처리 설정
            
        Returns:
            처리 결과 딕셔너리
        """
        # 실제 비즈니스 로직 구현
        processed_data = self._apply_processing_logic(data, settings)
        return {
            "processed": True,
            "rows_processed": len(processed_data),
            "processing_method": settings.get("method", "default")
        }
```

---

## 🏭 Factory 패턴 마스터하기

**Phase 4-4.5에서 완전 적용된 Factory 패턴 사용법**

### TestDataFactory 사용법

```python
from tests.factories.test_data_factory import TestDataFactory

# 분류 데이터 생성
classification_data = TestDataFactory.create_classification_data(
    n_samples=100,
    n_features=5,
    n_classes=2
)

# 회귀 데이터 생성  
regression_data = TestDataFactory.create_regression_data(
    n_samples=100,
    n_features=10,
    noise=0.1
)

# 복합 학습 데이터 생성
comprehensive_data = TestDataFactory.create_comprehensive_training_data(
    n_samples=200,
    include_categorical=True,
    missing_rate=0.05
)
```

### SettingsFactory 사용법

```python
from tests.factories.settings_factory import SettingsFactory

# 분류 작업 설정 생성
classification_settings = SettingsFactory.create_classification_settings("local")

# 회귀 작업 설정 생성
regression_settings = SettingsFactory.create_regression_settings("dev")

# 최소 설정 생성
minimal_settings = SettingsFactory.create_minimal_settings()

# 커스텀 설정 생성
custom_settings = SettingsFactory.create_custom_settings(
    task_type="classification",
    model_class="sklearn.ensemble.RandomForestClassifier",
    hyperparameters={"n_estimators": 50}
)
```

### MockComponentRegistry 사용법 (LRU 캐싱)

```python
from tests.mocks.component_registry import MockComponentRegistry

# Mock 컴포넌트 생성 (LRU 캐싱 적용)
augmenter = MockComponentRegistry.get_augmenter("pass_through")
preprocessor = MockComponentRegistry.get_preprocessor("simple_scaler")
model = MockComponentRegistry.get_model("classifier")
evaluator = MockComponentRegistry.get_evaluator("standard")

# 캐시 통계 확인 (Phase 4 고도화 기능)
stats = MockComponentRegistry.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Memory usage: {stats['memory_usage_kb']} KB")

# 캐시 리셋 (필요시)
MockComponentRegistry.reset_all()
```

---

## ⚡ 테스트 실행 전략 (Phase 4 최적화)

**77% 성능 향상 달성 - 최적화된 테스트 실행 전략**

### 1. 빠른 개발용 (핵심 테스트만 - 3.00초)

```bash
# 핵심 컴포넌트만 테스트 - 최고 속도
uv run pytest -m "core and unit" -v

# 병렬 실행으로 더 빠르게
uv run pytest -m "core and unit" -n auto -v
```

### 2. 표준 CI (기본 스위트)

```bash
# slow/integration 제외한 모든 단위 테스트
uv run pytest -q -m "not slow and not integration"

# 커버리지와 함께
uv run pytest --cov=src --cov-report=term-missing tests/unit/
```

### 3. 완전한 테스트 (전체)

```bash
# 모든 테스트 실행
uv run pytest tests/

# 성능 최적화 병렬 실행
uv run pytest -n auto tests/unit/ -v
```

### 4. 마커별 실행

```bash
# Blueprint 원칙 테스트만
uv run pytest -m "blueprint_principle_1"

# 핵심 기능 테스트만
uv run pytest -m "core"

# 느린 테스트만 (CI에서)
uv run pytest -m "slow"
```

### 5. 성과 검증

```bash
# Phase 4-4.5 종합 성과 검증
./scripts/verify_test_coverage.sh

# 테스트 품질 검증 (Phase 5.2)
uv run pytest tests/meta/test_quality_validator.py -v
```

---

## 📝 코딩 가이드라인

### Python 스타일 (PEP8 + 프로젝트 규칙)

```python
from typing import Dict, List, Optional, Any
import pandas as pd

class ExampleComponent:
    """Example 컴포넌트 - Google Style Docstring"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """컴포넌트 초기화
        
        Args:
            config: 설정 딕셔너리
            
        Raises:
            ValueError: 잘못된 설정값인 경우
        """
        self.config = config
        self._validate_config()
    
    def process_data(
        self, 
        data: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """데이터 처리 메서드
        
        Args:
            data: 입력 데이터프레임
            target_col: 타겟 컬럼명 (선택사항)
            
        Returns:
            처리된 데이터프레임
            
        Example:
            >>> component = ExampleComponent({"method": "standard"})
            >>> result = component.process_data(df, "target")
            >>> len(result) > 0
            True
        """
        # 구현 로직
        return data.copy()
    
    def _validate_config(self) -> None:
        """설정 검증 - 비공개 메서드"""
        if "method" not in self.config:
            raise ValueError("method는 필수 설정입니다")
```

### 테스트 작성 규칙

```python
import pytest
from tests.factories.test_data_factory import TestDataFactory

@pytest.mark.unit
@pytest.mark.core  # 핵심 기능은 core 마커
class TestExampleComponent:
    """Example 컴포넌트 테스트"""
    
    def test_process_data_should_return_dataframe_when_valid_input_given(self, test_factories):
        """test_<컴포넌트>_should_<행동>_when_<조건> 명명 규칙"""
        # Given: Factory 패턴으로 테스트 데이터 준비
        data = test_factories['data'].create_classification_data(n_samples=10)
        config = {"method": "standard"}
        component = ExampleComponent(config)
        
        # When: 메서드 실행
        result = component.process_data(data)
        
        # Then: 결과 검증
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert result.columns.tolist() == data.columns.tolist()
    
    def test_init_should_raise_error_when_invalid_config_given(self):
        """에러 케이스 테스트"""
        # Given: 잘못된 설정
        invalid_config = {}  # method 누락
        
        # When & Then: 예외 발생 확인
        with pytest.raises(ValueError, match="method는 필수 설정입니다"):
            ExampleComponent(invalid_config)
```

---

## 🎯 첫 번째 기여

### 1. 이슈 선택

```bash
# Good First Issues 라벨 확인
# https://github.com/wooshikwon/modern-ml-pipeline/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22

# 브랜치 생성
git checkout -b feature/add-new-processor
```

### 2. TDD로 구현

```bash
# 1. RED: 실패하는 테스트 작성
uv run pytest tests/unit/components/test_new_processor.py::TestNewProcessor::test_process -v
# FAILED (예상됨)

# 2. GREEN: 최소 구현으로 테스트 통과
# src/components/new_processor.py 구현

uv run pytest tests/unit/components/test_new_processor.py::TestNewProcessor::test_process -v
# PASSED

# 3. REFACTOR: 코드 개선
# 리팩토링 후 테스트 재실행
uv run pytest -m "core and unit" -v  # 빠른 검증
```

### 3. 코드 품질 확인

```bash
# 정적 검사 및 포매팅
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src

# 또는 pre-commit으로 한 번에
uv run pre-commit run --all-files
```

### 4. 커밋 및 PR

```bash
# 커밋 (Conventional Commits + Task ID)
git add .
git commit -m "feat(components): add new processor for data normalization (P05-4)"

# PR 생성
git push origin feature/add-new-processor
# GitHub에서 PR 생성
```

---

## 🔧 트러블슈팅

### 1. 테스트 실행 오류

**문제: pytest 실행 시 import 오류**
```bash
# 해결: PYTHONPATH 설정
export PYTHONPATH=$PWD/src:$PYTHONPATH
uv run pytest tests/unit/ -v
```

**문제: Factory 패턴 관련 오류**
```bash
# 해결: Factory 의존성 확인
uv run pytest tests/unit/factories/ -v  # Factory 자체 테스트
```

### 2. 성능 문제

**문제: 테스트가 너무 느림**
```bash
# 해결: 핵심 테스트만 실행
uv run pytest -m "core and unit" -v  # 3.00초

# 병렬 실행으로 최적화
uv run pytest -n auto tests/unit/ -v
```

**문제: Mock Registry 캐시 이슈**
```python
# 해결: 캐시 리셋
from tests.mocks.component_registry import MockComponentRegistry
MockComponentRegistry.reset_all()
```

### 3. 환경 설정 오류

**문제: MLflow 연결 실패**
```bash
# 해결: 로컬 파일 모드로 전환
echo "MLFLOW_TRACKING_URI=./mlruns" >> .env

# 또는 Graceful Degradation 확인
uv run python main.py system-check
```

**문제: 의존성 충돌**
```bash
# 해결: 환경 재구성
rm -rf .venv
uv sync
```

### 4. 도움 요청

**문제 해결이 어려운 경우:**

1. **GitHub Issues**: 버그 리포트 또는 질문 이슈 생성
2. **Pull Request**: Draft PR로 코드 리뷰 요청  
3. **Documentation**: CLAUDE.md, BLUEPRINT.md 참조
4. **Test Quality**: `uv run pytest tests/meta/` 품질 검증

---

## 📚 추가 학습 자료

### 필수 문서
- **[CLAUDE.md](../CLAUDE.md)**: Vibe Coding 프로젝트 지침
- **[BLUEPRINT.md](../.claude/BLUEPRINT.md)**: 시스템 설계 원칙
- **[TEST_STABILIZATION_PLAN.md](../.claude/TEST_STABILIZATION_PLAN.md)**: Phase 4-5 테스트 성과

### 실습 과제
1. **첫 번째 테스트**: Factory 패턴으로 간단한 컴포넌트 테스트 작성
2. **TDD 실습**: RED-GREEN-REFACTOR 사이클로 기능 구현
3. **성능 최적화**: 핵심 테스트 마커 적용 연습

### 고급 주제
- **Mock Registry LRU 캐싱**: 메모리 최적화 심화 학습
- **Meta Testing**: 테스트 품질 자동 검증 시스템 이해
- **Session-scoped Fixtures**: 성능 최적화 고급 기법

---

**🎉 축하합니다! 이제 Modern ML Pipeline의 개발자가 되었습니다.**

Phase 4-4.5 성과 (77% 성능 향상, 100% 테스트 안정화)를 바탕으로 더 나은 MLOps 플랫폼을 함께 만들어가세요!