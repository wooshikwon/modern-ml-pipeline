# Model.computed 필드 이슈 해결 방안

## 문제 분석

### 현재 상황
1. **스키마 정의**: `Model` 클래스(src/settings/recipe.py)에 `computed` 필드가 정의되지 않음
2. **동적 추가**: `SettingsFactory`가 런타임에 `computed` 필드를 동적으로 추가
3. **파이프라인 의존성**: `run_train_pipeline`이 `settings.recipe.model.computed["run_name"]` 접근
4. **테스트 문제**: 단위 테스트에서 전체 파이프라인 실행 없이 이 필드에 접근 시 오류

### 코드 분석
```python
# src/settings/recipe.py
class Model(BaseModel):
    class_path: str = Field(...)
    library: str = Field(...)
    hyperparameters: HyperparametersTuning
    calibration: Optional[Calibration] = Field(None)
    # computed 필드 없음!

# src/settings/factory.py
def _add_training_computed_fields(self, settings: Settings, ...):
    if not hasattr(settings.recipe.model, 'computed'):
        settings.recipe.model.computed = {}  # 동적으로 추가
    settings.recipe.model.computed.update({
        "run_name": run_name,
        "environment": settings.config.environment.name,
        ...
    })
```

## 해결 방안

### 방안 1: Test Helper Function (권장) ⭐
```python
# tests/conftest.py에 추가
def add_computed_fields_for_training(settings: Settings,
                                    recipe_path: str = "test_recipe.yaml") -> Settings:
    """테스트용 computed 필드 추가 헬퍼"""
    from datetime import datetime
    from pathlib import Path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recipe_name = Path(recipe_path).stem

    # SettingsFactory와 동일한 로직
    if not hasattr(settings.recipe.model, 'computed'):
        settings.recipe.model.computed = {}

    settings.recipe.model.computed.update({
        "run_name": f"{recipe_name}_{timestamp}",
        "environment": settings.config.environment.name,
        "recipe_file": recipe_path
    })

    return settings

# 사용 예시
def test_train_command(component_test_context, isolated_temp_directory):
    # Settings 생성
    settings = SettingsFactory.for_training(...)

    # Computed 필드 추가
    settings = add_computed_fields_for_training(settings, recipe_path)

    # 이제 settings.recipe.model.computed["run_name"] 접근 가능
```

**장점:**
- ✅ SettingsFactory 로직과 일관성 유지
- ✅ 재사용 가능한 단일 함수
- ✅ 테스트 가독성 향상
- ✅ 실제 파이프라인 동작과 유사

**단점:**
- ⚠️ SettingsFactory 로직 중복

### 방안 2: Fixture로 자동 추가
```python
# tests/conftest.py
@pytest.fixture
def settings_with_computed(settings_builder):
    """computed 필드가 추가된 Settings 반환"""
    def _create_settings(**kwargs):
        settings = settings_builder.build()

        # 자동으로 computed 필드 추가
        if not hasattr(settings.recipe.model, 'computed'):
            settings.recipe.model.computed = {}

        settings.recipe.model.computed.update({
            "run_name": f"test_run_{uuid.uuid4().hex[:8]}",
            "environment": settings.config.environment.name,
            "seed": 42
        })

        return settings

    return _create_settings

# 사용 예시
def test_something(settings_with_computed):
    settings = settings_with_computed()
    assert settings.recipe.model.computed["run_name"]  # OK
```

**장점:**
- ✅ 자동화된 처리
- ✅ 모든 테스트에서 일관된 방식

**단점:**
- ⚠️ 추가 fixture 필요
- ⚠️ 기존 settings_builder와 혼동 가능

### 방안 3: SettingsFactory 메서드 직접 호출
```python
def test_train_command(isolated_temp_directory):
    # Settings 생성
    factory = SettingsFactory()
    settings = factory.for_training(
        recipe_path=str(recipe_path),
        config_path=str(config_path),
        data_path=str(data_path)
    )

    # 이미 computed 필드가 추가되어 있음
    # SettingsFactory.for_training이 내부적으로 _add_training_computed_fields 호출
```

**장점:**
- ✅ 실제 코드와 동일한 경로
- ✅ 추가 코드 불필요

**단점:**
- ⚠️ 전체 SettingsFactory 로직 실행 (느림)
- ⚠️ 파일 I/O 필요

### 방안 4: Mock computed 필드만 추가
```python
def test_train_command():
    with patch('src.cli.commands.train_command.run_train_pipeline') as mock_pipeline:
        # Settings는 실제로 생성
        settings = SettingsFactory.for_training(...)

        # computed 필드만 Mock으로 추가
        with patch.object(settings.recipe.model, 'computed', {
            'run_name': 'test_run',
            'environment': 'test'
        }):
            result = runner.invoke(app, [...])
```

**장점:**
- ✅ 최소한의 Mocking
- ✅ 명확한 의도

**단점:**
- ⚠️ Mock 사용 (No Mock Hell 원칙 위배)
- ⚠️ 중첩된 patch 복잡도

### 방안 5: Model 클래스 수정 (소스 코드 변경)
```python
# src/settings/recipe.py
from pydantic import ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(extra='allow')  # 동적 필드 허용

    class_path: str = Field(...)
    library: str = Field(...)
    hyperparameters: HyperparametersTuning
    calibration: Optional[Calibration] = Field(None)
    computed: Optional[Dict[str, Any]] = Field(default_factory=dict)  # 또는 이렇게
```

**장점:**
- ✅ 근본적 해결
- ✅ 타입 안정성
- ✅ IDE 자동완성 지원

**단점:**
- ⚠️ 소스 코드 변경 필요
- ⚠️ 기존 동작 변경 위험

## 권장 구현

### 1단계: Helper Function 추가 (즉시 적용 가능)
```python
# tests/conftest.py에 추가
@pytest.fixture
def add_model_computed():
    """Model에 computed 필드를 추가하는 헬퍼"""
    def _add_computed(settings: Settings,
                     run_name: str = None,
                     environment: str = None) -> Settings:
        if not hasattr(settings.recipe.model, 'computed'):
            settings.recipe.model.computed = {}

        settings.recipe.model.computed.update({
            "run_name": run_name or f"test_run_{uuid.uuid4().hex[:8]}",
            "environment": environment or settings.config.environment.name,
            "seed": 42,
            "recipe_file": "test_recipe.yaml"
        })

        return settings

    return _add_computed
```

### 2단계: CLI 테스트에 적용
```python
def test_train_command_with_required_arguments(
    component_test_context,
    isolated_temp_directory,
    add_model_computed  # 새 fixture
):
    # ... recipe, config 파일 생성 ...

    with patch('src.cli.commands.train_command.run_train_pipeline') as mock_pipeline:
        # SettingsFactory 호출 전 computed 필드 처리를 위한 patch
        original_for_training = SettingsFactory.for_training

        def for_training_with_computed(*args, **kwargs):
            settings = original_for_training(*args, **kwargs)
            return add_model_computed(settings)

        with patch.object(SettingsFactory, 'for_training', for_training_with_computed):
            result = self.runner.invoke(self.app, [...])
```

### 3단계: 다른 파이프라인 테스트에도 적용
```python
# tests/unit/pipelines/test_train_pipeline.py
def test_train_pipeline(settings_builder, add_model_computed):
    settings = settings_builder.with_task("classification").build()
    settings = add_model_computed(settings)  # computed 필드 추가

    # 이제 안전하게 접근 가능
    assert settings.recipe.model.computed["run_name"]
```

## 장기 개선 방안

### 소스 코드 개선 제안
1. **Model 클래스에 computed 필드 추가**
   ```python
   class Model(BaseModel):
       # ... 기존 필드들 ...
       computed: Optional[Dict[str, Any]] = Field(
           default=None,
           description="런타임 계산 필드"
       )
   ```

2. **SettingsBuilder에 computed 지원 추가**
   ```python
   class SettingsBuilder:
       def with_computed_fields(self, **fields) -> "SettingsBuilder":
           if self._model.computed is None:
               self._model.computed = {}
           self._model.computed.update(fields)
           return self
   ```

3. **Factory 패턴 개선**
   - computed 필드 초기화를 Settings 생성 시점으로 이동
   - 테스트 모드 플래그 추가

## 결론

### 즉시 적용 방안
**방안 1 (Test Helper Function)** 적용 권장:
- 최소한의 변경으로 즉시 해결 가능
- tests/README.md의 "Real object testing" 원칙 준수
- 다른 테스트에도 쉽게 확장 가능

### 실행 계획
1. `tests/conftest.py`에 `add_model_computed` fixture 추가
2. `test_train_command.py`에 fixture 적용
3. 다른 CLI 테스트 파일 변환 시 동일 패턴 적용
4. 통합/E2E 테스트는 실제 SettingsFactory 사용

### 검증 기준
- ✅ Mock 사용 최소화
- ✅ 실제 Settings 객체 사용
- ✅ computed 필드 접근 오류 해결
- ✅ 테스트 가독성 유지