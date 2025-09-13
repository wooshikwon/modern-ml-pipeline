# 최종 CLI Settings 통합 및 아키텍처 재설계 계획

## 1. 현재 상황 재분석 및 추가 발견사항

### 1.1 기존 문제점 재확인

🚨 **핵심 문제들**:
1. **serve-api/batch-inference의 하드코딩된 더미 Recipe 사용**
2. **MLflow에 학습시 Recipe 전체가 저장되지 않음**
3. **train은 60+ 줄, serve/inference는 3줄로 CLI 일관성 부재**
4. **추론시 학습 환경을 완전히 재현할 수 없는 구조**

### 1.2 새로 발견된 아키텍처 문제들

#### 1.2.1 Pydantic model_config의 하드코딩 문제

**현재 상태**:
```python
# src/settings/recipe.py (420줄)
model_config = ConfigDict(
    json_schema_extra={
        "example": {
            "name": "classification_rf",
            "task_choice": "classification",
            # ... 50+ 줄의 하드코딩된 예제 데이터
```

**문제점**:
- 예제 데이터가 Pydantic 클래스 내부에 하드코딩됨
- 테스트 데이터와 중복되어 유지보수 어려움
- 실제 사용 패턴과 동기화되지 않음

#### 1.2.2 검증 로직의 책임 혼재

**현재 상태**:
```python
# settings/recipe.py에 8개의 복잡한 검증자
@field_validator('fixed', 'tunable')
@model_validator(mode='after')
# ...

# settings/loader.py에 런타임 검증
def validate_data_source_compatibility(self):
    # 파일 확장자 기반 어댑터 타입 추론 로직
```

**문제점**:
- 파일 확장자 어댑터 매핑은 Factory의 책임
- Pydantic 검증과 비즈니스 로직 검증이 혼재
- models/catalog의 우수한 하이퍼파라미터 구조를 활용하지 못함

#### 1.2.3 Recipe Template의 불일치

**현재 상태**:
```yaml
# src/cli/templates/recipes/recipe.yaml.j2:49
data:
  loader:
    # source_uri는 레시피에서 관리하지 않습니다. CLI --data-path로만 주입됩니다.
```

**문제점**:
- 주석으로만 명시되고 실제 구조는 빈 상태
- CLI --data-path 방식과 template 구조 불일치
- 학습-추론 간 데이터 경로 처리 방식 다름

## 2. 완전 재설계 아키텍처

### 2.1 레이어별 책임 재정의

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Layer                               │
│ 역할: 파라미터 파싱, 에러 핸들링, 사용자 인터페이스           │
│ 파일: src/cli/commands/*.py                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Settings Factory Layer                     │
│ 역할: 통합 Settings 생성, MLflow Recipe 복원               │
│ 파일: src/settings/factory.py                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Pydantic Schema Layer                       │
│ 역할: 데이터 구조 정의, 기본 타입 검증                       │
│ 파일: src/settings/config.py, recipe.py                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Validation Layer                            │
│ 역할: 비즈니스 로직 검증, 동적 카탈로그 검증                 │
│ 파일: src/settings/validation/*.py                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Factory Layer                             │
│ 역할: 컴포넌트 생성, 런타임 검증, 어댑터 매핑               │
│ 파일: src/factory/*.py                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Registry 기반 동적 시스템 (핵심 혁신)

#### 2.2.1 Components Registry 연동 시스템

```python
# src/settings/validation/registry_connector.py (신규)
class RegistryConnector:
    """Components Registry와 CLI/Validation 연동"""

    def __init__(self):
        self._trigger_component_imports()  # Registry 활성화

    def _trigger_component_imports(self):
        """모든 컴포넌트를 import하여 Registry 활성화"""
        import src.components.preprocessor
        import src.components.evaluator
        import src.components.adapter
        import src.components.trainer
        import src.components.fetcher
        import src.components.datahandler
        import src.components.calibration

    def get_available_preprocessors(self) -> Dict[str, Dict[str, str]]:
        """Registry에서 실제 사용 가능한 preprocessor 목록 가져오기"""
        from src.components.preprocessor.registry import PreprocessorStepRegistry

        # Registry에서 동적으로 가져오기
        available_steps = PreprocessorStepRegistry.preprocessor_steps.keys()

        # 카테고리별로 분류 (실제 Registry 기반)
        categorized = {
            "Missing Value Handling": {},
            "Encoder": {},
            "Feature Engineering": {},
            "Scaler": {}
        }

        for step_type in available_steps:
            step_class = PreprocessorStepRegistry.preprocessor_steps[step_type]
            description = self._get_step_description(step_class)

            # 타입별 카테고리 분류 (클래스 이름 기반)
            category = self._categorize_step(step_type, step_class)
            if category in categorized:
                categorized[category][step_type] = description

        return {k: v for k, v in categorized.items() if v}  # 빈 카테고리 제거

    def get_available_metrics_by_task(self) -> Dict[str, List[str]]:
        """Registry에서 Task별 사용 가능한 metric 목록 가져오기"""
        from src.components.evaluator.registry import EvaluatorRegistry

        metrics_by_task = {}

        # 등록된 모든 task 순회
        for task_type in EvaluatorRegistry.get_available_tasks():
            evaluator_class = EvaluatorRegistry.get_evaluator_class(task_type)

            # Evaluator 클래스에서 지원하는 metric 추출
            metrics = self._extract_metrics_from_evaluator(evaluator_class)
            metrics_by_task[task_type] = metrics

        return metrics_by_task

    def _extract_metrics_from_evaluator(self, evaluator_class) -> List[str]:
        """Evaluator 클래스에서 지원하는 metric 추출"""
        # Evaluator 클래스의 METRICS 속성 또는 메서드 분석
        if hasattr(evaluator_class, 'SUPPORTED_METRICS'):
            return evaluator_class.SUPPORTED_METRICS
        elif hasattr(evaluator_class, 'get_available_metrics'):
            return evaluator_class.get_available_metrics()
        else:
            # 기본 분석: evaluate 메서드의 파라미터나 내부 로직 분석
            return self._analyze_evaluator_metrics(evaluator_class)

    def _analyze_evaluator_metrics(self, evaluator_class) -> List[str]:
        """Evaluator 클래스 분석을 통한 지원 메트릭 추론"""
        import inspect

        # 1. 클래스의 상수 또는 속성에서 메트릭 찾기
        for attr_name in dir(evaluator_class):
            if 'METRIC' in attr_name.upper():
                attr_value = getattr(evaluator_class, attr_name)
                if isinstance(attr_value, (list, tuple)):
                    return list(attr_value)
                elif isinstance(attr_value, dict):
                    return list(attr_value.keys())

        # 2. evaluate 메서드 시그니처 분석
        if hasattr(evaluator_class, 'evaluate'):
            sig = inspect.signature(evaluator_class.evaluate)
            for param_name, param in sig.parameters.items():
                if 'metric' in param_name.lower() and param.annotation != inspect.Parameter.empty:
                    # Type hint에서 메트릭 추출 시도
                    if hasattr(param.annotation, '__args__'):
                        return list(param.annotation.__args__)

        # 3. 기본값 반환 (Task별 일반적인 메트릭)
        class_name = evaluator_class.__name__.lower()
        if 'classification' in class_name:
            return ['accuracy', 'precision', 'recall', 'f1']
        elif 'regression' in class_name:
            return ['mse', 'mae', 'r2']
        else:
            return ['custom_metric']

    def _get_step_description(self, step_class) -> str:
        """전처리 단계 클래스에서 설명 추출"""
        # 1. 클래스 docstring에서 설명 추출
        if step_class.__doc__:
            # 첫 번째 줄만 간단한 설명으로 사용
            first_line = step_class.__doc__.strip().split('\n')[0]
            if first_line and len(first_line) < 100:
                return first_line

        # 2. 클래스명 기반 기본 설명
        class_name = step_class.__name__

        # 일반적인 패턴 매칭
        if 'Imputer' in class_name:
            return f"{class_name}: 결측값을 처리합니다"
        elif 'Encoder' in class_name:
            return f"{class_name}: 범주형 데이터를 인코딩합니다"
        elif 'Scaler' in class_name:
            return f"{class_name}: 수치형 데이터를 정규화합니다"
        elif 'Discretizer' in class_name:
            return f"{class_name}: 연속형 데이터를 구간으로 나눕니다"
        elif 'Feature' in class_name:
            return f"{class_name}: 피처를 생성하거나 변환합니다"
        else:
            return f"{class_name}: 데이터 전처리를 수행합니다"

    def _categorize_step(self, step_type: str, step_class) -> str:
        """전처리 단계의 카테고리 분류"""
        # 클래스명이나 타입명 기반 분류
        step_name = step_type.lower()

        if 'imputer' in step_name or 'missing' in step_name:
            return "Missing Value Handling"
        elif 'encoder' in step_name:
            return "Encoder"
        elif 'scaler' in step_name:
            return "Scaler"
        elif any(x in step_name for x in ['feature', 'polynomial', 'discretizer']):
            return "Feature Engineering"
        else:
            return "Other"

# src/settings/validation/catalog_validator.py (강화)
class CatalogValidator:
    """Models Catalog + Registry 기반 동적 검증"""

    def __init__(self, catalog_path: str = "src/models/catalog"):
        self.catalog_path = Path(catalog_path)
        self.registry_connector = RegistryConnector()
        self._catalog_cache = {}

    def validate_preprocessor_steps(self, steps: List[Dict]) -> ValidationResult:
        """Registry 기반 전처리 단계 검증"""
        from src.components.preprocessor.registry import PreprocessorStepRegistry

        available_steps = set(PreprocessorStepRegistry.preprocessor_steps.keys())

        for step in steps:
            step_type = step.get('type')
            if step_type not in available_steps:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unknown preprocessor step: '{step_type}'. "
                                f"Available: {sorted(available_steps)}"
                )

        return ValidationResult(is_valid=True)

    def validate_metrics(self, task_type: str, metrics: List[str]) -> ValidationResult:
        """Registry 기반 metric 검증"""
        available_metrics = self.registry_connector.get_available_metrics_by_task()
        task_metrics = available_metrics.get(task_type, [])

        invalid_metrics = [m for m in metrics if m not in task_metrics]

        if invalid_metrics:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid metrics for {task_type}: {invalid_metrics}. "
                            f"Available: {task_metrics}"
            )

        return ValidationResult(is_valid=True)
```

#### 2.2.2 CLI Builder와 Validator 동기화

```python
# src/cli/utils/recipe_builder.py (개선)
class RecipeBuilder:
    """Registry 기반 동적 Recipe 빌더"""

    def __init__(self):
        self.ui = InteractiveUI()
        self.template_engine = TemplateEngine(templates_dir)
        self.registry_connector = RegistryConnector()  # 🆕 Registry 연동

    def _collect_preprocessor_steps(self) -> List[Dict[str, Any]]:
        """Registry에서 동적으로 전처리 옵션 생성"""

        # 🆕 하드코딩 제거 - Registry에서 동적으로 가져오기
        available_preprocessors = self.registry_connector.get_available_preprocessors()

        preprocessor_steps = []

        # Registry 기반 동적 처리
        for category, preprocessors in available_preprocessors.items():
            if not preprocessors:  # 빈 카테고리 스킵
                continue

            if not self.ui.confirm(f"\n{category} 전처리를 사용하시겠습니까?",
                                 default=category in ["Missing Value Handling", "Encoder", "Scaler"]):
                continue

            # 카테고리 내 각 전처리기 선택
            selected_preprocessors = []
            for step_type, description in preprocessors.items():
                if self.ui.confirm(f"  {description}를 사용하시겠습니까?", default=True):
                    # 각 전처리기의 파라미터 설정
                    step_config = self._configure_preprocessor_params(step_type)
                    step_config['type'] = step_type
                    selected_preprocessors.append(step_config)

            preprocessor_steps.extend(selected_preprocessors)

        return preprocessor_steps

    def _configure_preprocessor_params(self, step_type: str) -> Dict[str, Any]:
        """전처리기별 파라미터 설정"""
        from src.components.preprocessor.registry import PreprocessorStepRegistry

        step_class = PreprocessorStepRegistry.preprocessor_steps.get(step_type)
        if not step_class:
            return {}

        # step_class의 __init__ 파라미터 분석하여 사용자 입력 받기
        import inspect
        sig = inspect.signature(step_class.__init__)

        config = {}
        for param_name, param in sig.parameters.items():
            if param_name in ['self']:
                continue

            # 기본값이 있는 파라미터만 설정 가능하도록 제한
            if param.default != inspect.Parameter.empty:
                default_value = param.default

                # 타입에 따른 입력 받기
                if isinstance(default_value, bool):
                    config[param_name] = self.ui.confirm(f"    {param_name}", default=default_value)
                elif isinstance(default_value, (int, float)):
                    user_input = self.ui.prompt(f"    {param_name} (기본값: {default_value})")
                    if user_input.strip():
                        config[param_name] = type(default_value)(user_input)
                    else:
                        config[param_name] = default_value
                elif isinstance(default_value, str):
                    user_input = self.ui.prompt(f"    {param_name} (기본값: '{default_value}')")
                    config[param_name] = user_input if user_input.strip() else default_value

        return config

    def _get_available_metrics(self, task_type: str) -> List[str]:
        """Registry에서 Task별 사용 가능한 metric 가져오기"""
        metrics_by_task = self.registry_connector.get_available_metrics_by_task()
        return metrics_by_task.get(task_type, [])

    def _collect_evaluation_metrics(self, task_type: str) -> List[str]:
        """Registry 기반 동적 메트릭 선택"""
        available_metrics = self._get_available_metrics(task_type)

        if not available_metrics:
            self.ui.warning(f"Task '{task_type}'에 대한 메트릭을 찾을 수 없습니다.")
            return ['accuracy'] if task_type == 'classification' else ['mse']

        self.ui.info(f"\n{task_type} Task에 사용 가능한 메트릭:")
        selected_metrics = []

        for metric in available_metrics:
            if self.ui.confirm(f"  {metric}를 사용하시겠습니까?",
                             default=metric in ['accuracy', 'mse', 'mae']):
                selected_metrics.append(metric)

        # 최소 하나의 메트릭은 선택되어야 함
        if not selected_metrics:
            selected_metrics = [available_metrics[0]]
            self.ui.info(f"기본 메트릭 '{available_metrics[0]}'이 선택되었습니다.")

        return selected_metrics

# src/settings/validation/business_validator.py (개선)
class BusinessValidator:
    """Registry 동기화된 비즈니스 검증"""

    def __init__(self):
        self.registry_connector = RegistryConnector()
        self.catalog_validator = CatalogValidator()

    def validate_recipe_components(self, recipe: Recipe) -> ValidationResult:
        """Recipe의 모든 컴포넌트를 Registry 기반으로 검증"""

        # 1. Preprocessor 검증
        if recipe.preprocessor and recipe.preprocessor.steps:
            result = self.catalog_validator.validate_preprocessor_steps(recipe.preprocessor.steps)
            if not result.is_valid:
                return result

        # 2. Metrics 검증
        result = self.catalog_validator.validate_metrics(recipe.task_choice, recipe.evaluation.metrics)
        if not result.is_valid:
            return result

        # 3. 기타 컴포넌트 검증...

        return ValidationResult(is_valid=True)

    def validate_manual_recipe_edits(self, recipe: Recipe) -> ValidationResult:
        """사용자가 Recipe를 수기 수정한 경우의 검증"""

        # Registry 기반 전체 검증
        result = self.validate_recipe_components(recipe)
        if not result.is_valid:
            # 상세한 에러 메시지 제공
            available_preprocessors = self.registry_connector.get_available_preprocessors()
            available_metrics = self.registry_connector.get_available_metrics_by_task()

            enhanced_message = result.error_message + "\n\n사용 가능한 옵션:\n"

            if "preprocessor" in result.error_message.lower():
                enhanced_message += "전처리기:\n"
                for category, preprocessors in available_preprocessors.items():
                    enhanced_message += f"  {category}: {list(preprocessors.keys())}\n"

            if "metric" in result.error_message.lower():
                enhanced_message += f"메트릭 ({recipe.task_choice}): {available_metrics.get(recipe.task_choice, [])}\n"

            return ValidationResult(is_valid=False, error_message=enhanced_message)

        return result
```

### 2.3 완전 재설계된 파일 구조

```
src/settings/
├── config.py              # Config Pydantic 모델만 (검증 로직 제거)
├── recipe.py              # Recipe Pydantic 모델만 (검증 로직 제거)
├── factory.py             # SettingsFactory (핵심 통합 로직)
├── loader.py              # 기존 함수들 (단순화됨)
├── validation/            # 검증 로직 전담 (신규)
│   ├── __init__.py
│   ├── catalog_validator.py    # Models Catalog 기반 검증
│   ├── business_validator.py   # 비즈니스 로직 검증
│   └── compatibility_validator.py  # Config-Recipe 호환성 검증
├── mlflow_restore.py      # MLflow Recipe 복원 (신규)
└── fixtures/              # 테스트 데이터 (신규)
    ├── __init__.py
    ├── config_examples.py      # Config 예제들
    └── recipe_examples.py      # Recipe 예제들
```

## 3. 단계별 상세 구현 계획

### 3.1 Phase 1: Pydantic 모델 순수화 (1주)

#### 3.1.1 model_config 하드코딩 제거

**Before**:
```python
# src/settings/recipe.py
class Recipe(BaseModel):
    # ... 필드 정의 ...

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                # 50+ 줄의 하드코딩된 예제
            }
        }
    )
```

**After**:
```python
# src/settings/recipe.py - 순수한 데이터 모델만
class Recipe(BaseModel):
    """Recipe 스키마 - 순수 데이터 구조"""
    name: str
    task_choice: TaskType
    model: Model
    data: Data
    # ... 다른 필드들 ...

    # 검증 로직 모두 제거

# src/settings/fixtures/recipe_examples.py - 예제 데이터 분리
CLASSIFICATION_RF_EXAMPLE = {
    "name": "classification_rf",
    "task_choice": "classification",
    # ... 예제 데이터
}

REGRESSION_XGB_EXAMPLE = {
    # ... 다른 예제
}
```

#### 3.1.2 검증 로직 완전 분리

**검증 책임 재배치**:
```python
# src/settings/validation/catalog_validator.py
class CatalogValidator:
    """Models Catalog 기반 동적 검증"""

    def validate_model_hyperparameters(self, model_config: Dict) -> ValidationResult:
        """카탈로그에서 스펙을 동적으로 로드하여 검증"""
        catalog_spec = self._load_from_catalog(model_config['class_path'])
        return self._validate_against_spec(model_config, catalog_spec)

# src/settings/validation/business_validator.py
class BusinessValidator:
    """비즈니스 로직 검증"""

    def validate_feature_store_compatibility(self, config: Config, recipe: Recipe) -> ValidationResult:
        """Feature Store 설정 호환성 검증"""

    def validate_task_compatibility(self, recipe: Recipe) -> ValidationResult:
        """Task와 모델 호환성 검증"""

# src/settings/validation/compatibility_validator.py
class CompatibilityValidator:
    """Config-Recipe 호환성 검증"""

    def validate_config_recipe_compatibility(self, config: Config, recipe: Recipe) -> ValidationResult:
        """Config와 Recipe 간 호환성 검증"""
```

#### 3.1.3 Recipe Template 정리

**Before**:
```yaml
# recipe.yaml.j2:47-49
data:
  loader:
    # source_uri는 레시피에서 관리하지 않습니다. CLI --data-path로만 주입됩니다.
```

**After**:
```yaml
# recipe.yaml.j2 - data.loader 섹션 완전 제거
data:
  # loader 섹션 삭제 - CLI --data-path로만 처리

  # Fetcher 설정 (Feature Store 연동)
  fetcher:
    type: {{ fetcher_type }}
    # ... 기존 로직 유지
```

### 3.2 Phase 2: SettingsFactory 및 MLflow 복원 시스템 (1-2주)

#### 3.2.1 통합 SettingsFactory 구현

```python
# src/settings/factory.py
class SettingsFactory:
    """통합 Settings 생성 팩토리"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.compatibility_validator = CompatibilityValidator()

    @classmethod
    def for_training(cls, recipe_path: str, config_path: str,
                    data_path: str, context_params: Optional[Dict] = None) -> Settings:
        """학습용 Settings 생성"""
        factory = cls()

        # 1. 기본 로드
        config = factory._load_config(config_path)
        recipe = factory._load_recipe(recipe_path)

        # 2. 데이터 경로 처리 (기존 train_command 로직 이관)
        factory._process_training_data_path(recipe, data_path, context_params)

        # 3. 검증 실행 (레이어별 분리)
        factory._validate_all(config, recipe)

        # 4. Settings 생성
        settings = Settings(config, recipe)
        factory._add_training_computed_fields(settings, recipe_path)

        return settings

    @classmethod
    def for_inference(cls, config_path: str, run_id: str) -> Settings:
        """추론용 Settings 생성 - MLflow Recipe 복원"""
        factory = cls()

        # 1. 현재 Config 로드
        config = factory._load_config(config_path)

        # 2. MLflow에서 학습시 Recipe 복원
        recipe_restorer = MLflowRecipeRestorer(run_id)
        recipe = recipe_restorer.restore_recipe()

        # 3. 검증 실행 (추론용)
        factory._validate_for_inference(config, recipe)

        # 4. Settings 생성
        settings = Settings(config, recipe)
        factory._add_inference_computed_fields(settings, run_id)

        return settings

    def _validate_all(self, config: Config, recipe: Recipe) -> None:
        """레이어별 검증 실행"""
        # 1. Catalog 기반 하이퍼파라미터 검증
        self.catalog_validator.validate_model_hyperparameters(recipe.model.model_dump())

        # 2. 비즈니스 로직 검증
        self.business_validator.validate_feature_store_compatibility(config, recipe)
        self.business_validator.validate_task_compatibility(recipe)

        # 3. Config-Recipe 호환성 검증
        self.compatibility_validator.validate_config_recipe_compatibility(config, recipe)

        # 파일 확장자 기반 어댑터 검증은 Factory 레벨로 이관됨
```

#### 3.2.2 MLflow Recipe 복원 시스템

```python
# src/settings/mlflow_restore.py
class MLflowRecipeRestorer:
    """MLflow에서 학습시 Recipe 완전 복원"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = mlflow.tracking.MlflowClient()

    def restore_recipe(self) -> Recipe:
        """학습시 Recipe를 완전히 복원"""
        try:
            # 1. MLflow artifacts에서 recipe_snapshot.yaml 다운로드
            recipe_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/recipe_snapshot.yaml"
            )

            # 2. Recipe 객체로 복원
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = yaml.safe_load(f)

            # 3. 환경변수 치환 (현재 환경 기준)
            from src.settings.loader import resolve_env_variables
            recipe_data = resolve_env_variables(recipe_data)

            # 4. Recipe 객체 생성 (검증은 SettingsFactory에서)
            return Recipe(**recipe_data)

        except FileNotFoundError:
            # Legacy Run 감지 - 하위 호환성
            logger.warning(f"Recipe snapshot not found for run {self.run_id}. Using legacy fallback.")
            return self._create_legacy_recipe()
        except Exception as e:
            raise ValueError(f"Recipe 복원 실패 (run_id: {self.run_id}): {e}")

    def _create_legacy_recipe(self) -> Recipe:
        """기존 하드코딩 방식 fallback (임시)"""
        # 기존 create_settings_for_inference 로직을 여기로 이관
        # 점진적 마이그레이션을 위한 호환성 보장
```

### 3.3 Phase 3: Factory 레벨 어댑터 매핑 및 런타임 검증 (1주)

#### 3.3.1 어댑터 타입 매핑을 Factory로 이관

```python
# src/factory/adapter_mapper.py (신규)
class AdapterTypeMapper:
    """데이터 소스 기반 어댑터 타입 자동 매핑"""

    def __init__(self):
        self.mapping_rules = {
            'sql_patterns': ['.sql', 'select', 'from', 'where'],
            'storage_patterns': ['.csv', '.parquet', '.json', 's3://', 'gs://', 'az://'],
            'bigquery_patterns': ['bigquery://'],
        }

    def infer_adapter_type(self, source_uri: str) -> str:
        """소스 URI에서 어댑터 타입 추론"""
        source_lower = source_uri.lower()

        # SQL 패턴 검사
        if any(pattern in source_lower for pattern in self.mapping_rules['sql_patterns']):
            return 'sql'

        # Storage 패턴 검사
        if any(source_lower.startswith(pattern) or source_lower.endswith(pattern)
               for pattern in self.mapping_rules['storage_patterns']):
            return 'storage'

        # BigQuery 패턴 검사
        if any(source_lower.startswith(pattern) for pattern in self.mapping_rules['bigquery_patterns']):
            return 'bigquery'

        # 기본값
        return 'storage'

    def validate_adapter_compatibility(self, source_uri: str, config_adapter_type: str) -> bool:
        """설정된 어댑터 타입과 소스 URI 호환성 검증"""
        inferred_type = self.infer_adapter_type(source_uri)

        # SQL 계열 호환성
        if inferred_type == 'sql' and config_adapter_type in ['sql', 'bigquery']:
            return True

        # 정확한 매칭
        return inferred_type == config_adapter_type

# Factory에서 사용
class Factory(BaseFactory):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.adapter_mapper = AdapterTypeMapper()

    def create_data_adapter(self):
        # 런타임에 어댑터 호환성 검증
        source_uri = self.settings.recipe.data.loader.source_uri
        config_adapter_type = self.settings.config.data_source.adapter_type

        if not self.adapter_mapper.validate_adapter_compatibility(source_uri, config_adapter_type):
            inferred_type = self.adapter_mapper.infer_adapter_type(source_uri)
            logger.warning(f"어댑터 타입 불일치: 설정={config_adapter_type}, 추론={inferred_type}")
            # 자동 수정 또는 경고 후 계속 진행

        return super().create_data_adapter()
```

### 3.4 Phase 4: CLI 명령어 통합 및 레거시 정리 (1주)

#### 3.4.1 CLI 명령어 완전 통합

```python
# src/cli/commands/train_command.py - 대폭 단순화
def train_command(recipe_path: str, config_path: str, data_path: str,
                 context_params: Optional[str] = None, record_reqs: bool = False):
    """학습 파이프라인 실행 - 완전 단순화"""
    try:
        # 1. 파라미터 파싱
        params = json.loads(context_params) if context_params else None

        # 2. Settings 생성 (모든 복잡한 로직은 SettingsFactory에서)
        settings = SettingsFactory.for_training(
            recipe_path=recipe_path,
            config_path=config_path,
            data_path=data_path,
            context_params=params
        )
        setup_logging(settings)

        # 3. 메타데이터 전달 (MLflow 저장용)
        enhanced_params = params or {}
        enhanced_params.update({
            'original_recipe_path': recipe_path,
            'original_config_path': config_path,
            'original_data_path': data_path
        })

        # 4. 파이프라인 실행
        if record_reqs:
            run_train_pipeline(settings=settings, context_params=enhanced_params,
                             record_requirements=True)
        else:
            run_train_pipeline(settings=settings, context_params=enhanced_params)

        logger.info("✅ 학습이 성공적으로 완료되었습니다.")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"학습 실행 중 오류: {e}", exc_info=True)
        raise typer.Exit(code=1)

# src/cli/commands/serve_command.py - Recipe 복원 기반
def serve_api_command(run_id: str, config_path: str,
                     host: str = "0.0.0.0", port: int = 8000):
    """API 서버 실행 - 학습시 Recipe 완전 복원"""
    try:
        # Settings 생성 (MLflow Recipe 복원)
        settings = SettingsFactory.for_serving(config_path=config_path, run_id=run_id)
        setup_logging(settings)

        # 학습시 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Original Recipe: {settings.recipe.name}")
        logger.info(f"Task Type: {settings.recipe.task_choice}")
        logger.info(f"Model: {settings.recipe.model.class_path}")

        # API 서버 실행
        run_api_server(settings=settings, run_id=run_id, host=host, port=port)

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"API 서버 실행 중 오류: {e}", exc_info=True)
        raise typer.Exit(code=1)

# src/cli/commands/inference_command.py - 동일한 패턴
def batch_inference_command(run_id: str, config_path: str, data_path: str,
                           context_params: Optional[str] = None):
    """배치 추론 실행 - 학습시 Recipe 완전 복원"""
    try:
        # 파라미터 파싱
        params = json.loads(context_params) if context_params else None

        # Settings 생성 (MLflow Recipe 복원)
        settings = SettingsFactory.for_inference(config_path=config_path, run_id=run_id)
        setup_logging(settings)

        # 추론 정보 로깅
        logger.info(f"Config: {config_path}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Original Recipe: {settings.recipe.name}")
        logger.info(f"Task Type: {settings.recipe.task_choice}")

        # 배치 추론 실행
        run_inference_pipeline(
            settings=settings,
            run_id=run_id,
            data_path=data_path,
            context_params=params or {}
        )

        logger.info("✅ 배치 추론이 성공적으로 완료되었습니다.")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"설정 오류: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"배치 추론 실행 중 오류: {e}", exc_info=True)
        raise typer.Exit(code=1)
```

#### 3.4.2 레거시 코드 완전 정리

**삭제할 파일들**:
```bash
# 🗑️ 완전 삭제
src/factory/artifact.py                           # 하위호환성 라우터

# 🗑️ 함수 삭제
src/settings/loader.py:create_settings_for_inference()  # 더미 Recipe 생성
src/settings/loader.py:load_config_files()              # 중복 함수
src/settings/loader.py:validate_data_source_compatibility()  # Factory로 이관

# 🔄 리팩토링
src/settings/recipe.py                            # 검증 로직 모두 제거
src/settings/config.py                            # model_config 예제 데이터 제거
```

**Import 경로 정리**:
```python
# Before (삭제 예정)
from src.factory.artifact import PyfuncWrapper

# After (표준 경로로 통일)
from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper
```

## 4. 테스트 구조 통합 계획

### 4.1 Fixtures 기반 테스트 데이터 관리

```python
# src/settings/fixtures/config_examples.py
"""Config 예제 데이터 - Pydantic model_config에서 이관"""

LOCAL_DEV_CONFIG = {
    "environment": {"name": "local"},
    "mlflow": {
        "tracking_uri": "./mlruns",
        "experiment_name": "mmp-local"
    },
    "data_source": {
        "name": "PostgreSQL",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "postgresql://user:pass@localhost:5432/db"
        }
    }
}

PRODUCTION_CONFIG = {
    # ... 프로덕션 설정 예제
}

# src/settings/fixtures/recipe_examples.py
"""Recipe 예제 데이터 - Pydantic model_config에서 이관"""

CLASSIFICATION_RF_RECIPE = {
    "name": "classification_rf",
    "task_choice": "classification",
    "model": {
        "class_path": "sklearn.ensemble.RandomForestClassifier",
        "library": "sklearn",
        "hyperparameters": {
            "tuning_enabled": True,
            "fixed": {"random_state": 42},
            "tunable": {
                "n_estimators": {"type": "int", "range": [50, 200]},
                "max_depth": {"type": "int", "range": [5, 20]}
            }
        }
    },
    # ... 나머지 구조
}

# tests/fixtures/settings_fixtures.py
"""테스트용 Settings fixtures"""

@pytest.fixture
def sample_config():
    return Config(**LOCAL_DEV_CONFIG)

@pytest.fixture
def sample_recipe():
    return Recipe(**CLASSIFICATION_RF_RECIPE)

@pytest.fixture
def sample_settings(sample_config, sample_recipe):
    return Settings(sample_config, sample_recipe)
```

### 4.2 Catalog 기반 검증 테스트

```python
# tests/unit/settings/validation/test_catalog_validator.py
class TestCatalogValidator:
    """Models Catalog 기반 검증 테스트"""

    def test_valid_rf_hyperparameters(self, catalog_validator):
        """RandomForest 하이퍼파라미터 검증"""
        model_config = {
            "class_path": "sklearn.ensemble.RandomForestClassifier",
            "hyperparameters": {
                "tuning_enabled": True,
                "tunable": {
                    "n_estimators": {"type": "int", "range": [50, 200]},
                    "max_depth": {"type": "int", "range": [5, 20]}
                }
            }
        }

        result = catalog_validator.validate_model_hyperparameters(model_config)
        assert result.is_valid

    def test_invalid_hyperparameter_range(self, catalog_validator):
        """잘못된 하이퍼파라미터 범위 검증"""
        model_config = {
            "class_path": "sklearn.ensemble.RandomForestClassifier",
            "hyperparameters": {
                "tuning_enabled": True,
                "tunable": {
                    "n_estimators": {"type": "int", "range": [1000, 2000]}  # 카탈로그 범위 초과
                }
            }
        }

        result = catalog_validator.validate_model_hyperparameters(model_config)
        assert not result.is_valid
        assert "n_estimators range exceeds catalog limits" in result.error_message
```

## 5. MLflow 완전 저장/복원 시스템

### 5.1 학습시 완전한 저장

```python
# src/pipelines/train_pipeline.py 강화
def run_train_pipeline(settings: Settings, context_params: Optional[Dict] = None,
                      record_requirements: bool = False):
    """학습 파이프라인 + 완전한 재현성 저장"""

    # ... 기존 학습 로직 ...

    # 🆕 MLflow에 완전한 재현성 정보 저장
    log_complete_training_artifacts(
        settings=settings,
        original_recipe_path=context_params.get('original_recipe_path'),
        original_config_path=context_params.get('original_config_path'),
        original_data_path=context_params.get('original_data_path'),
        context_params=context_params,
        python_model=pyfunc_model,
        signature=signature,
        input_example=input_example
    )

# src/utils/integrations/mlflow_integration.py 강화
def log_complete_training_artifacts(
    settings: Settings,
    original_recipe_path: str,
    original_config_path: str,
    original_data_path: str,
    context_params: Optional[Dict] = None,
    python_model = None,
    signature = None,
    input_example = None
):
    """학습시 완전한 재현성을 위한 모든 정보 저장"""

    # 1. 기존 모델 저장 (호환성 유지)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=python_model,
        signature=signature,
        input_example=input_example
    )

    # 2. 🆕 완전한 Recipe/Config 스냅샷 저장
    mlflow.log_artifact(original_recipe_path, "training_artifacts/recipe_snapshot.yaml")
    mlflow.log_artifact(original_config_path, "training_artifacts/config_snapshot.yaml")

    # 3. 🆕 실행 컨텍스트 저장
    execution_context = {
        "data_path": original_data_path,
        "context_params": context_params or {},
        "mmp_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "cli_command": "train",
        "settings_factory_version": "1.0"
    }
    mlflow.log_dict(execution_context, "training_artifacts/execution_context.json")

    # 4. 🆕 Settings 메타데이터 저장 (검색/분석용)
    settings_metadata = {
        "recipe_name": settings.recipe.name,
        "task_type": settings.recipe.task_choice,
        "model_class": settings.recipe.model.class_path,
        "model_library": settings.recipe.model.library,
        "tuning_enabled": settings.recipe.model.hyperparameters.tuning_enabled,
        "fetcher_type": settings.recipe.data.fetcher.type,
        "target_column": settings.recipe.data.data_interface.target_column,
        "entity_columns": settings.recipe.data.data_interface.entity_columns
    }
    mlflow.log_dict(settings_metadata, "training_artifacts/settings_metadata.json")
```

### 5.2 추론시 완전한 복원

```python
# src/settings/mlflow_restore.py 완전 구현
class MLflowRecipeRestorer:
    """MLflow에서 완전한 학습 환경 복원"""

    def restore_complete_context(self) -> RestoreContext:
        """학습시 전체 컨텍스트 복원"""
        try:
            # 1. Recipe 복원
            recipe = self.restore_recipe()

            # 2. 실행 컨텍스트 복원
            execution_context = self._restore_execution_context()

            # 3. 메타데이터 복원
            metadata = self._restore_metadata()

            return RestoreContext(
                recipe=recipe,
                execution_context=execution_context,
                metadata=metadata,
                is_legacy=False
            )

        except FileNotFoundError:
            # Legacy Run 호환성
            return self._create_legacy_context()

    def _restore_execution_context(self) -> Dict:
        """실행 컨텍스트 복원"""
        context_path = mlflow.artifacts.download_artifacts(
            run_id=self.run_id,
            artifact_path="training_artifacts/execution_context.json"
        )

        with open(context_path, 'r') as f:
            return json.load(f)

    def validate_version_compatibility(self) -> bool:
        """버전 호환성 검증"""
        metadata = self._restore_metadata()

        stored_version = metadata.get('mmp_version', '1.0')
        current_version = "2.0"

        if stored_version != current_version:
            logger.warning(f"Version mismatch: stored={stored_version}, current={current_version}")
            return False

        return True

@dataclass
class RestoreContext:
    """복원된 컨텍스트 정보"""
    recipe: Recipe
    execution_context: Dict
    metadata: Dict
    is_legacy: bool
```

## 6. 성능 최적화 및 캐싱 시스템

### 6.1 Recipe 복원 캐싱

```python
# src/settings/cache.py (신규)
class RecipeCache:
    """Recipe 복원 캐싱 시스템"""

    def __init__(self, cache_dir: str = ".mmp_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}

    def get_cached_recipe(self, run_id: str) -> Optional[Recipe]:
        """캐시된 Recipe 반환"""
        # 1. 메모리 캐시 확인
        if run_id in self._memory_cache:
            return self._memory_cache[run_id]

        # 2. 디스크 캐시 확인
        cache_file = self.cache_dir / f"{run_id}_recipe.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    recipe_data = json.load(f)
                recipe = Recipe(**recipe_data)
                self._memory_cache[run_id] = recipe
                return recipe
            except Exception:
                # 손상된 캐시 파일 제거
                cache_file.unlink()

        return None

    def cache_recipe(self, run_id: str, recipe: Recipe) -> None:
        """Recipe 캐싱"""
        # 메모리 캐시
        self._memory_cache[run_id] = recipe

        # 디스크 캐시
        cache_file = self.cache_dir / f"{run_id}_recipe.json"
        with open(cache_file, 'w') as f:
            json.dump(recipe.model_dump(), f, indent=2)

# MLflowRecipeRestorer에 캐싱 통합
class MLflowRecipeRestorer:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.cache = RecipeCache()

    def restore_recipe(self) -> Recipe:
        # 캐시 확인
        cached_recipe = self.cache.get_cached_recipe(self.run_id)
        if cached_recipe:
            logger.debug(f"Recipe loaded from cache: {self.run_id}")
            return cached_recipe

        # MLflow에서 복원
        recipe = self._restore_from_mlflow()

        # 캐시에 저장
        self.cache.cache_recipe(self.run_id, recipe)

        return recipe
```

### 6.2 Catalog 검증 캐싱

```python
# src/settings/validation/catalog_validator.py 성능 최적화
class CatalogValidator:
    def __init__(self, catalog_path: str = "src/models/catalog"):
        self.catalog_path = Path(catalog_path)
        self._catalog_cache = {}  # 메모리 캐시
        self._last_modified = {}  # 파일 수정 시간 추적

    def _load_catalog_spec(self, class_path: str) -> Dict:
        """캐싱된 카탈로그 스펙 로드"""
        cache_key = class_path

        # 캐시 확인
        if cache_key in self._catalog_cache:
            catalog_file = self._get_catalog_file_path(class_path)
            if catalog_file.exists():
                current_mtime = catalog_file.stat().st_mtime
                cached_mtime = self._last_modified.get(cache_key, 0)

                # 파일이 수정되지 않았으면 캐시 사용
                if current_mtime == cached_mtime:
                    return self._catalog_cache[cache_key]

        # 파일에서 로드
        spec = self._load_from_file(class_path)

        # 캐시 업데이트
        self._catalog_cache[cache_key] = spec
        if catalog_file.exists():
            self._last_modified[cache_key] = catalog_file.stat().st_mtime

        return spec
```

## 7. 최종 결과 및 효과

### 7.1 아키텍처 개선 효과

**🎯 완전한 재현성 달성**:
- 학습 → 추론 → 서빙 전 단계에서 100% 동일한 Recipe 사용
- MLflow 기반 완전한 환경 복원
- 버전 호환성 및 마이그레이션 지원

**🏗️ 깔끔한 아키텍처**:
- 레이어별 책임 분리 (CLI → Factory → Validation → Pydantic)
- Models Catalog 기반 동적 검증 시스템
- 테스트 데이터와 예제 데이터 완전 분리

**⚡ 성능 최적화**:
- Recipe 복원 캐싱 시스템
- Catalog 검증 캐싱
- 불필요한 Pydantic 검증 제거

### 7.2 CLI 명령어 일관성

**Before (불일치)**:
```python
# train: 60+ 줄의 복잡한 로직
# serve: 3줄의 단순 호출 (더미 Recipe)
# inference: 5줄의 단순 호출 (더미 Recipe)
```

**After (완전 통일)**:
```python
# 모든 명령어가 10-15줄의 동일한 패턴
def any_command(...):
    settings = SettingsFactory.for_*(...)
    setup_logging(settings)
    run_*_pipeline(settings, ...)
```

### 7.3 개발자 경험 향상

**🔧 유지보수성**:
- 하나의 SettingsFactory로 모든 Settings 생성 통합
- 검증 로직 중앙화로 수정 포인트 최소화
- Import 경로 일관성

**🧪 테스트 용이성**:
- Fixtures 기반 테스트 데이터 관리
- 각 레이어별 독립적 테스트 가능
- Models Catalog 기반 실제 모델 검증

**📚 코드 가독성**:
- 책임별 파일 분리
- 하드코딩 제거
- 명확한 데이터 플로우

## 8. 마이그레이션 및 위험 관리

### 8.1 단계별 마이그레이션 전략

**Phase 1 (위험도: 낮음)**:
- Pydantic 모델 순수화
- 검증 로직 분리
- 테스트 데이터 Fixtures 이관

**Phase 2 (위험도: 중간)**:
- SettingsFactory 구현
- MLflow 저장 시스템 추가
- 캐싱 시스템 도입

**Phase 3 (위험도: 중간)**:
- MLflow 복원 시스템
- Legacy Run 호환성 구현
- Factory 어댑터 매핑 이관

**Phase 4 (위험도: 높음)**:
- CLI 명령어 대폭 수정
- 레거시 함수 삭제
- import 경로 정리

### 8.2 위험 완화 방안

**🛡️ 하위 호환성 보장**:
- Legacy MLflow Run 자동 감지 및 처리
- 기존 하드코딩 방식을 fallback으로 유지
- 점진적 마이그레이션 지원

**🔄 롤백 계획**:
- 각 Phase별 독립적 롤백 가능
- Feature Flag 기반 새로운 시스템 활성화
- 기존 함수들을 deprecated로 표시 후 단계적 제거

**📊 모니터링**:
- Recipe 복원 성공률 모니터링
- 성능 메트릭 추적 (복원 시간, 캐시 히트율)
- 에러 패턴 분석 및 개선

## 9. 최종 결론

이 재설계 계획을 통해 달성할 수 있는 것:

**🎯 핵심 목표 100% 달성**:
1. ✅ **완전 재현성**: 학습-추론-서빙 전 단계 동일 환경
2. ✅ **CLI 일관성**: 모든 명령어 동일한 패턴과 복잡도
3. ✅ **아키텍처 정리**: 레이어별 책임 분리, 하드코딩 제거
4. ✅ **성능 최적화**: 캐싱 시스템, 불필요한 검증 제거

**🚀 추가 달성 효과**:
- Models Catalog 기반 동적 검증으로 확장성 확보
- 테스트 구조 통합으로 품질 향상
- 개발자 경험 대폭 개선

**📈 비즈니스 가치**:
- Production-Ready MLOps 시스템으로 진화
- 실험 재현성 완전 보장
- 개발 생산성 향상 및 유지보수 비용 절감

이 계획을 통해 현재의 **불완전한 MLOps 시스템**을 **엔터프라이즈급 Production MLOps 플랫폼**으로 완전히 변화시킬 수 있습니다.