# Modern ML Pipeline Settings 아키텍처 완전 재설계 계획

## 🎯 목표 아키텍처 정의

### 핵심 설계 원칙

1. **단일 책임 원칙**: 각 모듈은 하나의 명확한 책임만 담당
2. **동적 검증 시스템**: Registry 및 Catalog 기반 실시간 유효성 검사
3. **CLI 명령어 통합**: train/serve-api/batch-inference 3개 명령어의 일관된 Settings 생성
4. **완전한 재현성**: MLflow 기반 학습-추론 환경 100% 복원

### 타겟 파일 구조

```
src/settings/
├── config.py              # Config Pydantic 스키마 (순수 검증만)
├── recipe.py              # Recipe Pydantic 스키마 (순수 검증만)
├── factory.py             # 통합 SettingsFactory (CLI 명령어별 Settings 생성)
├── validation/            # 동적 검증 시스템
│   ├── __init__.py
│   ├── catalog_validator.py    # src/models/catalog 기반 모델/하이퍼파라미터 검증
│   ├── business_validator.py   # src/components registry 기반 컴포넌트 검증
│   └── compatibility_validator.py  # Config-Recipe 호환성 검증
├── mlflow_restore.py      # MLflow Recipe 복원 시스템
└── recipe_builder.py      # Registry 기반 동적 Recipe 생성기 (CLI에서 이관)
```

## 📋 현재 상황 분석

### 기존 구조의 문제점

1. **검증 로직 분산**: Pydantic 모델, loader.py, CLI 명령어에 검증 로직 산재
2. **하드코딩 의존성**: recipe_builder.py의 전처리기 정보, CLI의 더미 Recipe 등
3. **CLI 일관성 부재**: train(복잡) vs serve-api/batch-inference(단순) 구조 차이
4. **재현성 부족**: 추론시 학습 환경 완전 복원 불가

### 활용 가능한 기존 자산

1. **src/models/catalog/**: Task별(Classification, Regression, etc.) 모델 스펙
2. **src/components/**/registry.py**: 실제 사용 가능한 컴포넌트 동적 등록 시스템
3. **기존 factory.py**: Pydantic 모델 기본값 생성 로직
4. **기존 loader.py**: Settings 클래스 및 환경변수 처리

## 🏗️ 상세 컴포넌트 설계

### 1. 순수 Pydantic 스키마 (config.py, recipe.py)

#### config.py - Config 스키마 순수화
```python
"""순수 Config Pydantic 스키마 - 검증 로직 완전 제거"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, Literal

class Environment(BaseModel):
    name: str = Field(..., description="환경 이름")

class MLflow(BaseModel):
    tracking_uri: str = Field(..., description="MLflow tracking URI")
    experiment_name: str = Field(..., description="실험 이름")
    tracking_username: Optional[str] = Field(None, description="MLflow 인증 사용자명")
    tracking_password: Optional[str] = Field(None, description="MLflow 인증 비밀번호")
    s3_endpoint_url: Optional[str] = Field(None, description="S3 호환 엔드포인트 URL")

# DataSource 어댑터별 Config 모델들
class PostgreSQLConfig(BaseModel):
    connection_uri: str = Field(..., description="PostgreSQL 연결 URI")
    query_timeout: int = Field(default=300, description="쿼리 타임아웃(초)")

class BigQueryConfig(BaseModel):
    connection_uri: str = Field(..., description="BigQuery 연결 URI")
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")
    location: str = Field(default="US", description="BigQuery 위치")
    use_pandas_gbq: bool = Field(default=True, description="pandas_gbq 사용 여부")
    query_timeout: int = Field(default=300, description="쿼리 타임아웃(초)")

class LocalFilesConfig(BaseModel):
    base_path: str = Field(..., description="로컬 파일 기본 경로")
    storage_options: Dict[str, Any] = Field(default_factory=dict, description="저장소 옵션")

class S3StorageOptions(BaseModel):
    aws_access_key_id: Optional[str] = Field(None, description="AWS Access Key ID")
    aws_secret_access_key: Optional[str] = Field(None, description="AWS Secret Access Key")
    region_name: str = Field(default="us-east-1", description="AWS 리전")

class S3Config(BaseModel):
    base_path: str = Field(..., description="S3 기본 경로")
    storage_options: S3StorageOptions = Field(..., description="S3 저장소 옵션")

class GCSStorageOptions(BaseModel):
    project: Optional[str] = Field(None, description="GCP 프로젝트 ID")
    token: Optional[str] = Field(None, description="인증 토큰")

class GCSConfig(BaseModel):
    base_path: str = Field(..., description="GCS 기본 경로")
    storage_options: GCSStorageOptions = Field(..., description="GCS 저장소 옵션")

class DataSource(BaseModel):
    name: str = Field(..., description="데이터 소스 이름")
    adapter_type: str = Field(..., description="어댑터 타입")
    config: Union[PostgreSQLConfig, BigQueryConfig, LocalFilesConfig, S3Config, GCSConfig] = Field(..., description="어댑터별 설정")

# Feast Online Store 타입별 모델들
class RedisOnlineStore(BaseModel):
    type: Literal["redis"] = "redis"
    connection_string: str = Field(..., description="Redis 연결 문자열")
    password: Optional[str] = Field(None, description="Redis 비밀번호")

class DynamoDBOnlineStore(BaseModel):
    type: Literal["dynamodb"] = "dynamodb"
    region: str = Field(..., description="AWS 리전")
    table_name: str = Field(..., description="DynamoDB 테이블 이름")

class SQLiteOnlineStore(BaseModel):
    type: Literal["sqlite"] = "sqlite"
    path: str = Field(..., description="SQLite 파일 경로")

# Feast Offline Store 타입별 모델들
class BigQueryOfflineStore(BaseModel):
    type: Literal["bigquery"] = "bigquery"
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")

class FileOfflineStore(BaseModel):
    type: Literal["file"] = "file"
    path: str = Field(..., description="파일 저장 경로")

# Union 타입 정의
FeastOnlineStore = Union[RedisOnlineStore, DynamoDBOnlineStore, SQLiteOnlineStore]
FeastOfflineStore = Union[BigQueryOfflineStore, FileOfflineStore]

class FeastConfig(BaseModel):
    project: str = Field(..., description="Feast 프로젝트 이름")
    registry: str = Field(..., description="Registry 경로")
    online_store: FeastOnlineStore
    offline_store: FeastOfflineStore
    entity_key_serialization_version: int = Field(default=2, description="엔티티 키 직렬화 버전")

class FeatureStore(BaseModel):
    provider: str = Field(default="none", description="Feature Store 제공자")
    feast_config: Optional[FeastConfig] = Field(None, description="Feast 설정")

class AuthConfig(BaseModel):
    enabled: bool = Field(default=False, description="인증 활성화")
    type: str = Field(default="jwt", description="인증 타입")
    secret_key: Optional[str] = Field(None, description="인증 시크릿 키")

class Serving(BaseModel):
    enabled: bool = Field(default=False, description="서빙 활성화")
    host: str = Field(default="0.0.0.0", description="서빙 호스트")
    port: int = Field(default=8000, description="서빙 포트")
    workers: int = Field(default=1, description="워커 수")
    model_stage: Optional[str] = Field(None, description="모델 스테이지")
    auth: Optional[AuthConfig] = Field(None, description="인증 설정")

class ArtifactStore(BaseModel):
    type: str = Field(..., description="아티팩트 저장소 타입")
    config: Dict[str, Any] = Field(..., description="저장소 설정")

# Output 어댑터별 Config 모델들
class StorageOutputConfig(BaseModel):
    base_path: str = Field(..., description="저장 기본 경로")

class SQLOutputConfig(BaseModel):
    connection_uri: str = Field(..., description="데이터베이스 연결 URI")
    table: str = Field(..., description="테이블 이름")

class BigQueryOutputConfig(BaseModel):
    connection_uri: str = Field(..., description="BigQuery 연결 URI")
    project_id: str = Field(..., description="GCP 프로젝트 ID")
    dataset_id: str = Field(..., description="BigQuery 데이터셋 ID")
    table: str = Field(..., description="테이블 이름")
    location: str = Field(default="US", description="BigQuery 위치")
    use_pandas_gbq: bool = Field(default=True, description="pandas_gbq 사용 여부")

class OutputTarget(BaseModel):
    name: str = Field(..., description="출력 대상 이름")
    enabled: bool = Field(default=True, description="출력 활성화")
    adapter_type: str = Field(..., description="어댑터 타입")
    config: Union[StorageOutputConfig, SQLOutputConfig, BigQueryOutputConfig] = Field(..., description="어댑터별 출력 설정")

class Output(BaseModel):
    inference: OutputTarget = Field(..., description="추론 결과 출력")

class Config(BaseModel):
    """Config 스키마 - 순수 데이터 구조만"""
    environment: Environment
    mlflow: Optional[MLflow] = None
    data_source: DataSource
    feature_store: FeatureStore
    serving: Optional[Serving] = None
    artifact_store: Optional[ArtifactStore] = None
    output: Output

    # 검증 로직은 validation/ 으로 완전 분리
```

#### recipe.py - Recipe 스키마 순수화
```python
"""순수 Recipe Pydantic 스키마 - 검증 로직 완전 제거"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class HyperparametersTuning(BaseModel):
    tuning_enabled: bool = Field(False, description="튜닝 활성화")
    optimization_metric: Optional[str] = Field(None, description="최적화 메트릭")
    n_trials: Optional[int] = Field(None, description="튜닝 시도 횟수")
    timeout: Optional[int] = Field(None, description="튜닝 타임아웃(초)")
    fixed: Optional[Dict[str, Any]] = Field(None, description="고정 파라미터")
    tunable: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="튜닝 파라미터")
    values: Optional[Dict[str, Any]] = Field(None, description="고정값")

class Calibration(BaseModel):
    enabled: bool = Field(False, description="캘리브레이션 활성화")
    method: Optional[str] = Field(None, description="캘리브레이션 방법")

class Model(BaseModel):
    class_path: str = Field(..., description="모델 클래스 경로")
    library: str = Field(..., description="라이브러리 이름")
    hyperparameters: HyperparametersTuning
    calibration: Optional[Calibration] = Field(None, description="캘리브레이션 설정")

class Loader(BaseModel):
    source_uri: Optional[str] = Field(None, description="데이터 소스 URI")

class FeatureView(BaseModel):
    join_key: str = Field(..., description="조인 키")
    features: List[str] = Field(..., description="피처 목록")

class Fetcher(BaseModel):
    type: str = Field(..., description="Fetcher 타입")
    feature_views: Optional[Dict[str, FeatureView]] = Field(None, description="Feature View 설정")
    timestamp_column: Optional[str] = Field(None, description="타임스탬프 컬럼")

class DataInterface(BaseModel):
    target_column: Optional[str] = Field(None, description="타겟 컬럼")
    treatment_column: Optional[str] = Field(None, description="처치 변수 컬럼 (Causal 태스크)")
    timestamp_column: Optional[str] = Field(None, description="타임스탬프 컬럼 (TimeSeries 태스크)")
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록")
    feature_columns: Optional[List[str]] = Field(None, description="피처 컬럼 목록")

class DataSplit(BaseModel):
    train: float = Field(..., description="학습 데이터 비율")
    test: float = Field(..., description="테스트 데이터 비율")
    validation: float = Field(..., description="검증 데이터 비율")
    calibration: Optional[float] = Field(None, description="캘리브레이션 데이터 비율")

class Data(BaseModel):
    loader: Loader
    fetcher: Fetcher
    data_interface: DataInterface
    split: DataSplit

class PreprocessorStep(BaseModel):
    type: str = Field(..., description="전처리 타입")
    columns: Optional[List[str]] = Field(None, description="적용 컬럼")
    strategy: Optional[str] = Field(None, description="전처리 전략")
    degree: Optional[int] = Field(None, description="다항식 차수")
    n_bins: Optional[int] = Field(None, description="구간 수")
    create_missing_indicators: Optional[bool] = Field(None, description="결측치 지시자 생성")

class Preprocessor(BaseModel):
    steps: List[PreprocessorStep] = Field(..., description="전처리 단계 목록")

class Evaluation(BaseModel):
    metrics: List[str] = Field(..., description="평가 메트릭 목록")
    random_state: int = Field(default=42, description="재현성을 위한 시드")

class Metadata(BaseModel):
    author: str = Field(default="CLI Recipe Builder", description="작성자")
    created_at: str = Field(..., description="생성 시간")
    description: str = Field(..., description="Recipe 설명")
    tuning_note: Optional[str] = Field(None, description="튜닝 관련 노트")

class Recipe(BaseModel):
    """Recipe 스키마 - 순수 데이터 구조만"""
    name: str
    task_choice: str
    model: Model
    data: Data
    preprocessor: Optional[Preprocessor] = None
    evaluation: Evaluation
    metadata: Metadata

    # 검증 로직은 validation/ 으로 완전 분리
```

### 2. 통합 Settings Factory (factory.py)

#### 핵심 설계 개념
- **CLI 명령어별 전용 메서드**: `for_training()`, `for_serving()`, `for_inference()`
- **기존 로직 통합**: loader.py의 Settings 클래스와 파일 로딩 로직 흡수
- **검증 시스템 연동**: validation/ 모듈과 긴밀한 연계

```python
"""통합 Settings Factory - 모든 CLI 명령어의 Settings 생성 중앙화"""

from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from .config import Config
from .recipe import Recipe
from .validation import ValidationOrchestrator
from .mlflow_restore import MLflowRecipeRestorer

class Settings:
    """통합 Settings 컨테이너"""
    def __init__(self, config: Config, recipe: Recipe):
        self.config = config
        self.recipe = recipe
        # 검증은 Factory에서 사전 실행됨

class SettingsFactory:
    """통합 Settings Factory - CLI 명령어별 Settings 생성"""

    def __init__(self):
        """검증 시스템 초기화"""
        self.validator = ValidationOrchestrator()

    @classmethod
    def for_training(cls, recipe_path: str, config_path: str,
                    data_path: str, context_params: Optional[Dict] = None) -> Settings:
        """
        train 명령어용 Settings 생성

        통합 기능:
        1. Recipe/Config 파일 로딩
        2. data_path 처리 (Jinja 템플릿 렌더링)
        3. 동적 검증 실행 (Catalog + Registry + Compatibility)
        4. 계산 필드 추가 (run_name 등)
        """
        factory = cls()

        # 1. 파일 로딩
        config = factory._load_config(config_path)
        recipe = factory._load_recipe(recipe_path)

        # 2. 학습 전용 데이터 경로 처리
        factory._process_training_data_path(recipe, data_path, context_params)

        # 3. 동적 검증 실행
        validation_result = factory.validator.validate_for_training(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"학습 설정 검증 실패: {validation_result.error_message}")

        # 4. Settings 생성 및 계산 필드 추가
        settings = Settings(config, recipe)
        factory._add_training_computed_fields(settings, recipe_path, context_params)

        return settings

    @classmethod
    def for_serving(cls, config_path: str, run_id: str) -> Settings:
        """
        serve-api 명령어용 Settings 생성

        핵심 기능:
        1. 현재 Config 로딩 (서빙 환경)
        2. MLflow에서 학습시 Recipe 완전 복원
        3. 서빙 호환성 검증
        """
        factory = cls()

        # 1. 현재 서빙 환경의 Config 로딩
        config = factory._load_config(config_path)

        # 2. MLflow Recipe 복원
        recipe_restorer = MLflowRecipeRestorer(run_id)
        recipe = recipe_restorer.restore_recipe()

        # 3. 서빙 호환성 검증
        validation_result = factory.validator.validate_for_serving(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"서빙 설정 검증 실패: {validation_result.error_message}")

        # 4. Settings 생성
        settings = Settings(config, recipe)
        factory._add_serving_computed_fields(settings, run_id)

        return settings

    @classmethod
    def for_inference(cls, config_path: str, run_id: str, data_path: str,
                     context_params: Optional[Dict] = None) -> Settings:
        """
        batch-inference 명령어용 Settings 생성

        핵심 기능:
        1. 현재 Config 로딩 (추론 환경)
        2. MLflow에서 학습시 Recipe 완전 복원
        3. 추론 데이터 경로 처리 (배치별 데이터)
        4. 추론 호환성 검증
        """
        factory = cls()

        # 1. 현재 추론 환경의 Config 로딩
        config = factory._load_config(config_path)

        # 2. MLflow Recipe 복원
        recipe_restorer = MLflowRecipeRestorer(run_id)
        recipe = recipe_restorer.restore_recipe()

        # 3. 추론 전용 데이터 경로 처리
        factory._process_inference_data_path(recipe, data_path, context_params)

        # 4. 추론 호환성 검증
        validation_result = factory.validator.validate_for_inference(config, recipe)
        if not validation_result.is_valid:
            raise ValueError(f"추론 설정 검증 실패: {validation_result.error_message}")

        # 5. Settings 생성
        settings = Settings(config, recipe)
        factory._add_inference_computed_fields(settings, run_id, data_path)

        return settings

    # === 내부 유틸리티 메서드들 ===
    def _load_config(self, config_path: str) -> Config:
        """Config 파일 로딩 및 환경변수 치환"""
        # 기존 loader.py의 _load_config() 로직 통합

    def _load_recipe(self, recipe_path: str) -> Recipe:
        """Recipe 파일 로딩 및 환경변수 치환"""
        # 기존 loader.py의 _load_recipe() 로직 통합

    def _process_training_data_path(self, recipe: Recipe, data_path: str,
                                   context_params: Optional[Dict]) -> None:
        """학습용 데이터 경로 처리 (Jinja 템플릿 렌더링)"""
        # 기존 train_command의 데이터 경로 처리 로직 통합

    def _process_inference_data_path(self, recipe: Recipe, data_path: str,
                                    context_params: Optional[Dict]) -> None:
        """추론용 데이터 경로 처리"""
        # batch-inference 전용 데이터 경로 처리

# 하위 호환성 편의 함수
def load_settings(recipe_path: str, config_path: str, **kwargs) -> Settings:
    """하위 호환성: 기존 load_settings() 지원"""
    return SettingsFactory.for_training(recipe_path, config_path,
                                       kwargs.get('data_path', ''))
```

### 3. 동적 검증 시스템 (validation/)

#### validation/__init__.py - 검증 오케스트레이터
```python
"""검증 시스템 통합 오케스트레이터"""

from .catalog_validator import CatalogValidator
from .business_validator import BusinessValidator
from .compatibility_validator import CompatibilityValidator
from typing import NamedTuple

class ValidationResult(NamedTuple):
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = []

class ValidationOrchestrator:
    """모든 검증 로직의 중앙 조정자"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.compatibility_validator = CompatibilityValidator()

    def validate_for_training(self, config: Config, recipe: Recipe) -> ValidationResult:
        """학습용 종합 검증"""
        # 1. Catalog 기반 모델/하이퍼파라미터 검증
        # 2. Registry 기반 컴포넌트 검증
        # 3. Config-Recipe 호환성 검증
        # 4. 학습 전용 검증 (데이터 소스 등)

    def validate_for_serving(self, config: Config, recipe: Recipe) -> ValidationResult:
        """서빙용 종합 검증"""
        # 서빙 환경 특화 검증

    def validate_for_inference(self, config: Config, recipe: Recipe) -> ValidationResult:
        """추론용 종합 검증"""
        # 추론 환경 특화 검증
```

#### validation/catalog_validator.py - 모델 카탈로그 기반 검증
```python
"""src/models/catalog 기반 동적 모델/하이퍼파라미터 검증"""

from pathlib import Path
from typing import Dict, List, Set
import yaml

class CatalogValidator:
    """Models Catalog 기반 동적 검증 시스템"""

    def __init__(self, catalog_path: str = "src/models/catalog"):
        self.catalog_path = Path(catalog_path)
        self._task_models_cache = {}  # 성능 최적화용 캐시

    def get_available_tasks(self) -> Set[str]:
        """사용 가능한 Task 목록 동적 추출"""
        # src/models/catalog/ 하위 디렉토리명 기반
        tasks = set()
        for task_dir in self.catalog_path.iterdir():
            if task_dir.is_dir() and not task_dir.name.startswith('.'):
                tasks.add(task_dir.name.lower())
        return tasks

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들 동적 추출"""
        task_dir = self.catalog_path / task_type.title()
        if not task_dir.exists():
            return {}

        models = {}
        for model_file in task_dir.glob("*.yaml"):
            with open(model_file, 'r') as f:
                model_spec = yaml.safe_load(f)
                models[model_file.stem] = model_spec

        return models

    def validate_model_specification(self, recipe_model: Dict) -> ValidationResult:
        """Recipe의 모델 스펙을 Catalog와 대조 검증"""
        class_path = recipe_model.get('class_path')
        task_type = recipe_model.get('task_choice', '').lower()

        # 1. Task 타입 검증
        available_tasks = self.get_available_tasks()
        if task_type not in available_tasks:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown task type: {task_type}. Available: {available_tasks}"
            )

        # 2. 모델 존재 검증
        available_models = self.get_available_models_for_task(task_type)
        model_found = None
        for model_name, model_spec in available_models.items():
            if model_spec.get('class_path') == class_path:
                model_found = model_spec
                break

        if not model_found:
            return ValidationResult(
                is_valid=False,
                error_message=f"Model {class_path} not found in {task_type} catalog"
            )

        # 3. 하이퍼파라미터 검증
        return self._validate_hyperparameters(recipe_model, model_found)

    def _validate_hyperparameters(self, recipe_model: Dict, catalog_spec: Dict) -> ValidationResult:
        """하이퍼파라미터 범위 및 타입 검증"""
        recipe_hyperparams = recipe_model.get('hyperparameters', {})
        catalog_hyperparams = catalog_spec.get('hyperparameters', {})

        if recipe_hyperparams.get('tuning_enabled', False):
            # 튜닝 모드: tunable 파라미터 검증
            recipe_tunable = recipe_hyperparams.get('tunable', {})
            catalog_tunable = catalog_hyperparams.get('tunable', {})

            for param_name, param_spec in recipe_tunable.items():
                if param_name not in catalog_tunable:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Tunable parameter '{param_name}' not supported by {catalog_spec['class_path']}"
                    )

                # 범위 검증
                recipe_range = param_spec.get('range', [])
                catalog_range = catalog_tunable[param_name].get('range', [])

                if (recipe_range and catalog_range and
                    (recipe_range[0] < catalog_range[0] or recipe_range[1] > catalog_range[1])):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Parameter '{param_name}' range {recipe_range} exceeds catalog limits {catalog_range}"
                    )

        else:
            # 고정값 모드: values 파라미터 검증
            recipe_values = recipe_hyperparams.get('values', {})
            # 기본값 및 타입 검증 로직

        return ValidationResult(is_valid=True)
```

#### validation/business_validator.py - 컴포넌트 Registry 기반 검증
```python
"""src/components Registry 기반 비즈니스 로직 검증"""

class BusinessValidator:
    """Components Registry 기반 동적 검증 시스템"""

    def __init__(self):
        self._trigger_component_imports()  # Registry 활성화

    def _trigger_component_imports(self):
        """모든 컴포넌트 Registry 활성화"""
        import src.components.preprocessor
        import src.components.evaluator
        import src.components.calibration
        # ... 기타 컴포넌트들

    def get_available_preprocessor_types(self) -> Set[str]:
        """실제 등록된 전처리기 타입들 동적 추출"""
        from src.components.preprocessor.registry import PreprocessorStepRegistry
        return set(PreprocessorStepRegistry.preprocessor_steps.keys())

    def get_available_calibrators(self) -> Set[str]:
        """실제 등록된 캘리브레이터들 동적 추출"""
        from src.components.calibration.registry import CalibrationRegistry
        return set(CalibrationRegistry.calibrators.keys())

    def get_available_evaluators_for_task(self, task_type: str) -> List[str]:
        """Task별 사용 가능한 Evaluator들 동적 추출"""
        from src.components.evaluator.registry import EvaluatorRegistry
        return EvaluatorRegistry.get_available_metrics_for_task(task_type)

    def validate_preprocessor_steps(self, recipe_preprocessor: Dict) -> ValidationResult:
        """Recipe의 전처리 단계를 Registry와 대조 검증"""
        if not recipe_preprocessor or not recipe_preprocessor.get('steps'):
            return ValidationResult(is_valid=True)

        available_types = self.get_available_preprocessor_types()

        for step in recipe_preprocessor['steps']:
            step_type = step.get('type')
            if step_type not in available_types:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unknown preprocessor type: '{step_type}'. Available: {sorted(available_types)}"
                )

        return ValidationResult(is_valid=True)

    def validate_calibration_settings(self, recipe_calibration: Dict, task_type: str) -> ValidationResult:
        """캘리브레이션 설정 검증"""
        if not recipe_calibration or not recipe_calibration.get('enabled', False):
            return ValidationResult(is_valid=True)

        # Classification 태스크만 캘리브레이션 지원
        if task_type.lower() != 'classification':
            return ValidationResult(
                is_valid=False,
                error_message=f"Calibration is only supported for classification tasks, got: {task_type}"
            )

        method = recipe_calibration.get('method')
        available_calibrators = self.get_available_calibrators()

        if method not in available_calibrators:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown calibration method: '{method}'. Available: {sorted(available_calibrators)}"
            )

        return ValidationResult(is_valid=True)

    def validate_evaluation_metrics(self, recipe_evaluation: Dict, task_type: str) -> ValidationResult:
        """평가 메트릭 검증"""
        if not recipe_evaluation or not recipe_evaluation.get('metrics'):
            return ValidationResult(is_valid=True)

        recipe_metrics = recipe_evaluation['metrics']
        available_metrics = self.get_available_evaluators_for_task(task_type)

        invalid_metrics = [m for m in recipe_metrics if m not in available_metrics]

        if invalid_metrics:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid metrics for {task_type}: {invalid_metrics}. Available: {available_metrics}"
            )

        return ValidationResult(is_valid=True)
```

#### validation/compatibility_validator.py - Config-Recipe 호환성 검증
```python
"""Config-Recipe 간 호환성 충돌 검증"""

class CompatibilityValidator:
    """Config와 Recipe 간 설정 충돌 검증"""

    def validate_feature_store_consistency(self, config: Config, recipe: Recipe) -> ValidationResult:
        """Feature Store 설정 일관성 검증"""
        recipe_fetcher_type = recipe.data.fetcher.type
        config_fs_provider = config.feature_store.provider

        if recipe_fetcher_type == "feature_store":
            # Recipe에서 feature_store 사용 시 Config에 설정 필요
            if config_fs_provider == "none":
                return ValidationResult(
                    is_valid=False,
                    error_message="Recipe uses feature_store fetcher but Config feature_store provider is 'none'"
                )

            # Feast 설정 존재 확인
            if config_fs_provider == "feast" and not config.feature_store.feast_config:
                return ValidationResult(
                    is_valid=False,
                    error_message="Recipe uses feature_store but Config lacks feast_config"
                )

        elif config_fs_provider != "none":
            # Config에 Feature Store 설정했지만 Recipe에서 사용 안함 - 경고만
            return ValidationResult(
                is_valid=True,
                warnings=["Config has feature_store settings but Recipe doesn't use feature_store fetcher"]
            )

        return ValidationResult(is_valid=True)

    def validate_data_source_compatibility(self, config: Config, recipe: Recipe) -> ValidationResult:
        """데이터 소스 어댑터 호환성 검증"""
        # source_uri가 주입된 이후에만 검증 가능
        if not hasattr(recipe.data.loader, 'source_uri') or not recipe.data.loader.source_uri:
            return ValidationResult(is_valid=True)

        source_uri = recipe.data.loader.source_uri.lower()
        config_adapter = config.data_source.adapter_type

        # URI 패턴 기반 어댑터 타입 추론
        if self._is_sql_pattern(source_uri):
            compatible_types = ['sql', 'bigquery']
            if config_adapter not in compatible_types:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"SQL data source requires adapter_type in {compatible_types}, got: {config_adapter}"
                )

        elif self._is_storage_pattern(source_uri):
            if config_adapter != 'storage':
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Storage file requires adapter_type 'storage', got: {config_adapter}"
                )

        elif source_uri.startswith('bigquery://'):
            if config_adapter != 'bigquery':
                return ValidationResult(
                    is_valid=False,
                    error_message=f"BigQuery URI requires adapter_type 'bigquery', got: {config_adapter}"
                )

        return ValidationResult(is_valid=True)

    def _is_sql_pattern(self, uri: str) -> bool:
        """SQL 패턴 검사"""
        return (uri.endswith('.sql') or 'select' in uri or 'from' in uri)

    def _is_storage_pattern(self, uri: str) -> bool:
        """Storage 패턴 검사"""
        return (uri.endswith(('.csv', '.parquet', '.json')) or
                uri.startswith(('s3://', 'gs://', 'az://')))
```

### 4. MLflow Recipe 복원 시스템 (mlflow_restore.py)

```python
"""MLflow 기반 완전한 Recipe 복원 시스템"""

import mlflow
import yaml
import json
from pathlib import Path
from typing import Optional
from .recipe import Recipe

class MLflowRecipeRestorer:
    """MLflow에서 학습시 Recipe 완전 복원"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.client = mlflow.tracking.MlflowClient()

    def restore_recipe(self) -> Recipe:
        """학습시 Recipe 완전 복원"""
        try:
            # 1. recipe_snapshot.yaml 다운로드
            recipe_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/recipe_snapshot.yaml"
            )

            # 2. Recipe 데이터 로드
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_data = yaml.safe_load(f)

            # 3. 환경변수 치환 (현재 추론 환경 기준)
            recipe_data = self._resolve_env_variables(recipe_data)

            # 4. Recipe 객체 생성
            return Recipe(**recipe_data)

        except FileNotFoundError:
            # Legacy Run 호환성 - 하위 호환 fallback
            raise ValueError(f"Recipe snapshot not found for run {self.run_id}. This run was created before Recipe restoration was implemented.")

    def get_training_context(self) -> Dict:
        """학습시 실행 컨텍스트 복원"""
        try:
            context_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="training_artifacts/execution_context.json"
            )

            with open(context_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _resolve_env_variables(self, data):
        """환경변수 치환 (현재 추론 환경 기준)"""
        # factory.py의 환경변수 처리 로직과 동일
        return data
```

### 5. Registry 기반 동적 Recipe Builder (recipe_builder.py)

#### 기존 CLI에서 settings/로 이관 및 동적화
```python
"""Registry 기반 동적 Recipe 생성기"""

from pathlib import Path
from typing import Dict, List, Set
from .validation.catalog_validator import CatalogValidator
from .validation.business_validator import BusinessValidator
from datetime import datetime

class RecipeBuilder:
    """Registry 기반 동적 Recipe 빌더 - 하드코딩 완전 제거"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        # InteractiveUI 등은 기존 방식 유지

    def get_available_tasks(self) -> Set[str]:
        """src/models/catalog 기반 동적 Task 목록"""
        return self.catalog_validator.get_available_tasks()

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들"""
        return self.catalog_validator.get_available_models_for_task(task_type)

    def get_available_preprocessors(self) -> Dict[str, List[str]]:
        """Registry 기반 전처리기 분류 및 제공"""
        available_types = self.business_validator.get_available_preprocessor_types()

        # 타입명 기반 자동 분류 (하드코딩 대신 규칙 기반)
        categorized = {
            "Missing Value Handling": [],
            "Encoders": [],
            "Scalers": [],
            "Feature Engineering": []
        }

        for step_type in available_types:
            category = self._categorize_preprocessor(step_type)
            if category in categorized:
                categorized[category].append(step_type)

        return {k: v for k, v in categorized.items() if v}  # 빈 카테고리 제거

    def get_available_metrics_for_task(self, task_type: str) -> List[str]:
        """Registry 기반 Task별 메트릭 제공"""
        return self.business_validator.get_available_evaluators_for_task(task_type)

    def build_recipe_interactively(self) -> Dict:
        """사용자 대화형 Recipe 생성 - 모든 옵션이 동적"""

        # 1. Task 선택 (동적)
        available_tasks = self.get_available_tasks()
        task_choice = self._select_from_options("Task를 선택하세요:", list(available_tasks))

        # 2. 모델 선택 (Task별 동적)
        available_models = self.get_available_models_for_task(task_choice)
        model_name = self._select_from_options("모델을 선택하세요:", list(available_models.keys()))
        selected_model = available_models[model_name]

        # 3. 전처리기 선택 (Registry 기반 동적)
        preprocessor_steps = self._collect_preprocessor_steps()

        # 4. 평가 메트릭 선택 (Task별 동적)
        available_metrics = self.get_available_metrics_for_task(task_choice)
        selected_metrics = self._select_metrics(available_metrics)

        # 5. Task별 특수 설정 처리
        model_config = self._build_model_config(selected_model, task_choice)
        data_interface_config = self._build_data_interface_config(task_choice)
        data_split_config = self._build_data_split_config(task_choice, model_config.get('calibration', {}))

        # 6. Recipe 데이터 구성
        recipe_data = {
            "name": f"{task_choice}_{model_name}_{datetime.now().strftime('%Y%m%d')}",
            "task_choice": task_choice,
            "model": model_config,
            "data": {
                "loader": {"source_uri": None},  # 나중에 주입
                "fetcher": {"type": "direct"},  # 기본값
                "data_interface": data_interface_config,
                "split": data_split_config
            },
            "preprocessor": {"steps": preprocessor_steps} if preprocessor_steps else None,
            "evaluation": {
                "metrics": selected_metrics,
                "random_state": 42
            },
            "metadata": {
                "author": "CLI Recipe Builder",
                "created_at": datetime.now().isoformat(),
                "description": f"{task_choice} task using {selected_model['library']}"
            }
        }

        return recipe_data

    def _build_model_config(self, selected_model: Dict, task_choice: str) -> Dict:
        """모델 설정 구성 (Task별 조건부 로직 포함)"""
        model_config = {
            "class_path": selected_model["class_path"],
            "library": selected_model["library"],
            "hyperparameters": self._configure_hyperparameters(selected_model)
        }

        # Classification task에서만 Calibration 지원
        if task_choice.lower() == 'classification':
            calibration_enabled = self.ui.confirm("캠리브레이션을 사용하시겠습니까?")
            if calibration_enabled:
                available_methods = self.business_validator.get_available_calibrators()
                calibration_method = self._select_from_options(
                    "캠리브레이션 방법을 선택하세요:",
                    list(available_methods)
                )
                model_config["calibration"] = {
                    "enabled": True,
                    "method": calibration_method
                }
            else:
                model_config["calibration"] = {"enabled": False}

        return model_config

    def _build_data_interface_config(self, task_choice: str) -> Dict:
        """데이터 인터페이스 설정 구성 (Task별 특수 필드 포함)"""
        data_interface = {
            "entity_columns": [],  # 나중에 사용자 입력
            "feature_columns": None  # 자동 추론
        }

        # Clustering에서만 target_column 없음
        if task_choice.lower() != 'clustering':
            data_interface["target_column"] = None  # 나중에 주입

        # Causal task에서만 treatment_column 사용
        if task_choice.lower() == 'causal':
            data_interface["treatment_column"] = None  # 나중에 주입

        # TimeSeries task에서만 timestamp_column 필수
        if task_choice.lower() == 'timeseries':
            data_interface["timestamp_column"] = None  # 나중에 주입

        return data_interface

    def _build_data_split_config(self, task_choice: str, calibration_config: Dict) -> Dict:
        """데이터 분할 설정 구성 (Calibration 조건부 포함)"""
        # 기본 데이터 분할 비율
        data_split = {
            "train": 0.8,
            "test": 0.1,
            "validation": 0.1
        }

        # Classification + Calibration 활성화시에만 calibration split 추가
        if (task_choice.lower() == 'classification' and
            calibration_config.get('enabled', False)):
            # Calibration을 위해 train 비율 조정
            data_split = {
                "train": 0.7,
                "test": 0.1,
                "validation": 0.1,
                "calibration": 0.1
            }

        return data_split

    def _collect_preprocessor_steps(self) -> List[Dict]:
        """Registry 기반 전처리기 수집"""
        available_preprocessors = self.get_available_preprocessors()
        selected_steps = []

        for category, preprocessors in available_preprocessors.items():
            if self.ui.confirm(f"\n{category} 전처리를 사용하시겠습니까?"):
                for preprocessor_type in preprocessors:
                    if self.ui.confirm(f"  {preprocessor_type}를 사용하시겠습니까?"):
                        step_config = {
                            "type": preprocessor_type,
                            # 동적 파라미터 설정 (Registry에서 클래스 분석)
                            "params": self._configure_preprocessor_params(preprocessor_type)
                        }
                        selected_steps.append(step_config)

        return selected_steps

    def _categorize_preprocessor(self, step_type: str) -> str:
        """전처리기 타입명 기반 카테고리 분류 (하드코딩 제거)"""
        step_lower = step_type.lower()

        if 'imputer' in step_lower or 'missing' in step_lower:
            return "Missing Value Handling"
        elif 'encoder' in step_lower:
            return "Encoders"
        elif 'scaler' in step_lower:
            return "Scalers"
        elif 'feature' in step_lower or 'discretizer' in step_lower:
            return "Feature Engineering"
        else:
            return "Other"

    def generate_template_variables(self, recipe_data: Dict,
                                   config_template_vars: Dict = None) -> Dict:
        """
Recipe 데이터로부터 Jinja 템플릿 변수들 생성

        모든 config.yaml.j2, recipe.yaml.j2에서 사용되는 변수들을
        Recipe 데이터에서 추출하여 생성
        """
        template_vars = {
            # Recipe 기본 정보
            "recipe_name": recipe_data["name"],
            "task": recipe_data["task_choice"].title(),
            "model_class": recipe_data["model"]["class_path"],
            "model_library": recipe_data["model"]["library"],
            "timestamp": recipe_data["metadata"]["created_at"],
            "author": recipe_data["metadata"]["author"],

            # 평가 메트릭
            "metrics": recipe_data["evaluation"]["metrics"],

            # 데이터 인터페이스
            "target_column": recipe_data["data"]["data_interface"].get("target_column"),
            "entity_columns": recipe_data["data"]["data_interface"]["entity_columns"],

            # 데이터 분할
            "train_ratio": recipe_data["data"]["split"]["train"],
            "test_ratio": recipe_data["data"]["split"]["test"],
            "validation_ratio": recipe_data["data"]["split"]["validation"],

            # 하이퍼파라미터 관련
            "enable_tuning": recipe_data["model"]["hyperparameters"]["tuning_enabled"],
        }

        # 하이퍼파라미터 체계 처리
        hyperparams = recipe_data["model"]["hyperparameters"]
        if hyperparams["tuning_enabled"]:
            template_vars.update({
                "optimization_metric": hyperparams.get("optimization_metric"),
                "n_trials": hyperparams.get("n_trials"),
                "tuning_timeout": hyperparams.get("timeout"),
                "fixed_params": hyperparams.get("fixed", {}),
                "tunable_specs": hyperparams.get("tunable", {})
            })
        else:
            template_vars["all_hyperparameters"] = hyperparams.get("values", {})

        # Task별 조건부 변수들
        task_lower = recipe_data["task_choice"].lower()

        # Classification 전용
        if task_lower == 'classification':
            calibration = recipe_data["model"].get("calibration", {})
            template_vars.update({
                "calibration_enabled": calibration.get("enabled", False),
                "calibration_method": calibration.get("method"),
                "calibration_ratio": recipe_data["data"]["split"].get("calibration")
            })

        # Causal 전용
        elif task_lower == 'causal':
            template_vars["treatment_column"] = recipe_data["data"]["data_interface"].get("treatment_column")

        # TimeSeries 전용
        elif task_lower == 'timeseries':
            template_vars["timeseries_timestamp_column"] = recipe_data["data"]["data_interface"].get("timestamp_column")

        # Feature Store 관련
        fetcher = recipe_data["data"]["fetcher"]
        template_vars.update({
            "fetcher_type": fetcher["type"],
            "feature_views": fetcher.get("feature_views"),
            "timestamp_column": fetcher.get("timestamp_column")
        })

        # 전처리기 관련
        preprocessor = recipe_data.get("preprocessor")
        if preprocessor:
            template_vars["preprocessor_steps"] = preprocessor["steps"]

        # Config 템플릿 변수들 병합 (사용자 제공)
        if config_template_vars:
            template_vars.update(config_template_vars)

        return template_vars
```

## 🔄 단계별 구현 계획

### Phase 1: Pydantic 스키마 순수화 (1주)
1. **config.py, recipe.py에서 검증 로직 완전 제거**
   - model_config, field_validator, model_validator 모두 삭제
   - 순수 데이터 구조 스키마만 유지

2. **validation/ 디렉토리 생성 및 기본 구조 구축**
   - `__init__.py`: ValidationOrchestrator
   - `catalog_validator.py`: 기본 Catalog 읽기 기능
   - `business_validator.py`: Registry import 및 기본 검증
   - `compatibility_validator.py`: 기존 Settings._validate() 로직 이관

### Phase 2: 통합 Factory 구현 (1-2주)
1. **기존 loader.py 로직을 factory.py로 완전 통합**
   - Settings 클래스, 파일 로딩, 환경변수 처리 모두 이관
   - `SettingsFactory` 클래스 구현

2. **CLI 명령어별 전용 메서드 구현**
   - `for_training()`: 기존 train_command 로직 통합
   - `for_serving()`, `for_inference()`: MLflow 복원 연동

3. **검증 시스템 통합**
   - Factory에서 ValidationOrchestrator 호출
   - CLI 명령어별 차별화된 검증 로직

### Phase 3: MLflow 완전 저장/복원 시스템 (1주)
1. **학습시 Recipe 완전 저장**
   - train_pipeline.py에 recipe_snapshot.yaml 저장 로직 추가
   - execution_context.json 생성

2. **MLflowRecipeRestorer 구현**
   - recipe_snapshot.yaml 기반 Recipe 복원
   - Legacy Run 호환성 처리

### Phase 4: 동적 검증 시스템 완성 (1주)
1. **CatalogValidator 완전 구현**
   - src/models/catalog 동적 스캔
   - 하이퍼파라미터 범위/타입 검증

2. **BusinessValidator 완전 구현**
   - Registry 기반 컴포넌트 검증
   - Task별 메트릭 검증

3. **CompatibilityValidator 완전 구현**
   - Feature Store 일관성 검증
   - 데이터 소스 어댑터 호환성 검증

### Phase 5: Recipe Builder 동적화 및 CLI 통합 (1주)
1. **recipe_builder.py를 settings/로 이관**
   - 기존 하드코딩 제거
   - Registry 기반 동적 옵션 제공

2. **CLI 명령어 단순화**
   - train/serve-api/batch-inference 모두 SettingsFactory 사용
   - 일관된 에러 처리 및 로깅

### Phase 6: 통합 테스트 및 최적화 (1주)
1. **전체 시스템 통합 테스트**
   - 3개 CLI 명령어별 E2E 테스트
   - 검증 시스템 종합 테스트

2. **성능 최적화**
   - Catalog/Registry 캐싱
   - 검증 로직 최적화

## 🎯 최종 달성 목표

### 완전한 재현성
- **학습 → 추론 → 서빙**: 100% 동일한 Recipe 환경
- **MLflow 기반**: 학습시 Recipe 완전 저장 및 복원

### 동적 검증 시스템
- **실시간 유효성**: Registry 및 Catalog 기반 동적 검증
- **충돌 방지**: Config-Recipe 간 설정 모순 사전 감지

### CLI 일관성
- **통일된 복잡도**: 모든 명령어가 동일한 패턴으로 15줄 내외
- **일관된 에러 처리**: 명확하고 도움이 되는 검증 메시지

### 확장성
- **신규 모델 자동 지원**: src/models/catalog에 추가만으로 자동 인식
- **신규 컴포넌트 자동 지원**: Registry 등록만으로 자동 사용 가능
- **하드코딩 완전 제거**: 모든 옵션이 동적으로 제공

이 리팩토링을 통해 **Production-Ready MLOps 시스템**으로 완전히 진화하며, 개발자 경험과 시스템 안정성을 모두 확보할 수 있습니다.