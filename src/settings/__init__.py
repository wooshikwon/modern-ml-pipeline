"""
Settings Module Public API (v4.0)
완전한 아키텍처 재설계 후 새로운 API
"""

# Config 모듈 - 인프라 설정 스키마 (순수 Pydantic)
from .config import (
    AuthConfig,
    BigQueryConfig,
    Config,
    DataSource,
    Environment,
    FeastConfig,
    FeatureStore,
    GCSConfig,
    LocalFilesConfig,
    MLflow,
    Output,
    OutputTarget,
    PostgresOfflineStore,
    PostgreSQLConfig,
    S3Config,
    Serving,
)

# Factory 모듈 - 통합 Settings 생성
from .factory import load_settings  # 하위 호환성
from .factory import (
    Settings,
    SettingsFactory,
)

# MLflow Artifact 저장/복원 시스템
from .mlflow_restore import (
    MLflowArtifactRestorer,
    MLflowArtifactSaver,
    restore_all_from_mlflow,
    restore_config_from_mlflow,
    restore_recipe_from_mlflow,
    save_training_artifacts_to_mlflow,
)

# Recipe 모듈 - 워크플로우 정의 스키마 (순수 Pydantic)
from .recipe import (
    Calibration,
    Data,
    DataInterface,
    DataSplit,
    Evaluation,
    FeatureView,
    Fetcher,
    HyperparametersTuning,
    Loader,
    Metadata,
    Model,
    Preprocessor,
    PreprocessorStep,
    Recipe,
)

# Validation 모듈 - 동적 검증 시스템
from .validation import (
    ValidationOrchestrator,
    ValidationResult,
)

# Recipe Builder - Registry 기반 동적 생성 (moved to src/cli/utils/)
# from .recipe_builder import (
#     RecipeBuilder,
#     build_recipe_interactive,
#     create_recipe_file,
# )

# 주요 클래스/함수 export
__all__ = [
    # ========== Core Classes ==========
    "Settings",
    "SettingsFactory",
    "Config",
    "Recipe",
    "ValidationOrchestrator",
    # ========== Loading Functions ==========
    "load_settings",  # 하위 호환성
    # ========== Config Schemas ==========
    "Environment",
    "MLflow",
    "DataSource",
    "PostgreSQLConfig",
    "BigQueryConfig",
    "LocalFilesConfig",
    "S3Config",
    "GCSConfig",
    "FeatureStore",
    "FeastConfig",
    "PostgresOfflineStore",
    "Serving",
    "AuthConfig",
    "OutputTarget",
    "Output",
    # ========== Recipe Schemas ==========
    "Model",
    "Calibration",
    "HyperparametersTuning",
    "Data",
    "DataSplit",
    "Loader",
    "FeatureView",
    "Fetcher",
    "DataInterface",
    "Preprocessor",
    "PreprocessorStep",
    "Evaluation",
    "Metadata",
    # ========== Validation ==========
    "ValidationResult",
    # ========== MLflow Integration ==========
    "MLflowArtifactRestorer",
    "MLflowArtifactSaver",
    "save_training_artifacts_to_mlflow",
    "restore_recipe_from_mlflow",
    "restore_config_from_mlflow",
    "restore_all_from_mlflow",
    # ========== Recipe Building (moved to src/cli/utils/) ==========
    # "build_recipe_interactive",
    # "create_recipe_file",
]

# 버전 정보 (패키지 메타데이터에서 동적으로 가져옴)
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("modern-ml-pipeline")
except Exception:
    __version__ = "unknown"

# 패키지 메타데이터
__author__ = "Modern ML Pipeline Team"
__description__ = "Unified Settings Factory with dynamic validation and MLflow integration"
