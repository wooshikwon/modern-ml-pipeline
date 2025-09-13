"""
Settings Module Public API (v4.0)
완전한 아키텍처 재설계 후 새로운 API
"""

# Factory 모듈 - 통합 Settings 생성
from .factory import (
    Settings,
    SettingsFactory,
    load_settings,  # 하위 호환성
)

# Config 모듈 - 인프라 설정 스키마 (순수 Pydantic)
from .config import (
    Config,
    Environment,
    MLflow,
    DataSource,
    PostgreSQLConfig,
    BigQueryConfig,
    LocalFilesConfig,
    S3Config,
    GCSConfig,
    FeatureStore,
    FeastConfig,
    FeastOnlineStore,
    FeastOfflineStore,
    Serving,
    AuthConfig,
    ArtifactStore,
    OutputTarget,
    Output,
)

# Recipe 모듈 - 워크플로우 정의 스키마 (순수 Pydantic)
from .recipe import (
    Recipe,
    Model,
    Calibration,
    HyperparametersTuning,
    Data,
    DataSplit,
    Loader,
    FeatureView,
    Fetcher,
    DataInterface,
    Preprocessor,
    PreprocessorStep,
    Evaluation,
    Metadata,
)

# Validation 모듈 - 동적 검증 시스템
from .validation import (
    ValidationOrchestrator,
    ValidationResult,
)

# MLflow 복원 시스템
from .mlflow_restore import (
    MLflowRecipeRestorer,
    MLflowRecipeSaver,
    save_recipe_to_mlflow,
    restore_recipe_from_mlflow,
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
    "FeastOnlineStore",
    "FeastOfflineStore",
    "Serving",
    "AuthConfig",
    "ArtifactStore",
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
    "MLflowRecipeRestorer",
    "MLflowRecipeSaver",
    "save_recipe_to_mlflow",
    "restore_recipe_from_mlflow",

    # ========== Recipe Building (moved to src/cli/utils/) ==========
    # "build_recipe_interactive",
    # "create_recipe_file",
]

# 버전 정보
__version__ = "4.0.0"

# 패키지 메타데이터
__author__ = "Modern ML Pipeline Team"
__description__ = "Unified Settings Factory with dynamic validation and MLflow integration"