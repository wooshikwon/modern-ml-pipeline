"""
Settings Module Public API (v3.0)
CLI 템플릿과 완벽 호환되는 새로운 API
완전히 재작성됨 - 모든 새로운 클래스 export
"""

# Loader 모듈 - Settings 컨테이너와 로딩 함수
from .loader import (
    Settings,
    load_settings,
    create_settings_for_inference,
    load_config_files,
    resolve_env_variables,
)

# Config 모듈 - 인프라 설정 스키마
from .config import (
    Config,
    Environment,
    MLflow,
    DataSource,
    FeatureStore,
    FeastConfig,
    FeastOnlineStore,
    FeastOfflineStore,
    Serving,
    AuthConfig,
    ArtifactStore,
)

# Recipe 모듈 - 워크플로우 정의 스키마
from .recipe import (
    Recipe,
    Model,
    HyperparametersTuning,
    Data,
    Loader,
    EntitySchema,
    Fetcher,
    FeatureNamespace,
    DataInterface,
    Preprocessor,
    PreprocessorStep,
    Evaluation,
    ValidationConfig,
    Metadata,
)

# Validator 모듈 - 검증 및 모델 카탈로그
from .validator import (
    Validator,
    ModelCatalog,
    ModelSpec,
    HyperparameterSpec,
    TunableParameter,
    validate,  # 호환성 별칭
)


# 추론/서빙 전용 함수 (기존 API 호환성)
def create_settings_for_inference(config_data):
    """
    추론/서빙용 최소 Settings 생성 (호환성 유지)
    
    Args:
        config_data: Config 딕셔너리 데이터
        
    Returns:
        최소 Settings 객체
    """
    from .loader import create_settings_for_inference as _create
    return _create(config_data)


# 주요 클래스/함수 export
__all__ = [
    # ========== Core Classes ==========
    "Settings",
    "Config",
    "Recipe",
    "Validator",
    
    # ========== Loading Functions ==========
    "load_settings",
    "create_settings_for_inference",
    "load_config_files",
    "resolve_env_variables",
    
    # ========== Config Schemas ==========
    "Environment",
    "MLflow",
    "DataSource",
    "FeatureStore",
    "FeastConfig",
    "FeastOnlineStore",
    "FeastOfflineStore",
    "Serving",
    "AuthConfig",
    "ArtifactStore",
    
    # ========== Recipe Schemas ==========
    "Model",
    "HyperparametersTuning",
    "Data",
    "Loader",
    "EntitySchema",
    "Fetcher",
    "FeatureNamespace",
    "DataInterface",
    "Preprocessor",
    "PreprocessorStep",
    "Evaluation",
    "ValidationConfig",
    "Metadata",
    
    # ========== Validator Classes ==========
    "ModelCatalog",
    "ModelSpec",
    "HyperparameterSpec",
    "TunableParameter",
    "validate",  # 호환성 별칭
]

# 버전 정보
__version__ = "3.0.0"

# 패키지 메타데이터
__author__ = "Modern ML Pipeline Team"
__description__ = "Settings management for ML pipeline with CLI template compatibility"