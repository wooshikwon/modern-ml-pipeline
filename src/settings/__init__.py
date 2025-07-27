"""
Settings Module
Blueprint v17.0 설정 시스템 통합 인터페이스 (27개 Recipe 완전 지원)

관심사별로 분리된 모듈들의 통합 진입점입니다.
27개 Recipe와 완전히 호환됩니다.
"""

# =============================================================================
# Core Models & Loaders (필수)
# =============================================================================

from .models import (
    # 통합 설정 모델
    Settings,
    
    # 🆕 27개 Recipe 모델들
    RecipeSettings,
    ModelConfigurationSettings,
    EvaluationSettings,
    ValidationMethodSettings,
    OptunaParameterConfig,
    HyperparametersSettings,  # 🔄 수정: ModernHyperparametersSettings → HyperparametersSettings
    EvaluatorSettings,
    
    # 🆕 27개 Recipe 추가 모델들
    EntitySchema,
    MLTaskSettings,
    FeatureNamespaceSettings,
    RecipeMetadataSettings,
    
    # 운영 환경 설정 모델들
    EnvironmentSettings,
    MlflowSettings,
    RealtimeFeatureStoreConnectionSettings,
    RealtimeFeatureStoreSettings,
    ServingSettings,
    ArtifactStoreSettings,
    
    # 🆕 Config 기반 Dynamic Factory (Blueprint v17.0)
    DataAdapterSettings,
    AdapterConfigSettings,
    PostgresStorageSettings,
    
    # 모델 논리 설정 모델들 (27개 Recipe 컴포넌트)
    LoaderSettings,
    AugmenterSettings,
    PreprocessorParamsSettings,
    PreprocessorSettings,
    HyperparameterTuningSettings,
    FeatureStoreSettings,
)

from .loaders import (
    # 메인 로딩 함수들
    load_settings,
    load_settings_by_file,
    
    # 개별 로딩 함수들
    load_config_files,
    load_recipe_file,
    
    # 편의 함수들
    get_app_env,
    is_local_env,
    is_dev_env,
    is_prod_env,
    get_feast_config,
)

# =============================================================================
# Public API Definition (27개 Recipe 완전 지원)
# =============================================================================

__all__ = [
    # 메인 클래스
    "Settings",
    
    # 🆕 27개 Recipe 핵심 모델들
    "RecipeSettings",
    "ModelConfigurationSettings", 
    "EntitySchema",
    "MLTaskSettings",
    "HyperparametersSettings",
    "FeatureNamespaceSettings",
    "RecipeMetadataSettings",
    
    # 로딩 함수들 (기존 호환성)
    "load_settings",
    "load_settings_by_file",
    "load_config_files", 
    "load_recipe_file",
    
    # 환경 설정 모델들
    "EnvironmentSettings",
    "MlflowSettings", 
    "RealtimeFeatureStoreConnectionSettings",
    "RealtimeFeatureStoreSettings",
    "ServingSettings",
    "ArtifactStoreSettings",
    
    # 컴포넌트 설정 모델들
    "LoaderSettings",
    "AugmenterSettings",
    "PreprocessorParamsSettings", 
    "PreprocessorSettings",
    "EvaluatorSettings",
    
    # 평가 및 튜닝 모델들
    "EvaluationSettings",
    "ValidationMethodSettings",
    "HyperparameterTuningSettings",
    "OptunaParameterConfig",
    
    # 기타 설정 모델들
    "FeatureStoreSettings",
    "DataAdapterSettings",
    "AdapterConfigSettings",
    "PostgresStorageSettings",
    
    # 편의 함수들
    "get_app_env",
    "is_local_env",
    "is_dev_env", 
    "is_prod_env",
    "get_feast_config",
]
