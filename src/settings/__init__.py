"""
Settings Module
Blueprint v17.0 설정 시스템 통합 인터페이스

관심사별로 분리된 모듈들의 통합 진입점입니다.
기존 settings.py 코드와 완전히 호환됩니다.
"""

# =============================================================================
# Core Models & Loaders (필수)
# =============================================================================

from .models import (
    # 통합 설정 모델
    Settings,
    
    # 🆕 현대화된 Recipe 모델들
    RecipeSettings,
    ModelConfigurationSettings,
    EvaluationSettings,
    ValidationMethodSettings,
    OptunaParameterConfig,
    ModernHyperparametersSettings,
    EvaluatorSettings,
    
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
    
    # 모델 논리 설정 모델들 (현대화된 Recipe 컴포넌트)
    LoaderSettings,
    AugmenterSettings,
    PreprocessorParamsSettings,
    PreprocessorSettings,
    HyperparameterTuningSettings,
    FeatureStoreSettings,
    DataInterfaceSettings,
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
# Optional Extensions (선택적 import)
# =============================================================================

# Blueprint v17.0 확장 기능들은 필요시에만 import
# Example: from src.settings.extensions import validate_environment_settings

# =============================================================================
# Public API Definition
# =============================================================================

__all__ = [
    # 메인 클래스
    "Settings",
    
    # 로딩 함수들 (기존 호환성)
    "load_settings",
    "load_settings_by_file",
    "load_config_files", 
    "load_recipe_file",
    
    # 모든 Pydantic 모델들
    "EnvironmentSettings",
    "MlflowSettings", 
    "RealtimeFeatureStoreConnectionSettings",
    "RealtimeFeatureStoreSettings",
    "ServingSettings",
    "ArtifactStoreSettings",
    "LoaderSettings",
    "AugmenterSettings",
    "PreprocessorParamsSettings", 
    "PreprocessorSettings",
    "HyperparameterTuningSettings",
    "FeatureStoreSettings",
    "DataInterfaceSettings",
    "ModelHyperparametersSettings",
    "ModelSettings",
    
    # 편의 함수들
    "get_app_env",
    "is_local_env",
    "is_dev_env", 
    "is_prod_env",
    "get_feast_config",
]
