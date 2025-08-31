"""
Settings Pydantic Models
Blueprint v17.0 - 27개 Recipe 완전 대응

Pydantic 모델 정의 모듈
"""

from pydantic import BaseModel
from typing import Dict, Optional

from ._config_schema import (
    EnvironmentSettings,
    MlflowSettings,
    DataAdapterSettings,
    ServingSettings,
    ArtifactStoreSettings,
    FeatureStoreSettings,
    HyperparameterTuningSettings as ConfigHyperparameterTuningSettings,
)
from ._recipe_schema import RecipeSettings

class Settings(BaseModel):
    """
    Blueprint v17.0 통합 설정 모델
    configs/*.yaml (인프라) + recipes/*.yaml (모델 논리)의 통합 인터페이스
    """
    # configs/*.yaml에서 오는 필드들 (인프라 설정)
    environment: EnvironmentSettings
    mlflow: MlflowSettings
    serving: ServingSettings
    artifact_stores: Dict[str, ArtifactStoreSettings]
    data_adapters: Optional[DataAdapterSettings] = None
    feature_store: Optional[FeatureStoreSettings] = None
    
    # 인프라 제약으로서의 HPO 설정
    hyperparameter_tuning: Optional[ConfigHyperparameterTuningSettings] = None
    
    # recipes/*.yaml에서 오는 필드 (모델 논리)
    recipe: RecipeSettings 