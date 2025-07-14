"""
Settings Pydantic Models
Blueprint v17.0 설정 모델 정의 모듈

이 모듈은 모든 Pydantic 모델들을 관리합니다.
최종적으로 settings/__init__.py에서 Settings 클래스로 통합됩니다.
"""

from pydantic import BaseModel, Field, RootModel
from typing import Dict, Any, List, Optional
from collections.abc import Mapping


# =============================================================================
# 1. 운영 환경 설정 (config/*.yaml)
# =============================================================================

class EnvironmentSettings(BaseModel):
    """환경별 기본 설정"""
    app_env: str
    gcp_project_id: str
    gcp_credential_path: Optional[str] = None


class MlflowSettings(BaseModel):
    """MLflow 실험 추적 설정"""
    tracking_uri: str
    experiment_name: str


class RealtimeFeatureStoreConnectionSettings(BaseModel):
    """실시간 Feature Store 연결 설정"""
    host: str
    port: int
    db: int = 0


class RealtimeFeatureStoreSettings(BaseModel):
    """실시간 Feature Store 설정"""
    store_type: str
    connection: RealtimeFeatureStoreConnectionSettings


class ServingSettings(BaseModel):
    """API 서빙 설정"""
    model_stage: str
    realtime_feature_store: RealtimeFeatureStoreSettings


class ArtifactStoreSettings(BaseModel):
    """아티팩트 저장소 설정"""
    enabled: bool
    base_uri: str


# =============================================================================
# 2. 모델 논리 설정 (recipes/*.yaml)
# =============================================================================

class LoaderSettings(BaseModel):
    """데이터 로더 설정"""
    name: str
    source_uri: str
    local_override_uri: Optional[str] = None


class AugmenterSettings(BaseModel):
    """피처 증강기 설정 (Blueprint v17.0: Feature Store 지원)"""
    
    # 🔄 기존 필드들 (하위 호환성 유지)
    name: Optional[str] = None
    source_uri: Optional[str] = None
    local_override_uri: Optional[str] = None
    
    # 🆕 Feature Store 방식 필드들 (Blueprint v17.0)
    type: Optional[str] = None  # "feature_store" or "sql" (기본값: sql)
    features: Optional[List[Dict[str, Any]]] = None  # Feature Store 피처 설정
    
    def validate_augmenter_config(self):
        """Augmenter 설정의 유효성 검증"""
        if self.type == "feature_store":
            # Feature Store 방식: features가 필요
            if not self.features:
                raise ValueError("Feature Store 방식 Augmenter에는 features 설정이 필요합니다.")
        else:
            # 기존 SQL 방식: source_uri가 필요 (기본값)
            if not self.source_uri:
                raise ValueError("기존 SQL 방식 Augmenter에는 source_uri가 필요합니다.")


class PreprocessorParamsSettings(BaseModel):
    """전처리기 파라미터 설정"""
    criterion_col: Optional[str] = None
    exclude_cols: List[str]


class PreprocessorSettings(BaseModel):
    """전처리기 설정"""
    name: str
    params: PreprocessorParamsSettings


class HyperparameterTuningSettings(BaseModel):
    """하이퍼파라미터 튜닝 설정 (Blueprint v17.0)"""
    enabled: bool = False  # 기본값: 기존 동작 유지
    engine: str = "optuna"
    n_trials: int = 10
    metric: str = "accuracy"
    direction: str = "maximize"
    timeout: Optional[int] = None  # 초 단위, None이면 제한 없음
    pruning: Optional[Dict[str, Any]] = None
    parallelization: Optional[Dict[str, Any]] = None


class FeatureStoreSettings(BaseModel):
    """Feature Store 설정 (Blueprint v17.0: config 통합)"""
    provider: str = "dynamic"
    
    # 🎯 Blueprint 원칙 1 준수: config 내 완전한 Feast 설정
    feast_config: Optional[Dict[str, Any]] = None
    
    # 연결 정보 (하위 호환성)
    connection_timeout: int = 5000
    retry_attempts: int = 3
    connection_info: Dict[str, Any] = {}


class DataInterfaceSettings(BaseModel):
    """데이터 인터페이스 설정 (다양한 ML 태스크 지원)"""
    
    # 필수 필드
    task_type: str  # "classification", "regression", "clustering", "causal"
    
    # 조건부 필수 필드들 (clustering 제외하고 필수)
    target_col: Optional[str] = None
    
    # Causal 전용 필드들 (기존 호환성 유지)
    treatment_col: Optional[str] = None
    treatment_value: Optional[Any] = None
    
    # Classification 전용 필드들
    class_weight: Optional[str] = None  # "balanced" 등
    pos_label: Optional[Any] = None  # 이진 분류용
    average: Optional[str] = "weighted"  # f1 계산 방식
    
    # Regression 전용 필드들
    sample_weight_col: Optional[str] = None
    
    # Clustering 전용 필드들
    n_clusters: Optional[int] = None
    true_labels_col: Optional[str] = None  # 평가용 실제 라벨
    
    # 기존 필드 유지 (Optional로 변경)
    features: Optional[Dict[str, str]] = None
    
    def validate_required_fields(self):
        """task_type에 따른 필수 필드 검증"""
        if self.task_type in ["classification", "regression", "causal"]:
            if not self.target_col:
                raise ValueError(f"{self.task_type} 태스크에는 target_col이 필요합니다.")
        
        if self.task_type == "causal":
            if not self.treatment_col:
                raise ValueError("causal 태스크에는 treatment_col이 필요합니다.")
            if self.treatment_value is None:
                raise ValueError("causal 태스크에는 treatment_value가 필요합니다.")


class ModelHyperparametersSettings(RootModel[Dict[str, Any]]):
    """모델 하이퍼파라미터 설정"""
    root: Dict[str, Any]


class ModelSettings(BaseModel):
    """모델 전체 설정"""
    class_path: str  # 새로 추가: 동적 모델 로딩용
    loader: LoaderSettings
    augmenter: Optional[AugmenterSettings] = None
    preprocessor: Optional[PreprocessorSettings] = None
    data_interface: DataInterfaceSettings
    hyperparameters: ModelHyperparametersSettings
    
    # 🆕 새로 추가 (Optional로 하위 호환성 보장)
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    computed: Optional[Dict[str, Any]] = None


# =============================================================================
# 3. 최종 통합 설정 모델
# =============================================================================

class Settings(BaseModel):
    """
    Blueprint v17.0 통합 설정 모델
    
    config/*.yaml (인프라 설정) + recipes/*.yaml (모델 논리)의 
    완전한 통합 인터페이스를 제공합니다.
    """
    
    # config/*.yaml에서 오는 필드들 (인프라 설정)
    environment: EnvironmentSettings
    mlflow: MlflowSettings
    serving: ServingSettings
    artifact_stores: Dict[str, ArtifactStoreSettings]
    
    # recipes/*.yaml에서 오는 필드 (모델 논리)
    model: ModelSettings
    
    # 🆕 Blueprint v17.0 새로 추가 (Optional로 하위 호환성 보장)
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    feature_store: Optional[FeatureStoreSettings] = None
    
    @classmethod
    def load(cls) -> "Settings":
        """편의 메서드: 기본 설정 로딩"""
        from .loaders import load_settings_by_file
        return load_settings_by_file("default") 