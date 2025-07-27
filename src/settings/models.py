"""
Settings Pydantic Models
Blueprint v17.0 설정 모델 정의 모듈

이 모듈은 모든 Pydantic 모델들을 관리합니다.
최종적으로 settings/__init__.py에서 Settings 클래스로 통합됩니다.
"""

from pydantic import BaseModel, Field, RootModel
from typing import Dict, Any, List, Optional, Union, Literal
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


class AdapterConfigSettings(BaseModel):
    """개별 어댑터 설정 (Blueprint v17.0: Config-driven Dynamic Factory)"""
    class_name: str  # e.g., "FileSystemAdapter", "BigQueryAdapter"
    config: Dict[str, Any] = {}  # 어댑터별 구체적 설정


class DataAdapterSettings(BaseModel):
    """
    데이터 어댑터 설정 (Blueprint v17.0: Config-driven Dynamic Factory)
    
    환경별 어댑터 매핑과 동적 어댑터 생성을 위한 설정 모델.
    Blueprint 원칙 1 "레시피는 논리, 설정은 인프라"를 완전히 구현합니다.
    """
    
    # 환경별 기본 어댑터 매핑
    default_loader: str = "filesystem"
    default_storage: str = "filesystem"
    default_feature_store: str = "filesystem"
    
    # 어댑터별 구체적 설정
    adapters: Dict[str, AdapterConfigSettings] = {}
    
    def get_adapter_config(self, adapter_name: str) -> AdapterConfigSettings:
        """어댑터 설정 조회"""
        if adapter_name not in self.adapters:
            raise ValueError(f"어댑터 설정을 찾을 수 없습니다: {adapter_name}")
        return self.adapters[adapter_name]
    
    def get_default_adapter(self, purpose: str) -> str:
        """목적별 기본 어댑터 조회"""
        purpose_mapping = {
            "loader": self.default_loader,
            "storage": self.default_storage,
            "feature_store": self.default_feature_store,
        }
        
        if purpose not in purpose_mapping:
            raise ValueError(f"지원하지 않는 어댑터 목적: {purpose}")
        
        return purpose_mapping[purpose]


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


class PostgresStorageSettings(BaseModel):
    """PostgreSQL 저장 설정"""
    enabled: bool = False
    table_name: str = "batch_predictions"
    connection_uri: str


class ArtifactStoreSettings(BaseModel):
    """아티팩트 저장소 설정"""
    enabled: bool
    base_uri: str
    postgres_storage: Optional[PostgresStorageSettings] = None


# =============================================================================
# 2. 모델 논리 설정 (recipes/*.yaml)
# =============================================================================

# 🆕 2.1 Optuna 하이퍼파라미터 지원 모델들
class OptunaParameterConfig(BaseModel):
    """Optuna 하이퍼파라미터 설정 (Blueprint v17.0: Automated HPO)"""
    type: Literal["int", "float", "categorical"]
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    log: Optional[bool] = None
    choices: Optional[List[Any]] = None
    
    def validate_optuna_config(self):
        """Optuna 파라미터 설정 유효성 검증"""
        if self.type in ["int", "float"]:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.type} 타입 파라미터에는 low, high 값이 필요합니다.")
            if self.low >= self.high:
                raise ValueError("low 값은 high 값보다 작아야 합니다.")
        elif self.type == "categorical":
            if not self.choices or len(self.choices) == 0:
                raise ValueError("categorical 타입 파라미터에는 choices가 필요합니다.")


class ModernHyperparametersSettings(BaseModel):
    """현대화된 하이퍼파라미터 설정 (Optuna + 고정값 지원)"""
    root: Dict[str, Union[OptunaParameterConfig, Any]]
    
    def get_optuna_params(self) -> Dict[str, OptunaParameterConfig]:
        """Optuna 최적화 대상 파라미터만 추출"""
        return {
            key: value for key, value in self.root.items() 
            if isinstance(value, OptunaParameterConfig)
        }
    
    def get_fixed_params(self) -> Dict[str, Any]:
        """고정값 파라미터만 추출"""
        return {
            key: value for key, value in self.root.items() 
            if not isinstance(value, OptunaParameterConfig)
        }


# 🆕 2.2 평가 설정 모델들
class ValidationMethodSettings(BaseModel):
    """검증 방법 설정"""
    method: Literal["train_test_split", "cross_validation", "unsupervised"]
    test_size: Optional[float] = 0.2
    stratify: Optional[bool] = None
    random_state: Optional[int] = 42
    cv_folds: Optional[int] = 5  # cross_validation용
    
    def validate_method_config(self):
        """검증 방법별 설정 유효성 검증"""
        if self.method == "train_test_split":
            if self.test_size is None or not (0.0 < self.test_size < 1.0):
                raise ValueError("train_test_split에는 0과 1 사이의 test_size가 필요합니다.")
        elif self.method == "cross_validation":
            if self.cv_folds is None or self.cv_folds < 2:
                raise ValueError("cross_validation에는 2 이상의 cv_folds가 필요합니다.")


class EvaluationSettings(BaseModel):
    """평가 설정 (Blueprint v17.0: 작업별 특화 메트릭 지원)"""
    metrics: List[str]
    validation: ValidationMethodSettings
    
    def validate_task_metrics(self, task_type: str):
        """작업 타입별 메트릭 유효성 검증"""
        valid_metrics = {
            "classification": {
                "accuracy", "precision", "recall", "f1_score", "roc_auc",
                "precision_weighted", "recall_weighted", "f1_weighted"
            },
            "regression": {
                "r2_score", "mean_squared_error", "mean_absolute_error", 
                "root_mean_squared_error", "neg_mean_squared_error", "neg_root_mean_squared_error"
            },
            "causal": {
                "uplift_auc", "uplift_at_k", "qini_coefficient", "treatment_effect"
            },
            "clustering": {
                "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", 
                "inertia", "adjusted_rand_score"
            }
        }
        
        if task_type not in valid_metrics:
            raise ValueError(f"지원하지 않는 작업 타입: {task_type}")
        
        invalid_metrics = set(self.metrics) - valid_metrics[task_type]
        if invalid_metrics:
            raise ValueError(f"{task_type}에서 지원하지 않는 메트릭: {invalid_metrics}")


# 2.3 모델 논리 컴포넌트 설정들
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
        elif self.type == "pass_through":
            # Blueprint 원칙 9: LOCAL 환경의 의도적 제약 - 추가 검증 불필요
            pass
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


class EvaluatorSettings(BaseModel):
    """평가자 설정 (Blueprint v17.0: 새로 추가)"""
    name: str


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


# 🆕 2.4 현대화된 모델 설정 (Recipe 구조 지원)
class ModelConfigurationSettings(BaseModel):
    """모델 구성 설정 (Blueprint v17.0: Recipe model 섹션)"""
    class_path: str  # 동적 모델 로딩용
    
    # 하이퍼파라미터 (현대화된/기존 형식 모두 지원)
    hyperparameters: Union[ModernHyperparametersSettings, Dict[str, Any]]
    
    # 기본 컴포넌트들
    loader: LoaderSettings
    augmenter: Optional[AugmenterSettings] = None
    preprocessor: Optional[PreprocessorSettings] = None
    data_interface: DataInterfaceSettings
    evaluator: Optional[EvaluatorSettings] = None
    
    # 튜닝 설정
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    
    # 계산된 필드들
    computed: Optional[Dict[str, Any]] = None


# 🆕 2.5 최상위 Recipe 설정
class RecipeSettings(BaseModel):
    """
    완전한 Recipe 설정 (Blueprint v17.0: Recipe YAML 완전 매핑)
    
    현대화된 25개 Recipe 파일의 구조를 완전히 지원합니다:
    - name: Recipe 식별자
    - model: 모델 구성 설정
    - evaluation: 평가 설정
    """
    name: str
    model: ModelConfigurationSettings
    evaluation: EvaluationSettings
    
    def validate_recipe_consistency(self):
        """Recipe 전체의 일관성 검증"""
        # Task type과 evaluation metrics 일관성 확인
        self.evaluation.validate_task_metrics(self.model.data_interface.task_type)
        
        # Data interface 필수 필드 확인
        self.model.data_interface.validate_required_fields()
        
        # Augmenter 설정 검증
        if self.model.augmenter:
            self.model.augmenter.validate_augmenter_config()
        
        # Validation method 설정 검증
        self.evaluation.validation.validate_method_config()
        
        # Optuna 파라미터 검증 (현대화된 형식인 경우)
        if isinstance(self.model.hyperparameters, ModernHyperparametersSettings):
            for param_name, param_config in self.model.hyperparameters.get_optuna_params().items():
                param_config.validate_optuna_config()


# =============================================================================
# 3. 최종 통합 설정 모델
# =============================================================================

class Settings(BaseModel):
    """
    Blueprint v17.0 통합 설정 모델
    
    config/*.yaml (인프라 설정) + recipes/*.yaml (모델 논리)의 
    완전한 통합 인터페이스를 제공합니다.
    
    현대화된 Recipe 구조 전용 (레거시 지원 제거)
    """
    
    # config/*.yaml에서 오는 필드들 (인프라 설정)
    environment: EnvironmentSettings
    mlflow: MlflowSettings
    serving: ServingSettings
    artifact_stores: Dict[str, ArtifactStoreSettings]
    
    # 🆕 Blueprint v17.0: Config-driven Dynamic Factory
    data_adapters: Optional[DataAdapterSettings] = None
    
    # recipes/*.yaml에서 오는 필드 (모델 논리) - 현대화된 Recipe 전용
    recipe: RecipeSettings
    
    # 🆕 Blueprint v17.0 새로 추가 (Optional로 하위 호환성 보장)
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    feature_store: Optional[FeatureStoreSettings] = None
    
    @classmethod
    def load(cls) -> "Settings":
        """편의 메서드: 기본 설정 로딩"""
        from .loaders import load_settings_by_file
        return load_settings_by_file("default") 