"""
Settings Pydantic Models
Blueprint v17.0 - 27개 Recipe 완전 대응

Pydantic 모델 정의 모듈
"""

from pydantic import BaseModel, Field, RootModel, validator
from typing import Dict, Any, List, Optional, Union, Literal


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
    """개별 어댑터 설정"""
    class_name: str  # e.g., "FileSystemAdapter", "BigQueryAdapter"
    config: Dict[str, Any] = {}


class DataAdapterSettings(BaseModel):
    """데이터 어댑터 설정 - Config-driven Dynamic Factory"""
    
    # 환경별 기본 어댑터 매핑
    default_loader: str = "filesystem"
    default_storage: str = "filesystem"
    default_feature_store: str = "filesystem"
    
    # 어댑터별 세부 설정
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
# 2. 모델 논리 설정 (recipes/*.yaml) - 27개 Recipe 대응
# =============================================================================

class OptunaParameterConfig(BaseModel):
    """Optuna 하이퍼파라미터 설정"""
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


class HyperparametersSettings(RootModel):
    """Dictionary 형태 하이퍼파라미터 (27개 Recipe 표준)"""
    root: Dict[str, Union[OptunaParameterConfig, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparametersSettings":
        """Dictionary에서 HyperparametersSettings 생성"""
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, dict) and "type" in value:
                # Optuna 형태 파라미터
                processed_data[key] = OptunaParameterConfig(**value)
            else:
                # 고정값 파라미터
                processed_data[key] = value
        return cls(root=processed_data)
    
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


class EntitySchema(BaseModel):
    """Entity + Timestamp 기반 Point-in-Time 정합성 스키마"""
    entity_columns: List[str]    # ["user_id", "product_id"] - PK 정의
    timestamp_column: str        # "event_timestamp" - Point-in-Time 기준
    
    @validator('entity_columns')
    def validate_entity_columns(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Entity 컬럼은 최소 1개 필요합니다.")
        return v
    
    @validator('timestamp_column')
    def validate_timestamp_column(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Timestamp 컬럼이 필요합니다.")
        return v
    
    def get_key_columns(self) -> List[str]:
        """JOIN 키 컬럼들 반환"""
        return self.entity_columns + [self.timestamp_column]


class MLTaskSettings(BaseModel):
    """ML 작업별 세부 설정 - 27개 Recipe data_interface 대응"""
    
    # 필수 필드
    task_type: str  # "classification", "regression", "clustering", "causal"
    
    # 조건부 필수 필드들
    target_column: Optional[str] = None
    
    # Causal 전용 필드들
    treatment_column: Optional[str] = None
    treatment_value: Optional[Any] = None
    
    # Classification 전용 필드들
    class_weight: Optional[str] = None
    pos_label: Optional[Any] = None
    average: Optional[str] = "weighted"
    
    # Regression 전용 필드들
    sample_weight_column: Optional[str] = None
    
    # Clustering 전용 필드들
    n_clusters: Optional[int] = None
    
    def validate_required_fields(self):
        """task_type에 따른 필수 필드 검증"""
        if self.task_type in ["classification", "regression", "causal"]:
            if not self.target_column:
                raise ValueError(f"{self.task_type} 태스크에는 target_column이 필요합니다.")
        
        if self.task_type == "causal":
            if not self.treatment_column:
                raise ValueError("causal 태스크에는 treatment_column이 필요합니다.")
            if self.treatment_value is None:
                raise ValueError("causal 태스크에는 treatment_value가 필요합니다.")
    
    def get_target_fields(self) -> List[str]:
        """타겟 관련 모든 필드 반환"""
        fields = []
        if self.target_column:
            fields.append(self.target_column)
        if self.treatment_column:
            fields.append(self.treatment_column)
        return fields


# 🆕 2.4 Feature Store 설정 (27개 Recipe 대응)
class FeatureNamespaceSettings(BaseModel):
    """Feature Store 네임스페이스 설정"""
    feature_namespace: str
    features: List[str]


class AugmenterSettings(BaseModel):
    """피처 증강기 설정 (27개 Recipe 대응)"""
    
    # 🆕 Feature Store 방식 (27개 Recipe 표준)
    type: str = "feature_store"                           # "feature_store" 표준
    features: Optional[List[FeatureNamespaceSettings]] = None  # Feature 네임스페이스들
    
    # 🔄 기존 필드들 (하위 호환성)
    name: Optional[str] = None
    source_uri: Optional[str] = None
    local_override_uri: Optional[str] = None
    
    def validate_augmenter_config(self):
        """Augmenter 설정의 유효성 검증"""
        if self.type == "feature_store":
            if not self.features:
                raise ValueError("Feature Store 방식 Augmenter에는 features 설정이 필요합니다.")
            for feature_ns in self.features:
                if not feature_ns.features:
                    raise ValueError(f"네임스페이스 '{feature_ns.feature_namespace}'에 피처가 정의되지 않았습니다.")


# 🆕 2.5 전처리기 설정 (27개 Recipe 대응)
class PreprocessorParamsSettings(BaseModel):
    """전처리기 파라미터 설정"""
    criterion_col: Optional[str] = None
    exclude_cols: List[str] = []
    handle_missing: Optional[str] = "median"
    scale_features: Optional[bool] = False
    encode_categorical: Optional[str] = "onehot"


class PreprocessorSettings(BaseModel):
    """전처리기 설정"""
    name: str
    params: PreprocessorParamsSettings


# 🆕 2.6 평가 설정 (27개 Recipe 대응)
class ValidationMethodSettings(BaseModel):
    """검증 방법 설정"""
    method: str = "train_test_split"
    test_size: Optional[float] = 0.2
    stratify: Optional[bool] = None
    random_state: Optional[int] = 42
    cv_folds: Optional[int] = 5
    
    def validate_method_config(self):
        """검증 방법별 설정 유효성 검증"""
        if self.method == "train_test_split":
            if self.test_size is None or not (0.0 < self.test_size < 1.0):
                raise ValueError("train_test_split에는 0과 1 사이의 test_size가 필요합니다.")
        elif self.method == "cross_validation":
            if self.cv_folds is None or self.cv_folds < 2:
                raise ValueError("cross_validation에는 2 이상의 cv_folds가 필요합니다.")


class EvaluationSettings(BaseModel):
    """평가 설정"""
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
                "root_mean_squared_error", "neg_mean_squared_error", 
                "neg_root_mean_squared_error", "neg_mean_absolute_error",
                "neg_mean_absolute_percentage_error"
            },
            "causal": {
                "uplift_auc", "uplift_at_k", "qini_coefficient", "treatment_effect"
            },
            "clustering": {
                "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "inertia"
            }
        }
        
        if task_type not in valid_metrics:
            raise ValueError(f"지원하지 않는 작업 타입: {task_type}")
        
        invalid_metrics = set(self.metrics) - valid_metrics[task_type]
        if invalid_metrics:
            raise ValueError(f"{task_type}에서 지원하지 않는 메트릭: {invalid_metrics}")


# 🆕 2.7 기타 컴포넌트 설정들
class LoaderSettings(BaseModel):
    """데이터 로더 설정 (27개 Recipe 완전 대응)"""
    name: str
    source_uri: str
    entity_schema: EntitySchema                   # Point-in-Time 스키마
    local_override_uri: Optional[str] = None


class EvaluatorSettings(BaseModel):
    """평가자 설정"""
    name: str


class HyperparameterTuningSettings(BaseModel):
    """하이퍼파라미터 튜닝 설정 (27개 Recipe 완전 대응)"""
    enabled: bool = False
    n_trials: int = 10
    metric: str = "accuracy"
    direction: str = "maximize"
    timeout: Optional[int] = None


# 🆕 2.8 Recipe 메타데이터 (새로 추가)
class RecipeMetadataSettings(BaseModel):
    """Recipe 메타데이터 (선택적)"""
    description: Optional[str] = None
    use_cases: Optional[List[str]] = None
    performance_baseline: Optional[Dict[str, Any]] = None
    data_requirements: Optional[Dict[str, Any]] = None


# 🆕 2.9 모델 구성 설정 (27개 Recipe 완전 대응)
class ModelConfigurationSettings(BaseModel):
    """모델 구성 설정 (27개 Recipe 완전 대응)"""
    
    # 🎯 필수 필드들
    class_path: str                                       # 동적 모델 로딩용
    
    # 🎯 하이퍼파라미터 (Dictionary 형태 표준)
    hyperparameters: Union[HyperparametersSettings, Dict[str, Any]]
    
    # 🎯 컴포넌트들 (27개 Recipe 구조)
    loader: LoaderSettings                                # Point-in-Time 데이터 로딩
    data_interface: MLTaskSettings                        # ML 작업별 세부 설정
    augmenter: Optional[AugmenterSettings] = None         # Feature Store 증강
    preprocessor: Optional[PreprocessorSettings] = None   # 전처리
    evaluator: Optional[EvaluatorSettings] = None         # 평가
    
    # 🎯 튜닝 설정
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    
    # 🆕 동적 계산 필드 (loaders.py에서 설정)
    computed: Optional[Dict[str, Any]] = None             # 런타임 계산 필드들
    
    def get_all_exclude_columns(self) -> List[str]:
        """모든 제외 컬럼들 반환"""
        exclude_cols = []
        
        # Entity + Timestamp 컬럼들
        exclude_cols.extend(self.loader.entity_schema.get_key_columns())
        
        # Target 관련 컬럼들
        exclude_cols.extend(self.data_interface.get_target_fields())
        
        # 전처리기 제외 컬럼들
        if self.preprocessor:
            exclude_cols.extend(self.preprocessor.params.exclude_cols)
        
        return list(set(exclude_cols))  # 중복 제거


# 🆕 2.10 Recipe 설정 (27개 Recipe 완전 대응)
class RecipeSettings(BaseModel):
    """
    완전한 Recipe 설정 (27개 Recipe 완전 대응)
    
    모든 Recipe YAML 파일의 구조를 완벽하게 지원:
    - name: Recipe 식별자
    - model: 모델 구성 (class_path, hyperparameters, 컴포넌트들)
    - evaluation: 평가 설정
    - metadata: 메타데이터 (선택적)
    """
    name: str
    model: ModelConfigurationSettings
    evaluation: EvaluationSettings
    metadata: Optional[RecipeMetadataSettings] = None    # 🆕 메타데이터 지원
    
    def validate_recipe_consistency(self):
        """Recipe 전체의 일관성 검증 (27개 Recipe 대응)"""
        # Task type과 evaluation metrics 일관성 확인
        self.evaluation.validate_task_metrics(self.model.data_interface.task_type)
        
        # ML Task 필수 필드 확인
        self.model.data_interface.validate_required_fields()
        
        # Augmenter 설정 검증
        if self.model.augmenter:
            self.model.augmenter.validate_augmenter_config()
        
        # Validation method 설정 검증
        self.evaluation.validation.validate_method_config()
        
        # Optuna 파라미터 검증
        if isinstance(self.model.hyperparameters, HyperparametersSettings):
            for param_name, param_config in self.model.hyperparameters.get_optuna_params().items():
                param_config.validate_optuna_config()


# =============================================================================
# 3. Feature Store 설정 (config 통합)
# =============================================================================

class FeatureStoreSettings(BaseModel):
    """Feature Store 설정"""
    provider: str = "dynamic"
    feast_config: Optional[Dict[str, Any]] = None
    connection_timeout: int = 5000
    retry_attempts: int = 3
    connection_info: Dict[str, Any] = {}


# =============================================================================
# 4. 최종 통합 설정 모델 (27개 Recipe 완전 지원)
# =============================================================================

class Settings(BaseModel):
    """
    Blueprint v17.0 통합 설정 모델 (27개 Recipe 완전 지원)
    
    config/*.yaml (인프라 설정) + recipes/*.yaml (모델 논리)의 
    완전한 통합 인터페이스를 제공합니다.
    """
    
    # config/*.yaml에서 오는 필드들 (인프라 설정)
    environment: EnvironmentSettings
    mlflow: MlflowSettings
    serving: ServingSettings
    artifact_stores: Dict[str, ArtifactStoreSettings]
    
    # 🆕 Config-driven Dynamic Factory
    data_adapters: Optional[DataAdapterSettings] = None
    
    # 🎯 recipes/*.yaml에서 오는 필드 (27개 Recipe 완전 지원)
    recipe: RecipeSettings
    
    # 🆕 추가 설정들
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    feature_store: Optional[FeatureStoreSettings] = None
    
    # 🗑️ 순환 의존성 제거: Settings.load() 편의 메서드 삭제
    # 이유: 어디서도 사용되지 않으며, loaders.py와 순환 의존성을 만듦
    # 대신 직접 load_settings_by_file() 사용 