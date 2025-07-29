# src/settings/_recipe_schema.py
from pydantic import BaseModel, Field, RootModel, validator
from typing import Dict, Any, List, Optional, Union, Literal

class JinjaVariable(BaseModel):
    """Jinja 템플릿 변수에 대한 명세서"""
    name: str
    type: Literal["date", "string", "integer", "float", "boolean"]
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None

class OptunaParameterConfig(BaseModel):
    """Optuna 하이퍼파라미터 설정"""
    type: Literal["int", "float", "categorical"]
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    log: Optional[bool] = None
    choices: Optional[List[Any]] = None
    
    @validator('type')
    def validate_optuna_config(cls, v, values):
        if v in ["int", "float"]:
            if 'low' not in values or 'high' not in values:
                raise ValueError(f"{v} 타입 파라미터에는 low, high 값이 필요합니다.")
            if values['low'] >= values['high']:
                raise ValueError("low 값은 high 값보다 작아야 합니다.")
        elif v == "categorical":
            if not values.get('choices'):
                raise ValueError("categorical 타입 파라미터에는 choices가 필요합니다.")
        return v

class HyperparametersSettings(RootModel):
    """Dictionary 형태 하이퍼파라미터 (27개 Recipe 표준)"""
    root: Dict[str, Union[OptunaParameterConfig, Any]]
    
    def get_optuna_params(self) -> Dict[str, OptunaParameterConfig]:
        return {
            key: value for key, value in self.root.items() 
            if isinstance(value, OptunaParameterConfig)
        }
    
    def get_fixed_params(self) -> Dict[str, Any]:
        return {
            key: value for key, value in self.root.items() 
            if not isinstance(value, OptunaParameterConfig)
        }

class EntitySchema(BaseModel):
    """Entity + Timestamp 기반 Point-in-Time 정합성 스키마"""
    entity_columns: List[str]
    timestamp_column: str
    
    @validator('entity_columns', pre=True, always=True)
    def validate_entity_columns(cls, v):
        if not v:
            raise ValueError("Entity 컬럼은 최소 1개 필요합니다.")
        return v

class MLTaskSettings(BaseModel):
    """ML 작업별 세부 설정 - 27개 Recipe data_interface 대응"""
    task_type: str
    target_column: Optional[str] = None
    treatment_column: Optional[str] = None
    treatment_value: Optional[Any] = None
    class_weight: Optional[str] = None
    pos_label: Optional[Any] = None
    average: Optional[str] = "weighted"
    sample_weight_column: Optional[str] = None
    n_clusters: Optional[int] = None
    
    def validate_required_fields(self):
        if self.task_type in ["classification", "regression", "causal"] and not self.target_column:
            raise ValueError(f"{self.task_type} 태스크에는 target_column이 필요합니다.")
        if self.task_type == "causal":
            if not self.treatment_column:
                raise ValueError("causal 태스크에는 treatment_column이 필요합니다.")
            if self.treatment_value is None:
                raise ValueError("causal 태스크에는 treatment_value가 필요합니다.")

class FeatureNamespaceSettings(BaseModel):
    """Feature Store 네임스페이스 설정"""
    feature_namespace: str
    features: List[str]

class AugmenterSettings(BaseModel):
    """피처 증강기 설정 (27개 Recipe 대응)"""
    type: str = "feature_store"
    features: Optional[List[FeatureNamespaceSettings]] = None
    name: Optional[str] = None
    source_uri: Optional[str] = None
    local_override_uri: Optional[str] = None
    
    @validator('type')
    def validate_augmenter_config(cls, v, values):
        if v == "feature_store" and not values.get('features'):
            raise ValueError("Feature Store 방식 Augmenter에는 features 설정이 필요합니다.")
        return v

class PreprocessorStepSettings(BaseModel):
    """단일 전처리 단계 설정"""
    name: str
    transformer: str # column_transforms에 정의된 이름

class ColumnTransformSettings(BaseModel):
    """단일 컬럼 변환 설정"""
    type: str  # Registry에 등록된 '별명' (e.g., "simple_imputer")
    params: Dict[str, Any] = {}
    columns: List[str]

class PreprocessorSettings(BaseModel):
    """전처리기 설정 (Pipeline Builder용)"""
    column_transforms: Dict[str, ColumnTransformSettings] = {}
    steps: List[PreprocessorStepSettings] = []

class ValidationMethodSettings(BaseModel):
    """검증 방법 설정"""
    method: str = "train_test_split"
    test_size: Optional[float] = 0.2
    stratify: Optional[bool] = None
    random_state: Optional[int] = 42
    cv_folds: Optional[int] = 5
    
    @validator('method')
    def validate_method_config(cls, v, values):
        if v == "train_test_split" and (values.get('test_size') is None or not (0.0 < values.get('test_size', 0) < 1.0)):
            raise ValueError("train_test_split에는 0과 1 사이의 test_size가 필요합니다.")
        if v == "cross_validation" and (values.get('cv_folds') is None or values.get('cv_folds', 0) < 2):
            raise ValueError("cross_validation에는 2 이상의 cv_folds가 필요합니다.")
        return v

class EvaluationSettings(BaseModel):
    """평가 설정"""
    metrics: List[str]
    validation: ValidationMethodSettings

class LoaderSettings(BaseModel):
    """데이터 로더 설정 (27개 Recipe 완전 대응)"""
    name: str
    source_uri: str
    entity_schema: EntitySchema
    jinja_variables: Optional[List[JinjaVariable]] = None # [신규] Jinja 변수 명세서
    local_override_uri: Optional[str] = None

class EvaluatorSettings(BaseModel):
    """평가자 설정"""
    name: str

class HyperparameterTuningSettings(BaseModel):
    """하이퍼파라미터 튜닝 설정 (Recipe 논리)"""
    enabled: bool = False
    n_trials: int = 10
    metric: str = "accuracy"
    direction: str = "maximize"

class RecipeMetadataSettings(BaseModel):
    """Recipe 메타데이터 (선택적)"""
    description: Optional[str] = None
    use_cases: Optional[List[str]] = None

class ModelConfigurationSettings(BaseModel):
    """모델 구성 설정 (27개 Recipe 완전 대응)"""
    class_path: str
    hyperparameters: Union[HyperparametersSettings, Dict[str, Any]]
    loader: LoaderSettings
    data_interface: MLTaskSettings
    augmenter: Optional[AugmenterSettings] = None
    preprocessor: Optional[PreprocessorSettings] = None
    evaluator: Optional[EvaluatorSettings] = None
    hyperparameter_tuning: Optional[HyperparameterTuningSettings] = None
    computed: Optional[Dict[str, Any]] = None

class RecipeSettings(BaseModel):
    """완전한 Recipe 설정 (27개 Recipe 완전 대응)"""
    name: str
    model: ModelConfigurationSettings
    evaluation: EvaluationSettings
    metadata: Optional[RecipeMetadataSettings] = None
    
    def validate_recipe_consistency(self):
        self.model.data_interface.validate_required_fields() 