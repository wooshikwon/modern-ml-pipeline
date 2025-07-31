# src/settings/_recipe_schema.py
from pydantic import BaseModel, Field, RootModel, validator, model_validator
from typing import Dict, Any, List, Optional, Union, Literal
import importlib
import inspect

from .compatibility_maps import TASK_METRIC_COMPATIBILITY
from src.utils.system.logger import logger

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
    name: Optional[str] = None
    source_uri: str
    adapter: str  # 데이터 로딩에 사용할 어댑터의 타입(예: 'sql', 'storage')
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

class ModelSettings(BaseModel):
    """모델 설정 (27개 Recipe 완전 대응)"""
    class_path: str
    hyperparameters: Union[HyperparametersSettings, Dict[str, Any]]
    
    @model_validator(mode='after')
    def validate_hyperparameters_are_valid_for_model(cls, v: "ModelSettings") -> "ModelSettings":
        """
        레시피의 하이퍼파라미터가 class_path에 명시된 실제 모델 클래스의
        유효한 초기화 인자인지 동적으로 검증합니다.
        """
        # 1. 검증할 하이퍼파라미터가 없으면 즉시 통과
        if not v.hyperparameters:
            return v

        # 2. 모델 클래스를 동적으로 로드
        try:
            module_path, class_name = v.class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError, ValueError):
            # 클래스를 임포트할 수 없는 경우, 이 검증기는 통과시키고 경고를 남김.
            # (다른 검증 단계나 파이프라인 실행 시점에서 오류가 발생할 것임)
            logger.warning(f"'{v.class_path}' 모델 클래스를 임포트할 수 없어 하이퍼파라미터 검증을 건너뜁니다.")
            return v

        # 3. 모델의 __init__ 시그니처에서 유효한 파라미터 목록을 추출
        try:
            init_signature = inspect.signature(model_class.__init__)
            valid_params = set(init_signature.parameters.keys())
        except (TypeError, ValueError):
            # 일부 C++ 기반 모델처럼 시그니처 분석이 불가능한 경우, 검증을 건너뜀
            logger.warning(f"'{v.class_path}' 모델의 시그니처를 분석할 수 없어 하이퍼파라미터 검증을 건너뜁니다.")
            return v

        # 4. 레시피의 각 하이퍼파라미터가 유효한지 확인
        for param_name in v.hyperparameters.keys():
            if param_name not in valid_params:
                raise ValueError(
                    f"잘못된 하이퍼파라미터: '{param_name}'은(는) 모델 '{v.class_path}'의 유효한 파라미터가 아닙니다. "
                    f"사용 가능한 파라미터: {sorted(list(valid_params))}"
                )

        return v

class RecipeSettings(BaseModel):
    """완전한 Recipe 설정 (27개 Recipe 완전 대응)"""
    name: str
    model: ModelConfigurationSettings
    evaluation: EvaluationSettings
    metadata: Optional[RecipeMetadataSettings] = None
    
    def validate_recipe_consistency(self):
        self.model.data_interface.validate_required_fields() 

    @model_validator(mode='after')
    def validate_recipe_consistency(cls, v: "RecipeSettings") -> "RecipeSettings":
        """
        레시피의 여러 섹션 간의 논리적 일관성을 검증합니다.
        (기존 로직 + 신규 로직)
        """
        # 1. 태스크 타입과 평가지표 호환성 검증
        task_type = v.model.data_interface.task_type
        
        if task_type in TASK_METRIC_COMPATIBILITY:
            allowed_metrics = TASK_METRIC_COMPATIBILITY[task_type]
            for metric in v.evaluation.metrics:
                # 'precision_weighted' -> 'precision' 으로 기본 이름만 검사
                base_metric = metric.split('_')[0]
                if base_metric not in allowed_metrics and metric not in allowed_metrics:
                    raise ValueError(
                        f"평가 지표 '{metric}'은(는) '{task_type}' 태스크 타입과 호환되지 않습니다. "
                        f"'{task_type}'에 사용 가능한 지표: {allowed_metrics}"
                    )
        
        # 2. (기존) Augmenter-Feature Store 호환성 검증
        if v.model.augmenter and v.model.augmenter.type == "feature_store":
            if not v.model.loader.feature_retrieval:
                raise ValueError(
                    "Augmenter 타입이 'feature_store'일 경우, "
                    "loader.feature_retrieval 섹션이 반드시 필요합니다."
                )
        return v 