"""
Recipe Schema - Workflow Definitions (v3.0)
Optuna 튜닝, Feature Store 통합 등 신규 기능 포함
완전히 재작성됨 - CLI recipe.yaml.j2와 100% 호환
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Any, Optional, Literal


class HyperparametersTuning(BaseModel):
    """
    하이퍼파라미터 설정 - Optuna 튜닝 지원
    CLI 템플릿과 정확히 일치하는 구조
    """
    tuning_enabled: bool = Field(False, description="Optuna 하이퍼파라미터 튜닝 활성화")
    
    # Optuna 튜닝 메타 설정 (tuning_enabled=True일 때만)
    optimization_metric: Optional[str] = Field(None, description="최적화할 평가 지표")
    direction: Optional[Literal["minimize", "maximize"]] = Field(None, description="최적화 방향")
    n_trials: Optional[int] = Field(None, ge=1, le=1000, description="튜닝 trial 수")
    timeout: Optional[int] = Field(None, ge=10, description="튜닝 타임아웃(초)")
    
    # 튜닝 활성화시 사용
    fixed: Optional[Dict[str, Any]] = Field(None, description="튜닝시에도 고정할 파라미터")
    tunable: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="튜닝 가능 파라미터 (type, range 포함)"
    )
    
    # 튜닝 비활성화시 사용
    values: Optional[Dict[str, Any]] = Field(None, description="튜닝 비활성화시 사용할 고정값")
    


class Calibration(BaseModel):
    """캘리브레이션 설정 (Classification 태스크에서만 사용)"""
    enabled: bool = Field(False, description="캘리브레이션 활성화 여부")
    method: Optional[str] = Field(None, description="캘리브레이션 방법 ('beta', 'isotonic', 'temperature')")
    


class Model(BaseModel):
    """모델 설정"""
    class_path: str = Field(..., description="모델 클래스 전체 경로 (예: sklearn.ensemble.RandomForestClassifier)")
    library: str = Field(..., description="라이브러리 이름 (sklearn, xgboost, lightgbm, catboost 등)")
    hyperparameters: HyperparametersTuning = Field(
        default_factory=lambda: HyperparametersTuning(tuning_enabled=False, values={}),
        description="하이퍼파라미터 설정"
    )
    calibration: Optional[Calibration] = Field(None, description="캘리브레이션 설정 (Classification 태스크에서만 사용)")
    
    # 런타임에 추가되는 필드
    computed: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="런타임 계산 필드 (run_name 등)"
    )


class FeatureView(BaseModel):
    """Feast FeatureView 정의 (개별 피처 그룹)"""
    join_key: str = Field(..., description="Join할 기준 컬럼 (user_id, item_id 등)")
    features: List[str] = Field(..., description="해당 FeatureView에서 가져올 피처 목록")


class Loader(BaseModel):
    """데이터 로더 설정"""
    source_uri: Optional[str] = Field(None, description="데이터 소스 URI (SQL 파일 경로 또는 데이터 파일 경로)")
    
    def get_adapter_type(self) -> str:
        """source_uri에서 어댑터 타입 자동 추론"""
        if not self.source_uri:
            # 런타임 인자(예: context_params['data_source_uri'])로 주입하는 구조를 우선
            # 미지정 시 기본은 SQL로 간주하여 어댑터 선택
            return 'sql'
        uri = self.source_uri.lower()
        
        # .sql 파일은 SQL adapter
        if uri.endswith('.sql'):
            return 'sql'
        # 데이터 파일은 Storage adapter
        elif any(uri.endswith(ext) for ext in ['.csv', '.parquet', '.json', '.feather']):
            return 'storage'
        # 기본값은 SQL (쿼리 문자열로 가정)
        else:
            return 'sql'


class Fetcher(BaseModel):
    """피처 페처 설정 - Feature Store 통합"""
    type: Literal["feature_store", "pass_through"] = Field(..., description="페처 타입")
    
    # 새로운 구조: feature_views
    feature_views: Optional[Dict[str, FeatureView]] = Field(
        None, 
        description="Feast FeatureView 설정 (feature_store 타입에서 사용)"
    )
    
    # 새로운 필드: timestamp_column
    timestamp_column: Optional[str] = Field(
        None,
        description="Point-in-time join 기준 타임스탬프 컬럼"
    )
    


class DataInterface(BaseModel):
    """데이터 인터페이스 설정"""
    target_column: Optional[str] = Field(None, description="타겟 컬럼 이름 (clustering에서는 None)")
    
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="피처 컬럼 목록 (None이면 target, treatment, entity 제외 모든 컬럼 사용)"
    )
    
    treatment_column: Optional[str] = Field(
        None, 
        description="처치 변수 컬럼 (causal task에서만 사용)"
    )
    
    # id_column → entity_columns 변경
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록 (user_id, item_id 등)")
    
    # ✅ Timeseries 전용 필드들
    timestamp_column: Optional[str] = Field(None, description="시계열 타임스탬프 컬럼 (timeseries task에서 필수)")
    
    # validation 로직은 Recipe 레벨로 이동


class DataSplit(BaseModel):
    """데이터 분할 비율 설정"""
    train: float = Field(..., description="학습용 데이터 비율", ge=0, le=1)
    validation: float = Field(..., description="검증용 데이터 비율", ge=0, le=1)
    test: float = Field(..., description="테스트용 데이터 비율", ge=0, le=1)
    calibration: float = Field(0.0, description="보정용 데이터 비율", ge=0, le=1)
    


class Data(BaseModel):
    """데이터 설정"""
    loader: Loader = Field(..., description="데이터 로더 설정")
    fetcher: Fetcher = Field(..., description="피처 페처 설정")
    data_interface: DataInterface = Field(..., description="데이터 인터페이스 설정")
    split: DataSplit = Field(..., description="데이터 분할 설정")


class PreprocessorStep(BaseModel):
    """
    전처리 단계 - 다양한 전처리 타입 지원
    CLI 템플릿의 모든 전처리 타입 포함
    """
    type: Literal[
        # Scaler
        "standard_scaler",
        "min_max_scaler", 
        "robust_scaler",
        # Encoder
        "one_hot_encoder",
        "ordinal_encoder",
        "catboost_encoder",
        # Imputer (missing_indicator 기능은 simple_imputer의 create_missing_indicators 옵션으로 통합)
        "simple_imputer",
        # Feature Engineering
        "polynomial_features",
        "tree_based_feature_generator",
        "kbins_discretizer"
    ] = Field(..., description="전처리 타입")
    
    columns: Optional[List[str]] = Field(None, description="적용할 컬럼 목록 (Global 전처리기는 선택사항)")
    
    # 타입별 추가 파라미터
    strategy: Optional[Literal[
        # SimpleImputer 전략
        "mean", "median", "most_frequent", "constant",
        # KBinsDiscretizer 전략
        "uniform", "quantile", "kmeans"
    ]] = Field(
        None, 
        description="SimpleImputer 또는 KBinsDiscretizer 전략"
    )
    degree: Optional[int] = Field(None, ge=2, le=5, description="PolynomialFeatures 차수")
    n_bins: Optional[int] = Field(None, ge=2, le=20, description="KBinsDiscretizer 구간 개수")
    sigma: Optional[float] = Field(None, ge=0.0, le=1.0, description="CatBoostEncoder regularization")
    create_missing_indicators: Optional[bool] = Field(
        None, 
        description="SimpleImputer에서 imputation 전 결측값 위치를 나타내는 indicator 컬럼 생성 여부"
    )
    


class Preprocessor(BaseModel):
    """전처리 파이프라인"""
    steps: List[PreprocessorStep] = Field(
        default_factory=list,
        description="전처리 단계 목록 (순서대로 적용)"
    )


class ValidationConfig(BaseModel):
    """검증 설정"""
    method: Literal["train_test_split", "cross_validation"] = Field(
        "train_test_split",
        description="검증 방법"
    )
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="테스트 세트 비율")
    random_state: int = Field(42, description="랜덤 시드")
    
    # Cross validation용 (선택)
    n_folds: Optional[int] = Field(None, ge=2, le=10, description="Cross validation fold 수")
    


class Evaluation(BaseModel):
    """평가 설정"""
    metrics: List[str] = Field(..., description="평가 메트릭 목록")
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="검증 설정"
    )
    


class Metadata(BaseModel):
    """메타데이터"""
    author: Optional[str] = Field("CLI Recipe Builder", description="작성자")
    created_at: str = Field(..., description="생성 시간")
    description: Optional[str] = Field(None, description="Recipe 설명")
    tuning_note: Optional[str] = Field(None, description="튜닝 관련 메모")


class Recipe(BaseModel):
    """
    루트 레시피 설정 (recipes/*.yaml)
    CLI recipe.yaml.j2 템플릿과 100% 호환
    """
    name: str = Field(..., description="레시피 이름")
    task_choice: Literal["classification", "regression", "clustering", "causal", "timeseries"] = Field(
        ..., 
        description="사용자가 Recipe Builder에서 선택한 ML 태스크"
    )
    model: Model = Field(..., description="모델 설정")
    data: Data = Field(..., description="데이터 설정")
    preprocessor: Optional[Preprocessor] = Field(None, description="전처리 파이프라인")
    evaluation: Evaluation = Field(..., description="평가 설정")
    metadata: Optional[Metadata] = Field(None, description="메타데이터")
    
    
    def get_task_type(self) -> str:
        """ML 태스크 타입 반환 (task_choice 기반)"""
        return self.task_choice
    
    def get_metrics(self) -> List[str]:
        """평가 메트릭 목록 반환"""
        return self.evaluation.metrics
    
    def is_tuning_enabled(self) -> bool:
        """하이퍼파라미터 튜닝 활성화 여부"""
        return self.model.hyperparameters.tuning_enabled
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """현재 하이퍼파라미터 반환 (튜닝 여부에 따라 다름)"""
        hp = self.model.hyperparameters
        
        if hp.tuning_enabled:
            # 튜닝 활성화시 fixed 파라미터만 반환
            return hp.fixed or {}
        else:
            # 튜닝 비활성화시 values 반환
            return hp.values or {}
    
    def get_tunable_params(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """튜닝 가능한 파라미터 반환"""
        if self.is_tuning_enabled():
            return self.model.hyperparameters.tunable
        return None
    
