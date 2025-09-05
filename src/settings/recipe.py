"""
Recipe Schema - Workflow Definitions (v3.0)
Optuna 튜닝, Feature Store 통합 등 신규 기능 포함
완전히 재작성됨 - CLI recipe.yaml.j2와 100% 호환
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Any, Optional, Union, Literal


class HyperparametersTuning(BaseModel):
    """
    하이퍼파라미터 설정 - Optuna 튜닝 지원
    CLI 템플릿과 정확히 일치하는 구조
    """
    tuning_enabled: bool = Field(False, description="Optuna 하이퍼파라미터 튜닝 활성화")
    
    # 튜닝 활성화시 사용
    fixed: Optional[Dict[str, Any]] = Field(None, description="튜닝시에도 고정할 파라미터")
    tunable: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="튜닝 가능 파라미터 (type, range 포함)"
    )
    
    # 튜닝 비활성화시 사용
    values: Optional[Dict[str, Any]] = Field(None, description="튜닝 비활성화시 사용할 고정값")
    
    @field_validator('fixed', 'tunable')
    def validate_tuning_params(cls, v, info):
        """튜닝 활성화시 fixed/tunable 검증"""
        if info.data.get('tuning_enabled'):
            if info.field_name == 'tunable' and v:
                # tunable 파라미터 구조 검증
                for param, spec in v.items():
                    if 'type' not in spec:
                        raise ValueError(f"{param}에 'type'이 필요합니다")
                    if 'range' not in spec:
                        raise ValueError(f"{param}에 'range'가 필요합니다")
                    # type 검증
                    if spec['type'] not in ['int', 'float', 'categorical']:
                        raise ValueError(f"{param}의 type은 int/float/categorical 중 하나여야 합니다")
        return v
    
    @field_validator('values')
    def validate_fixed_values(cls, v, info):
        """튜닝 비활성화시 values 검증"""
        if not info.data.get('tuning_enabled') and not v:
            # 튜닝이 비활성화되었는데 values가 없으면 경고
            pass  # 빈 딕셔너리라도 허용
        return v


class Model(BaseModel):
    """모델 설정"""
    class_path: str = Field(..., description="모델 클래스 전체 경로 (예: sklearn.ensemble.RandomForestClassifier)")
    library: str = Field(..., description="라이브러리 이름 (sklearn, xgboost, lightgbm, catboost 등)")
    hyperparameters: HyperparametersTuning = Field(
        default_factory=lambda: HyperparametersTuning(tuning_enabled=False, values={}),
        description="하이퍼파라미터 설정"
    )
    
    # 런타임에 추가되는 필드
    computed: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="런타임 계산 필드 (run_name 등)"
    )


class EntitySchema(BaseModel):
    """엔티티 스키마 - Feature Store Point-in-time join용"""
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록 (예: user_id, item_id)")
    timestamp_column: str = Field(..., description="타임스탬프 컬럼 (point-in-time join 기준)")


class Loader(BaseModel):
    """데이터 로더 설정"""
    source_uri: str = Field(..., description="데이터 소스 URI (SQL 파일 경로 또는 데이터 파일 경로)")
    entity_schema: EntitySchema = Field(..., description="엔티티 스키마 정의")
    
    def get_adapter_type(self) -> str:
        """source_uri에서 어댑터 타입 자동 추론"""
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


class FeatureNamespace(BaseModel):
    """Feature Store 네임스페이스"""
    feature_namespace: str = Field(..., description="피처 네임스페이스 이름")
    features: List[str] = Field(..., description="해당 네임스페이스에서 가져올 피처 목록")


class Fetcher(BaseModel):
    """
    피처 페처 설정 - Feature Store 통합
    """
    type: Literal["feature_store", "pass_through"] = Field(..., description="페처 타입")
    features: Optional[List[FeatureNamespace]] = Field(
        None, 
        description="Feature Store에서 가져올 피처 정의"
    )
    
    @field_validator('features')
    def validate_features(cls, v, info):
        """feature_store 타입일 때 features 필수"""
        if info.data.get('type') == 'feature_store':
            if not v:
                # 빈 리스트라도 허용
                return []
        elif info.data.get('type') == 'pass_through':
            if v:
                # pass_through인데 features가 있으면 무시
                pass
        return v


class DataInterface(BaseModel):
    """데이터 인터페이스 설정"""
    task_type: Literal["classification", "regression", "clustering", "causal"] = Field(
        ..., 
        description="ML 태스크 타입"
    )
    target_column: str = Field(..., description="타겟 컬럼 이름")
    feature_columns: Optional[List[str]] = Field(
        None, 
        description="피처 컬럼 목록 (None이면 target 제외 모든 컬럼)"
    )
    id_column: Optional[str] = Field(None, description="ID 컬럼 (추적용)")


class Data(BaseModel):
    """데이터 설정"""
    loader: Loader = Field(..., description="데이터 로더 설정")
    fetcher: Fetcher = Field(..., description="피처 페처 설정")
    data_interface: DataInterface = Field(..., description="데이터 인터페이스 설정")


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
        # Imputer
        "simple_imputer",
        # Feature Engineering
        "polynomial_features",
        "tree_based_feature_generator",
        "missing_indicator",
        "kbins_discretizer"
    ] = Field(..., description="전처리 타입")
    
    columns: List[str] = Field(..., description="적용할 컬럼 목록")
    
    # 타입별 추가 파라미터
    strategy: Optional[Literal["mean", "median", "most_frequent", "constant"]] = Field(
        None, 
        description="SimpleImputer 전략"
    )
    degree: Optional[int] = Field(None, ge=2, le=5, description="PolynomialFeatures 차수")
    n_bins: Optional[int] = Field(None, ge=2, le=20, description="KBinsDiscretizer 구간 개수")
    sigma: Optional[float] = Field(None, ge=0.0, le=1.0, description="CatBoostEncoder regularization")
    
    @model_validator(mode='after')
    def validate_step_params(self):
        """전처리 타입별 필수 파라미터 검증 및 기본값 설정"""
        # simple_imputer는 strategy가 필수
        if self.type == 'simple_imputer' and not self.strategy:
            raise ValueError("simple_imputer는 strategy가 필요합니다")
        
        # polynomial_features는 degree 기본값 설정
        if self.type == 'polynomial_features' and self.degree is None:
            self.degree = 2
        
        # kbins_discretizer는 n_bins 기본값 설정
        if self.type == 'kbins_discretizer' and self.n_bins is None:
            self.n_bins = 5
        
        return self


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
    
    @model_validator(mode='after')
    def validate_cross_validation(self):
        """cross_validation일 때 n_folds 기본값 설정"""
        if self.method == 'cross_validation' and self.n_folds is None:
            self.n_folds = 5  # 기본값
        return self


class Evaluation(BaseModel):
    """평가 설정"""
    metrics: List[str] = Field(..., description="평가 메트릭 목록")
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="검증 설정"
    )
    
    @field_validator('metrics')
    def validate_metrics(cls, v):
        """메트릭 이름 정규화"""
        if not v:
            raise ValueError("최소 하나의 메트릭이 필요합니다")
        # 소문자로 정규화
        return [m.lower() for m in v]


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
    model: Model = Field(..., description="모델 설정")
    data: Data = Field(..., description="데이터 설정")
    preprocessor: Optional[Preprocessor] = Field(None, description="전처리 파이프라인")
    evaluation: Evaluation = Field(..., description="평가 설정")
    metadata: Optional[Metadata] = Field(None, description="메타데이터")
    
    def get_task_type(self) -> str:
        """ML 태스크 타입 반환"""
        return self.data.data_interface.task_type
    
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
    
    class Config:
        """Pydantic 설정"""
        json_schema_extra = {
            "example": {
                "name": "classification_rf",
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "library": "sklearn",
                    "hyperparameters": {
                        "tuning_enabled": True,
                        "fixed": {
                            "random_state": 42,
                            "n_jobs": -1
                        },
                        "tunable": {
                            "n_estimators": {
                                "type": "int",
                                "range": [50, 200]
                            },
                            "max_depth": {
                                "type": "int", 
                                "range": [5, 20]
                            }
                        }
                    }
                },
                "data": {
                    "loader": {
                        "source_uri": "sql/train_data.sql",
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "event_timestamp"
                        }
                    },
                    "fetcher": {
                        "type": "feature_store",
                        "features": [
                            {
                                "feature_namespace": "user_features",
                                "features": ["age", "gender", "location"]
                            }
                        ]
                    },
                    "data_interface": {
                        "task_type": "classification",
                        "target_column": "label",
                        "feature_columns": None
                    }
                },
                "preprocessor": {
                    "steps": [
                        {
                            "type": "standard_scaler",
                            "columns": ["age", "income"]
                        },
                        {
                            "type": "one_hot_encoder",
                            "columns": ["gender", "location"]
                        }
                    ]
                },
                "evaluation": {
                    "metrics": ["accuracy", "f1", "roc_auc"],
                    "validation": {
                        "method": "train_test_split",
                        "test_size": 0.2,
                        "random_state": 42
                    }
                },
                "metadata": {
                    "author": "Data Scientist",
                    "created_at": "2024-01-01 12:00:00",
                    "description": "Random Forest classifier with Optuna tuning",
                    "tuning_note": "Optuna will optimize n_estimators and max_depth"
                }
            }
        }