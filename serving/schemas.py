from pydantic import BaseModel, Field, create_model
from typing import Any, List, Type, Dict
import re

# Jinja2 템플릿에서 변수를 추출하기 위한 정규식
# 예: {{ campaign_id }}, {{ member_id }}
JINJA_VAR_PATTERN = re.compile(r"{{\s*(\w+)\s*}}")


def get_pk_from_loader_sql(sql_template: str) -> List[str]:
    """
    Loader의 SQL 템플릿 문자열에서 Jinja2 변수를 추출하여 API의 PK 목록으로 사용합니다.
    환경 변수인 gcp_project_id는 제외합니다.
    """
    # 정규식을 사용하여 모든 Jinja2 변수 찾기
    variables = JINJA_VAR_PATTERN.findall(sql_template)
    # 중복 제거 및 환경 변수 제외
    pk_list = sorted(list(set(v for v in variables if v != "gcp_project_id")))
    return pk_list


def create_dynamic_prediction_request(
    model_name: str, pk_fields: List[str]
) -> Type[BaseModel]:
    """
    추출된 PK 필드 목록을 기반으로 Pydantic 모델을 동적으로 생성합니다.
    """
    # 필드 딕셔너리 생성
    field_annotations = {}
    field_defaults = {}
    
    for field in pk_fields:
        field_annotations[field] = Any
        field_defaults[field] = Field(..., description=f"Primary Key: {field}")
    
    # type()을 사용하여 동적 클래스 생성
    class_name = f"{model_name}PredictionRequest"
    
    # 클래스 속성 딕셔너리
    class_dict = {
        '__annotations__': field_annotations,
        **field_defaults
    }
    
    # BaseModel을 상속받는 동적 클래스 생성
    DynamicModel = type(class_name, (BaseModel,), class_dict)
    
    return DynamicModel


class PredictionResponse(BaseModel):
    """
    단일 예측 결과를 위한 응답 스키마입니다.
    """
    uplift_score: float = Field(..., json_schema_extra={"example": 0.123}, description="계산된 Uplift 점수")
    model_uri: str = Field(
        ...,
        example="models:/uplift-model/Production",
        description="예측에 사용된 모델의 MLflow URI",
    )
    # 🆕 Blueprint v17.0: 최적화 정보 포함 (Optional로 하위 호환성 보장)
    optimization_enabled: bool = Field(default=False, description="하이퍼파라미터 최적화 여부")
    best_score: float = Field(default=0.0, description="최적화 달성 점수 (활성화된 경우)")


def create_batch_prediction_request(
    prediction_request_model: Type[BaseModel],
) -> Type[BaseModel]:
    """
    동적으로 생성된 PredictionRequest 모델을 사용하여 배치 요청 스키마를 생성합니다.
    """
    model_name = prediction_request_model.__name__
    return create_model(
        f"Batch{model_name}",
        samples=(
            List[prediction_request_model],
            Field(..., description="예측을 위한 샘플 리스트"),
        ),
    )


class BatchPredictionResponse(BaseModel):
    """
    배치 예측 결과를 위한 응답 스키마입니다.
    """
    predictions: List[Dict[str, Any]] = Field(
        ..., description="Uplift 점수 리스트 (PK 포함)"
    )
    model_uri: str = Field(
        ...,
        example="models:/uplift-model/Production",
        description="예측에 사용된 모델의 MLflow URI",
    )
    sample_count: int = Field(..., json_schema_extra={"example": 100}, description="처리된 샘플 수")
    # 🆕 Blueprint v17.0: 최적화 정보 포함 (Optional로 하위 호환성 보장)
    optimization_enabled: bool = Field(default=False, description="하이퍼파라미터 최적화 여부")
    best_score: float = Field(default=0.0, description="최적화 달성 점수 (활성화된 경우)")


class HealthCheckResponse(BaseModel):
    """
    헬스 체크 응답 스키마입니다.
    """
    status: str = Field(..., json_schema_extra={"example": "healthy"}, description="서비스 상태")
    model_uri: str = Field(
        ...,
        example="models:/uplift-model/Production",
        description="현재 로드된 모델의 MLflow URI",
    )
    model_name: str = Field(..., json_schema_extra={"example": "xgboost_x_learner"}, description="로드된 모델 이름")


# 🆕 Blueprint v17.0: 새로운 메타데이터 응답 스키마들

class HyperparameterOptimizationInfo(BaseModel):
    """
    하이퍼파라미터 최적화 결과 정보
    """
    enabled: bool = Field(..., description="하이퍼파라미터 최적화 수행 여부")
    engine: str = Field(default="", description="사용된 최적화 엔진 (optuna 등)")
    best_params: Dict[str, Any] = Field(default={}, description="최적 하이퍼파라미터 조합")
    best_score: float = Field(default=0.0, description="달성한 최고 점수")
    total_trials: int = Field(default=0, description="수행된 총 trial 수")
    pruned_trials: int = Field(default=0, description="조기 중단된 trial 수")
    optimization_time: str = Field(default="", description="총 최적화 소요 시간")


class TrainingMethodologyInfo(BaseModel):
    """
    학습 방법론 및 Data Leakage 방지 정보
    """
    train_test_split_method: str = Field(default="", description="데이터 분할 방법")
    train_ratio: float = Field(default=0.8, description="학습 데이터 비율")
    validation_strategy: str = Field(default="", description="검증 전략")
    preprocessing_fit_scope: str = Field(default="", description="전처리 fit 범위 (Data Leakage 방지)")
    random_state: int = Field(default=42, description="재현성을 위한 시드값")


class ModelMetadataResponse(BaseModel):
    """
    모델의 완전한 메타데이터 응답
    """
    model_uri: str = Field(..., description="모델 MLflow URI")
    model_class_path: str = Field(default="", description="모델 클래스 경로")
    hyperparameter_optimization: HyperparameterOptimizationInfo = Field(..., description="하이퍼파라미터 최적화 정보")
    training_methodology: TrainingMethodologyInfo = Field(..., description="학습 방법론 정보")
    training_metadata: Dict[str, Any] = Field(default={}, description="기타 학습 메타데이터")
    api_schema: Dict[str, Any] = Field(default={}, description="동적 생성된 API 스키마 정보")


class OptimizationHistoryResponse(BaseModel):
    """
    하이퍼파라미터 최적화 과정 상세 히스토리
    """
    enabled: bool = Field(..., description="최적화 수행 여부")
    optimization_history: List[Dict[str, Any]] = Field(default=[], description="전체 최적화 과정 기록")
    search_space: Dict[str, Any] = Field(default={}, description="탐색한 하이퍼파라미터 공간")
    convergence_info: Dict[str, Any] = Field(default={}, description="수렴 정보")
    timeout_occurred: bool = Field(default=False, description="타임아웃 발생 여부")