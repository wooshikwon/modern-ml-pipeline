from pydantic import BaseModel, Field, create_model
from typing import Any, List, Type

from src.settings.settings import Settings


def create_dynamic_prediction_request(settings: Settings) -> Type[BaseModel]:
    """
    주어진 settings 객체의 피처 정보를 기반으로 Pydantic 모델을 동적으로 생성합니다.
    """
    features = settings.model.data_interface.features
    
    # API 요청 본문에 포함될 필요가 없는 컬럼들
    exclude_cols = {
        'member_id',  # 일반적으로 요청 시점에는 알 수 없는 내부 식별자
        settings.model.data_interface.target_col,
        settings.model.data_interface.treatment_col
    }
    
    fields_to_create = {
        feature: (Any, Field(..., description=f"Feature: {feature}"))
        for feature in features if feature not in exclude_cols
    }
    
    # 동적으로 생성할 필드가 없으면 빈 모델을 반환
    if not fields_to_create:
        return create_model('PredictionRequest')

    return create_model('PredictionRequest', **fields_to_create)


class PredictionResponse(BaseModel):
    """
    단일 예측 결과를 위한 응답 스키마입니다.
    """
    uplift_score: float = Field(..., example=0.123, description="계산된 Uplift 점수")
    model_uri: str = Field(..., example="models:/uplift-model/Production", description="예측에 사용된 모델의 MLflow URI")


def create_batch_prediction_request(prediction_request_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    동적으로 생성된 PredictionRequest 모델을 사용하여 배치 요청 스키마를 생성합니다.
    """
    return create_model(
        'BatchPredictionRequest',
        samples=(List[prediction_request_model], Field(..., description="예측을 위한 샘플 리스트"))
    )


class BatchPredictionResponse(BaseModel):
    """
    배치 예측 결과를 위한 응답 스키마입니다.
    """
    predictions: List[float] = Field(..., description="Uplift 점수 리스트")
    model_uri: str = Field(..., example="models:/uplift-model/Production", description="예측에 사용된 모델의 MLflow URI")
    sample_count: int = Field(..., example=100, description="처리된 샘플 수")


class HealthCheckResponse(BaseModel):
    """
    헬스 체크 응답 스키마입니다.
    """
    status: str = Field(..., example="healthy", description="서비스 상태")
    model_uri: str = Field(..., example="models:/uplift-model/Production", description="현재 로드된 모델의 MLflow URI")
    model_name: str = Field(..., example="xgboost_x_learner", description="로드된 모델 이름")
