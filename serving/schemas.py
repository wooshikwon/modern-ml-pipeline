from pydantic import BaseModel, Field, create_model
from typing import Any, List

# config.settings에서 통합 설정 객체를 import
from config.settings import settings

def create_dynamic_prediction_request() -> BaseModel:
    """
    settings.model.data_interface.features를 기반으로
    Pydantic 모델을 동적으로 생성합니다.
    """
    features = settings.model.data_interface.features
    
    # 학습에 사용되지 않는 id, outcome, treatment 컬럼은 API 요청에서 제외
    exclude_cols = [
        'member_id',
        settings.model.data_interface.target_col,
        settings.model.data_interface.treatment_col
    ]
    
    fields_to_create = {
        feature: (Any, Field(..., description=f"Feature: {feature}"))
        for feature in features if feature not in exclude_cols
    }
    
    if not fields_to_create:
        return BaseModel

    return create_model('PredictionRequest', **fields_to_create)

# 동적으로 PredictionRequest 모델 생성
PredictionRequest = create_dynamic_prediction_request()


class PredictionResponse(BaseModel):
    """
    예측 결과를 ���한 응답 데이터 스키마입니다.
    """
    uplift_score: float = Field(..., example=0.123, description="계산된 Uplift 점수")
    model_uri: str = Field(..., example="models:/uplift-virtual-coupon/Production", description="예측에 사용된 모델의 MLflow Model URI")


class BatchPredictionRequest(BaseModel):
    """
    배치 예측 요청을 위한 스키마.
    """
    samples: List[PredictionRequest] = Field(..., description="예측을 위한 샘플 리스트")


class BatchPredictionResponse(BaseModel):
    """
    배치 예측 응답을 위한 스키마.
    """
    predictions: List[float] = Field(..., description="Uplift 점수 리스트")
    model_uri: str = Field(..., example="models:/uplift-virtual-coupon/Production", description="예측에 사용된 모델의 MLflow Model URI")
    sample_count: int = Field(..., example=100, description="처리된 샘플 수")

class HealthCheckResponse(BaseModel):
    """
    헬스 체크 응답 스키마입니다.
    """
    status: str = Field(..., example="healthy", description="서비스 상태")
    model_uri: str = Field(..., example="models:/uplift-virtual-coupon/Production", description="현재 로드된 모델의 MLflow Model URI")
    model_name: str = Field(..., example="xgboost_x_learner", description="로드된 모델 이름")
