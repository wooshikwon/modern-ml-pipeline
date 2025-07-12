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
    fields_to_create = {
        field: (Any, Field(..., description=f"Primary Key: {field}"))
        for field in pk_fields
    }

    # 동적으로 생성할 필드가 없으면 빈 모델을 반환
    if not fields_to_create:
        return create_model(f"{model_name}PredictionRequest")

    return create_model(f"{model_name}PredictionRequest", **fields_to_create)


class PredictionResponse(BaseModel):
    """
    단일 예측 결과를 위한 응답 스키마입니다.
    """
    uplift_score: float = Field(..., example=0.123, description="계산된 Uplift 점수")
    model_uri: str = Field(
        ...,
        example="models:/uplift-model/Production",
        description="예측에 사용된 모델의 MLflow URI",
    )


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
    sample_count: int = Field(..., example=100, description="처리된 샘플 수")


class HealthCheckResponse(BaseModel):
    """
    헬스 체크 응답 스키마입니다.
    """
    status: str = Field(..., example="healthy", description="서비스 상태")
    model_uri: str = Field(
        ...,
        example="models:/uplift-model/Production",
        description="현재 로드된 모델의 MLflow URI",
    )
    model_name: str = Field(..., example="xgboost_x_learner", description="로드된 모델 이름")