from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field, create_model


def create_dynamic_prediction_request(model_name: str, pk_fields: List[str]) -> Type[BaseModel]:
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
    class_dict = {"__annotations__": field_annotations, **field_defaults}

    # BaseModel을 상속받는 동적 클래스 생성
    DynamicModel = type(class_name, (BaseModel,), class_dict)

    return DynamicModel


# create_datainterface_based_prediction_request v1 삭제됨
# v2 버전 사용 (target_column 자동 제외 기능 포함)


class MinimalPredictionResponse(BaseModel):
    """일반 태스크에 공통적인 최소 예측 응답 스키마"""

    model_config = {"protected_namespaces": ()}

    prediction: Any = Field(..., description="모델 예측 결과")
    model_uri: str = Field(..., description="예측에 사용된 모델의 MLflow URI")


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

    model_config = {"protected_namespaces": ()}

    predictions: List[Dict[str, Any]] = Field(..., description="예측 결과 리스트 (PK 포함)")
    model_uri: str = Field(
        ...,
        json_schema_extra={"example": "runs:/<run_id>/model"},
        description="예측에 사용된 모델의 MLflow URI",
    )
    sample_count: int = Field(..., json_schema_extra={"example": 100}, description="처리된 샘플 수")
    #  최적화 정보 포함 (Optional로 하위 호환성 보장)
    optimization_enabled: bool = Field(default=False, description="하이퍼파라미터 최적화 여부")
    best_score: float = Field(default=0.0, description="최적화 달성 점수 (활성화된 경우)")


class HealthCheckResponse(BaseModel):
    """
    Liveness 체크 응답 스키마 (K8s livenessProbe용).
    프로세스 생존 여부만 확인하는 경량 응답.
    """

    status: str = Field(
        default="ok", json_schema_extra={"example": "ok"}, description="프로세스 상태"
    )


class ReadyCheckResponse(BaseModel):
    """
    Readiness 체크 응답 스키마 (K8s readinessProbe용).
    모델 로드 상태까지 확인하여 트래픽 수신 준비 여부 반환.
    """

    model_config = {"protected_namespaces": ()}

    status: str = Field(..., json_schema_extra={"example": "ready"}, description="서비스 상태")
    model_uri: str = Field(
        ...,
        json_schema_extra={"example": "runs:/<run_id>/model"},
        description="현재 로드된 모델의 MLflow URI",
    )
    model_name: str = Field(
        ..., json_schema_extra={"example": "your_model_name"}, description="로드된 모델 이름"
    )


#  새로운 메타데이터 응답 스키마들


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
    preprocessing_fit_scope: str = Field(
        default="", description="전처리 fit 범위 (Data Leakage 방지)"
    )
    random_state: int = Field(default=42, description="재현성을 위한 시드값")


class ModelMetadataResponse(BaseModel):
    """
    모델의 완전한 메타데이터 응답
    """

    model_config = {"protected_namespaces": ()}

    model_uri: str = Field(..., description="모델 MLflow URI")
    model_class_path: str = Field(default="", description="모델 클래스 경로")
    hyperparameter_optimization: HyperparameterOptimizationInfo = Field(
        ..., description="하이퍼파라미터 최적화 정보"
    )
    training_methodology: TrainingMethodologyInfo = Field(..., description="학습 방법론 정보")
    training_metadata: Dict[str, Any] = Field(default={}, description="기타 학습 메타데이터")
    api_schema: Dict[str, Any] = Field(default={}, description="동적 생성된 API 스키마 정보")


class OptimizationHistoryResponse(BaseModel):
    """
    하이퍼파라미터 최적화 과정 상세 히스토리
    """

    enabled: bool = Field(..., description="최적화 수행 여부")
    optimization_history: List[Dict[str, Any]] = Field(
        default=[], description="전체 최적화 과정 기록"
    )
    search_space: Dict[str, Any] = Field(default={}, description="탐색한 하이퍼파라미터 공간")
    convergence_info: Dict[str, Any] = Field(default={}, description="수렴 정보")
    timeout_occurred: bool = Field(default=False, description="타임아웃 발생 여부")


def create_datainterface_based_prediction_request_v2(
    model_name: str, data_interface_schema: Dict[str, Any], exclude_target: bool = True
) -> Type[BaseModel]:
    """
    🚀 Improved: DataInterface 스키마를 기반으로 API 요청 모델을 자동 생성합니다.
    Target 컬럼은 자동으로 제외됩니다.

    Args:
        model_name: 생성할 모델의 이름
        data_interface_schema: PyfuncWrapper에 저장된 DataInterface 스키마
        exclude_target: target_column 자동 제외 여부 (기본: True)

    Returns:
        동적으로 생성된 Pydantic 모델 클래스
    """
    field_annotations = {}
    field_defaults = {}

    # Target column 추출 (제외용)
    target_column = data_interface_schema.get("target_column")

    # 1. Entity columns (항상 필요, target 제외)
    entity_columns = data_interface_schema.get("entity_columns", []) or []
    for col in entity_columns:
        if exclude_target and col == target_column:
            continue  # target column 자동 제외
        field_annotations[col] = Any
        field_defaults[col] = Field(..., description=f"Entity column: {col}")

    # 2. Feature columns (명시된 경우)
    feature_columns = data_interface_schema.get("feature_columns", []) or []
    if feature_columns:
        for col in feature_columns:
            if exclude_target and col == target_column:
                continue  # target column 자동 제외
            if col not in field_annotations:  # 중복 방지
                field_annotations[col] = Any
                field_defaults[col] = Field(..., description=f"Feature column: {col}")

    # 3. Task-specific columns
    task_type = data_interface_schema.get("task_type", "")

    # Timeseries: timestamp column
    if task_type == "timeseries":
        timestamp_col = data_interface_schema.get("timestamp_column")
        if timestamp_col and timestamp_col != target_column:
            field_annotations[timestamp_col] = Any
            field_defaults[timestamp_col] = Field(
                ..., description=f"Timestamp column: {timestamp_col}"
            )

    # Causal: treatment column
    elif task_type == "causal":
        treatment_col = data_interface_schema.get("treatment_column")
        if treatment_col and treatment_col != target_column:
            field_annotations[treatment_col] = Any
            field_defaults[treatment_col] = Field(
                ..., description=f"Treatment column: {treatment_col}"
            )

    # 4. Required columns from training (학습 시 사용된 컬럼들)
    required_columns = data_interface_schema.get("required_columns", []) or []
    for col in required_columns:
        if exclude_target and col == target_column:
            continue  # target column 자동 제외
        if col not in field_annotations:  # 중복 방지
            field_annotations[col] = Any
            field_defaults[col] = Field(..., description=f"Required column: {col}")

    # 5. 모든 사용 가능한 컬럼 (feature_columns가 None인 경우)
    all_columns = data_interface_schema.get("all_columns", []) or []
    if not feature_columns and all_columns:  # feature_columns가 명시되지 않은 경우
        exclude_cols = set([target_column] if exclude_target else [])
        exclude_cols.update(entity_columns)  # entity는 이미 추가됨
        if task_type == "causal":
            exclude_cols.add(data_interface_schema.get("treatment_column"))

        for col in all_columns:
            if col not in exclude_cols and col not in field_annotations:
                field_annotations[col] = Any
                field_defaults[col] = Field(..., description=f"Feature: {col}")

    # 클래스 이름 생성
    class_name = f"{model_name}PredictionRequest"

    # 클래스 속성 딕셔너리
    class_dict = {
        "__annotations__": field_annotations,
        **field_defaults,
        "__doc__": f"Auto-generated prediction request schema for {model_name} (target_column '{target_column}' excluded)",
    }

    # BaseModel을 상속받는 동적 클래스 생성
    DynamicModel = type(class_name, (BaseModel,), class_dict)

    return DynamicModel
