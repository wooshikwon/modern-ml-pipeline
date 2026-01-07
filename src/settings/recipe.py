"""순수 Recipe Pydantic 스키마 - 검증 로직 완전 제거"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HyperparametersTuning(BaseModel):
    tuning_enabled: bool = Field(False, description="튜닝 활성화")
    optimization_metric: Optional[str] = Field(None, description="최적화 메트릭")
    n_trials: Optional[int] = Field(None, description="튜닝 시도 횟수")
    timeout: Optional[int] = Field(None, description="튜닝 타임아웃(초)")
    fixed: Optional[Dict[str, Any]] = Field(None, description="고정 파라미터")
    tunable: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="튜닝 파라미터")
    values: Optional[Dict[str, Any]] = Field(None, description="고정값")


class Calibration(BaseModel):
    enabled: bool = Field(False, description="캘리브레이션 활성화")
    method: Optional[str] = Field(None, description="캘리브레이션 방법")


class Model(BaseModel):
    class_path: str = Field(..., description="모델 클래스 경로")
    library: str = Field(..., description="라이브러리 이름")
    hyperparameters: HyperparametersTuning
    calibration: Optional[Calibration] = Field(None, description="캘리브레이션 설정")
    computed: Optional[Dict[str, Any]] = Field(default_factory=dict, description="런타임 계산 필드")


class Loader(BaseModel):
    source_uri: Optional[str] = Field(None, description="데이터 소스 URI")


class FeatureView(BaseModel):
    join_key: str = Field(..., description="조인 키")
    features: List[str] = Field(..., description="피처 목록")


class Fetcher(BaseModel):
    type: str = Field(..., description="Fetcher 타입")
    feature_views: Optional[Dict[str, FeatureView]] = Field(None, description="Feature View 설정")
    timestamp_column: Optional[str] = Field(None, description="타임스탬프 컬럼")


class DataInterface(BaseModel):
    target_column: Optional[str] = Field(None, description="타겟 컬럼")
    treatment_column: Optional[str] = Field(None, description="처치 변수 컬럼 (Causal 태스크)")
    timestamp_column: Optional[str] = Field(None, description="타임스탬프 컬럼 (TimeSeries 태스크)")
    entity_columns: List[str] = Field(..., description="엔티티 컬럼 목록")
    feature_columns: Optional[List[str]] = Field(None, description="피처 컬럼 목록")
    sequence_length: Optional[int] = Field(None, description="시퀀스 길이 (TimeSeries LSTM용)")


class DataSplit(BaseModel):
    train: float = Field(..., description="학습 데이터 비율")
    test: float = Field(..., description="테스트 데이터 비율")
    validation: float = Field(..., description="검증 데이터 비율")
    calibration: Optional[float] = Field(None, description="캘리브레이션 데이터 비율")


class Data(BaseModel):
    loader: Loader
    fetcher: Fetcher
    data_interface: DataInterface
    split: DataSplit


class PreprocessorStep(BaseModel):
    type: str = Field(..., description="전처리 타입")
    columns: Optional[List[str]] = Field(None, description="적용 컬럼")
    strategy: Optional[str] = Field(None, description="전처리 전략")
    degree: Optional[int] = Field(None, description="다항식 차수")
    n_bins: Optional[int] = Field(None, description="구간 수")
    create_missing_indicators: Optional[bool] = Field(None, description="결측치 지시자 생성")
    handle_unknown: Optional[str] = Field(None, description="미지 범주 처리 방식")
    unknown_value: Optional[Any] = Field(None, description="미지 범주 대체값")


class Preprocessor(BaseModel):
    steps: List[PreprocessorStep] = Field(..., description="전처리 단계 목록")


class Evaluation(BaseModel):
    metrics: Optional[List[str]] = Field(
        default=None, description="평가 메트릭 목록 (미지정 시 Task별 기본값 사용)"
    )
    random_state: int = Field(default=42, description="재현성을 위한 시드")


class Metadata(BaseModel):
    author: str = Field(default="CLI Recipe Builder", description="작성자")
    created_at: Optional[str] = Field(None, description="생성 시간")
    description: str = Field(..., description="Recipe 설명")
    tuning_note: Optional[str] = Field(None, description="튜닝 관련 노트")


class Recipe(BaseModel):
    """Recipe 스키마 - 순수 데이터 구조만"""

    name: str
    task_choice: str
    model: Model
    data: Data
    preprocessor: Optional[Preprocessor] = None
    evaluation: Evaluation
    metadata: Metadata
