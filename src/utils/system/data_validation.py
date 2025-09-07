"""
DataInterface 기반 데이터 검증 모듈

Phase 5.1에서 도입된 DataInterface 기반 컬럼 검증 로직을 제공합니다.
기존 schema_utils.py보다 단순하고 명확한 DataInterface 필수 컬럼 검증에 집중합니다.

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 단일 책임 원칙
"""

from typing import List, Set, Dict, Any
import pandas as pd

from src.settings.recipe import DataInterface
from src.utils.system.logger import logger


def get_required_columns_from_data_interface(
    data_interface: DataInterface, 
    actual_training_df: pd.DataFrame = None
) -> List[str]:
    """
    DataInterface에서 필수 컬럼 목록을 추출합니다.
    
    DataInterface 검증 규칙 (추론용):
    - entity_columns: 항상 필수 (어떤 entity에 대한 예측인지 식별)
    - timestamp_column: timeseries task에서 필수 (언제 시점 예측인지)
    - treatment_column: causal task에서 필수 (처치 변수)
    - feature_columns: 명시된 경우 포함, null인 경우 실제 학습 데이터에서 자동 추출
    - target_column: 추론시 불필요하므로 제외
    
    Args:
        data_interface: Recipe의 DataInterface 설정
        actual_training_df: 실제 학습 데이터프레임 (feature_columns=null인 경우 필요)
        
    Returns:
        List[str]: 필수 컬럼 목록 (중복 제거됨)
        
    Examples:
        >>> interface = DataInterface(
        ...     task_type="regression",
        ...     target_column="price", 
        ...     entity_columns=["user_id"],
        ...     feature_columns=["age", "income"]
        ... )
        >>> get_required_columns_from_data_interface(interface)
        ['user_id', 'age', 'income']  # target_column 제외
        
        >>> # feature_columns=null인 경우 실제 데이터에서 추출
        >>> interface_null = DataInterface(
        ...     task_type="regression",
        ...     target_column="price",
        ...     entity_columns=["user_id"], 
        ...     feature_columns=None
        ... )
        >>> df = pd.DataFrame({'user_id': [1,2], 'price': [100,200], 'age': [25,30], 'income': [5000,6000]})
        >>> get_required_columns_from_data_interface(interface_null, df)
        ['user_id', 'age', 'income']  # target 제외, entity + 나머지 feature 자동 추출
    """
    required = []
    
    # 1. Entity 컬럼들 (항상 필수 - 어떤 entity에 대한 예측인지 식별)
    required.extend(data_interface.entity_columns)
    
    # 2. Task별 특수 컬럼 (추론에 필요한 컬럼들)
    if data_interface.task_type == "timeseries" and data_interface.timestamp_column:
        required.append(data_interface.timestamp_column)
    elif data_interface.task_type == "causal" and data_interface.treatment_column:
        required.append(data_interface.treatment_column)
    
    # 참고: target_column은 추론시 불필요하므로 제외
    
    # 4. Feature columns 처리
    if data_interface.feature_columns:
        # 명시적 feature_columns 추가
        required.extend(data_interface.feature_columns)
    elif actual_training_df is not None:
        # feature_columns=null인 경우: 실제 학습 데이터에서 자동 추출
        # target, entity, timestamp, treatment 제외한 나머지 모든 컬럼 사용
        exclude_columns = set()
        
        # 제외할 컬럼들 수집
        if data_interface.target_column:
            exclude_columns.add(data_interface.target_column)
        exclude_columns.update(data_interface.entity_columns)
        if data_interface.timestamp_column:
            exclude_columns.add(data_interface.timestamp_column)
        if data_interface.treatment_column:
            exclude_columns.add(data_interface.treatment_column)
        
        # 실제 데이터에서 제외 대상 컬럼들을 빼고 나머지 추가
        feature_columns_from_data = [
            col for col in actual_training_df.columns 
            if col not in exclude_columns
        ]
        required.extend(feature_columns_from_data)
        
        logger.info(
            f"feature_columns=null 감지 - 실제 학습 데이터에서 {len(feature_columns_from_data)}개 "
            f"feature 컬럼 자동 추출: {feature_columns_from_data}"
        )
    else:
        # feature_columns=null이지만 actual_training_df가 없는 경우 (추론 시점)
        logger.warning(
            "feature_columns=null이지만 실제 학습 데이터가 제공되지 않았습니다. "
            "저장된 data_interface_schema를 사용해야 합니다."
        )
    
    # 중복 제거하여 반환
    unique_required = list(set(required))
    
    logger.debug(
        f"DataInterface 필수 컬럼 추출 완료 - "
        f"Task: {data_interface.task_type}, "
        f"컬럼 수: {len(unique_required)}, "
        f"컬럼들: {unique_required}"
    )
    
    return unique_required


def validate_data_interface_columns(
    df: pd.DataFrame, 
    data_interface: DataInterface, 
    stored_required_columns: List[str] = None
) -> None:
    """
    DataFrame이 DataInterface 필수 컬럼을 모두 포함하는지 검증합니다.
    
    Phase 5 수정된 검증 정책:
    - 학습시: data.data_interface + 실제 학습 데이터에서 필수 컬럼 추출
    - 추론시: 저장된 필수 컬럼 목록과 --data-path 데이터 비교
    - 완전한 컬럼 일치가 아닌, 필수 컬럼 포함 여부만 검증
    - 추가 컬럼 존재는 허용 (무시됨)
    
    Args:
        df: 검증할 데이터프레임
        data_interface: Recipe의 DataInterface 설정
        stored_required_columns: 학습시 저장된 필수 컬럼 목록 (추론시 사용)
        
    Raises:
        ValueError: 필수 컬럼이 누락된 경우 상세한 진단 메시지와 함께 발생
        
    Examples:
        >>> # 추론용 검증 (target_column 불필요)
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 2, 3],
        ...     'age': [25, 30, 35],
        ...     'income': [5000, 6000, 7000]
        ... })
        >>> interface = DataInterface(...)
        >>> validate_data_interface_columns(df, interface)  # 통과 (target_column 없어도 됨)
        
        >>> # 추론시 검증 (저장된 컬럼 목록 사용)
        >>> inference_df = pd.DataFrame({'user_id': [4, 5], 'age': [40, 45], 'income': [8000, 9000]})
        >>> validate_data_interface_columns(inference_df, interface, stored_required_columns=['user_id', 'age', 'income'])  # 통과
    """
    # 추론시에는 저장된 컬럼 목록 우선 사용
    if stored_required_columns:
        required_columns = stored_required_columns
        logger.debug(f"저장된 필수 컬럼 목록 사용: {required_columns}")
    else:
        # 학습시에는 DataInterface에서 추출 (실제 데이터 필요할 수 있음)
        required_columns = get_required_columns_from_data_interface(data_interface, df)
        logger.debug(f"DataInterface에서 필수 컬럼 추출: {required_columns}")
    
    actual_columns = set(df.columns.tolist())
    missing_columns = set(required_columns) - actual_columns
    
    if missing_columns:
        # 상세한 진단 메시지 생성
        error_message = (
            f"DataInterface 필수 컬럼 누락 감지:\n\n"
            f"📋 Task Type: {data_interface.task_type}\n"
            f"❌ 누락된 컬럼: {sorted(missing_columns)}\n"
            f"✅ 필요한 전체 컬럼: {sorted(required_columns)}\n"
            f"📊 실제 데이터 컬럼: {sorted(actual_columns)}\n\n"
            f"💡 해결방안:\n"
            f"1. 데이터 소스에 누락된 컬럼을 추가하세요\n"
            f"2. Recipe의 DataInterface 설정을 확인하세요\n"
            f"3. Fetcher 설정에서 컬럼 매핑을 확인하세요"
        )
        
        logger.error(f"DataInterface 컬럼 검증 실패: {missing_columns}")
        raise ValueError(error_message)
    
    # 추가 컬럼이 있는 경우 정보성 로그 (에러 아님)
    extra_columns = actual_columns - set(required_columns)
    if extra_columns:
        logger.info(
            f"DataInterface에 정의되지 않은 추가 컬럼 발견: {sorted(extra_columns)} "
            f"(허용됨, 학습/추론에서 무시됨)"
        )
    
    logger.info(
        f"DataInterface 컬럼 검증 통과 - "
        f"Task: {data_interface.task_type}, "
        f"필수 컬럼: {len(required_columns)}개, "
        f"실제 컬럼: {len(actual_columns)}개"
    )


def create_data_interface_schema_for_storage(
    data_interface: DataInterface, 
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    PyfuncWrapper 저장용 DataInterface 스키마 메타데이터를 생성합니다.
    
    이 스키마는 추론 시점에 데이터 검증을 위해 MLflow 모델과 함께 저장됩니다.
    학습 시점의 DataInterface 설정과 실제 데이터 타입 정보를 포함합니다.
    
    **핵심 기능:**
    - feature_columns=null인 경우 실제 학습 데이터에서 사용된 모든 컬럼 저장
    - 추론시 이 저장된 컬럼들과 --data-path 데이터의 일치성 검증에 사용
    
    Args:
        data_interface: Recipe의 DataInterface 설정
        df: 실제 학습 데이터프레임
        
    Returns:
        Dict[str, Any]: PyfuncWrapper 저장용 스키마 메타데이터
        
    Examples:
        >>> # feature_columns가 명시된 경우
        >>> schema = create_data_interface_schema_for_storage(interface, train_df)
        >>> schema['required_columns']
        ['price', 'user_id', 'age', 'income']
        
        >>> # feature_columns=null인 경우 실제 데이터에서 추출
        >>> schema_null = create_data_interface_schema_for_storage(interface_null, train_df)
        >>> schema_null['required_columns']  # target, entity 제외한 모든 컬럼
        ['price', 'user_id', 'age', 'income', 'location', 'category']
    """
    # 🆕 핵심: 실제 학습 데이터를 기반으로 필수 컬럼 추출
    required_columns = get_required_columns_from_data_interface(data_interface, df)
    
    # 실제 존재하는 컬럼들의 데이터 타입 수집
    column_dtypes = {}
    for col in required_columns:
        if col in df.columns:
            column_dtypes[col] = str(df[col].dtype)
        else:
            # 이론적으로는 validate_data_interface_columns()를 먼저 호출해야 하므로
            # 이 상황은 발생하지 않아야 함
            logger.warning(f"필수 컬럼 '{col}'이 DataFrame에 없습니다. 타입 정보를 'unknown'으로 설정합니다.")
            column_dtypes[col] = 'unknown'
    
    # 추론 시 검증을 위한 완전한 스키마 메타데이터 생성
    schema_metadata = {
        # DataInterface 전체 설정 (Pydantic 모델을 dict로 변환)
        'data_interface': data_interface.model_dump(),
        
        # 🆕 핵심: 실제 학습시 사용된 필수 컬럼들 (feature_columns=null 처리 포함)
        'required_columns': required_columns,
        'column_dtypes': column_dtypes,
        
        # feature_columns=null 여부 메타데이터 (디버깅용)
        'feature_columns_was_null': data_interface.feature_columns is None,
        'original_dataframe_columns': df.columns.tolist(),  # 전체 컬럼 기록
        
        # 메타데이터
        'schema_version': '5.1',  # Phase 5.1 버전
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'total_required_columns': len(required_columns),
        'validation_policy': 'data_interface_based',  # 새로운 정책 명시
        
        # 디버깅 정보
        'actual_dataframe_shape': list(df.shape)
    }
    
    logger.info(
        f"DataInterface 저장용 스키마 생성 완료 - "
        f"Task: {data_interface.task_type}, "
        f"필수 컬럼: {len(required_columns)}개, "
        f"feature_columns_was_null: {data_interface.feature_columns is None}, "
        f"스키마 버전: 5.1"
    )
    
    return schema_metadata