import pandas as pd
from src.settings import Settings
from src.utils.system.logger import logger

def validate_schema(df: pd.DataFrame, settings: Settings):
    """
    입력 데이터프레임이 data_interface에 정의된 스키마와 일치하는지 검증합니다.

    Args:
        df (pd.DataFrame): 검증할 데이터프레임.
        settings (Settings): 스키마 정보가 포함된 설정 객체.

    Raises:
        TypeError: 스키마 검증에 실패할 경우 발생합니다.
    """
    expected_schema = settings.model.data_interface.features
    errors = []
    logger.info("모델 입력 데이터 스키마를 검증합니다...")

    # features 스키마가 정의되지 않은 경우 검증 스킵 (LOCAL 환경에서 주로 발생)
    if expected_schema is None:
        logger.info("features 스키마가 정의되지 않아 검증을 스킵합니다.")
        return

    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            errors.append(f"- 필수 컬럼 누락: '{col}'")
            continue
        
        actual_type = str(df[col].dtype)
        is_valid = False
        if expected_type == "numeric" and pd.api.types.is_numeric_dtype(df[col]):
            is_valid = True
        elif expected_type == "category" and (pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            is_valid = True
        
        if not is_valid:
            errors.append(f"- 컬럼 '{col}' 타입 불일치: 예상='{expected_type}', 실제='{actual_type}'")

    if errors:
        error_message = "모델 입력 데이터 스키마 검증 실패:\n" + "\n".join(errors)
        error_message += "\n\n'preprocessor' 또는 'augmenter' 설정을 확인하여 스키마를 맞추세요."
        raise TypeError(error_message)
    
    logger.info("스키마 검증 성공.")


def convert_schema(df: pd.DataFrame, expected_schema: dict) -> pd.DataFrame:
    """
    데이터프레임을 예상 스키마에 맞게 변환합니다.
    
    Args:
        df (pd.DataFrame): 변환할 데이터프레임
        expected_schema (dict): 예상 스키마 딕셔너리
    
    Returns:
        pd.DataFrame: 변환된 데이터프레임
    """
    logger.info("데이터 타입 변환을 시작합니다...")
    
    converted_df = df.copy()
    
    for col, expected_type in expected_schema.items():
        if col not in converted_df.columns:
            continue
            
        if expected_type == "numeric":
            converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce')
        elif expected_type == "category":
            converted_df[col] = converted_df[col].astype('category')
    
    logger.info("데이터 타입 변환 완료.")
    return converted_df 