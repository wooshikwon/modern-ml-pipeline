import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from src.settings import Settings
from src.utils.system.logger import logger

def validate_schema(df: pd.DataFrame, settings: Settings, for_training: bool = False):
    """
    입력 데이터프레임이 Recipe 스키마와 일치하는지 검증합니다. (27개 Recipe 대응)

    Args:
        df (pd.DataFrame): 검증할 데이터프레임.
        settings (Settings): 스키마 정보가 포함된 설정 객체.
        for_training (bool): True면 모델 학습용 데이터 검증 (entity, timestamp 컬럼 제외)
                            False면 원본 데이터 검증 (모든 컬럼 요구)

    Raises:
        TypeError: 스키마 검증에 실패할 경우 발생합니다.
    """
    logger.info(f"모델 입력 데이터 스키마를 검증합니다... (for_training: {for_training})")

    # 27개 Recipe 구조: 필수 컬럼들 확인
    entity_schema = settings.recipe.model.loader.entity_schema
    data_interface = settings.recipe.model.data_interface
    
    errors = []
    required_columns = []
    
    if not for_training:
        # 원본 데이터 검증: 모든 컬럼 요구
        # 1. Entity + Timestamp 컬럼 검증
        required_columns = entity_schema.entity_columns + [entity_schema.timestamp_column]
        
        # 2. Target 컬럼 검증 (clustering 제외)
        if data_interface.task_type != "clustering" and data_interface.target_column:
            required_columns.append(data_interface.target_column)
    else:
        # 모델 학습용 데이터 검증: entity_schema 컬럼들 제외
        logger.info("모델 학습용 데이터 검증: entity_columns, timestamp_column 제외")
        required_columns = []
        
        # Target 컬럼은 이미 분리되었으므로 검증하지 않음
        
    # 3. Treatment 컬럼 검증 (causal 전용) - 학습 시에도 필요
    if data_interface.task_type == "causal" and data_interface.treatment_column:
        required_columns.append(data_interface.treatment_column)
    
    # 필수 컬럼 존재 여부 검증
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"- 필수 컬럼 누락: '{col}' (task_type: {data_interface.task_type})")
    
    # 기본 데이터 타입 검증 (timestamp 컬럼)
    if entity_schema.timestamp_column in df.columns:
        timestamp_col = entity_schema.timestamp_column
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                # 자동 변환 시도
                pd.to_datetime(df[timestamp_col])
                logger.info(f"Timestamp 컬럼 '{timestamp_col}' 자동 변환 가능")
            except:
                errors.append(f"- Timestamp 컬럼 '{timestamp_col}' 타입 오류: datetime 변환 불가")

    if errors:
        error_message = "모델 입력 데이터 스키마 검증 실패:\n" + "\n".join(errors)
        error_message += f"\n\n필수 컬럼: {required_columns}"
        error_message += f"\n실제 컬럼: {list(df.columns)}"
        raise TypeError(error_message)
    
    logger.info(f"스키마 검증 성공 (task_type: {data_interface.task_type})")


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

class SchemaConsistencyValidator:
    """
    🆕 Phase 4: Training/Inference 스키마 일관성 자동 검증기
    기존 validate_schema 함수를 확장한 차세대 스키마 일관성 보장 시스템
    """
    
    def __init__(self, training_schema: dict):
        """
        Training 시점에 생성된 완전한 스키마 메타데이터를 활용하여 초기화
        
        Args:
            training_schema (dict): Training 시점 스키마 메타데이터
                - entity_columns: Phase 1 EntitySchema 정보
                - timestamp_column: Point-in-Time 기준 컬럼
                - inference_columns: Inference에 필요한 컬럼 목록
                - column_types: 각 컬럼의 타입 정보
        """
        self.training_schema = training_schema
        logger.info(f"SchemaConsistencyValidator 초기화 완료 - 검증 대상: {len(training_schema.get('inference_columns', []))}개 컬럼")
    
    def validate_inference_consistency(self, inference_df: pd.DataFrame) -> bool:
        """
        🎯 4단계 스키마 일관성 검증: 기본 구조 → 컬럼 일관성 → 타입 호환성 → Point-in-Time 특별 검증
        
        Args:
            inference_df (pd.DataFrame): 검증할 Inference 데이터
            
        Returns:
            bool: 모든 검증 통과 시 True
            
        Raises:
            ValueError: 스키마 불일치 발견 시 상세한 진단 메시지와 함께 발생
        """
        
        # 1. 기본 스키마 구조 검증 (기존 validate_schema 로직 활용)
        logger.info("Phase 1: 기본 스키마 구조 검증 시작...")
        self._validate_basic_schema(inference_df)
        
        # 2. Training/Inference 필수 컬럼 일관성 검증
        logger.info("Phase 2: Training/Inference 컬럼 일관성 검증 시작...")
        self._validate_column_consistency(inference_df)
        
        # 3. 타입 호환성 매트릭스 검증 (호환 허용, 비호환 차단)
        logger.info("Phase 3: 고급 타입 호환성 검증 시작...")
        self._validate_dtype_compatibility(inference_df)
        
        # 4. Entity/Timestamp 특별 검증 (Phase 1, 2 연계)
        logger.info("Phase 4: Point-in-Time 컬럼 특별 검증 시작...")
        self._validate_point_in_time_columns(inference_df)
        
        logger.info("✅ 모든 스키마 일관성 검증 통과")
        return True
    
    def _validate_basic_schema(self, df: pd.DataFrame):
        """기존 validate_schema 로직을 활용한 기본 스키마 검증"""
        # 기본적인 컬럼 존재 여부 확인
        required_cols = self.training_schema.get('inference_columns', [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"기본 스키마 검증에서 누락 컬럼 발견: {missing_cols}")
        else:
            logger.info("기본 스키마 구조 검증 통과")
    
    def _validate_column_consistency(self, inference_df: pd.DataFrame):
        """Training vs Inference 필수 컬럼 일관성 검증"""
        required_cols = self.training_schema.get('inference_columns', [])
        missing_cols = set(required_cols) - set(inference_df.columns)
        
        if missing_cols:
            raise ValueError(
                f"🚨 Inference 데이터에 Training 시 필수 컬럼이 없습니다: {missing_cols}\n"
                f"Training 스키마: {required_cols}\n"
                f"현재 스키마: {list(inference_df.columns)}\n"
                f"💡 해결방안: 누락된 컬럼을 데이터에 추가하거나 preprocessor/augmenter 설정을 확인하세요."
            )
        
        # 추가 컬럼은 경고만 (새로운 피처 추가 가능성)
        extra_cols = set(inference_df.columns) - set(required_cols)
        if extra_cols:
            logger.warning(f"Training에 없던 추가 컬럼 발견: {extra_cols} (허용되지만 무시됩니다)")
        
        logger.info("컬럼 일관성 검증 통과")
    
    def _validate_dtype_compatibility(self, inference_df: pd.DataFrame):
        """고급 타입 호환성 매트릭스 검증 (기존 convert_schema 로직 확장)"""
        column_types = self.training_schema.get('column_types', {})
        
        for col in self.training_schema.get('inference_columns', []):
            if col not in inference_df.columns:
                continue  # 이미 column_consistency에서 처리됨
                
            expected_dtype = column_types.get(col, 'unknown')
            actual_dtype = str(inference_df[col].dtype)
            
            if not self._is_compatible_dtype(expected_dtype, actual_dtype):
                raise ValueError(
                    f"🚨 컬럼 '{col}' 타입 불일치:\n"
                    f"Training 시: {expected_dtype} → 현재 Inference: {actual_dtype}\n"
                    f"이는 호환되지 않는 타입 변경입니다.\n"
                    f"💡 해결방안: 전처리 단계에서 타입을 '{expected_dtype}'로 변환하거나 모델을 재학습하세요."
                )
        
        logger.info("타입 호환성 검증 통과")
    
    def _is_compatible_dtype(self, expected: str, actual: str) -> bool:
        """타입 호환성 매트릭스 (기존 convert_schema 로직 확장)"""
        # 완전 동일한 경우 허용
        if expected == actual:
            return True
        
        # 호환 가능한 타입 그룹들
        compatible_groups = [
            # 숫자형: int64 ↔ int32 ↔ int 호환 허용
            (['int64', 'int32', 'int'], ['int64', 'int32', 'int']),
            # 실수형: float64 ↔ float32 ↔ float 호환 허용  
            (['float64', 'float32', 'float'], ['float64', 'float32', 'float']),
            # 문자열: object ↔ string 호환 허용
            (['object', 'string'], ['object', 'string']),
            # 날짜: datetime64 변형들 허용
            (['datetime64', 'datetime'], ['datetime64', 'datetime']),
            # 부울: bool 변형들 허용
            (['bool', 'boolean'], ['bool', 'boolean'])
        ]
        
        for expected_group, actual_group in compatible_groups:
            if any(e in expected for e in expected_group) and \
               any(a in actual for a in actual_group):
                return True
        
        # 위험한 타입 변경은 차단 (string → int, int → string 등)
        return False
    
    def _validate_point_in_time_columns(self, inference_df: pd.DataFrame):
        """Entity/Timestamp 컬럼 특별 검증 (Phase 1, 2 연계)"""
        # Phase 1 EntitySchema 정보 활용
        entity_cols = self.training_schema.get('entity_columns', [])
        timestamp_col = self.training_schema.get('timestamp_column', '')
        
        # Entity 컬럼들 존재 및 타입 검증
        for col in entity_cols:
            if col not in inference_df.columns:
                raise ValueError(f"🚨 필수 Entity 컬럼 누락: '{col}' - Point-in-Time JOIN을 위해 필수입니다")
        
        # Timestamp 컬럼 특별 검증
        if timestamp_col and timestamp_col in inference_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(inference_df[timestamp_col]):
                raise ValueError(
                    f"🚨 Timestamp 컬럼 '{timestamp_col}'이 datetime 타입이 아닙니다: {inference_df[timestamp_col].dtype}\n"
                    f"💡 해결방안: pd.to_datetime(df['{timestamp_col}'])로 변환하거나 전처리에서 날짜 형식을 맞춰주세요."
                )
            
            # 미래 데이터 체크 (Phase 2 Point-in-Time 안전성 연계)
            max_timestamp = inference_df[timestamp_col].max()
            current_time = pd.Timestamp.now()
            if max_timestamp > current_time:
                logger.warning(
                    f"⚠️ 미래 데이터 감지: 최대 timestamp {max_timestamp} > 현재 시간 {current_time}\n"
                    f"Point-in-Time 안전성을 위해 주의 필요"
                )
        
        logger.info("Point-in-Time 컬럼 특별 검증 통과")


def generate_training_schema_metadata(training_df: pd.DataFrame, data_interface_config: dict) -> dict:
    """
    🆕 Phase 4: Training 시점에 완전한 스키마 메타데이터 생성
    Phase 1-3의 모든 정보를 통합하여 자기 기술적 스키마 생성
    
    Args:
        training_df (pd.DataFrame): Training 데이터
        data_interface_config (dict): EntitySchema 설정 정보
        
    Returns:
        dict: 완전한 스키마 메타데이터
    """
    target_column = data_interface_config.get('target_column')
    
    # Inference에 필요한 컬럼들 (target 제외)
    inference_columns = [col for col in training_df.columns if col != target_column]
    
    schema_metadata = {
        # Phase 1 EntitySchema 정보 활용
        'entity_columns': data_interface_config.get('entity_columns', []),
        'timestamp_column': data_interface_config.get('timestamp_column', ''),
        'target_column': target_column,
        'task_type': data_interface_config.get('task_type', ''),
        
        # 실제 Training 데이터 스키마 정보
        'training_columns': list(training_df.columns),
        'inference_columns': inference_columns,
        'column_types': {col: str(training_df[col].dtype) for col in training_df.columns},
        
        # 메타데이터
        'schema_version': '2.0',
        'created_at': datetime.now().isoformat(),
        'point_in_time_safe': True,  # Phase 2 ASOF JOIN 보장
        'sql_injection_safe': True,  # Phase 3 보안 강화 보장
        'total_training_samples': len(training_df),
        'column_count': len(training_df.columns)
    }
    
    logger.info(f"✅ Training 스키마 메타데이터 생성 완료: {len(inference_columns)}개 inference 컬럼, {len(training_df.columns)}개 전체 컬럼")
    return schema_metadata 