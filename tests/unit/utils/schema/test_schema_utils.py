"""
Schema Utils 테스트 (27개 Recipe 호환성 검증 핵심 모듈)
tests/README.md 전략 준수: 컨텍스트 기반, 퍼블릭 API, 실제 객체, 결정론적

테스트 대상 핵심 기능:
- validate_schema() - 27개 Recipe 대응 스키마 검증 (training vs inference)
- convert_schema() - 데이터 타입 자동 변환
- SchemaConsistencyValidator - 고급 training/inference 일관성 검증
- generate_training_schema_metadata() - 완전한 스키마 메타데이터 생성

핵심 Edge Cases:
- Entity/Timestamp 컬럼 처리 (Point-in-Time 안전성)
- Task type별 필수 컬럼 검증 (clustering 제외)
- Causal task의 treatment 컬럼 검증
- Training vs Inference 데이터 스키마 차이
- 타입 호환성 매트릭스 검증
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.utils.schema.schema_utils import (
    validate_schema,
    convert_schema,
    SchemaConsistencyValidator,
    generate_training_schema_metadata
)


class TestValidateSchema:
    """validate_schema 함수 핵심 테스트 (27개 Recipe 호환성)"""
    
    def test_validate_schema_classification_original_data(self, settings_builder):
        """케이스 A: Classification - 원본 데이터 검증 (entity + timestamp + target 필수)"""
        # Given: Classification 설정과 완전한 원본 데이터
        settings = (
            settings_builder
            .with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["customer_id", "product_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # 완전한 원본 데이터 (Entity + Timestamp + Target 포함)
        np.random.seed(42)
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'product_id': ['A', 'B', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature_1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # When: 원본 데이터 검증 (for_training=False)
        # Then: 에러 없이 통과
        validate_schema(df, settings, for_training=False)
    
    def test_validate_schema_classification_training_data(self, settings_builder):
        """케이스 B: Classification - 학습용 데이터 검증 (entity/timestamp 제외)"""
        # Given: Classification 설정
        settings = (
            settings_builder
            .with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["customer_id", "product_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # 학습용 데이터 (Entity/Timestamp 제거됨, Target도 분리됨)
        np.random.seed(42)
        training_df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [0.5, 1.5, 2.5],
            'processed_feature': [10, 20, 30]
        })
        
        # When: 학습용 데이터 검증 (for_training=True)
        # Then: Entity/Timestamp 없어도 통과 (제외되기 때문)
        validate_schema(training_df, settings, for_training=True)
    
    def test_validate_schema_causal_with_treatment(self, settings_builder):
        """케이스 C: Causal - Treatment 컬럼 필수 검증"""
        # Given: Causal 설정 (treatment 컬럼 포함)
        settings = (
            settings_builder
            .with_task("causal")
            .with_target_column("outcome")
            .with_treatment_column("treatment")
            .with_entity_columns(["user_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # Treatment 컬럼 포함된 데이터
        np.random.seed(42)
        df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'treatment': [0, 1, 0],  # Causal 필수
            'feature_1': [1.0, 2.0, 3.0],
            'outcome': [0.1, 0.8, 0.3]
        })
        
        # When: 원본 데이터 검증
        # Then: Treatment 컬럼 포함으로 통과
        validate_schema(df, settings, for_training=False)
        
        # 학습용 데이터에서도 treatment는 필요 (causal 특성상)
        training_df = df[['treatment', 'feature_1']].copy()  # entity/timestamp 제외
        validate_schema(training_df, settings, for_training=True)
    
    def test_validate_schema_clustering_no_target(self, settings_builder):
        """케이스 D: Clustering - Target 컬럼 없어도 통과"""
        # Given: Clustering 설정 (target 없음)
        settings = (
            settings_builder
            .with_task("clustering")
            .with_entity_columns(["item_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # Target 없는 데이터
        np.random.seed(42)
        df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [0.5, 1.5, 2.5]
        })
        
        # When: Clustering 검증
        # Then: Target 없어도 통과
        validate_schema(df, settings, for_training=False)
    
    def test_validate_schema_missing_required_columns_error(self, settings_builder):
        """케이스 E: 필수 컬럼 누락 시 에러 발생"""
        # Given: Entity 필수인 설정
        settings = (
            settings_builder
            .with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["customer_id", "product_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # 필수 컬럼 누락된 데이터 (customer_id 없음)
        df = pd.DataFrame({
            'product_id': ['A', 'B', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature_1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # When & Then: 필수 컬럼 누락으로 TypeError 발생
        with pytest.raises(TypeError, match="필수 컬럼 누락: 'customer_id'"):
            validate_schema(df, settings, for_training=False)
    
    def test_validate_schema_invalid_timestamp_type_error(self, settings_builder):
        """케이스 F: Timestamp 타입 오류 시 에러 발생"""
        # Given: Timestamp 필수인 설정
        settings = (
            settings_builder
            .with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["customer_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # 잘못된 timestamp 타입 (변환 불가능한 문자열)
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'timestamp': ['invalid_date', 'another_invalid', 'also_invalid'],  # 변환 불가
            'feature_1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # When & Then: Timestamp 타입 오류로 TypeError 발생
        with pytest.raises(TypeError, match="Timestamp 컬럼 'timestamp' 타입 오류"):
            validate_schema(df, settings, for_training=False)
    
    def test_validate_schema_convertible_timestamp_success(self, settings_builder):
        """케이스 G: 변환 가능한 Timestamp는 통과"""
        # Given: Timestamp 필수인 설정
        settings = (
            settings_builder
            .with_task("classification")
            .with_target_column("target")
            .with_entity_columns(["customer_id"])
            .with_timestamp_column("timestamp")
            .build()
        )
        
        # 변환 가능한 timestamp 문자열
        df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],  # 변환 가능
            'feature_1': [1.0, 2.0, 3.0],
            'target': [0, 1, 0]
        })
        
        # When: 검증 (변환 가능한 timestamp)
        # Then: 통과 (자동 변환 로그 출력)
        validate_schema(df, settings, for_training=False)


class TestConvertSchema:
    """convert_schema 함수 테스트 (데이터 타입 자동 변환)"""
    
    def test_convert_schema_numeric_conversion(self):
        """케이스 A: 숫자형 변환 테스트"""
        # Given: 문자열 숫자 데이터
        df = pd.DataFrame({
            'numeric_col': ['1.5', '2.0', '3.5'],
            'category_col': ['A', 'B', 'C'],
            'other_col': [10, 20, 30]
        })
        
        expected_schema = {
            'numeric_col': 'numeric',
            'category_col': 'category'
        }
        
        # When: 스키마 변환
        result = convert_schema(df, expected_schema)
        
        # Then: 
        # 1. numeric_col이 숫자형으로 변환됨
        assert pd.api.types.is_numeric_dtype(result['numeric_col'])
        assert result['numeric_col'].iloc[0] == 1.5
        
        # 2. category_col이 category 타입으로 변환됨
        assert pd.api.types.is_categorical_dtype(result['category_col'])
        
        # 3. 스키마에 없는 컬럼은 그대로 유지
        assert result['other_col'].dtype == df['other_col'].dtype
    
    def test_convert_schema_invalid_numeric_with_coerce(self):
        """케이스 B: 변환 불가능한 숫자는 NaN으로 처리"""
        # Given: 변환 불가능한 문자열 포함
        df = pd.DataFrame({
            'numeric_col': ['1.5', 'invalid', '3.5'],
            'normal_col': [1, 2, 3]
        })
        
        expected_schema = {
            'numeric_col': 'numeric'
        }
        
        # When: 스키마 변환 (errors='coerce')
        result = convert_schema(df, expected_schema)
        
        # Then: invalid 값은 NaN으로 변환됨
        assert pd.api.types.is_numeric_dtype(result['numeric_col'])
        assert result['numeric_col'].iloc[0] == 1.5
        assert pd.isna(result['numeric_col'].iloc[1])  # 'invalid' -> NaN
        assert result['numeric_col'].iloc[2] == 3.5
    
    def test_convert_schema_missing_columns_ignored(self):
        """케이스 C: 스키마에 있지만 데이터에 없는 컬럼은 무시"""
        # Given: 스키마에 정의되었지만 데이터에 없는 컬럼
        df = pd.DataFrame({
            'existing_col': [1, 2, 3]
        })
        
        expected_schema = {
            'existing_col': 'numeric',
            'missing_col': 'category'  # 데이터에 없음
        }
        
        # When: 스키마 변환
        result = convert_schema(df, expected_schema)
        
        # Then: 에러 없이 처리, existing_col만 변환됨
        assert 'existing_col' in result.columns
        assert 'missing_col' not in result.columns
        assert pd.api.types.is_numeric_dtype(result['existing_col'])


class TestSchemaConsistencyValidator:
    """SchemaConsistencyValidator 클래스 테스트 (고급 Training/Inference 일관성 검증)"""
    
    def test_validator_initialization(self):
        """케이스 A: Validator 초기화 테스트"""
        # Given: Training 스키마 메타데이터
        training_schema = {
            'entity_columns': ['customer_id'],
            'timestamp_column': 'timestamp',
            'inference_columns': ['feature_1', 'feature_2'],
            'column_types': {'feature_1': 'float64', 'feature_2': 'int64'},
            'target_column': 'target',
            'task_choice': 'classification'
        }
        
        # When: Validator 초기화
        validator = SchemaConsistencyValidator(training_schema)
        
        # Then: 정상 초기화
        assert validator.training_schema == training_schema
    
    def test_validate_inference_consistency_success(self):
        """케이스 B: Inference 일관성 검증 성공"""
        # Given: Training 스키마와 호환되는 Inference 데이터
        training_schema = {
            'entity_columns': ['customer_id'],
            'timestamp_column': 'timestamp',
            'inference_columns': ['feature_1', 'feature_2'],
            'column_types': {'feature_1': 'float64', 'feature_2': 'int64'},
            'target_column': 'target',
            'task_choice': 'classification'
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 호환되는 Inference 데이터
        inference_df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [10, 20, 30]
        })
        
        # When: 일관성 검증
        result = validator.validate_inference_consistency(inference_df)
        
        # Then: 성공
        assert result is True
    
    def test_validate_inference_missing_columns_error(self):
        """케이스 C: Inference 데이터에 필수 컬럼 누락 시 에러"""
        # Given: Training 스키마
        training_schema = {
            'inference_columns': ['feature_1', 'feature_2'],
            'column_types': {'feature_1': 'float64', 'feature_2': 'int64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 필수 컬럼 누락된 Inference 데이터
        inference_df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0]
            # feature_2 누락
        })
        
        # When & Then: ValueError 발생
        with pytest.raises(ValueError, match="Inference 데이터에 Training 시 필수 컬럼이 없습니다"):
            validator.validate_inference_consistency(inference_df)
    
    def test_validate_inference_dtype_incompatibility_error(self):
        """케이스 D: 호환되지 않는 타입 변경 시 에러"""
        # Given: Training 스키마
        training_schema = {
            'inference_columns': ['feature_1', 'feature_2'],
            'column_types': {'feature_1': 'float64', 'feature_2': 'int64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 호환되지 않는 타입의 Inference 데이터
        inference_df = pd.DataFrame({
            'feature_1': ['string_value', 'another_string', 'text'],  # float64 -> object (비호환)
            'feature_2': [10, 20, 30]
        })
        
        # When & Then: ValueError 발생 (타입 불일치)
        with pytest.raises(ValueError, match="컬럼 'feature_1' 타입 불일치"):
            validator.validate_inference_consistency(inference_df)
    
    def test_validate_inference_compatible_dtype_success(self):
        """케이스 E: 호환되는 타입 변경은 통과"""
        # Given: Training 스키마
        training_schema = {
            'inference_columns': ['feature_1', 'feature_2'],
            'column_types': {'feature_1': 'float64', 'feature_2': 'int64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 호환되는 타입의 Inference 데이터 (float64 -> float32, int64 -> int32 호환)
        inference_df = pd.DataFrame({
            'feature_1': np.array([1.0, 2.0, 3.0], dtype='float32'),  # float64 -> float32 (호환)
            'feature_2': np.array([10, 20, 30], dtype='int32')         # int64 -> int32 (호환)
        })
        
        # When: 일관성 검증
        result = validator.validate_inference_consistency(inference_df)
        
        # Then: 성공 (호환되는 타입 변경)
        assert result is True
    
    def test_validate_point_in_time_columns_success(self):
        """케이스 F: Point-in-Time 컬럼 특별 검증 성공"""
        # Given: Entity/Timestamp 정보 포함 스키마
        training_schema = {
            'entity_columns': ['customer_id', 'product_id'],
            'timestamp_column': 'timestamp',
            'inference_columns': ['customer_id', 'product_id', 'timestamp', 'feature_1'],
            'column_types': {'customer_id': 'int64', 'product_id': 'object', 'timestamp': 'datetime64[ns]', 'feature_1': 'float64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # Entity/Timestamp 포함된 올바른 데이터
        inference_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'product_id': ['A', 'B', 'C'],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'feature_1': [1.0, 2.0, 3.0]
        })
        
        # When: Point-in-Time 검증
        result = validator.validate_inference_consistency(inference_df)
        
        # Then: 성공
        assert result is True
    
    def test_validate_point_in_time_invalid_timestamp_error(self):
        """케이스 G: 잘못된 Timestamp 타입 시 에러"""
        # Given: Timestamp 포함 스키마
        training_schema = {
            'entity_columns': ['customer_id'],
            'timestamp_column': 'timestamp',
            'inference_columns': ['customer_id', 'timestamp', 'feature_1'],
            'column_types': {'customer_id': 'int64', 'timestamp': 'datetime64[ns]', 'feature_1': 'float64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 잘못된 timestamp 타입 데이터
        inference_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'timestamp': ['invalid_date', 'also_invalid', 'bad_date'],  # datetime이 아님
            'feature_1': [1.0, 2.0, 3.0]
        })
        
        # When & Then: ValueError 발생 (Timestamp 타입 오류)
        with pytest.raises(ValueError, match="Timestamp 컬럼 'timestamp'이 datetime 타입이 아닙니다"):
            validator.validate_inference_consistency(inference_df)
    
    def test_validate_point_in_time_future_data_warning(self):
        """케이스 H: 미래 데이터 감지 시 경고 (통과하지만 경고)"""
        # Given: Timestamp 포함 스키마
        training_schema = {
            'entity_columns': ['customer_id'],
            'timestamp_column': 'timestamp',
            'inference_columns': ['customer_id', 'timestamp', 'feature_1'],
            'column_types': {'customer_id': 'int64', 'timestamp': 'datetime64[ns]', 'feature_1': 'float64'}
        }
        
        validator = SchemaConsistencyValidator(training_schema)
        
        # 미래 데이터 포함 (경고용)
        future_date = pd.Timestamp.now() + pd.Timedelta(days=1)
        inference_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'timestamp': [pd.Timestamp.now(), future_date, pd.Timestamp.now()],
            'feature_1': [1.0, 2.0, 3.0]
        })
        
        # When: Point-in-Time 검증 (경고 발생하지만 통과)
        with patch('src.utils.core.logger.logger') as mock_logger:
            result = validator.validate_inference_consistency(inference_df)
            
            # Then: 성공하지만 경고 로그 발생
            assert result is True
            mock_logger.warning.assert_called()  # 경고 로그 호출됨


class TestGenerateTrainingSchemaMetadata:
    """generate_training_schema_metadata 함수 테스트 (완전한 스키마 메타데이터 생성)"""
    
    def test_generate_metadata_with_entity_columns(self):
        """케이스 A: Entity 컬럼 있는 경우 - Entity만 Inference에 사용"""
        # Given: Entity 컬럼 포함 설정
        np.random.seed(42)
        training_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'product_id': ['A', 'B', 'C'],
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [10, 20, 30],
            'target': [0, 1, 0]
        })
        
        data_interface_config = {
            'entity_columns': ['customer_id', 'product_id'],
            'timestamp_column': 'timestamp',
            'target_column': 'target',
            'task_choice': 'classification'
        }
        
        # When: 스키마 메타데이터 생성
        metadata = generate_training_schema_metadata(training_df, data_interface_config)
        
        # Then: Entity 컬럼만 Inference에 사용됨 (서빙 최소 입력)
        assert metadata['entity_columns'] == ['customer_id', 'product_id']
        assert metadata['inference_columns'] == ['customer_id', 'product_id']
        assert metadata['target_column'] == 'target'
        assert metadata['task_choice'] == 'classification'
        assert metadata['training_columns'] == ['customer_id', 'product_id', 'feature_1', 'feature_2', 'target']
        assert len(metadata['column_types']) == 5  # 모든 컬럼 타입 정보
        assert metadata['schema_version'] == '2.0'
        assert 'created_at' in metadata
    
    def test_generate_metadata_without_entity_columns(self):
        """케이스 B: Entity 컬럼 없는 경우 - 타겟 제외한 모든 피처 사용"""
        # Given: Entity 컬럼 없는 설정
        np.random.seed(42)
        training_df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [10, 20, 30],
            'feature_3': [0.5, 1.5, 2.5],
            'target': [0, 1, 0]
        })
        
        data_interface_config = {
            'entity_columns': [],  # Entity 없음
            'target_column': 'target',
            'task_choice': 'regression'
        }
        
        # When: 스키마 메타데이터 생성
        metadata = generate_training_schema_metadata(training_df, data_interface_config)
        
        # Then: 타겟 제외한 모든 피처가 Inference에 사용됨
        assert metadata['entity_columns'] == []
        assert set(metadata['inference_columns']) == {'feature_1', 'feature_2', 'feature_3'}
        assert metadata['target_column'] == 'target'
        assert metadata['task_choice'] == 'regression'
        assert len(metadata['training_columns']) == 4
        assert metadata['total_training_samples'] == 3
        assert metadata['column_count'] == 4
    
    def test_generate_metadata_clustering_no_target(self):
        """케이스 C: Clustering - Target 없는 경우"""
        # Given: Clustering (target 없음)
        np.random.seed(42)
        training_df = pd.DataFrame({
            'item_id': [1, 2, 3],
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [10, 20, 30]
        })
        
        data_interface_config = {
            'entity_columns': ['item_id'],
            'target_column': None,  # Clustering은 target 없음
            'task_choice': 'clustering'
        }
        
        # When: 스키마 메타데이터 생성
        metadata = generate_training_schema_metadata(training_df, data_interface_config)
        
        # Then: Entity만 Inference에 사용, target 없음
        assert metadata['entity_columns'] == ['item_id']
        assert metadata['inference_columns'] == ['item_id']
        assert metadata['target_column'] is None
        assert metadata['task_choice'] == 'clustering'
        assert 'point_in_time_safe' in metadata
        assert 'sql_injection_safe' in metadata
    
    def test_generate_metadata_comprehensive_types(self):
        """케이스 D: 다양한 타입의 컬럼들이 올바르게 메타데이터에 반영되는지 확인"""
        # Given: 다양한 타입의 데이터
        np.random.seed(42)
        training_df = pd.DataFrame({
            'entity_int': [1, 2, 3],
            'entity_str': ['A', 'B', 'C'],
            'float_feature': [1.5, 2.5, 3.5],
            'int_feature': [10, 20, 30],
            'bool_feature': [True, False, True],
            'datetime_feature': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'target': [0.1, 0.8, 0.3]
        })
        
        data_interface_config = {
            'entity_columns': ['entity_int', 'entity_str'],
            'timestamp_column': 'datetime_feature',
            'target_column': 'target',
            'task_choice': 'regression'
        }
        
        # When: 스키마 메타데이터 생성
        metadata = generate_training_schema_metadata(training_df, data_interface_config)
        
        # Then: 모든 타입이 올바르게 캡처됨
        column_types = metadata['column_types']
        assert 'int' in column_types['entity_int']
        assert 'object' in column_types['entity_str']
        assert 'float' in column_types['float_feature']
        assert 'int' in column_types['int_feature']
        assert 'bool' in column_types['bool_feature']
        assert 'datetime' in column_types['datetime_feature']
        assert 'float' in column_types['target']
        
        # 메타데이터 완성도 확인
        assert metadata['schema_version'] == '2.0'
        assert isinstance(metadata['created_at'], str)
        assert metadata['point_in_time_safe'] is True
        assert metadata['sql_injection_safe'] is True