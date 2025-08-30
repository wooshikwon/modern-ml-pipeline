"""TestDataFactory 검증 테스트"""
import pytest
import pandas as pd
import numpy as np

from tests.factories.test_data_factory import TestDataFactory


@pytest.mark.core
@pytest.mark.unit
class TestTestDataFactory:
    """TestDataFactory 기능 검증"""
    
    def test_create_classification_data_basic(self):
        """분류 데이터 생성 기본 검증"""
        df = TestDataFactory.create_classification_data(n_samples=10)
        
        # 기본 구조 검증
        assert len(df) == 10
        assert isinstance(df, pd.DataFrame)
        
        # Entity 스키마 검증 (Blueprint 계약)
        assert 'user_id' in df.columns
        assert 'event_timestamp' in df.columns
        assert 'target' in df.columns
        
        # 데이터 타입 검증
        assert df['target'].dtype in ['int64', 'object']
        assert set(df['target'].unique()).issubset({0, 1})
        assert pd.api.types.is_datetime64_any_dtype(df['event_timestamp'])
    
    def test_create_regression_data_basic(self):
        """회귀 데이터 생성 기본 검증"""
        df = TestDataFactory.create_regression_data(n_samples=10)
        
        assert len(df) == 10
        assert 'user_id' in df.columns
        assert 'event_timestamp' in df.columns
        assert 'target' in df.columns
        
        # 회귀 타겟은 연속값
        assert df['target'].dtype in ['float64']
        assert not df['target'].isna().all()
    
    def test_create_entity_data(self):
        """Entity 데이터 생성 검증"""
        entity_ids = ['user_1', 'user_2', 'user_3']
        df = TestDataFactory.create_entity_data(entity_ids)
        
        assert len(df) == 3
        assert list(df['user_id']) == entity_ids
        assert 'event_timestamp' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['event_timestamp'])
    
    def test_data_consistency_with_seed(self):
        """시드 일관성 검증 - 동일한 시드로 동일한 데이터 생성"""
        df1 = TestDataFactory.create_classification_data(n_samples=50, seed=123)
        df2 = TestDataFactory.create_classification_data(n_samples=50, seed=123)
        
        # 동일한 시드로 생성된 데이터는 동일해야 함
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_comprehensive_training_data_features(self):
        """포괄적 학습 데이터의 다양한 피처 타입 검증"""
        df = TestDataFactory.create_comprehensive_training_data(n_samples=100)
        
        assert len(df) == 100
        
        # 숫자형 피처 존재
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 3, "Should have multiple numeric features"
        
        # 범주형 피처 존재
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_data_cols = [col for col in categorical_cols if col not in ['user_id']]
        assert len(categorical_data_cols) >= 2, "Should have categorical features"
        
        # 결측치 포함 검증
        assert df.isna().sum().sum() > 0, "Should contain missing values"
    
    def test_minimal_entity_data(self):
        """최소한의 Entity 데이터 검증"""
        df = TestDataFactory.create_minimal_entity_data(entity_count=3)
        
        assert len(df) == 3
        assert 'user_id' in df.columns
        assert 'event_ts' in df.columns  # 다른 timestamp 컬럼명
        assert 'label' in df.columns
        
        expected_users = ['u1', 'u2', 'u3']
        assert list(df['user_id']) == expected_users
    
    def test_time_series_data(self):
        """시계열 데이터 생성 검증"""
        df = TestDataFactory.create_time_series_data(n_samples=30, freq='D')
        
        assert len(df) == 30
        assert 'event_timestamp' in df.columns
        assert 'value' in df.columns
        assert 'target' in df.columns
        
        # 날짜 순서 확인
        timestamps = df['event_timestamp']
        assert timestamps.is_monotonic_increasing, "Timestamps should be in order"
    
    def test_standard_exclude_columns(self):
        """표준 제외 컬럼 리스트 검증"""
        exclude_cols = TestDataFactory.get_standard_exclude_columns()
        
        expected_cols = ['user_id', 'event_timestamp', 'event_ts', 'target', 'approved', 'label']
        assert set(exclude_cols) == set(expected_cols)
    
    def test_entity_and_timestamp_helpers(self):
        """Entity 및 Timestamp 헬퍼 메서드 검증"""
        entity_cols = TestDataFactory.get_entity_columns()
        timestamp_col = TestDataFactory.get_timestamp_column()
        
        assert entity_cols == ['user_id']
        assert timestamp_col == 'event_timestamp'
    
    @pytest.mark.extended
    def test_large_data_generation_performance(self):
        """대용량 데이터 생성 성능 테스트"""
        # 1000개 샘플 생성이 합리적 시간 내 완료되는지 확인
        df = TestDataFactory.create_classification_data(n_samples=1000)
        
        assert len(df) == 1000
        assert df.memory_usage(deep=True).sum() > 0
        
        # 메모리 효율성 확인 (너무 크지 않아야 함)
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        assert memory_mb < 10, f"Memory usage too high: {memory_mb:.2f}MB"