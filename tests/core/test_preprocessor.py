"""
Preprocessor 컴포넌트 테스트 (Blueprint v17.0 현대화)

Blueprint 원칙 검증:
- 원칙 8: 자동화된 HPO + Data Leakage 완전 방지
- Data Leakage 방지: Train 데이터에만 fit, Validation/Test에는 transform만
"""

import pandas as pd
import pytest
import numpy as np
from src.core.preprocessor import Preprocessor
from src.settings import Settings

class TestPreprocessorModernized:
    """Preprocessor 컴포넌트 단위 테스트 (Blueprint v17.0, 완전 현대화)"""

    def test_preprocessor_initialization(self, local_test_settings: Settings):
        """
        Preprocessor가 주입된 설정(Settings)으로 올바르게 초기화되는지 테스트합니다.
        """
        preprocessor = Preprocessor(settings=local_test_settings)
        assert preprocessor.settings == local_test_settings
        
        # local_classification_test.yaml에 정의된 exclude_cols 검증
        expected_exclude_cols = local_test_settings.model.preprocessor.exclude_cols
        assert preprocessor.exclude_cols == expected_exclude_cols
        assert "user_id" in preprocessor.exclude_cols

    def test_preprocessor_fit_transform_integration(self, local_test_settings: Settings):
        """
        Preprocessor의 fit_transform 메서드 통합 테스트
        """
        # 테스트용 샘플 데이터 생성 (local_classification_test.yaml 스키마 기반)
        sample_data = {
            'user_id': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'event_timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'age': [25, 30, 25, 40, 35],
            'income': [50000, 80000, 52000, 120000, 75000],
            'region': ['A', 'B', 'A', 'C', 'B'],
            'approved': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)

        preprocessor = Preprocessor(settings=local_test_settings)
        
        # fit_transform 실행
        transformed_df = preprocessor.fit_transform(df)

        # 1. fit 결과 확인
        assert preprocessor._is_fitted()
        assert "region" in preprocessor.categorical_cols_
        assert "age" in preprocessor.numerical_cols_
        assert "income" in preprocessor.numerical_cols_
        
        # 2. transform 결과 확인
        assert isinstance(transformed_df, pd.DataFrame)
        assert all(col in transformed_df.columns for col in preprocessor.feature_names_out_)
        assert "user_id" not in transformed_df.columns, "제외되어야 할 user_id 컬럼이 남아있습니다."
        assert "event_timestamp" not in transformed_df.columns, "제외되어야 할 event_timestamp 컬럼이 남아있습니다."
        
        # 3. 스케일링 확인 (수치형 데이터의 평균이 0에 가까운지)
        assert abs(transformed_df['age'].mean()) < 1e-9
        assert abs(transformed_df['income'].mean()) < 1e-9
        
        # 4. 인코딩 확인 (범주형 데이터가 수치로 변환되었는지)
        assert pd.api.types.is_numeric_dtype(transformed_df['region'])

    # 🆕 Blueprint v17.0: fit과 transform 분리 테스트
    def test_preprocessor_fit_and_transform_separately(self, local_test_settings: Settings):
        """
        fit과 transform을 분리하여 테스트한다.
        Blueprint 원칙 8: Data Leakage 방지 - Train 데이터에만 fit
        """
        # Train 데이터
        train_data = {
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000],
            'region': ['A', 'B', 'A', 'B'],
            'user_id': ['u1', 'u2', 'u3', 'u4']
        }
        train_df = pd.DataFrame(train_data)
        
        # Validation 데이터 (다른 분포)
        val_data = {
            'age': [45, 50, 28, 32],
            'income': [90000, 100000, 55000, 65000],
            'region': ['A', 'B', 'C', 'A'],  # 'C'는 train에 없던 새로운 범주
            'user_id': ['u5', 'u6', 'u7', 'u8']
        }
        val_df = pd.DataFrame(val_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        
        # 1. Train 데이터에만 fit (Data Leakage 방지)
        preprocessor.fit(train_df)
        assert preprocessor._is_fitted()
        
        # 2. Train 데이터 transform
        train_transformed = preprocessor.transform(train_df)
        
        # 3. Validation 데이터 transform (fit 없이)
        val_transformed = preprocessor.transform(val_df)
        
        # 검증: 두 변환 결과가 동일한 컬럼 구조를 가져야 함
        assert list(train_transformed.columns) == list(val_transformed.columns)
        
        # 검증: Train 데이터의 스케일링 기준이 Val 데이터에도 적용되었는지
        # (train 평균은 0에 가깝지만, val 평균은 다를 수 있음)
        assert abs(train_transformed['age'].mean()) < 1e-9  # Train은 평균 0
        assert abs(val_transformed['age'].mean()) > 0.1     # Val은 평균이 다를 수 있음

    # 🆕 Blueprint v17.0: 엣지 케이스 - 새로운 범주형 데이터 처리
    def test_preprocessor_handles_unseen_categories(self, local_test_settings: Settings):
        """
        fit 시점에 없던 새로운 범주형 데이터에 대한 처리를 검증한다.
        """
        # Train 데이터 (A, B 범주만 포함)
        train_data = {
            'region': ['A', 'B', 'A', 'B'],
            'age': [25, 30, 35, 40],
            'user_id': ['u1', 'u2', 'u3', 'u4']
        }
        train_df = pd.DataFrame(train_data)
        
        # Test 데이터 (C, D라는 새로운 범주 포함)
        test_data = {
            'region': ['A', 'C', 'D', 'B'],  # C, D는 train에 없던 범주
            'age': [28, 32, 45, 38],
            'user_id': ['u5', 'u6', 'u7', 'u8']
        }
        test_df = pd.DataFrame(test_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        
        # Train 데이터에 fit
        preprocessor.fit(train_df)
        
        # Test 데이터 transform (새로운 범주가 포함됨)
        # 이때 오류가 발생하지 않고, 적절히 처리되어야 함
        test_transformed = preprocessor.transform(test_df)
        
        # 검증: transform이 성공적으로 수행되었는지
        assert test_transformed is not None
        assert len(test_transformed) == len(test_df)
        assert 'region' in test_transformed.columns
        
        # 검증: 새로운 범주가 숫자로 변환되었는지
        assert pd.api.types.is_numeric_dtype(test_transformed['region'])

    # 🆕 Blueprint v17.0: 엣지 케이스 - 빈 데이터프레임 처리
    def test_preprocessor_handles_empty_dataframe(self, local_test_settings: Settings):
        """
        빈 데이터프레임에 대한 처리를 검증한다.
        """
        # 정상 데이터로 fit
        normal_data = {
            'age': [25, 30, 35],
            'region': ['A', 'B', 'A'],
            'user_id': ['u1', 'u2', 'u3']
        }
        normal_df = pd.DataFrame(normal_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        preprocessor.fit(normal_df)
        
        # 빈 데이터프레임 transform
        empty_df = pd.DataFrame(columns=['age', 'region', 'user_id'])
        
        # 빈 데이터프레임도 적절히 처리되어야 함
        result = preprocessor.transform(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == preprocessor.feature_names_out_

    # 🆕 Blueprint v17.0: 엣지 케이스 - 모든 수치형 또는 모든 범주형 데이터
    def test_preprocessor_handles_only_numerical_data(self, local_test_settings: Settings):
        """
        모든 컬럼이 수치형인 경우를 검증한다.
        """
        # 수치형 데이터만 포함
        numerical_data = {
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000],
            'score': [85.5, 90.2, 78.8, 92.1],
            'user_id': ['u1', 'u2', 'u3', 'u4']  # 제외될 컬럼
        }
        numerical_df = pd.DataFrame(numerical_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        
        result = preprocessor.fit_transform(numerical_df)
        
        # 검증: 모든 출력 컬럼이 수치형이어야 함
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        
        # 검증: user_id는 제외되어야 함
        assert 'user_id' not in result.columns
        
        # 검증: 범주형 컬럼이 없어야 함
        assert len(preprocessor.categorical_cols_) == 0
        assert len(preprocessor.numerical_cols_) == 3  # age, income, score

    def test_preprocessor_handles_only_categorical_data(self, local_test_settings: Settings):
        """
        모든 컬럼이 범주형인 경우를 검증한다.
        """
        # 범주형 데이터만 포함
        categorical_data = {
            'region': ['A', 'B', 'C', 'A'],
            'category': ['X', 'Y', 'X', 'Z'],
            'type': ['premium', 'basic', 'premium', 'standard'],
            'user_id': ['u1', 'u2', 'u3', 'u4']  # 제외될 컬럼
        }
        categorical_df = pd.DataFrame(categorical_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        
        result = preprocessor.fit_transform(categorical_df)
        
        # 검증: 모든 출력 컬럼이 수치형으로 변환되어야 함
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        
        # 검증: user_id는 제외되어야 함
        assert 'user_id' not in result.columns
        
        # 검증: 수치형 컬럼이 없어야 함
        assert len(preprocessor.numerical_cols_) == 0
        assert len(preprocessor.categorical_cols_) == 3  # region, category, type

    # 🆕 Blueprint v17.0: Data Leakage 방지 검증
    def test_preprocessor_prevents_data_leakage(self, local_test_settings: Settings):
        """
        Preprocessor가 Data Leakage를 방지하는지 검증한다.
        Train 데이터의 통계만 사용하여 스케일링해야 함.
        """
        # Train 데이터 (평균: age=30, income=60000)
        train_data = {
            'age': [20, 30, 40],
            'income': [40000, 60000, 80000],
            'user_id': ['u1', 'u2', 'u3']
        }
        train_df = pd.DataFrame(train_data)
        
        # Test 데이터 (평균: age=50, income=100000 - 완전히 다른 분포)
        test_data = {
            'age': [45, 50, 55],
            'income': [90000, 100000, 110000],
            'user_id': ['u4', 'u5', 'u6']
        }
        test_df = pd.DataFrame(test_data)
        
        preprocessor = Preprocessor(settings=local_test_settings)
        
        # Train 데이터에만 fit
        preprocessor.fit(train_df)
        
        # 각각 transform
        train_transformed = preprocessor.transform(train_df)
        test_transformed = preprocessor.transform(test_df)
        
        # 검증: Train 데이터는 평균이 0에 가까워야 함 (자기 자신으로 fit했으므로)
        assert abs(train_transformed['age'].mean()) < 1e-9
        assert abs(train_transformed['income'].mean()) < 1e-9
        
        # 검증: Test 데이터는 평균이 0에서 멀어야 함 (Train 통계로 변환했으므로)
        # Test 데이터가 Train보다 큰 값들이므로, 양수 평균을 가져야 함
        assert test_transformed['age'].mean() > 1.0
        assert test_transformed['income'].mean() > 1.0
        
        # 이것이 Data Leakage 방지의 핵심: Test 데이터의 통계가 변환에 영향을 주지 않음
