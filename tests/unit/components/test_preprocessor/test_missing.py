"""
Missing Indicator 모듈 테스트

결측값 지시자를 생성하는 변환기를 테스트:
- MissingIndicatorWrapper: sklearn MissingIndicator 래퍼
  * features: 'missing-only' (기본값), 'all'
  * 결측값이 있었던 위치를 바이너리로 표시
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.impute import MissingIndicator

from src.components.preprocessor.modules.missing import MissingIndicatorWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestMissingIndicatorWrapper:
    """MissingIndicatorWrapper 테스트 클래스"""
    
    def test_missing_indicator_initialization_default(self):
        """MissingIndicator 기본 초기화 테스트"""
        # Given: 기본 초기화
        indicator = MissingIndicatorWrapper()
        
        # Then: 올바른 초기화 확인
        assert indicator.features == 'missing-only'
        assert indicator.columns is None
        assert isinstance(indicator.indicator, MissingIndicator)
        assert hasattr(indicator, 'fit')
        assert hasattr(indicator, 'transform')
        assert hasattr(indicator, 'get_feature_names_out')
    
    def test_missing_indicator_initialization_with_parameters(self):
        """파라미터 지정 MissingIndicator 초기화"""
        # Given: 파라미터 지정
        features = 'all'
        columns = ['feature_1', 'feature_2']
        
        indicator = MissingIndicatorWrapper(
            features=features,
            columns=columns
        )
        
        # Then: 파라미터가 올바르게 설정됨
        assert indicator.features == features
        assert indicator.columns == columns
    
    def test_missing_indicator_missing_only_features(self):
        """missing-only 옵션 테스트 (기본값)"""
        # Given: 결측값이 있는 데이터
        data = DataFrameBuilder.build_missing_values_data(50)
        # 결측값이 없는 컬럼과 있는 컬럼 선택
        test_data = data[['numeric_complete', 'numeric_few_missing', 'numeric_many_missing']]
        
        indicator = MissingIndicatorWrapper(features='missing-only')
        
        # When: fit 및 transform 수행
        indicator.fit(test_data)
        result = indicator.transform(test_data)
        
        # Then: 결측값이 있던 컬럼에 대해서만 indicator 생성
        assert result.shape[0] == test_data.shape[0]  # 행 수는 동일
        
        # numeric_complete는 결측값이 없으므로 indicator가 생성되지 않아야 함
        # numeric_few_missing, numeric_many_missing은 결측값이 있으므로 indicator 생성됨
        expected_cols = 2  # 결측값이 있는 컬럼 2개
        assert result.shape[1] == expected_cols
        
        # 결과는 0 또는 1 값만 가져야 함 (바이너리 indicator)
        unique_values = np.unique(result.ravel())
        assert set(unique_values).issubset({0, 1})
    
    def test_missing_indicator_all_features(self):
        """all features 옵션 테스트"""
        # Given: 결측값이 일부만 있는 데이터
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_complete', 'numeric_few_missing']]
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 및 transform 수행
        indicator.fit(test_data)
        result = indicator.transform(test_data)
        
        # Then: 모든 컬럼에 대해 indicator 생성
        assert result.shape[0] == test_data.shape[0]
        assert result.shape[1] == test_data.shape[1]  # 원본과 동일한 컬럼 수
        
        # 결과는 0 또는 1 값만 가져야 함
        unique_values = np.unique(result.ravel())
        assert set(unique_values).issubset({0, 1})
        
        # numeric_complete 컬럼 (결측값 없음)의 indicator는 모두 0이어야 함
        complete_col_idx = 0  # 'numeric_complete'는 첫 번째 컬럼
        complete_indicators = result[:, complete_col_idx]
        assert np.all(complete_indicators == 0)
    
    def test_missing_indicator_transform_preserves_missing_pattern(self):
        """결측값 패턴이 올바르게 보존되는지 테스트"""
        # Given: 특정 패턴의 결측값을 가진 데이터 생성
        np.random.seed(42)
        data = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5, np.nan, 7],
            'feature_2': [np.nan, 2, 3, 4, np.nan, 6, 7]
        })
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 및 transform 수행
        indicator.fit(data)
        result = indicator.transform(data)
        
        # Then: 결측값 위치와 indicator 결과가 일치
        assert result.shape == data.shape
        
        # feature_1의 결측값 위치 (인덱스 2, 5)
        feature1_missing = result[:, 0]
        expected_feature1 = [0, 0, 1, 0, 0, 1, 0]  # 인덱스 2, 5에서 1
        assert np.array_equal(feature1_missing, expected_feature1)
        
        # feature_2의 결측값 위치 (인덱스 0, 4)
        feature2_missing = result[:, 1]
        expected_feature2 = [1, 0, 0, 0, 1, 0, 0]  # 인덱스 0, 4에서 1
        assert np.array_equal(feature2_missing, expected_feature2)
    
    def test_missing_indicator_no_missing_values(self):
        """결측값이 없는 데이터에 대한 테스트"""
        # Given: 결측값이 전혀 없는 데이터
        data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [6, 7, 8, 9, 10]
        })
        
        indicator = MissingIndicatorWrapper(features='missing-only')
        
        # When: fit 및 transform 수행
        indicator.fit(data)
        result = indicator.transform(data)
        
        # Then: 결측값이 없으므로 빈 결과 (컬럼 수 0)
        assert result.shape[0] == data.shape[0]
        assert result.shape[1] == 0  # 결측값이 있는 컬럼이 없음
    
    def test_missing_indicator_all_missing_values(self):
        """모든 값이 결측값인 컬럼 테스트"""
        # Given: 한 컬럼이 모두 결측값인 데이터
        data = pd.DataFrame({
            'all_missing': [np.nan] * 10,
            'some_missing': [1, np.nan, 3, np.nan, 5] * 2
        })
        
        indicator = MissingIndicatorWrapper(features='missing-only')
        
        # When: fit 및 transform 수행
        indicator.fit(data)
        result = indicator.transform(data)
        
        # Then: 결측값이 있는 컬럼에 대해 indicator 생성
        assert result.shape[0] == data.shape[0]
        assert result.shape[1] == 2  # 두 컬럼 모두 결측값 존재
        
        # all_missing 컬럼의 indicator는 모두 1이어야 함
        all_missing_indicators = result[:, 0]
        assert np.all(all_missing_indicators == 1)
    
    def test_missing_indicator_get_feature_names_out(self):
        """피처 이름 출력 테스트"""
        # Given: 결측값이 있는 데이터
        data = pd.DataFrame({
            'numeric_1': [1, 2, np.nan, 4],
            'numeric_2': [1, np.nan, 3, 4]
        })
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 후 피처 이름 확인
        indicator.fit(data)
        feature_names = indicator.get_feature_names_out(['numeric_1', 'numeric_2'])
        
        # Then: 적절한 피처 이름 생성
        assert len(feature_names) == 2
        # sklearn의 MissingIndicator가 생성하는 이름 패턴 확인
        assert all('missingindicator' in name.lower() for name in feature_names)
    
    def test_missing_indicator_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 missing indicator 생성
        indicator = PreprocessorStepRegistry.create("missing_indicator")
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_few_missing', 'numeric_many_missing']]
        
        # When: fit 및 transform 수행
        indicator.fit(test_data)
        result = indicator.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(indicator, MissingIndicatorWrapper)
        assert result.shape[0] == test_data.shape[0]
        # missing-only 기본 설정으로 결측값이 있는 컬럼들에 대해 indicator 생성
        assert result.shape[1] == 2
    
    def test_missing_indicator_custom_parameters_registry(self):
        """Registry를 통한 커스텀 파라미터 테스트"""
        # Given: 커스텀 파라미터로 missing indicator 생성
        indicator = PreprocessorStepRegistry.create(
            "missing_indicator",
            features='all',
            columns=['col1', 'col2']
        )
        
        # Then: 파라미터가 올바르게 적용됨
        assert indicator.features == 'all'
        assert indicator.columns == ['col1', 'col2']


class TestMissingIndicatorEdgeCases:
    """MissingIndicator 경계 사례 테스트"""
    
    def test_missing_indicator_single_column(self):
        """단일 컬럼 테스트"""
        # Given: 하나의 컬럼만 있는 데이터
        data = pd.DataFrame({
            'single_feature': [1, np.nan, 3, np.nan, 5]
        })
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 및 transform 수행
        indicator.fit(data)
        result = indicator.transform(data)
        
        # Then: 단일 컬럼도 올바르게 처리됨
        assert result.shape == data.shape
        expected = np.array([[0], [1], [0], [1], [0]])  # 인덱스 1, 3에서 결측값
        assert np.array_equal(result, expected)
    
    def test_missing_indicator_mixed_data_types(self):
        """혼합 데이터 타입에 대한 테스트"""
        # Given: 숫자형과 범주형이 혼합된 데이터
        data = pd.DataFrame({
            'numeric': [1.5, np.nan, 3.7, np.nan],
            'category': ['A', np.nan, 'C', 'D'],
            'integer': [10, 20, np.nan, 40]
        })
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 및 transform 수행
        indicator.fit(data)
        result = indicator.transform(data)
        
        # Then: 모든 데이터 타입에 대해 올바르게 처리됨
        assert result.shape == data.shape
        assert result.dtype == np.int64  # indicator 결과는 정수형
        
        # 각 컬럼의 결측값 패턴 확인
        expected = np.array([
            [0, 0, 0],  # 첫 번째 행: 결측값 없음
            [1, 1, 0],  # 두 번째 행: numeric, category에서 결측값
            [0, 0, 1],  # 세 번째 행: integer에서 결측값
            [1, 0, 0]   # 네 번째 행: numeric에서 결측값
        ])
        assert np.array_equal(result, expected)
    
    def test_missing_indicator_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        # Given: 빈 데이터프레임
        empty_data = pd.DataFrame()
        indicator = MissingIndicatorWrapper()
        
        # When/Then: 빈 데이터에 대한 적절한 처리
        try:
            indicator.fit(empty_data)
            result = indicator.transform(empty_data)
            assert result.shape == empty_data.shape
        except (ValueError, IndexError):
            # sklearn이 빈 데이터를 처리하지 못하는 것은 정상적 동작
            pass


class TestMissingIndicatorComparison:
    """MissingIndicator 비교 테스트"""
    
    def test_missing_only_vs_all_features(self):
        """missing-only vs all features 비교 테스트"""
        # Given: 일부 컬럼에만 결측값이 있는 데이터
        data = pd.DataFrame({
            'no_missing': [1, 2, 3, 4, 5],
            'some_missing': [1, np.nan, 3, np.nan, 5],
            'more_missing': [np.nan, 2, np.nan, 4, np.nan]
        })
        
        # When: 각 옵션으로 처리
        missing_only_indicator = MissingIndicatorWrapper(features='missing-only')
        all_indicator = MissingIndicatorWrapper(features='all')
        
        missing_only_indicator.fit(data)
        all_indicator.fit(data)
        
        missing_only_result = missing_only_indicator.transform(data)
        all_result = all_indicator.transform(data)
        
        # Then: 결과 비교
        # missing-only: 결측값이 있는 컬럼 2개에 대해서만 indicator
        assert missing_only_result.shape == (5, 2)
        
        # all: 모든 컬럼 3개에 대해 indicator
        assert all_result.shape == data.shape
        
        # no_missing 컬럼의 all_result는 모두 0이어야 함
        no_missing_indicators = all_result[:, 0]
        assert np.all(no_missing_indicators == 0)
    
    def test_missing_indicator_consistency(self):
        """동일 데이터에 대한 일관성 테스트"""
        # Given: 테스트 데이터
        data = DataFrameBuilder.build_missing_values_data(30)
        test_data = data[['numeric_few_missing', 'category_missing']]
        
        # When: 같은 설정으로 여러 번 처리
        indicator1 = MissingIndicatorWrapper(features='all')
        indicator2 = MissingIndicatorWrapper(features='all')
        
        indicator1.fit(test_data)
        indicator2.fit(test_data)
        
        result1 = indicator1.transform(test_data)
        result2 = indicator2.transform(test_data)
        
        # Then: 동일한 결과 생성
        assert np.array_equal(result1, result2)
        assert result1.shape == result2.shape


class TestMissingIndicatorErrorHandling:
    """MissingIndicator 오류 처리 테스트"""
    
    def test_missing_indicator_invalid_features_parameter(self):
        """잘못된 features 파라미터 처리 테스트"""
        # Given & When/Then: 잘못된 features 파라미터
        with pytest.raises(ValueError):
            # sklearn MissingIndicator가 잘못된 features 값을 처리할 때 발생하는 오류
            indicator = MissingIndicatorWrapper(features='invalid_option')
            data = pd.DataFrame({'feature': [1, np.nan, 3]})
            indicator.fit(data)
    
    def test_missing_indicator_transform_before_fit(self):
        """fit 전에 transform 호출 시 오류 처리"""
        # Given: fit되지 않은 indicator
        indicator = MissingIndicatorWrapper()
        data = pd.DataFrame({'feature': [1, np.nan, 3]})
        
        # When/Then: fit 전에 transform 호출하면 오류 발생
        with pytest.raises((ValueError, AttributeError)):
            indicator.transform(data)
    
    def test_missing_indicator_very_small_dataset(self):
        """매우 작은 데이터셋 처리 테스트"""
        # Given: 매우 작은 데이터셋
        tiny_data = pd.DataFrame({
            'feature': [np.nan, 1.0]  # 2개 데이터 포인트
        })
        
        indicator = MissingIndicatorWrapper(features='all')
        
        # When: fit 및 transform 수행
        indicator.fit(tiny_data)
        result = indicator.transform(tiny_data)
        
        # Then: 작은 데이터셋도 적절히 처리됨
        assert result.shape == tiny_data.shape
        expected = np.array([[1], [0]])  # 첫 번째 행에서 결측값
        assert np.array_equal(result, expected)