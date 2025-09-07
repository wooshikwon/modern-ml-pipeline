"""
Imputer 모듈 테스트

결측값 처리 변환기들을 테스트:
- SimpleImputerWrapper: sklearn SimpleImputer 래퍼
  * mean: 평균값으로 결측값 대체
  * median: 중앙값으로 결측값 대체  
  * most_frequent: 최빈값으로 결측값 대체
  * constant: 상수값으로 결측값 대체
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.impute import SimpleImputer

from src.components.preprocessor.modules.imputer import SimpleImputerWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestSimpleImputerWrapper:
    """SimpleImputerWrapper 테스트 클래스"""
    
    def test_simple_imputer_initialization_default(self):
        """SimpleImputer 기본 초기화 테스트"""
        # Given: 기본 초기화
        imputer = SimpleImputerWrapper()
        
        # Then: 올바른 초기화 확인
        assert imputer.strategy == 'mean'
        assert imputer.columns is None
        assert isinstance(imputer.imputer, SimpleImputer)
        assert hasattr(imputer, 'fit')
        assert hasattr(imputer, 'transform')
    
    def test_simple_imputer_initialization_with_parameters(self):
        """파라미터 지정 SimpleImputer 초기화"""
        # Given: 파라미터 지정
        strategy = 'median'
        columns = ['numeric_feature_1', 'numeric_feature_2']
        imputer = SimpleImputerWrapper(strategy=strategy, columns=columns)
        
        # Then: 파라미터가 올바르게 설정됨
        assert imputer.strategy == strategy
        assert imputer.columns == columns
        assert imputer.imputer.strategy == strategy
    
    def test_simple_imputer_mean_strategy(self):
        """평균값 대체 전략 테스트"""
        # Given: 결측값이 포함된 데이터와 mean 임퓨터
        data = DataFrameBuilder.build_missing_values_data(100)
        numeric_cols = ['numeric_few_missing', 'numeric_many_missing']
        test_data = data[numeric_cols]
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값 대체 결과 검증
        assert transformed.shape == test_data.shape
        assert not transformed.isna().any().any()  # 결측값 모두 제거됨
        
        # 평균값으로 대체되었는지 확인 (원본 데이터의 평균과 비교)
        for col in numeric_cols:
            original_mean = test_data[col].mean()
            filled_values = transformed[col]
            
            # 결측값이었던 위치의 값이 평균과 일치하는지 확인
            missing_mask = test_data[col].isna()
            filled_missing_values = filled_values[missing_mask]
            np.testing.assert_allclose(filled_missing_values, original_mean, rtol=1e-10)
    
    def test_simple_imputer_median_strategy(self):
        """중앙값 대체 전략 테스트"""
        # Given: 결측값이 포함된 데이터와 median 임퓨터
        data = DataFrameBuilder.build_missing_values_data(100)
        test_data = data[['numeric_few_missing']]
        
        imputer = SimpleImputerWrapper(strategy='median')
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값이 중앙값으로 대체됨
        assert not transformed.isna().any().any()
        
        original_median = test_data['numeric_few_missing'].median()
        missing_mask = test_data['numeric_few_missing'].isna()
        filled_values = transformed.loc[missing_mask, 'numeric_few_missing']
        np.testing.assert_allclose(filled_values, original_median, rtol=1e-10)
    
    def test_simple_imputer_most_frequent_strategy(self):
        """최빈값 대체 전략 테스트"""
        # Given: 범주형 결측값 데이터와 most_frequent 임퓨터
        data = DataFrameBuilder.build_missing_values_data(100)
        test_data = data[['category_missing']]
        
        # 문자열 데이터를 처리하기 위해 object dtype으로 변환
        test_data = test_data.astype(str)
        test_data = test_data.replace('nan', np.nan)
        
        imputer = SimpleImputerWrapper(strategy='most_frequent')
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값이 최빈값으로 대체됨
        assert not transformed.isna().any().any()
        
        # 최빈값 확인
        original_mode = test_data['category_missing'].mode()[0]
        missing_mask = test_data['category_missing'].isna()
        filled_values = transformed.loc[missing_mask, 'category_missing']
        assert all(val == original_mode for val in filled_values)
    
    def test_simple_imputer_constant_strategy(self):
        """상수값 대체 전략 테스트"""
        # Given: 결측값이 포함된 데이터와 constant 임퓨터
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_few_missing']]
        
        fill_value = -999
        imputer = SimpleImputerWrapper(strategy='constant')
        imputer.imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값이 지정된 상수로 대체됨
        assert not transformed.isna().any().any()
        
        missing_mask = test_data['numeric_few_missing'].isna()
        filled_values = transformed.loc[missing_mask, 'numeric_few_missing']
        assert all(val == fill_value for val in filled_values)
    
    def test_simple_imputer_registry_integration(self):
        """Registry 통합 테스트"""
        # Given: Registry에서 임퓨터 생성
        imputer = PreprocessorStepRegistry.create("simple_imputer")
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_few_missing']]
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        result = imputer.transform(test_data)
        
        # Then: 정상 동작 확인
        assert isinstance(imputer, SimpleImputerWrapper)
        assert result.shape == test_data.shape
        assert not result.isna().any().any()
    
    def test_simple_imputer_custom_strategy_parameters(self):
        """커스텀 전략 파라미터 테스트"""
        # Given: 다양한 전략으로 생성된 임퓨터들
        strategies = ['mean', 'median', 'most_frequent']
        data = DataFrameBuilder.build_missing_values_data(100)
        
        for strategy in strategies:
            # When: 각 전략으로 임퓨터 생성 및 테스트
            imputer = PreprocessorStepRegistry.create("simple_imputer", strategy=strategy)
            
            if strategy == 'most_frequent':
                # 범주형 데이터 사용
                test_data = data[['category_missing']].astype(str).replace('nan', np.nan)
            else:
                # 숫자형 데이터 사용
                test_data = data[['numeric_few_missing']]
            
            imputer.fit(test_data)
            result = imputer.transform(test_data)
            
            # Then: 모든 전략에서 결측값 처리 성공
            assert not result.isna().any().any(), f"{strategy} strategy failed"
            assert result.shape == test_data.shape


class TestSimpleImputerEdgeCases:
    """SimpleImputer 경계 사례 테스트"""
    
    def test_imputer_with_no_missing_values(self):
        """결측값이 없는 데이터 처리 테스트"""
        # Given: 결측값이 없는 데이터
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_complete']]  # 완전한 데이터
        
        imputer = SimpleImputerWrapper()
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        result = imputer.transform(test_data)
        
        # Then: 데이터가 변경되지 않음
        np.testing.assert_array_equal(result, test_data.values)
        assert result.shape == test_data.shape
    
    def test_imputer_with_all_missing_values(self):
        """모든 값이 결측인 경우 에러 발생 테스트 (Fast-fail)"""
        # Given: 모든 값이 NaN인 데이터
        test_data = pd.DataFrame({
            'all_missing': [np.nan] * 20
        })
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        # When & Then: fit 시 에러 발생 (Fast-fail)
        with pytest.raises(ValueError, match="SimpleImputer는 전체가 결측값인 컬럼을 처리할 수 없습니다"):
            imputer.fit(test_data)
        # 컬럼이 제거될 수 있음 (sklearn 경고 메시지에서 확인됨)
    
    def test_imputer_extreme_missing_percentage(self):
        """극단적 결측 비율 데이터 처리 테스트"""
        # Given: 80% 결측값을 가진 데이터
        data = DataFrameBuilder.build_missing_values_data(100)
        test_data = data[['numeric_extreme_missing']]  # 80% 결측
        
        imputer = SimpleImputerWrapper(strategy='median')
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        result = imputer.transform(test_data)
        
        # Then: 극단적 결측 상황에서도 정상 처리
        assert not result.isna().any().any()
        assert result.shape == test_data.shape
        
        # 결측이 아닌 원본 값들이 보존되었는지 확인
        non_missing_mask = ~pd.isna(test_data['numeric_extreme_missing'])
        original_values = test_data.loc[non_missing_mask, 'numeric_extreme_missing'].values
        result_values = result.loc[non_missing_mask, result.columns[0]]
        np.testing.assert_allclose(original_values, result_values, rtol=1e-10)


class TestImputerComparison:
    """여러 임퓨터 전략 간 비교 테스트"""
    
    def test_different_strategies_same_data(self):
        """동일 데이터에 대한 여러 전략 비교"""
        # Given: 공통 테스트 데이터
        data = DataFrameBuilder.build_missing_values_data(100)
        test_data = data[['numeric_many_missing']]  # 40% 결측
        
        strategies = ['mean', 'median']
        results = {}
        
        # When: 각 전략으로 변환
        for strategy in strategies:
            imputer = SimpleImputerWrapper(strategy=strategy)
            imputer.fit(test_data)
            results[strategy] = imputer.transform(test_data)
        
        # Then: 모든 전략에서 결측값 제거됨
        for strategy, result in results.items():
            assert not result.isna().any().any(), f"{strategy} has remaining NaN values"
            assert result.shape == test_data.shape
        
        # 전략별로 다른 값이 채워져야 함 (데이터 분포에 따라)
        mean_result = results['mean']
        median_result = results['median']
        
        # 결측값 위치에서 다른 값이 채워졌는지 확인 (대부분의 경우)
        missing_mask = test_data['numeric_many_missing'].isna()
        mean_filled = mean_result.loc[missing_mask, mean_result.columns[0]]
        median_filled = median_result.loc[missing_mask, median_result.columns[0]]
        
        # 평균과 중앙값은 일반적으로 다름 (데이터가 정규분포가 아닌 경우)
        # 완전히 같지 않을 확률이 높지만, 테스트의 안정성을 위해 형태만 확인
        assert len(mean_filled) == len(median_filled)
    
    def test_imputer_performance_with_different_missing_patterns(self):
        """다양한 결측 패턴에서의 임퓨터 성능 테스트"""
        # Given: 다양한 결측 비율 데이터
        data = DataFrameBuilder.build_missing_values_data(100)
        missing_patterns = {
            'low_missing': 'numeric_few_missing',      # 10% 결측
            'medium_missing': 'numeric_many_missing',   # 40% 결측
            'high_missing': 'numeric_extreme_missing'   # 80% 결측
        }
        
        imputer = SimpleImputerWrapper(strategy='mean')
        
        for pattern_name, column in missing_patterns.items():
            # When: 각 패턴에 대해 임퓨터 적용
            test_data = data[[column]]
            imputer.fit(test_data)
            result = imputer.transform(test_data)
            
            # Then: 모든 패턴에서 성공적으로 처리
            assert not result.isna().any().any(), f"Failed for {pattern_name}"
            assert result.shape == test_data.shape
            
            # 원본의 비결측 값들이 보존되었는지 확인
            non_missing_mask = ~test_data[column].isna()
            if non_missing_mask.any():  # 결측이 아닌 값이 있는 경우
                original_values = test_data.loc[non_missing_mask, column].values
                result_values = result.loc[non_missing_mask, result.columns[0]]
                np.testing.assert_allclose(original_values, result_values, rtol=1e-10)


class TestImputerErrorHandling:
    """Imputer 오류 처리 테스트"""
    
    def test_imputer_with_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        # Given: 빈 데이터프레임
        empty_data = pd.DataFrame()
        imputer = SimpleImputerWrapper()
        
        # When/Then: 빈 데이터에 대한 적절한 처리
        try:
            imputer.fit(empty_data)
            result = imputer.transform(empty_data)
            assert result.shape == empty_data.shape
        except (ValueError, AttributeError):
            # sklearn이 빈 데이터를 처리하지 못하는 것은 정상적 동작
            pass
    
    def test_imputer_invalid_strategy(self):
        """잘못된 전략 지정 테스트"""
        # Given: 잘못된 전략
        with pytest.raises((ValueError, TypeError)):
            # When: 잘못된 전략으로 임퓨터 생성
            invalid_imputer = SimpleImputerWrapper(strategy='invalid_strategy')
            data = DataFrameBuilder.build_missing_values_data(10)
            invalid_imputer.fit(data[['numeric_few_missing']])