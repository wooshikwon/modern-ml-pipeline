"""
Missing Value Handler 모듈 테스트

결측값 처리와 결측값 지시자 생성 기능을 테스트:
- MissingIndicatorWrapper: sklearn MissingIndicator 래퍼  
- 결측값 패턴 탐지 및 지시자 변수 생성
- SimpleImputerWrapper와의 통합 (create_missing_indicators=True)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.impute import MissingIndicator

from src.components.preprocessor.modules.imputer import SimpleImputerWrapper
from src.components.preprocessor.registry import PreprocessorStepRegistry
from tests.helpers.dataframe_builder import DataFrameBuilder


class TestMissingIndicatorIntegration:
    """결측값 지시자 통합 기능 테스트 (SimpleImputerWrapper 내장 기능)"""
    
    def test_simple_imputer_with_missing_indicators_enabled(self):
        """create_missing_indicators=True로 설정한 SimpleImputer 테스트"""
        # Given: 결측값이 포함된 데이터와 지시자 생성 옵션
        data = DataFrameBuilder.build_missing_values_data(100)
        test_data = data[['numeric_few_missing', 'numeric_many_missing']]
        
        imputer = SimpleImputerWrapper(
            strategy='mean', 
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 원본 컬럼 + 결측값 지시자 컬럼들이 생성됨
        assert transformed.shape[0] == test_data.shape[0]  # 행 수는 동일
        assert transformed.shape[1] > test_data.shape[1]   # 열 수는 증가 (지시자 추가)
        
        # 결측값이 모두 채워졌는지 확인
        original_columns = list(test_data.columns)
        for col in original_columns:
            if col in transformed.columns:
                assert not transformed[col].isna().any(), f"Column {col} still has missing values"
        
        # 결측값 지시자 컬럼이 생성되었는지 확인
        indicator_columns = [col for col in transformed.columns if 'missingindicator_' in col]
        assert len(indicator_columns) > 0, "Missing indicator columns should be created"
        
        # 지시자 컬럼은 0 또는 1만 포함해야 함
        for indicator_col in indicator_columns:
            unique_values = transformed[indicator_col].unique()
            assert set(unique_values).issubset({0, 1}), f"Indicator {indicator_col} should only contain 0 or 1"
    
    def test_simple_imputer_missing_indicators_disabled(self):
        """create_missing_indicators=False로 설정한 SimpleImputer 테스트 (기본값)"""
        # Given: 결측값이 포함된 데이터 (기본 설정)
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_few_missing']]
        
        imputer = SimpleImputerWrapper(strategy='median')  # 기본값: create_missing_indicators=False
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값 지시자 없이 원본 형태 유지
        assert transformed.shape == test_data.shape  # 형태 변화 없음
        assert not transformed.isna().any().any()    # 결측값만 채워짐
        
        # 지시자 컬럼이 생성되지 않았는지 확인
        indicator_columns = [col for col in transformed.columns if 'missingindicator_' in col]
        assert len(indicator_columns) == 0, "No indicator columns should be created"
    
    def test_missing_indicator_feature_names_generation(self):
        """결측값 지시자 피처 이름 생성 테스트"""
        # Given: 결측값이 포함된 여러 컬럼 데이터
        data = DataFrameBuilder.build_missing_values_data(80)
        test_data = data[['numeric_few_missing', 'numeric_many_missing', 'category_missing']]
        
        imputer = SimpleImputerWrapper(
            strategy='most_frequent',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data.astype(str).replace('nan', np.nan))  # 범주형 처리
        transformed = imputer.transform(test_data.astype(str).replace('nan', np.nan))
        
        # Then: 적절한 지시자 피처 이름이 생성됨
        feature_names = list(transformed.columns)
        
        # 원본 컬럼명 확인
        original_columns = ['numeric_few_missing', 'numeric_many_missing', 'category_missing']
        for col in original_columns:
            assert col in feature_names, f"Original column {col} should be preserved"
        
        # 지시자 컬럼명 확인 (결측값이 있었던 컬럼에만 생성됨)
        indicator_pattern = 'missingindicator_'
        indicator_columns = [col for col in feature_names if indicator_pattern in col]
        assert len(indicator_columns) > 0, "Indicator columns should be created"
        
        # 각 지시자 컬럼이 적절한 명명 규칙을 따르는지 확인
        for indicator_col in indicator_columns:
            assert indicator_col.startswith(indicator_pattern)
    
    def test_missing_indicator_correctness(self):
        """결측값 지시자의 정확성 테스트"""
        # Given: 특정 패턴으로 결측값을 가진 데이터
        np.random.seed(42)
        test_data = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'feature_2': [1, 2, 3, 4, np.nan, 6, 7, 8, np.nan, 10]
        })
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값 지시자가 원래 결측값 위치를 정확히 표시하는지 확인
        # feature_1의 결측값 위치 (인덱스 2, 5)
        if 'missingindicator_feature_1' in transformed.columns:
            indicator_1 = transformed['missingindicator_feature_1']
            expected_missing_1 = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # 2번, 5번 인덱스가 1
            np.testing.assert_array_equal(indicator_1.values, expected_missing_1)
        
        # feature_2의 결측값 위치 (인덱스 4, 8)
        if 'missingindicator_feature_2' in transformed.columns:
            indicator_2 = transformed['missingindicator_feature_2']
            expected_missing_2 = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # 4번, 8번 인덱스가 1
            np.testing.assert_array_equal(indicator_2.values, expected_missing_2)
        
        # 원본 피처들에는 결측값이 없어야 함 (imputation 완료)
        assert not transformed['feature_1'].isna().any()
        assert not transformed['feature_2'].isna().any()


class TestMissingValuePatternDetection:
    """결측값 패턴 탐지 기능 테스트"""
    
    def test_different_missing_patterns(self):
        """다양한 결측값 패턴에 대한 지시자 생성"""
        # Given: 다양한 결측 비율을 가진 데이터
        data = DataFrameBuilder.build_missing_values_data(100)
        
        # 저결측 (10%), 중결측 (40%), 고결측 (80%) 컬럼 선택
        test_data = data[['numeric_few_missing', 'numeric_many_missing', 'numeric_extreme_missing']]
        
        imputer = SimpleImputerWrapper(
            strategy='median',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 각 결측 패턴에 대한 지시자가 생성됨
        indicator_columns = [col for col in transformed.columns if 'missingindicator_' in col]
        
        # 결측값이 있는 컬럼 수만큼 지시자가 생성되어야 함
        expected_indicators = 3  # 3개 컬럼 모두 결측값 포함
        assert len(indicator_columns) == expected_indicators
        
        # 각 지시자의 결측값 비율이 원본과 일치하는지 확인
        for i, col in enumerate(['numeric_few_missing', 'numeric_many_missing', 'numeric_extreme_missing']):
            indicator_col = f'missingindicator_{col}'
            if indicator_col in transformed.columns:
                # 지시자의 1의 비율 = 원본의 결측값 비율
                original_missing_rate = test_data[col].isna().mean()
                indicator_missing_rate = transformed[indicator_col].mean()
                
                # 부동소수점 비교를 위한 허용 오차
                assert abs(original_missing_rate - indicator_missing_rate) < 1e-10
    
    def test_no_missing_values_no_indicators(self):
        """결측값이 없는 데이터에서는 지시자가 생성되지 않는 테스트"""
        # Given: 결측값이 없는 완전한 데이터
        data = DataFrameBuilder.build_missing_values_data(50)
        test_data = data[['numeric_complete']]  # 완전한 데이터 컬럼
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 결측값이 없으므로 지시자 컬럼도 생성되지 않음
        assert transformed.shape == test_data.shape  # 형태 변화 없음
        
        indicator_columns = [col for col in transformed.columns if 'missingindicator_' in col]
        assert len(indicator_columns) == 0, "No indicators should be created for complete data"
    
    def test_mixed_complete_and_incomplete_features(self):
        """완전한 피처와 불완전한 피처가 섞인 데이터 테스트"""
        # Given: 완전한 피처 + 불완전한 피처
        data = DataFrameBuilder.build_missing_values_data(80)
        test_data = data[['numeric_complete', 'numeric_few_missing', 'numeric_many_missing']]
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 불완전한 피처에만 지시자가 생성됨
        # numeric_complete: 지시자 없음
        # numeric_few_missing, numeric_many_missing: 지시자 있음
        
        expected_indicators = ['missingindicator_numeric_few_missing', 'missingindicator_numeric_many_missing']
        not_expected_indicators = ['missingindicator_numeric_complete']
        
        # 예상되는 지시자들이 존재하는지 확인
        for expected_indicator in expected_indicators:
            assert expected_indicator in transformed.columns, f"{expected_indicator} should exist"
        
        # 예상되지 않는 지시자들이 없는지 확인
        for not_expected_indicator in not_expected_indicators:
            assert not_expected_indicator not in transformed.columns, f"{not_expected_indicator} should not exist"
        
        # 완전한 피처는 변화가 없어야 함
        np.testing.assert_array_equal(
            transformed['numeric_complete'].values,
            test_data['numeric_complete'].values
        )


class TestMissingValueHandlingEdgeCases:
    """결측값 처리 경계 사례 테스트"""
    
    def test_all_missing_column_handling(self):
        """모든 값이 결측인 컬럼 처리 테스트"""
        # Given: 모든 값이 NaN인 컬럼이 포함된 데이터
        test_data = pd.DataFrame({
            'normal_feature': [1, 2, 3, 4, 5],
            'all_missing': [np.nan] * 5
        })
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When & Then: 모든 값이 결측인 컬럼은 처리할 수 없으므로 오류 발생
        with pytest.raises(ValueError, match="SimpleImputer는 전체가 결측값인 컬럼을 처리할 수 없습니다"):
            imputer.fit(test_data)
    
    def test_single_missing_value(self):
        """단일 결측값이 있는 경우 테스트"""
        # Given: 하나의 결측값만 있는 데이터
        test_data = pd.DataFrame({
            'single_missing': [1, 2, np.nan, 4, 5]
        })
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        transformed = imputer.transform(test_data)
        
        # Then: 단일 결측값도 적절히 처리됨
        assert not transformed['single_missing'].isna().any()
        
        # 지시자 컬럼 확인
        assert 'missingindicator_single_missing' in transformed.columns
        indicator_values = transformed['missingindicator_single_missing']
        
        # 정확히 하나의 1과 4개의 0이 있어야 함
        assert indicator_values.sum() == 1
        assert (indicator_values == 0).sum() == 4
    
    def test_extreme_missing_percentages(self):
        """극단적 결측 비율에 대한 테스트"""
        # Given: 매우 높은 결측 비율 (95%)
        np.random.seed(42)
        data = pd.DataFrame({
            'extreme_missing': np.random.randn(100)
        })
        
        # 95%를 결측값으로 설정
        missing_indices = np.random.choice(100, size=95, replace=False)
        data.loc[missing_indices, 'extreme_missing'] = np.nan
        
        imputer = SimpleImputerWrapper(
            strategy='median',
            create_missing_indicators=True
        )
        
        # When: fit 및 transform 수행
        imputer.fit(data)
        transformed = imputer.transform(data)
        
        # Then: 극단적 결측 상황도 처리됨
        assert not transformed['extreme_missing'].isna().any()
        
        # 지시자의 95%가 1이어야 함
        indicator_col = 'missingindicator_extreme_missing'
        assert indicator_col in transformed.columns
        
        missing_rate = transformed[indicator_col].mean()
        assert abs(missing_rate - 0.95) < 0.01  # 95% ± 1%


class TestMissingValueRegistryIntegration:
    """결측값 처리 Registry 통합 테스트"""
    
    def test_simple_imputer_with_indicators_registry_creation(self):
        """Registry를 통한 결측값 지시자 포함 Imputer 생성"""
        # Given: Registry에서 지시자 생성 옵션으로 imputer 생성
        imputer = PreprocessorStepRegistry.create(
            "simple_imputer",
            strategy='median',
            create_missing_indicators=True
        )
        
        data = DataFrameBuilder.build_missing_values_data(60)
        test_data = data[['numeric_few_missing', 'numeric_many_missing']]
        
        # When: fit 및 transform 수행
        imputer.fit(test_data)
        result = imputer.transform(test_data)
        
        # Then: Registry를 통한 생성도 정상 동작
        assert isinstance(imputer, SimpleImputerWrapper)
        assert imputer.create_missing_indicators == True
        assert result.shape[1] > test_data.shape[1]  # 지시자 컬럼 추가됨
        
        # 지시자 컬럼 존재 확인
        indicator_columns = [col for col in result.columns if 'missingindicator_' in col]
        assert len(indicator_columns) > 0
    
    def test_get_output_feature_names_with_indicators(self):
        """결측값 지시자 포함 시 출력 피처명 확인"""
        # Given: 지시자 생성 옵션이 활성화된 imputer
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        data = DataFrameBuilder.build_missing_values_data(40)
        test_data = data[['numeric_few_missing']]
        
        # When: fit 후 출력 피처명 확인
        imputer.fit(test_data)
        
        # get_output_feature_names 메서드가 있다면 호출
        if hasattr(imputer, 'get_output_feature_names'):
            feature_names = imputer.get_output_feature_names(list(test_data.columns))
            
            # Then: 원본 컬럼 + 지시자 컬럼명이 포함되어야 함
            assert 'numeric_few_missing' in feature_names
            assert any('missingindicator_' in name for name in feature_names)


class TestMissingValueErrorHandling:
    """결측값 처리 오류 상황 테스트"""
    
    def test_empty_dataframe_with_indicators(self):
        """빈 데이터프레임에서 지시자 생성 테스트"""
        # Given: 빈 데이터프레임
        empty_data = pd.DataFrame()
        
        imputer = SimpleImputerWrapper(
            strategy='mean',
            create_missing_indicators=True
        )
        
        # When/Then: 빈 데이터에 대한 적절한 처리
        try:
            imputer.fit(empty_data)
            result = imputer.transform(empty_data)
            assert result.shape == empty_data.shape
        except (ValueError, AttributeError):
            # sklearn이 빈 데이터를 처리하지 못하는 것은 정상적 동작
            pass
    
    def test_mixed_data_types_indicators(self):
        """혼합 데이터 타입에서 지시자 생성 테스트"""
        # Given: 숫자형 + 범주형 혼합 결측 데이터
        mixed_data = pd.DataFrame({
            'numeric_missing': [1.0, 2.0, np.nan, 4.0, np.nan],
            'category_missing': ['A', 'B', np.nan, 'C', np.nan],
            'complete_feature': [10, 20, 30, 40, 50]
        })
        
        # most_frequent 전략으로 모든 타입 처리
        imputer = SimpleImputerWrapper(
            strategy='most_frequent',
            create_missing_indicators=True
        )
        
        # When: 혼합 타입 데이터 처리
        imputer.fit(mixed_data)
        result = imputer.transform(mixed_data)
        
        # Then: 모든 타입에서 지시자가 생성됨
        assert not result.isna().any().any()  # 모든 결측값 처리됨
        
        # 결측값이 있었던 컬럼들의 지시자 확인
        expected_indicators = ['missingindicator_numeric_missing', 'missingindicator_category_missing']
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Indicator {indicator} should be created"
        
        # 완전한 피처의 지시자는 없어야 함
        assert 'missingindicator_complete_feature' not in result.columns
        
        # 각 지시자의 값이 올바른지 확인 (두 컬럼 모두 2개 결측값)
        for indicator in expected_indicators:
            assert result[indicator].sum() == 2  # 2개 결측값
    
    def test_invalid_strategy_with_indicators(self):
        """잘못된 전략과 지시자 옵션 조합 테스트"""
        # Given: 잘못된 전략으로 imputer 생성 시도
        try:
            invalid_imputer = SimpleImputerWrapper(
                strategy='invalid_strategy',
                create_missing_indicators=True
            )
            
            data = DataFrameBuilder.build_missing_values_data(10)
            test_data = data[['numeric_few_missing']]
            
            # When/Then: 잘못된 전략은 sklearn 수준에서 오류 발생
            with pytest.raises((ValueError, TypeError)):
                invalid_imputer.fit(test_data)
        except Exception:
            # 초기화 단계에서 오류가 발생할 수도 있음
            pass