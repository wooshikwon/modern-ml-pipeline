"""
메인 Preprocessor 클래스를 위한 종합적인 테스트 모듈

DataFrame-First 순차적 전처리 아키텍처의 핵심 기능을 테스트:
- Settings 기반 초기화
- Recipe에서 전처리 단계 실행
- fit/transform 워크플로우
- 동적 step 구성
- Global vs Targeted 전처리 정책
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from src.components.preprocessor.preprocessor import Preprocessor
from src.components.preprocessor.registry import PreprocessorStepRegistry  
from src.interface import BasePreprocessor
from src.settings import Settings
from tests.helpers.config_builder import ConfigBuilder
from tests.helpers.recipe_builder import RecipeBuilder
from tests.helpers.dataframe_builder import DataFrameBuilder

# Preprocessor 모듈들을 import하여 Registry에 자동 등록되도록 함
import src.components.preprocessor.modules.scaler  # 스케일러 등록
import src.components.preprocessor.modules.imputer  # 임퓨터 등록

# 인코더는 category_encoders 의존성이 있어서 별도 테스트에서 처리
try:
    import src.components.preprocessor.modules.encoder  # 인코더 등록  
except ImportError:
    # category_encoders가 없는 경우 건너뜀
    pass


class PreprocessorRecipeBuilder:
    """Preprocessor Recipe 빌더 (RecipeBuilder 확장)"""
    
    @staticmethod  
    def build_preprocessor_recipe_config(**overrides) -> Dict[str, Any]:
        """Preprocessor 설정이 포함된 Recipe 구성"""
        base_config = {
            'name': 'test_preprocessor_recipe',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 100}
                }
            },
            'preprocessor': {
                'steps': [
                    {
                        'type': 'standard_scaler',
                        'columns': ['num_feature_1', 'num_feature_2', 'num_feature_3']
                    },
                    {
                        'type': 'simple_imputer',
                        'strategy': 'mean',
                        'columns': ['missing_feature']
                    }
                ]
            },
            'data': {
                'loader': {'source_uri': './data/test.csv'},
                'fetcher': {'type': 'pass_through'},
                'data_interface': {
                    'task_choice': 'classification',
                    'target_column': 'target',
                    'entity_columns': ['user_id']
                }
            },
            'evaluation': {
                'metrics': ['accuracy'],
                'validation': {'method': 'train_test_split', 'test_size': 0.2}
            }
        }
        
        # 오버라이드 적용
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                current = base_config
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = value
            else:
                base_config[key] = value
                
        return base_config


class TestPreprocessorInitialization:
    """Preprocessor 초기화 테스트"""
    
    def test_preprocessor_initialization_with_settings(self):
        """Settings를 사용한 정상적인 초기화 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # Act
        preprocessor = Preprocessor(settings)
        
        # Assert
        assert preprocessor.settings == settings
        assert preprocessor.config == settings.recipe.preprocessor  # recipe 루트의 preprocessor 참조
        assert preprocessor.pipeline is None  # fit 전에는 None
        
    def test_preprocessor_initialization_without_config(self):
        """preprocessor 설정이 없는 경우 (None) 초기화 테스트"""
        # Arrange
        recipe = RecipeBuilder.build()  # 기본 recipe (preprocessor=None)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # Act
        preprocessor = Preprocessor(settings)
        
        # Assert
        assert preprocessor.config is None  # Recipe.preprocessor는 Optional이므로 None 허용
        assert preprocessor.pipeline is None


class TestPreprocessorPipelineCreation:
    """Preprocessor 파이프라인 생성 테스트 (DataFrame-First 아키텍처)"""
    
    def test_pipeline_creation_from_recipe_steps(self):
        """Recipe 설정에서 DataFrame-First 전처리 실행 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(50)
        preprocessor = Preprocessor(settings)
        
        # Act
        fitted_preprocessor = preprocessor.fit(test_data)
        
        # Assert
        # DataFrame-First 아키텍처에서는 sklearn 호환성을 위한 identity pipeline만 존재
        assert fitted_preprocessor.pipeline is not None
        assert len(fitted_preprocessor.pipeline.steps) == 1  # identity step만
        assert fitted_preprocessor.pipeline.steps[0][0] == 'identity'
        
        # 실제 전처리는 _fitted_transformers에서 관리
        assert hasattr(fitted_preprocessor, '_fitted_transformers')
        assert len(fitted_preprocessor._fitted_transformers) == 2  # scaler, imputer
        
        # 각 transformer 정보 확인
        transformer_types = [t['step_type'] for t in fitted_preprocessor._fitted_transformers]
        assert 'standard_scaler' in transformer_types
        assert 'simple_imputer' in transformer_types
        
    def test_dynamic_step_configuration(self):
        """동적으로 다른 step 구성 테스트 - Global 전처리기 특성 고려"""
        # Arrange - MinMaxScaler는 Global 전처리기이므로 모든 수치형 컬럼에 적용됨
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe_config['preprocessor']['steps'] = [
            {
                'type': 'min_max_scaler',
                # columns를 지정해도 Global 전처리기는 모든 수치형 컬럼에 적용
            }
        ]
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()  
        settings = Settings(config=config, recipe=recipe)
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(30)
        preprocessor = Preprocessor(settings)
        
        # Act
        fitted_preprocessor = preprocessor.fit(test_data)
        
        # Assert
        # DataFrame-First 아키텍처에서는 _fitted_transformers로 확인
        assert len(fitted_preprocessor._fitted_transformers) == 1  # min_max_scaler만
        assert fitted_preprocessor._fitted_transformers[0]['step_type'] == 'min_max_scaler'
        
        # MinMaxScaler는 Global 전처리기이므로 모든 수치형 컬럼에 적용됨
        target_columns = fitted_preprocessor._fitted_transformers[0]['target_columns']
        assert len(target_columns) >= 3  # 최소 3개의 수치형 컬럼
        assert 'num_feature_1' in target_columns
        assert 'num_feature_2' in target_columns
        assert 'num_feature_3' in target_columns


class TestPreprocessorFitTransform:
    """Preprocessor fit/transform 워크플로우 테스트"""
    
    def test_fit_transform_pipeline_execution(self):
        """전체 fit -> transform 파이프라인 실행 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        train_data = DataFrameBuilder.build_mixed_preprocessor_data(80)
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(20)
        
        preprocessor = Preprocessor(settings)
        
        # Act
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_data = fitted_preprocessor.transform(test_data)
        
        # Assert
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(test_data)
        
        # DataFrame-First에서는 컬럼 구조가 유지되거나 변경될 수 있음
        # 최소한 숫자형 컬럼들은 존재해야 함
        assert transformed_data.shape[1] > 0
        
        # 결측값이 처리되었는지 확인 (imputer 적용)
        if 'missing_feature' in transformed_data.columns:
            assert not transformed_data['missing_feature'].isnull().any()
        
    def test_transform_before_fit_raises_error(self):
        """fit 전에 transform 호출 시 에러 발생 테스트"""
        # Arrange  
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(20)
        preprocessor = Preprocessor(settings)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="아직 학습되지 않았습니다"):
            preprocessor.transform(test_data)
            
    def test_transform_with_missing_columns_handled(self):
        """transform 시 필요 컬럼이 없을 때 자동 생성 테스트 (Targeted 전처리기용)"""
        # Arrange - SimpleImputer는 Targeted이므로 컬럼 누락 시 0으로 생성
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe_config['preprocessor']['steps'] = [
            {
                'type': 'simple_imputer',
                'strategy': 'mean',
                'columns': ['missing_column_not_in_test']  # 테스트 데이터에 없는 컬럼
            }
        ]
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # fit용 데이터는 해당 컬럼을 포함, transform용 데이터는 제외
        full_data = DataFrameBuilder.build_mixed_preprocessor_data(50)
        full_data['missing_column_not_in_test'] = np.random.randn(50)
        
        incomplete_data = DataFrameBuilder.build_mixed_preprocessor_data(20)  # 해당 컬럼 없음
        
        preprocessor = Preprocessor(settings)
        fitted_preprocessor = preprocessor.fit(full_data)
        
        # Act - missing column은 자동으로 0으로 생성되어야 함 (Targeted 전처리기 특성)
        transformed_data = fitted_preprocessor.transform(incomplete_data)
        
        # Assert
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(incomplete_data)


class TestPreprocessorErrorHandling:
    """Preprocessor 에러 처리 테스트"""
    
    def test_invalid_step_type_raises_error(self):
        """존재하지 않는 step type 사용 시 에러 테스트"""
        # Arrange - Mock을 사용하여 Recipe validation을 우회
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # Mock으로 잘못된 step을 주입
        from src.settings.recipe import PreprocessorStep
        invalid_step = Mock(spec=PreprocessorStep)
        invalid_step.type = 'nonexistent_transformer'
        invalid_step.columns = ['num_feature_1']
        invalid_step.model_dump.return_value = {'strategy': None, 'degree': None, 'n_bins': None, 'sigma': None, 'create_missing_indicators': None}
        
        settings.recipe.preprocessor.steps = [invalid_step]
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(30)
        preprocessor = Preprocessor(settings)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown preprocessor step type"):
            preprocessor.fit(test_data)
            
    def test_empty_column_transforms_configuration(self):
        """빈 steps 설정 처리 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe_config['preprocessor']['steps'] = []  # 빈 steps 리스트
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(20)
        preprocessor = Preprocessor(settings)
        
        # Act
        fitted_preprocessor = preprocessor.fit(test_data)
        transformed_data = fitted_preprocessor.transform(test_data)
        
        # Assert - 빈 설정에서는 변환 없이 원본 데이터 반환
        assert isinstance(transformed_data, pd.DataFrame)
        assert len(fitted_preprocessor._fitted_transformers) == 0  # 변환기 없음
        pd.testing.assert_frame_equal(transformed_data, test_data)  # 원본과 동일


class TestPreprocessorIntegration:
    """Preprocessor 통합 시나리오 테스트"""
    
    def test_complete_preprocessing_workflow_classification(self):
        """분류 태스크 전체 전처리 워크플로우 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 현실적인 크기의 데이터
        train_data = DataFrameBuilder.build_mixed_preprocessor_data(200)
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(50)
        
        preprocessor = Preprocessor(settings)
        
        # Act - 전체 워크플로우 실행
        fitted_preprocessor = preprocessor.fit(train_data)
        train_transformed = fitted_preprocessor.transform(train_data)
        test_transformed = fitted_preprocessor.transform(test_data)
        
        # Assert
        assert isinstance(train_transformed, pd.DataFrame)
        assert isinstance(test_transformed, pd.DataFrame) 
        assert len(train_transformed) == len(train_data)
        assert len(test_transformed) == len(test_data)
        
        # 데이터 품질 확인
        assert not train_transformed.isnull().any().any()  # 결측값 없음
        assert not test_transformed.isnull().any().any()   # 결측값 없음
        
        # DataFrame-First에서는 _fitted_transformers로 전처리 단계 확인
        assert len(fitted_preprocessor._fitted_transformers) == 2  # scaler, imputer
        
    def test_global_vs_targeted_preprocessing(self):
        """Global과 Targeted 전처리 정책 테스트"""
        # Arrange - Global(StandardScaler)과 Targeted(SimpleImputer) 혼합
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe_config['preprocessor']['steps'] = [
            {
                'type': 'standard_scaler',  # Global 전처리기
                # columns 없음 - Global은 모든 수치형 컬럼에 적용
            },
            {
                'type': 'simple_imputer',   # Targeted 전처리기
                'strategy': 'mean',
                'columns': ['missing_feature']  # 특정 컬럼에만 적용
            }
        ]
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(100)
        preprocessor = Preprocessor(settings)
        
        # Act
        fitted_preprocessor = preprocessor.fit(test_data)
        transformed_data = fitted_preprocessor.transform(test_data)
        
        # Assert
        assert len(fitted_preprocessor._fitted_transformers) == 2
        
        # Global 전처리기 확인 (StandardScaler)
        global_transformer = fitted_preprocessor._fitted_transformers[0]
        assert global_transformer['step_type'] == 'standard_scaler'
        # Global은 모든 수치형 컬럼에 적용되므로 target_columns가 여러 개
        assert len(global_transformer['target_columns']) >= 3
        
        # Targeted 전처리기 확인 (SimpleImputer)  
        targeted_transformer = fitted_preprocessor._fitted_transformers[1]
        assert targeted_transformer['step_type'] == 'simple_imputer'
        assert targeted_transformer['target_columns'] == ['missing_feature']
        
        # 변환 결과 검증
        assert isinstance(transformed_data, pd.DataFrame)
        assert not transformed_data.isnull().any().any()  # 모든 결측값 처리됨