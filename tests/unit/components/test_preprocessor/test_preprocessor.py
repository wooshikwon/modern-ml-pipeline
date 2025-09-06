"""
메인 Preprocessor 클래스를 위한 종합적인 테스트 모듈

Recipe 기반 동적 파이프라인 빌더의 핵심 기능을 테스트:
- Settings 기반 초기화
- Recipe에서 파이프라인 생성  
- fit/transform 워크플로우
- 동적 step 구성
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
                    'task_type': 'classification',
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
    """Preprocessor 파이프라인 생성 테스트"""
    
    def test_pipeline_creation_from_recipe_steps(self):
        """Recipe 설정에서 파이프라인 생성 테스트"""
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
        assert fitted_preprocessor.pipeline is not None
        assert len(fitted_preprocessor.pipeline.steps) == 1  # preprocessor step
        assert fitted_preprocessor.pipeline.steps[0][0] == 'preprocessor'
        
        # ColumnTransformer 확인
        column_transformer = fitted_preprocessor.pipeline.steps[0][1]
        assert len(column_transformer.transformers) == 2  # scaler, imputer (steps 개수)
        
    def test_dynamic_step_configuration(self):
        """동적으로 다른 step 구성 테스트"""
        # Arrange - 스케일러만 포함
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe_config['preprocessor']['steps'] = [
            {
                'type': 'min_max_scaler',
                'columns': ['num_feature_1', 'num_feature_2']
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
        column_transformer = fitted_preprocessor.pipeline.steps[0][1]
        assert len(column_transformer.transformers) == 1  # scaler만
        # 동적 이름 생성되므로 min_max_scaler로 시작하는지 확인
        assert column_transformer.transformers[0][0].startswith('min_max_scaler_')


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
        # 변환 후 컬럼 수는 원본과 다를 수 있음 (OneHot 등으로 확장)
        assert transformed_data.shape[1] >= len(test_data.columns) - 1  # target 제외
        
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
        """transform 시 필요 컬럼이 없을 때 자동 생성 테스트"""
        # Arrange
        recipe_config = PreprocessorRecipeBuilder.build_preprocessor_recipe_config()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        full_data = DataFrameBuilder.build_mixed_preprocessor_data(50)
        incomplete_data = full_data.drop(columns=['num_feature_3'])  # 필요 컬럼 제거
        
        preprocessor = Preprocessor(settings)
        fitted_preprocessor = preprocessor.fit(full_data)
        
        # Act - missing column은 자동으로 0으로 생성되어야 함
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
        invalid_step.model_dump.return_value = {'strategy': None, 'degree': None, 'n_bins': None, 'sigma': None}
        
        settings.recipe.preprocessor.steps = [invalid_step]
        
        test_data = DataFrameBuilder.build_mixed_preprocessor_data(30)
        preprocessor = Preprocessor(settings)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown preprocessor step type"):
            preprocessor.fit(test_data)
            
    def test_empty_column_transforms_configuration(self):
        """빈 column_transforms 설정 처리 테스트"""
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
        
        # Assert - 변환 없이 passthrough만 동작
        assert isinstance(transformed_data, pd.DataFrame)
        # 빈 설정에서도 정상 동작해야 함


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