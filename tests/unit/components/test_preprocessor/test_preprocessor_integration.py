"""
Preprocessor 통합 테스트 (Phase 3: 새로운 모듈들)

새로 구현된 preprocessor 모듈들의 통합 테스트:
- feature_generator (TreeBased, Polynomial)
- discretizer (KBinsDiscretizer)  
- missing (MissingIndicator)
- 기존 모듈들과의 조합 테스트
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from typing import Dict, Any

from src.components.preprocessor.preprocessor import Preprocessor
from src.components.preprocessor.registry import PreprocessorStepRegistry
from src.settings import Settings
from tests.helpers.config_builder import ConfigBuilder
from tests.helpers.recipe_builder import RecipeBuilder
from tests.helpers.dataframe_builder import DataFrameBuilder

# 새로운 모듈들을 import하여 Registry에 등록
import src.components.preprocessor.modules.scaler
import src.components.preprocessor.modules.imputer
import src.components.preprocessor.modules.feature_generator
import src.components.preprocessor.modules.discretizer
import src.components.preprocessor.modules.missing

# 인코더는 선택적 import
try:
    import src.components.preprocessor.modules.encoder
except ImportError:
    pass


class IntegratedPreprocessorRecipeBuilder:
    """통합 테스트용 Recipe 빌더"""
    
    @staticmethod
    def build_feature_pipeline_recipe(**overrides) -> Dict[str, Any]:
        """피처 생성 + 전처리 파이프라인 Recipe"""
        base_config = {
            'name': 'feature_pipeline_test',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 50}
                }
            },
            'preprocessor': {
                'steps': [
                    # 1. 결측값 지시자 생성
                    {
                        'type': 'missing_indicator',
                        'columns': ['feature_1', 'feature_2', 'feature_3']
                    },
                    # 2. 결측값 대체
                    {
                        'type': 'simple_imputer',
                        'strategy': 'mean',
                        'columns': ['feature_1', 'feature_2', 'feature_3']
                    },
                    # 3. Polynomial 피처 생성 추가
                    {
                        'type': 'polynomial_features',
                        'columns': ['feature_1', 'feature_2'],
                        'degree': 2
                    },
                    # 4. Polynomial 피처 생성
                    {
                        'type': 'polynomial_features',
                        'columns': ['feature_1', 'feature_2'],
                        'degree': 2,
                        'interaction_only': True
                    },
                    # 5. 연속형 변수 구간화
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['feature_3'],
                        'n_bins': 4
                    },
                    # 6. 최종 스케일링
                    {
                        'type': 'standard_scaler',
                        'columns': ['feature_1', 'feature_2', 'feature_3']
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
    
    @staticmethod
    def build_discretization_pipeline_recipe(**overrides) -> Dict[str, Any]:
        """구간화 중심 파이프라인 Recipe"""
        base_config = {
            'name': 'discretization_pipeline_test',
            'model': {
                'class_path': 'sklearn.tree.DecisionTreeClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'max_depth': 5}
                }
            },
            'preprocessor': {
                'steps': [
                    # 1. 다양한 전략의 구간화
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['uniform_dist'],
                        'n_bins': 5
                    },
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['normal_dist'],
                        'n_bins': 4
                    },
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['exponential_dist'],
                        'n_bins': 3,
                        'encode': 'ordinal',
                        'strategy': 'kmeans'
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


class TestFeatureGenerationIntegration:
    """피처 생성 통합 테스트"""
    
    def test_polynomial_with_discretization_features(self):
        """Polynomial + Discretization 피처 생성 통합 테스트 (unsupervised)"""
        # Given: Polynomial과 Discretization를 함께 사용하는 설정
        recipe_config = {
            'name': 'poly_discrete_test',
            'model': {
                'class_path': 'sklearn.ensemble.RandomForestClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 10}
                }
            },
            'preprocessor': {
                'steps': [
                    {
                        'type': 'polynomial_features',
                        'columns': ['feature_1', 'feature_2'],
                        'degree': 2
                    },
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['feature_3'],
                        'n_bins': 4
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 피처 생성에 적합한 데이터
        train_data = DataFrameBuilder.build_feature_generation_data(100)
        test_data = DataFrameBuilder.build_feature_generation_data(50)
        
        preprocessor = Preprocessor(settings)
        
        # When: fit 및 transform 수행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 피처가 증가했는지 확인
        assert transformed_train.shape[1] > train_data.shape[1] - 1  # target 제외
        assert transformed_test.shape[1] == transformed_train.shape[1]
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        
        # 모든 변환이 성공적으로 적용됨
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
    
    def test_feature_generation_with_missing_values_handling(self):
        """피처 생성 + 결측값 처리 통합 테스트"""
        # Given: 결측값이 있는 데이터에 대한 피처 생성
        recipe_config = {
            'name': 'feature_missing_test',
            'model': {
                'class_path': 'sklearn.linear_model.LogisticRegression',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'max_iter': 100}
                }
            },
            'preprocessor': {
                'steps': [
                    # 1. 결측값 지시자 생성
                    {
                        'type': 'missing_indicator',
                        'columns': ['numeric_few_missing', 'numeric_many_missing']
                    },
                    # 2. 결측값 대체
                    {
                        'type': 'simple_imputer',
                        'strategy': 'median',
                        'columns': ['numeric_few_missing', 'numeric_many_missing']
                    },
                    # 3. Polynomial 피처 생성 (unsupervised)
                    {
                        'type': 'polynomial_features',
                        'columns': ['numeric_few_missing', 'numeric_many_missing'],
                        'degree': 2,
                        'interaction_only': True
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 결측값이 있는 데이터
        train_data = DataFrameBuilder.build_missing_values_data(80)
        # target 컬럼 추가
        train_data['target'] = np.random.choice([0, 1], len(train_data))
        
        test_data = DataFrameBuilder.build_missing_values_data(30)
        test_data['target'] = np.random.choice([0, 1], len(test_data))
        
        preprocessor = Preprocessor(settings)
        
        # When: fit 및 transform 수행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 결측값이 완전히 처리되고 피처가 생성됨
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
        
        # Missing indicator + imputed values + polynomial features
        assert transformed_train.shape[1] > train_data.shape[1] - 1  # target 제외
        assert transformed_test.shape[1] == transformed_train.shape[1]


class TestDiscretizationIntegration:
    """구간화 통합 테스트"""
    
    def test_multiple_discretization_strategies(self):
        """여러 구간화 전략 통합 테스트"""
        # Given: 여러 전략의 구간화
        recipe_config = IntegratedPreprocessorRecipeBuilder.build_discretization_pipeline_recipe()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 구간화용 데이터
        train_data = DataFrameBuilder.build_discretization_data(120)
        # target 추가
        train_data['target'] = np.random.choice([0, 1, 2], len(train_data))
        
        test_data = DataFrameBuilder.build_discretization_data(40)
        test_data['target'] = np.random.choice([0, 1, 2], len(test_data))
        
        preprocessor = Preprocessor(settings)
        
        # When: fit 및 transform 수행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 모든 구간화 적용됨
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        
        # onehot encoding으로 인해 컬럼 수 증가
        assert transformed_train.shape[1] > train_data.shape[1] - 1  # target 제외
        assert transformed_test.shape[1] == transformed_train.shape[1]
        
        # 모든 값이 수치형이어야 함
        assert transformed_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        assert transformed_test.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_discretization_with_scaling(self):
        """구간화 + 스케일링 통합 테스트"""
        # Given: 구간화 후 스케일링 적용
        recipe_config = {
            'name': 'discretize_scale_test',
            'model': {
                'class_path': 'sklearn.svm.SVC',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'kernel': 'linear'}
                }
            },
            'preprocessor': {
                'steps': [
                    # 1. 구간화
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['uniform_dist', 'normal_dist'],
                        'n_bins': 5
                    },
                    # 2. 스케일링
                    {
                        'type': 'standard_scaler',
                        'columns': ['uniform_dist', 'normal_dist']
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        train_data = DataFrameBuilder.build_discretization_data(100)
        train_data['target'] = np.random.choice([0, 1], len(train_data))
        
        test_data = DataFrameBuilder.build_discretization_data(30)
        test_data['target'] = np.random.choice([0, 1], len(test_data))
        
        preprocessor = Preprocessor(settings)
        
        # When: fit 및 transform 수행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 구간화된 값들이 스케일링됨
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        
        # 스케일링 후 평균이 0 근처인지 확인 (구간화된 값이므로 정확히 0은 아닐 수 있음)
        mean_values = transformed_train.mean()
        assert all(abs(mean) < 1.0 for mean in mean_values)  # 대략적인 확인


class TestComplexIntegratedPipeline:
    """복합 통합 파이프라인 테스트"""
    
    def test_full_feature_pipeline_integration(self):
        """전체 피처 파이프라인 통합 테스트"""
        # Given: 모든 새로운 모듈을 사용하는 복합 파이프라인
        recipe_config = IntegratedPreprocessorRecipeBuilder.build_feature_pipeline_recipe()
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 복합 데이터 생성 (결측값 + 피처 생성용 + 구간화용)
        base_data = DataFrameBuilder.build_feature_generation_data(150)
        # 일부 결측값 추가
        missing_mask = np.random.random(len(base_data)) > 0.85
        base_data.loc[missing_mask, 'feature_1'] = np.nan
        
        train_data = base_data.copy()
        
        test_data = DataFrameBuilder.build_feature_generation_data(50)
        test_missing_mask = np.random.random(len(test_data)) > 0.85
        test_data.loc[test_missing_mask, 'feature_1'] = np.nan
        
        preprocessor = Preprocessor(settings)
        
        # When: 전체 파이프라인 실행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 모든 단계가 성공적으로 실행됨
        assert isinstance(transformed_train, pd.DataFrame)
        assert isinstance(transformed_test, pd.DataFrame)
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        
        # 결측값 완전 처리
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
        
        # 피처 수 증가 (missing indicator + tree features + polynomial features + discretization)
        assert transformed_train.shape[1] > train_data.shape[1] - 1  # target 제외
        assert transformed_test.shape[1] == transformed_train.shape[1]
        
        # 모든 값이 수치형
        assert transformed_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        assert transformed_test.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_pipeline_order_dependency(self):
        """파이프라인 단계 순서 의존성 테스트"""
        # Given: 순서가 중요한 파이프라인 (결측값 처리 → 피처 생성 → 구간화 → 스케일링)
        recipe_config = {
            'name': 'order_test',
            'model': {
                'class_path': 'sklearn.ensemble.GradientBoostingClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'n_estimators': 10}
                }
            },
            'preprocessor': {
                'steps': [
                    # 1. 결측값 처리 (필수 먼저)
                    {
                        'type': 'simple_imputer',
                        'strategy': 'mean',
                        'columns': ['feature_1', 'feature_2']
                    },
                    # 2. Polynomial 피처 생성 (unsupervised)
                    {
                        'type': 'polynomial_features',
                        'columns': ['feature_1', 'feature_2'],
                        'degree': 2
                    },
                    # 3. 구간화 (연속값을 이산값으로)
                    {
                        'type': 'kbins_discretizer',
                        'columns': ['feature_1'],
                        'n_bins': 3
                    },
                    # 4. 스케일링 (마지막)
                    {
                        'type': 'min_max_scaler',
                        'columns': ['feature_1', 'feature_2']
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 결측값이 있는 데이터
        train_data = DataFrameBuilder.build_feature_generation_data(100)
        # 의도적으로 결측값 추가
        train_data.loc[:10, 'feature_1'] = np.nan
        train_data.loc[5:15, 'feature_2'] = np.nan
        
        test_data = DataFrameBuilder.build_feature_generation_data(30)
        test_data.loc[:3, 'feature_1'] = np.nan
        
        preprocessor = Preprocessor(settings)
        
        # When: 순서대로 파이프라인 실행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 모든 단계가 순서대로 성공적으로 실행됨
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        
        # 결측값 완전 제거
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
        
        # 최종 스케일링 결과 확인 (MinMaxScaler: 0~1 범위)
        for col in transformed_train.columns:
            if col in ['feature_1', 'feature_2']:  # 스케일링 대상 컬럼
                assert transformed_train[col].min() >= -0.1  # 약간의 오차 허용
                assert transformed_train[col].max() <= 1.1   # 약간의 오차 허용


class TestRobustnessAndEdgeCases:
    """견고성 및 경계 사례 테스트"""
    
    def test_small_dataset_integration(self):
        """작은 데이터셋에 대한 통합 테스트"""
        # Given: 매우 작은 데이터셋 (피처 수보다 샘플 수가 적음)
        recipe_config = {
            'name': 'small_data_test',
            'model': {
                'class_path': 'sklearn.naive_bayes.GaussianNB',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {}
                }
            },
            'preprocessor': {
                'steps': [
                    {
                        'type': 'polynomial_features',
                        'columns': ['feature_1', 'feature_2'],
                        'degree': 2,
                        'interaction_only': True
                    },
                    {
                        'type': 'standard_scaler',
                        'columns': ['feature_1', 'feature_2']
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 매우 작은 데이터셋
        train_data = DataFrameBuilder.build_feature_generation_data(10)
        test_data = DataFrameBuilder.build_feature_generation_data(5)
        
        preprocessor = Preprocessor(settings)
        
        # When: 작은 데이터로 파이프라인 실행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 작은 데이터에서도 정상 동작
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
    
    def test_all_missing_values_handling(self):
        """모든 값이 결측인 경우 처리 테스트"""
        # Given: 특정 컬럼이 모두 결측값인 데이터
        recipe_config = {
            'name': 'all_missing_test',
            'model': {
                'class_path': 'sklearn.dummy.DummyClassifier',
                'library': 'sklearn',
                'hyperparameters': {
                    'tuning_enabled': False,
                    'values': {'strategy': 'most_frequent'}
                }
            },
            'preprocessor': {
                'steps': [
                    {
                        'type': 'missing_indicator',
                        'columns': ['all_missing', 'partial_missing']
                    },
                    {
                        'type': 'simple_imputer',
                        'strategy': 'constant',
                        'columns': ['all_missing', 'partial_missing']
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
        
        recipe = RecipeBuilder.build(**recipe_config)
        config = ConfigBuilder.build()
        settings = Settings(config=config, recipe=recipe)
        
        # 모든 값이 결측인 컬럼을 포함한 데이터
        train_data = pd.DataFrame({
            'all_missing': [np.nan] * 50,
            'partial_missing': [1, 2, np.nan, 4, 5] * 10,
            'target': np.random.choice([0, 1], 50)
        })
        
        test_data = pd.DataFrame({
            'all_missing': [np.nan] * 20,
            'partial_missing': [1, np.nan, 3, 4, np.nan] * 4,
            'target': np.random.choice([0, 1], 20)
        })
        
        preprocessor = Preprocessor(settings)
        
        # When: 모든 결측값 컬럼이 포함된 파이프라인 실행
        fitted_preprocessor = preprocessor.fit(train_data)
        transformed_train = fitted_preprocessor.transform(train_data)
        transformed_test = fitted_preprocessor.transform(test_data)
        
        # Then: 모든 결측값도 적절히 처리됨
        assert len(transformed_train) == len(train_data)
        assert len(transformed_test) == len(test_data)
        assert not transformed_train.isnull().any().any()
        assert not transformed_test.isnull().any().any()
        
        # Missing indicator가 적절히 생성됨
        assert transformed_train.shape[1] > train_data.shape[1] - 1  # target 제외