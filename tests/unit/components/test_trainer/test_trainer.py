"""
test_trainer.py - Phase 3 Day 3-4 개발
메인 Trainer 클래스의 완전한 테스트 구현

핵심 테스트 케이스:
1. Factory Provider 패턴 테스트
2. Optuna 활성화 시 전체 훈련 플로우  
3. 고정 하이퍼파라미터 훈련 플로우
4. Task별 워크플로우 (classification/regression/clustering/causal)
5. Data Leakage 방지를 위한 3단계 분할
6. Task별 모델 fitting 패턴 (causal의 treatment 파라미터 등)
7. 훈련 방법론 메타데이터 생성
8. 컴포넌트 간 오케스트레이션 에러 처리
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

from src.components.trainer.trainer import Trainer
from src.settings import Settings
from src.interface import BaseModel
from tests.helpers.config_builder import SettingsBuilder
from tests.helpers.trainer_data_builder import TrainerDataBuilder


class TestTrainerInitialization:
    """Trainer 초기화 및 Factory Provider 패턴 테스트"""
    
    def test_trainer_initialization_with_factory_provider(self):
        """Factory Provider 패턴이 올바르게 주입되는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        mock_factory_provider = Mock()
        
        # Act
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        # Assert
        assert trainer.settings == settings
        assert trainer.factory_provider == mock_factory_provider
        assert trainer.training_results == {}
        
    def test_trainer_initialization_without_factory_provider(self):
        """Factory Provider가 없이 초기화되는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        
        # Act
        trainer = Trainer(settings)
        
        # Assert
        assert trainer.settings == settings
        assert trainer.factory_provider is None
        
    def test_get_factory_without_provider_raises_error(self):
        """Factory Provider가 주입되지 않았을 때 에러가 발생하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        trainer = Trainer(settings)
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Factory provider가 주입되지 않았습니다"):
            trainer._get_factory()
            
    def test_get_factory_with_provider_returns_factory(self):
        """Factory Provider가 올바르게 factory를 반환하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        # Act
        result = trainer._get_factory()
        
        # Assert
        assert result == mock_factory
        mock_factory_provider.assert_called_once()


class TestTrainerTuningWorkflows:
    """Optuna 튜닝 활성화/비활성화에 따른 전체 워크플로우 테스트"""
    
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    @patch('src.components.trainer.modules.trainer.OptunaOptimizer')
    def test_train_with_optuna_enabled(self, mock_optimizer_class, mock_prepare_data, mock_split_data):
        """Optuna 활성화 시 전체 훈련 플로우 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        
        # Mock 데이터 준비
        df = TrainerDataBuilder.build_classification_data()
        train_df = TrainerDataBuilder.build_classification_data(n_samples=80)
        test_df = TrainerDataBuilder.build_classification_data(n_samples=20)
        
        mock_split_data.return_value = (train_df, test_df)
        mock_prepare_data.side_effect = [
            (train_df.drop(['target', 'user_id'], axis=1), train_df['target'], None),
            (test_df.drop(['target', 'user_id'], axis=1), test_df['target'], None)
        ]
        
        # Mock 컴포넌트들
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x  # 변환 없이 반환
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'accuracy': 0.85, 'f1': 0.82}
        
        # Mock OptunaOptimizer
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            'enabled': True,
            'engine': 'optuna',
            'optimization_metric': 'accuracy',
            'best_params': {'n_estimators': 100},
            'best_score': 0.87,
            'model': mock_model
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock factory
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        # Act
        trained_model, fitted_preprocessor, metrics, results = trainer.train(
            df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator
        )
        
        # Assert
        # 1. OptunaOptimizer가 생성되었는지 확인
        mock_optimizer_class.assert_called_once_with(
            settings=settings, 
            factory_provider=trainer._get_factory
        )
        
        # 2. 최적화가 실행되었는지 확인
        mock_optimizer.optimize.assert_called_once_with(train_df, trainer._single_training_iteration)
        
        # 3. 결과 검증
        assert trained_model == mock_model
        assert fitted_preprocessor == mock_preprocessor
        assert metrics == {'accuracy': 0.85, 'f1': 0.82}
        
        # 4. training_results 검증
        assert 'hyperparameter_optimization' in results
        assert 'evaluation_metrics' in results
        assert 'training_methodology' in results
        
        assert results['hyperparameter_optimization']['enabled'] == True
        assert results['hyperparameter_optimization']['best_score'] == 0.87
        
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_train_with_fixed_hyperparameters(self, mock_prepare_data, mock_split_data):
        """고정 하이퍼파라미터로 훈련하는 플로우 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        
        # Mock 데이터 준비
        df = TrainerDataBuilder.build_classification_data()
        train_df = TrainerDataBuilder.build_classification_data(n_samples=80)
        test_df = TrainerDataBuilder.build_classification_data(n_samples=20)
        
        mock_split_data.return_value = (train_df, test_df)
        mock_prepare_data.side_effect = [
            (train_df.drop(['target', 'user_id'], axis=1), train_df['target'], None),
            (test_df.drop(['target', 'user_id'], axis=1), test_df['target'], None)
        ]
        
        # Mock 컴포넌트들
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'accuracy': 0.80}
        
        # Mock factory
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        # Act
        trained_model, fitted_preprocessor, metrics, results = trainer.train(
            df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator
        )
        
        # Assert
        # 1. 모델이 직접 학습되었는지 확인
        mock_model.fit.assert_called_once()
        
        # 2. 결과 검증
        assert trained_model == mock_model
        assert fitted_preprocessor == mock_preprocessor
        assert metrics == {'accuracy': 0.80}
        
        # 3. training_results에서 Optuna가 비활성화되었는지 확인
        assert results['hyperparameter_optimization']['enabled'] == False


class TestTrainerTaskSpecificWorkflows:
    """Task별 (classification/regression/clustering/causal) 워크플로우 테스트"""
    
    @pytest.mark.parametrize("task_choice,settings_builder", [
        ("classification", "build_classification_config"),
        ("regression", "build_regression_config"),
        ("clustering", "build_clustering_config"),
        ("causal", "build_causal_config")
    ])
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_train_task_specific_workflows(self, mock_prepare_data, mock_split_data, task_choice, settings_builder):
        """Task별 워크플로우가 올바르게 동작하는지 테스트"""
        # Arrange
        settings = getattr(SettingsBuilder, settings_builder)()
        
        # Task별 데이터 준비
        if task_choice == "classification":
            df = TrainerDataBuilder.build_classification_data()
            train_df = TrainerDataBuilder.build_classification_data(n_samples=80)
            test_df = TrainerDataBuilder.build_classification_data(n_samples=20)
            target_col = 'target'
        elif task_choice == "regression":
            df = TrainerDataBuilder.build_regression_data()
            train_df = TrainerDataBuilder.build_regression_data(n_samples=80)
            test_df = TrainerDataBuilder.build_regression_data(n_samples=20)
            target_col = 'target'
        elif task_choice == "clustering":
            df = TrainerDataBuilder.build_clustering_data()
            train_df = TrainerDataBuilder.build_clustering_data(n_samples=80)
            test_df = TrainerDataBuilder.build_clustering_data(n_samples=20)
            target_col = None  # clustering에는 target이 없음
        elif task_choice == "causal":
            df = TrainerDataBuilder.build_causal_data()
            train_df = TrainerDataBuilder.build_causal_data(n_samples=80)
            test_df = TrainerDataBuilder.build_causal_data(n_samples=20)
            target_col = 'outcome'
        
        mock_split_data.return_value = (train_df, test_df)
        
        # prepare_training_data 모킹
        if target_col:
            feature_cols = [col for col in train_df.columns if col not in [target_col, 'user_id', 'treatment']]
            X_train, y_train = train_df[feature_cols], train_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col]
        else:
            feature_cols = [col for col in train_df.columns if col not in ['user_id']]
            X_train, y_train = train_df[feature_cols], None
            X_test, y_test = test_df[feature_cols], None
        
        additional_data = {'treatment': train_df.get('treatment')} if task_choice == "causal" else None
        
        mock_prepare_data.side_effect = [
            (X_train, y_train, additional_data),
            (X_test, y_test, None)
        ]
        
        # Mock 컴포넌트들
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        mock_evaluator = Mock()
        
        # Task별 평가 메트릭 설정
        if task_choice == "classification":
            metrics = {'accuracy': 0.85, 'f1': 0.82}
        elif task_choice == "regression":
            metrics = {'mse': 0.15, 'r2': 0.85}
        elif task_choice == "clustering":
            metrics = {'silhouette_score': 0.7}
        elif task_choice == "causal":
            metrics = {'ate': 0.12, 'confidence_intervals': [0.08, 0.16]}
        
        mock_evaluator.evaluate.return_value = metrics
        
        # Mock factory
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        # Act
        trained_model, fitted_preprocessor, eval_metrics, results = trainer.train(
            df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator
        )
        
        # Assert
        # 1. 모델이 올바르게 학습되었는지 확인
        mock_model.fit.assert_called_once()
        
        # 2. Task별 특성 확인
        assert eval_metrics == metrics
        assert results['evaluation_metrics'] == metrics
        
        # 3. 훈련 방법론 메타데이터 확인
        methodology = results['training_methodology']
        assert methodology['train_test_split_method'] == 'stratified'
        assert methodology['train_ratio'] == 0.8


class TestTrainerDataLeakagePrevention:
    """Data Leakage 방지를 위한 3단계 분할 테스트"""
    
    @patch('src.components.trainer.modules.trainer.train_test_split')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_single_training_iteration_data_leakage_prevention(self, mock_prepare_data, mock_train_test_split):
        """Optuna 튜닝 시 Data Leakage 방지를 위한 3단계 분할 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()
        trainer = Trainer(settings)
        
        # 입력 데이터 준비
        train_df = TrainerDataBuilder.build_classification_data(n_samples=100)
        inner_train_df = TrainerDataBuilder.build_classification_data(n_samples=80)
        val_df = TrainerDataBuilder.build_classification_data(n_samples=20)
        
        # train_test_split 모킹 (Train을 다시 Train/Val로 분할)
        mock_train_test_split.return_value = (inner_train_df, val_df)
        
        # prepare_training_data 모킹
        X_train = inner_train_df.drop(['target', 'user_id'], axis=1)
        y_train = inner_train_df['target']
        X_val = val_df.drop(['target', 'user_id'], axis=1)
        y_val = val_df['target']
        
        mock_prepare_data.side_effect = [
            (X_train, y_train, None),
            (X_val, y_val, None)
        ]
        
        # Mock factory와 컴포넌트들
        mock_factory = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        
        mock_model_instance = Mock()
        mock_model_instance.set_params.return_value = None
        
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'accuracy': 0.85}
        
        mock_factory.create_preprocessor.return_value = mock_preprocessor
        mock_factory.create_model.return_value = mock_model_instance
        mock_factory.create_evaluator.return_value = mock_evaluator
        
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer.factory_provider = mock_factory_provider
        
        # Mock _fit_model 메서드
        trainer._fit_model = Mock()
        
        params = {'n_estimators': 100, 'max_depth': 5}
        seed = 42
        
        # Act
        result = trainer._single_training_iteration(train_df, params, seed)
        
        # Assert
        # 1. 3단계 데이터 분할이 올바르게 이루어졌는지 확인
        mock_train_test_split.assert_called_once_with(
            train_df, 
            test_size=0.2, 
            random_state=seed, 
            stratify=train_df.get('target')  # classification task이므로 target으로 stratify
        )
        
        # 2. 전처리가 Train 데이터에서만 fit되었는지 확인
        mock_preprocessor.fit.assert_called_once_with(X_train)
        
        # 3. 모델 파라미터가 설정되었는지 확인
        mock_model_instance.set_params.assert_called_once_with(**params)
        
        # 4. 모델이 학습되었는지 확인
        trainer._fit_model.assert_called_once_with(mock_model_instance, X_train, y_train, None)
        
        # 5. 검증 데이터로만 평가되었는지 확인
        mock_evaluator.evaluate.assert_called_once_with(mock_model_instance, X_val, y_val, val_df)
        
        # 6. 결과 구조 확인
        assert 'model' in result
        assert 'preprocessor' in result
        assert 'score' in result
        assert result['model'] == mock_model_instance
        assert result['preprocessor'] == mock_preprocessor
        assert result['score'] == 0.85


class TestTrainerTaskSpecificFitting:
    """Task별 모델 fitting 패턴 테스트 (causal의 treatment 파라미터 등)"""
    
    def test_fit_model_classification(self):
        """Classification task에서 모델 fitting 패턴 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        trainer = Trainer(settings)
        
        # BaseModel을 상속한 Mock 모델 생성
        class MockModel(BaseModel):
            def fit(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                pass
        
        mock_model = MockModel()
        mock_model.fit = Mock()  # fit을 Mock으로 교체
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        additional_data = None
        
        # Act
        trainer._fit_model(mock_model, X, y, additional_data)
        
        # Assert
        mock_model.fit.assert_called_once_with(X, y)
        
    def test_fit_model_regression(self):
        """Regression task에서 모델 fitting 패턴 테스트"""
        # Arrange
        settings = SettingsBuilder.build_regression_config()
        trainer = Trainer(settings)
        
        # BaseModel을 상속한 Mock 모델 생성
        class MockModel(BaseModel):
            def fit(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                pass
        
        mock_model = MockModel()
        mock_model.fit = Mock()  # fit을 Mock으로 교체
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([1.5, 2.3, 3.1])
        additional_data = None
        
        # Act
        trainer._fit_model(mock_model, X, y, additional_data)
        
        # Assert
        mock_model.fit.assert_called_once_with(X, y)
        
    def test_fit_model_clustering(self):
        """Clustering task에서 모델 fitting 패턴 테스트 (y 없음)"""
        # Arrange
        settings = SettingsBuilder.build_clustering_config()
        trainer = Trainer(settings)
        
        # BaseModel을 상속한 Mock 모델 생성
        class MockModel(BaseModel):
            def fit(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                pass
        
        mock_model = MockModel()
        mock_model.fit = Mock()  # fit을 Mock으로 교체
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = None  # clustering에서는 y가 None
        additional_data = None
        
        # Act
        trainer._fit_model(mock_model, X, y, additional_data)
        
        # Assert
        mock_model.fit.assert_called_once_with(X)  # y 없이 호출
        
    def test_fit_model_causal_with_treatment(self):
        """Causal task에서 treatment 파라미터를 포함한 모델 fitting 테스트"""
        # Arrange
        settings = SettingsBuilder.build_causal_config()
        trainer = Trainer(settings)
        
        # BaseModel을 상속한 Mock 모델 생성
        class MockModel(BaseModel):
            def fit(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                pass
        
        mock_model = MockModel()
        mock_model.fit = Mock()  # fit을 Mock으로 교체
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([1.5, 2.3, 3.1])  # outcome
        treatment = pd.Series([0, 1, 0])
        additional_data = {'treatment': treatment}
        
        # Act
        trainer._fit_model(mock_model, X, y, additional_data)
        
        # Assert
        mock_model.fit.assert_called_once_with(X, treatment, y)  # X, treatment, y 순서
        
    def test_fit_model_unsupported_task_type_raises_error(self):
        """지원하지 않는 task_type에서 에러가 발생하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        # task_type을 지원하지 않는 값으로 변경
        settings.recipe.data.data_interface.task_choice="unsupported_task"
        trainer = Trainer(settings)
        
        # BaseModel을 상속한 Mock 모델 생성
        class MockModel(BaseModel):
            def fit(self, *args, **kwargs):
                pass
            def predict(self, *args, **kwargs):
                pass
        
        mock_model = MockModel()
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        additional_data = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="지원하지 않는 task_type: unsupported_task"):
            trainer._fit_model(mock_model, X, y, additional_data)
            
    def test_fit_model_invalid_model_raises_error(self):
        """BaseModel 인터페이스를 따르지 않는 모델에서 에러가 발생하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        trainer = Trainer(settings)
        
        # fit 메서드가 없는 잘못된 모델 객체
        invalid_model = object()
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        additional_data = None
        
        # Act & Assert
        with pytest.raises(TypeError, match="BaseModel 인터페이스를 따르거나 scikit-learn 호환 모델이어야 합니다"):
            trainer._fit_model(invalid_model, X, y, additional_data)


class TestTrainerMetadataGeneration:
    """훈련 방법론 메타데이터 생성 검증 테스트"""
    
    def test_get_training_methodology_metadata(self):
        """훈련 방법론 메타데이터가 올바르게 생성되는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        trainer = Trainer(settings)
        
        # Act
        methodology = trainer._get_training_methodology()
        
        # Assert
        expected_metadata = {
            'train_test_split_method': 'stratified',
            'train_ratio': 0.8,
            'validation_strategy': 'train_validation_split',
            'random_state': 42,
            'preprocessing_fit_scope': 'train_only',
            'note': 'Optuna 사용 시 Train을 다시 Train(80%)/Val(20%)로 분할'
        }
        
        assert methodology == expected_metadata
        
    def test_training_methodology_included_in_results(self):
        """training_results에 훈련 방법론 메타데이터가 포함되는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        
        # Mock 데이터와 컴포넌트들
        df = TrainerDataBuilder.build_classification_data()
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'accuracy': 0.80}
        
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        
        trainer = Trainer(settings, factory_provider=mock_factory_provider)
        
        with patch('src.components.trainer.modules.trainer.split_data') as mock_split:
            with patch('src.components.trainer.modules.trainer.prepare_training_data') as mock_prepare:
                mock_split.return_value = (df[:80], df[80:])
                mock_prepare.side_effect = [
                    (df.drop(['target', 'user_id'], axis=1), df['target'], None),
                    (df.drop(['target', 'user_id'], axis=1), df['target'], None)
                ]
                
                # Act
                _, _, _, results = trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)
        
        # Assert
        assert 'training_methodology' in results
        methodology = results['training_methodology']
        assert methodology['train_test_split_method'] == 'stratified'
        assert methodology['train_ratio'] == 0.8
        assert methodology['preprocessing_fit_scope'] == 'train_only'


class TestTrainerErrorHandling:
    """컴포넌트 간 오케스트레이션 에러 처리 테스트"""
    
    @patch('src.components.trainer.modules.trainer.split_data')
    def test_component_orchestration_factory_error(self, mock_split_data):
        """Factory 생성 시 에러 처리 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_enabled_config()  # 튜닝 활성화로 Factory 사용 보장
        
        def failing_factory_provider():
            raise RuntimeError("Factory creation failed")
        
        trainer = Trainer(settings, factory_provider=failing_factory_provider)
        
        # Mock 데이터
        df = TrainerDataBuilder.build_classification_data()
        mock_split_data.return_value = (df[:80], df[80:])  # split_data 모킹
        
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_evaluator = Mock()
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Factory creation failed"):
            trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)
            
    @patch('src.components.trainer.modules.trainer.split_data')
    def test_component_orchestration_data_split_error(self, mock_split_data):
        """데이터 분할 시 에러 처리 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        trainer = Trainer(settings)
        
        # split_data에서 에러 발생하도록 설정
        mock_split_data.side_effect = ValueError("Data split failed")
        
        # Mock 데이터와 컴포넌트들
        df = TrainerDataBuilder.build_classification_data()
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_evaluator = Mock()
        
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer.factory_provider = mock_factory_provider
        
        # Act & Assert
        with pytest.raises(ValueError, match="Data split failed"):
            trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)
            
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_component_orchestration_preprocessing_error(self, mock_prepare_data, mock_split_data):
        """전처리 시 에러 처리 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        trainer = Trainer(settings)
        
        # Mock 데이터 분할
        df = TrainerDataBuilder.build_classification_data()
        mock_split_data.return_value = (df[:80], df[80:])
        mock_prepare_data.side_effect = [
            (df.drop(['target', 'user_id'], axis=1), df['target'], None),
            (df.drop(['target', 'user_id'], axis=1), df['target'], None)
        ]
        
        # 전처리에서 에러 발생하도록 설정
        mock_preprocessor = Mock()
        mock_preprocessor.fit.side_effect = RuntimeError("Preprocessing failed")
        
        # Mock 다른 컴포넌트들
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_evaluator = Mock()
        
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer.factory_provider = mock_factory_provider
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Preprocessing failed"):
            trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)
            
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_component_orchestration_model_training_error(self, mock_prepare_data, mock_split_data):
        """모델 훈련 시 에러 처리 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        trainer = Trainer(settings)
        
        # Mock 데이터 준비
        df = TrainerDataBuilder.build_classification_data()
        mock_split_data.return_value = (df[:80], df[80:])
        mock_prepare_data.side_effect = [
            (df.drop(['target', 'user_id'], axis=1), df['target'], None),
            (df.drop(['target', 'user_id'], axis=1), df['target'], None)
        ]
        
        # 모델 훈련에서 에러 발생하도록 설정
        mock_model = Mock()
        mock_model.fit.side_effect = RuntimeError("Model training failed")
        
        # Mock 다른 컴포넌트들
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        mock_evaluator = Mock()
        
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer.factory_provider = mock_factory_provider
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model training failed"):
            trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)
            
    @patch('src.components.trainer.modules.trainer.split_data')
    @patch('src.components.trainer.modules.trainer.prepare_training_data')
    def test_component_orchestration_evaluation_error(self, mock_prepare_data, mock_split_data):
        """모델 평가 시 에러 처리 테스트"""
        # Arrange
        settings = SettingsBuilder.build_tuning_disabled_config()
        trainer = Trainer(settings)
        
        # Mock 데이터 준비
        df = TrainerDataBuilder.build_classification_data()
        mock_split_data.return_value = (df[:80], df[80:])
        mock_prepare_data.side_effect = [
            (df.drop(['target', 'user_id'], axis=1), df['target'], None),
            (df.drop(['target', 'user_id'], axis=1), df['target'], None)
        ]
        
        # 평가에서 에러 발생하도록 설정
        mock_evaluator = Mock()
        mock_evaluator.evaluate.side_effect = RuntimeError("Evaluation failed")
        
        # Mock 다른 컴포넌트들
        mock_model = Mock()
        mock_fetcher = Mock()
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.side_effect = lambda x: x
        
        mock_factory = Mock()
        mock_factory_provider = Mock(return_value=mock_factory)
        trainer.factory_provider = mock_factory_provider
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Evaluation failed"):
            trainer.train(df, mock_model, mock_fetcher, mock_preprocessor, mock_evaluator)


class TestTrainerStratifyColumn:
    """Stratify 컬럼 선택 로직 테스트"""
    
    def test_get_stratify_col_classification(self):
        """Classification task에서 target_column을 stratify로 사용하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_classification_config()
        trainer = Trainer(settings)
        
        # Act
        stratify_col = trainer._get_stratify_col()
        
        # Assert
        assert stratify_col == "target"
        
    def test_get_stratify_col_causal(self):
        """Causal task에서 treatment_column을 stratify로 사용하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_causal_config()
        trainer = Trainer(settings)
        
        # Act
        stratify_col = trainer._get_stratify_col()
        
        # Assert
        assert stratify_col == "treatment"
        
    def test_get_stratify_col_regression(self):
        """Regression task에서 None을 반환하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_regression_config()
        trainer = Trainer(settings)
        
        # Act
        stratify_col = trainer._get_stratify_col()
        
        # Assert
        assert stratify_col is None
        
    def test_get_stratify_col_clustering(self):
        """Clustering task에서 None을 반환하는지 테스트"""
        # Arrange
        settings = SettingsBuilder.build_clustering_config()
        trainer = Trainer(settings)
        
        # Act
        stratify_col = trainer._get_stratify_col()
        
        # Assert
        assert stratify_col is None