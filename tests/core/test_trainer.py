"""
Trainer 컴포넌트 테스트

학습 프로세스, 컴포넌트 조합, 메트릭 수집 테스트
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.core.trainer import Trainer
from src.settings.settings import Settings


class TestTrainer:
    """Trainer 컴포넌트 테스트"""
    
    def test_trainer_initialization(self, xgboost_settings: Settings):
        """Trainer가 올바른 설정으로 초기화되는지 테스트"""
        trainer = Trainer(xgboost_settings)
        assert trainer.settings == xgboost_settings
        assert trainer.settings.model.name == "xgboost_x_learner"
    
    @patch('src.core.trainer.Factory')
    def test_trainer_component_creation(self, mock_factory, xgboost_settings: Settings):
        """Trainer가 필요한 컴포넌트들을 생성하는지 테스트"""
        # Mock Factory 설정
        mock_factory_instance = Mock()
        mock_augmenter = Mock()
        mock_preprocessor = Mock()
        mock_model = Mock()
        
        mock_factory_instance.create_augmenter.return_value = mock_augmenter
        mock_factory_instance.create_preprocessor.return_value = mock_preprocessor
        mock_factory_instance.create_model.return_value = mock_model
        mock_factory.return_value = mock_factory_instance
        
        trainer = Trainer(xgboost_settings)
        
        # Factory가 올바르게 생성되었는지 확인
        mock_factory.assert_called_once_with(xgboost_settings)
        assert trainer.factory == mock_factory_instance
    
    def test_train_method(self, xgboost_settings: Settings):
        """train 메서드의 전체 학습 과정 테스트"""
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c', 'd'],
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'outcome': [0, 1, 0, 1]
        })
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_preprocessor = Mock()
        mock_model = Mock()
        
        # 증강된 데이터
        augmented_data = sample_data.copy()
        augmented_data['feature3'] = [10, 20, 30, 40]
        
        # 전처리된 데이터
        preprocessed_data = pd.DataFrame({
            'feature1_scaled': [0.1, 0.2, 0.3, 0.4],
            'feature2_scaled': [0.1, 0.2, 0.3, 0.4],
            'feature3_scaled': [0.1, 0.2, 0.3, 0.4]
        })
        
        # Mock 동작 설정
        mock_augmenter.augment.return_value = augmented_data
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.return_value = preprocessed_data
        mock_model.fit.return_value = None
        
        # Factory mock 설정
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        trainer.factory.create_model.return_value = mock_model
        
        # 학습 실행
        trained_model, trained_preprocessor, metrics = trainer.train(sample_data)
        
        # 각 단계가 올바르게 실행되었는지 확인
        mock_augmenter.augment.assert_called_once_with(
            sample_data, 
            run_mode="batch", 
            context_params={}
        )
        mock_preprocessor.fit.assert_called_once()
        mock_preprocessor.transform.assert_called_once()
        mock_model.fit.assert_called_once()
        
        # 반환값 확인
        assert trained_model == mock_model
        assert trained_preprocessor == mock_preprocessor
        assert isinstance(metrics, dict)
    
    def test_train_with_context_params(self, xgboost_settings: Settings):
        """컨텍스트 파라미터와 함께 학습 테스트"""
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b'],
            'feature1': [1, 2],
            'outcome': [0, 1]
        })
        
        context_params = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = Mock()
        trainer.factory.create_model.return_value = Mock()
        
        # 컨텍스트 파라미터와 함께 학습 실행
        trainer.train(sample_data, context_params=context_params)
        
        # 컨텍스트 파라미터가 올바르게 전달되었는지 확인
        mock_augmenter.augment.assert_called_once_with(
            sample_data, 
            run_mode="batch", 
            context_params=context_params
        )
    
    def test_train_augmentation_error_handling(self, xgboost_settings: Settings):
        """데이터 증강 단계에서 오류 처리 테스트"""
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.side_effect = Exception("Augmentation failed")
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        
        # 오류가 적절히 전파되는지 확인
        with pytest.raises(Exception, match="Augmentation failed"):
            trainer.train(sample_data)
    
    def test_train_preprocessing_error_handling(self, xgboost_settings: Settings):
        """전처리 단계에서 오류 처리 테스트"""
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.side_effect = Exception("Preprocessing failed")
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        
        # 오류가 적절히 전파되는지 확인
        with pytest.raises(Exception, match="Preprocessing failed"):
            trainer.train(sample_data)
    
    def test_train_model_fitting_error_handling(self, xgboost_settings: Settings):
        """모델 학습 단계에서 오류 처리 테스트"""
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.return_value = pd.DataFrame()
        
        mock_model = Mock()
        mock_model.fit.side_effect = Exception("Model training failed")
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        trainer.factory.create_model.return_value = mock_model
        
        # 오류가 적절히 전파되는지 확인
        with pytest.raises(Exception, match="Model training failed"):
            trainer.train(sample_data)
    
    def test_train_empty_data_handling(self, xgboost_settings: Settings):
        """빈 데이터 처리 테스트"""
        empty_data = pd.DataFrame()
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = empty_data
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = Mock()
        trainer.factory.create_model.return_value = Mock()
        
        # 빈 데이터가 적절히 처리되는지 확인
        trained_model, trained_preprocessor, metrics = trainer.train(empty_data)
        
        # 각 컴포넌트가 호출되었는지 확인
        mock_augmenter.augment.assert_called_once()
        assert trained_model is not None
        assert trained_preprocessor is not None
        assert isinstance(metrics, dict)
    
    def test_blueprint_principle_context_injection(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: 컨텍스트 주입"""
        trainer = Trainer(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = Mock()
        trainer.factory.create_model.return_value = Mock()
        
        trainer.train(sample_data)
        
        # Augmenter가 올바른 컨텍스트(run_mode="batch")로 호출되었는지 확인
        mock_augmenter.augment.assert_called_once_with(
            sample_data, 
            run_mode="batch", 
            context_params={}
        )
    
    def test_metrics_collection(self, xgboost_settings: Settings):
        """메트릭 수집 테스트"""
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3],
            'outcome': [0, 1, 0]
        })
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.return_value = None
        mock_preprocessor.transform.return_value = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        
        mock_model = Mock()
        mock_model.fit.return_value = None
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        trainer.factory.create_model.return_value = mock_model
        
        # 학습 실행
        trained_model, trained_preprocessor, metrics = trainer.train(sample_data)
        
        # 메트릭이 올바르게 수집되었는지 확인
        assert isinstance(metrics, dict)
        assert 'training_samples' in metrics
        assert 'training_features' in metrics
        assert metrics['training_samples'] == 3
        assert metrics['training_features'] == 2
    
    def test_component_lifecycle(self, xgboost_settings: Settings):
        """컴포넌트 라이프사이클 테스트"""
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b'],
            'feature1': [1, 2],
            'outcome': [0, 1]
        })
        
        trainer = Trainer(xgboost_settings)
        
        # Mock 컴포넌트들이 올바른 순서로 호출되는지 확인
        call_order = []
        
        def track_augment(*args, **kwargs):
            call_order.append('augment')
            return sample_data
        
        def track_fit(*args, **kwargs):
            call_order.append('fit')
            return None
        
        def track_transform(*args, **kwargs):
            call_order.append('transform')
            return pd.DataFrame([[1, 2], [3, 4]])
        
        def track_model_fit(*args, **kwargs):
            call_order.append('model_fit')
            return None
        
        # Mock 컴포넌트 설정
        mock_augmenter = Mock()
        mock_augmenter.augment.side_effect = track_augment
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.side_effect = track_fit
        mock_preprocessor.transform.side_effect = track_transform
        
        mock_model = Mock()
        mock_model.fit.side_effect = track_model_fit
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        trainer.factory.create_model.return_value = mock_model
        
        # 학습 실행
        trainer.train(sample_data)
        
        # 올바른 순서로 호출되었는지 확인
        expected_order = ['augment', 'fit', 'transform', 'model_fit']
        assert call_order == expected_order 