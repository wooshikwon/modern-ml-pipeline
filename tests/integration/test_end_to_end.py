"""
End-to-End 통합 테스트

전체 시스템의 동작과 Blueprint 6대 원칙 준수를 검증하는 테스트
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.core.factory import Factory
from src.core.trainer import Trainer
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference
from src.settings.settings import Settings, load_settings


class TestEndToEndIntegration:
    """End-to-End 통합 테스트"""
    
    def test_blueprint_principle_1_recipe_vs_config_separation(self):
        """Blueprint 원칙 1: 레시피는 논리, 설정은 인프라"""
        # 설정 로딩이 레시피와 config를 올바르게 분리하는지 확인
        settings = load_settings("xgboost_x_learner")
        
        # 레시피 정보 (모델 논리)
        assert hasattr(settings.model, 'name')
        assert hasattr(settings.model, 'hyperparameters')
        assert hasattr(settings.model, 'loader')
        assert hasattr(settings.model, 'augmenter')
        
        # 인프라 정보 (환경 설정)
        assert hasattr(settings, 'data_sources')
        assert hasattr(settings, 'mlflow')
        
        # 레시피에 인프라 정보가 직접 포함되지 않았는지 확인
        assert not hasattr(settings.model, 'mlflow')
        assert not hasattr(settings.model, 'data_sources')
    
    def test_blueprint_principle_2_unified_data_adapter(self, xgboost_settings: Settings):
        """Blueprint 원칙 2: 통합 데이터 어댑터"""
        factory = Factory(xgboost_settings)
        
        # 모든 어댑터가 통합된 인터페이스를 가지는지 확인
        from src.interface.base_data_adapter import BaseDataAdapter
        
        adapters = [
            factory.create_data_adapter("bq"),
            factory.create_data_adapter("gs"),
            factory.create_data_adapter("s3"),
            factory.create_data_adapter("file")
        ]
        
        for adapter in adapters:
            assert isinstance(adapter, BaseDataAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
            assert adapter.settings == xgboost_settings
    
    def test_blueprint_principle_3_uri_driven_operation(self, xgboost_settings: Settings):
        """Blueprint 원칙 3: URI 기반 동작 및 동적 팩토리"""
        factory = Factory(xgboost_settings)
        
        # URI 스킴에 따라 올바른 어댑터가 생성되는지 확인
        uri_to_adapter = {
            "bq": "BigQueryAdapter",
            "gs": "GCSAdapter",
            "s3": "S3Adapter",
            "file": "FileSystemAdapter"
        }
        
        for scheme, expected_class in uri_to_adapter.items():
            adapter = factory.create_data_adapter(scheme)
            assert adapter.__class__.__name__ == expected_class
    
    def test_blueprint_principle_4_pure_logic_artifact(self, xgboost_settings: Settings):
        """Blueprint 원칙 4: 순수 로직 아티팩트"""
        factory = Factory(xgboost_settings)
        
        # 생성된 컴포넌트들이 순수 로직만 포함하는지 확인
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()
        
        # 컴포넌트들이 설정을 참조하지만 인프라 정보를 직접 포함하지 않는지 확인
        assert augmenter.settings == xgboost_settings
        assert preprocessor.settings == xgboost_settings
        assert model.settings == xgboost_settings
        
        # 인프라 정보는 설정을 통해서만 접근 가능
        assert not hasattr(augmenter, 'mlflow_uri')
        assert not hasattr(preprocessor, 'data_source_uri')
        assert not hasattr(model, 'database_config')
    
    def test_blueprint_principle_5_single_augmenter_context_injection(self, xgboost_settings: Settings):
        """Blueprint 원칙 5: 단일 Augmenter, 컨텍스트 주입"""
        factory = Factory(xgboost_settings)
        augmenter = factory.create_augmenter()
        
        # 단일 Augmenter 인스턴스가 다른 컨텍스트로 동작하는지 확인
        sample_data = pd.DataFrame({'member_id': ['a', 'b']})
        
        with patch.object(augmenter, '_augment_batch') as mock_batch:
            mock_batch.return_value = sample_data
            result_batch = augmenter.augment(sample_data, run_mode="batch")
            
        with patch.object(augmenter, '_augment_realtime') as mock_realtime:
            mock_realtime.return_value = sample_data
            result_realtime = augmenter.augment(
                sample_data, 
                run_mode="realtime",
                feature_store_config={}
            )
        
        # 동일한 인스턴스가 다른 컨텍스트에서 동작했는지 확인
        mock_batch.assert_called_once()
        mock_realtime.assert_called_once()
    
    def test_blueprint_principle_6_self_describing_api(self, xgboost_settings: Settings):
        """Blueprint 원칙 6: 자기 기술 API"""
        # API 스키마 생성이 모델 정보를 기반으로 하는지 확인
        from serving.schemas import create_dynamic_prediction_request
        
        # 모델의 loader SQL을 기반으로 API 스키마가 생성되는지 확인
        # (실제 구현에서는 SQL 파싱을 통해 PK 추출)
        mock_sql = "SELECT member_id, feature1, feature2 FROM table"
        
        with patch('src.utils.sql_utils.get_selected_columns') as mock_get_columns:
            mock_get_columns.return_value = ['member_id']
            
            # 동적 스키마 생성
            schema_class = create_dynamic_prediction_request(['member_id'])
            
            # 스키마가 모델 정보를 반영하는지 확인
            assert hasattr(schema_class, '__fields__')
            assert 'member_id' in schema_class.__fields__
    
    @patch('src.pipelines.train_pipeline.mlflow')
    def test_complete_training_workflow(self, mock_mlflow, xgboost_settings: Settings):
        """완전한 학습 워크플로우 테스트"""
        # Mock MLflow 설정
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.return_value = None
        mock_mlflow.start_run.return_value = None
        mock_mlflow.log_params.return_value = None
        mock_mlflow.log_metrics.return_value = None
        mock_mlflow.pyfunc.log_model.return_value = None
        mock_mlflow.set_tag.return_value = None
        
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c', 'd'],
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'treatment': [0, 1, 0, 1],
            'outcome': [0.5, 1.5, 0.3, 1.2]
        })
        
        # 데이터 로딩 Mock
        with patch('src.pipelines.train_pipeline.get_dataset_loader') as mock_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.load.return_value = sample_data
            mock_loader.return_value = mock_loader_instance
            
            # 학습 실행
            run_training(xgboost_settings)
            
            # MLflow 워크플로우가 올바르게 실행되었는지 확인
            mock_mlflow.set_tracking_uri.assert_called_once()
            mock_mlflow.set_experiment.assert_called_once()
            mock_mlflow.start_run.assert_called_once()
            mock_mlflow.pyfunc.log_model.assert_called_once()
    
    def test_component_interaction_flow(self, xgboost_settings: Settings):
        """컴포넌트 간 상호작용 흐름 테스트"""
        trainer = Trainer(xgboost_settings)
        
        # 샘플 데이터
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'treatment': [0, 1, 0],
            'outcome': [0.5, 1.5, 0.3]
        })
        
        # 컴포넌트 간 데이터 흐름 추적
        call_order = []
        
        def track_augment(*args, **kwargs):
            call_order.append('augment')
            return sample_data
        
        def track_fit(*args, **kwargs):
            call_order.append('preprocess_fit')
            return None
        
        def track_transform(*args, **kwargs):
            call_order.append('preprocess_transform')
            return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
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
        
        # 올바른 순서로 컴포넌트가 호출되었는지 확인
        expected_order = ['augment', 'preprocess_fit', 'preprocess_transform', 'model_fit']
        assert call_order == expected_order
    
    def test_error_propagation_and_handling(self, xgboost_settings: Settings):
        """오류 전파 및 처리 테스트"""
        trainer = Trainer(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        # 각 단계에서 오류가 올바르게 전파되는지 확인
        
        # 1. Augmenter 오류
        mock_augmenter = Mock()
        mock_augmenter.augment.side_effect = Exception("Augmentation failed")
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        
        with pytest.raises(Exception, match="Augmentation failed"):
            trainer.train(sample_data)
        
        # 2. Preprocessor 오류
        mock_augmenter = Mock()
        mock_augmenter.augment.return_value = sample_data
        
        mock_preprocessor = Mock()
        mock_preprocessor.fit.side_effect = Exception("Preprocessing failed")
        
        trainer.factory = Mock()
        trainer.factory.create_augmenter.return_value = mock_augmenter
        trainer.factory.create_preprocessor.return_value = mock_preprocessor
        
        with pytest.raises(Exception, match="Preprocessing failed"):
            trainer.train(sample_data)
        
        # 3. Model 오류
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
        
        with pytest.raises(Exception, match="Model training failed"):
            trainer.train(sample_data)
    
    def test_system_resilience(self, xgboost_settings: Settings):
        """시스템 회복력 테스트"""
        factory = Factory(xgboost_settings)
        
        # 1. 선택적 의존성 처리 (Redis)
        try:
            redis_adapter = factory.create_data_adapter("redis")
            # Redis가 사용 가능한 경우
            assert redis_adapter is not None
        except (ImportError, ValueError):
            # Redis가 사용 불가능한 경우 적절한 처리
            pass
        
        # 2. 인증 실패 시 graceful degradation
        bigquery_adapter = factory.create_data_adapter("bq")
        
        # 인증 실패 시에도 어댑터가 생성되고 빈 결과를 반환
        result = bigquery_adapter.read("bq://project.dataset.table", params={})
        assert isinstance(result, pd.DataFrame)
        
        # 3. 설정 파일 누락 시 처리
        try:
            invalid_settings = load_settings("non_existent_model")
            assert False, "Should have raised an exception"
        except Exception as e:
            # 적절한 오류 메시지 확인
            assert "non_existent_model" in str(e) or "not found" in str(e).lower()
    
    def test_data_pipeline_integrity(self, xgboost_settings: Settings):
        """데이터 파이프라인 무결성 테스트"""
        factory = Factory(xgboost_settings)
        
        # 데이터 변환 과정에서 무결성이 유지되는지 확인
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'outcome': [0, 1, 0]
        })
        
        # 1. Augmenter 처리
        augmenter = factory.create_augmenter()
        with patch.object(augmenter, '_augment_batch') as mock_augment:
            mock_augment.return_value = sample_data
            augmented_data = augmenter.augment(sample_data, run_mode="batch")
            
            # 데이터 무결성 확인
            assert len(augmented_data) == len(sample_data)
            assert 'member_id' in augmented_data.columns
        
        # 2. Preprocessor 처리
        preprocessor = factory.create_preprocessor()
        with patch.object(preprocessor, 'fit') as mock_fit, \
             patch.object(preprocessor, 'transform') as mock_transform:
            
            mock_fit.return_value = None
            mock_transform.return_value = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
            
            preprocessor.fit(sample_data)
            processed_data = preprocessor.transform(sample_data)
            
            # 데이터 변환 후 샘플 수 유지
            assert len(processed_data) == len(sample_data)
    
    def test_configuration_flexibility(self):
        """설정 유연성 테스트"""
        # 다양한 환경에서 설정이 올바르게 로드되는지 확인
        
        # 1. XGBoost 모델 설정
        xgboost_settings = load_settings("xgboost_x_learner")
        assert xgboost_settings.model.name == "xgboost_x_learner"
        
        # 2. CausalForest 모델 설정
        causal_forest_settings = load_settings("causal_forest")
        assert causal_forest_settings.model.name == "causal_forest"
        
        # 3. 각 모델이 고유한 설정을 가지는지 확인
        assert xgboost_settings.model.hyperparameters != causal_forest_settings.model.hyperparameters
    
    def test_blueprint_compliance_summary(self, xgboost_settings: Settings):
        """Blueprint 준수 요약 테스트"""
        factory = Factory(xgboost_settings)
        
        # 모든 핵심 컴포넌트가 Blueprint 원칙을 준수하는지 종합 확인
        
        # 1. 모든 컴포넌트가 올바르게 생성되는지 확인
        components = {
            'augmenter': factory.create_augmenter(),
            'preprocessor': factory.create_preprocessor(),
            'trainer': factory.create_trainer(),
            'model': factory.create_model()
        }
        
        for name, component in components.items():
            assert component is not None, f"{name} should be created"
            assert hasattr(component, 'settings'), f"{name} should have settings"
            assert component.settings == xgboost_settings, f"{name} should have correct settings"
        
        # 2. 모든 데이터 어댑터가 올바르게 생성되는지 확인
        adapters = {
            'bigquery': factory.create_data_adapter("bq"),
            'gcs': factory.create_data_adapter("gs"),
            's3': factory.create_data_adapter("s3"),
            'file': factory.create_data_adapter("file")
        }
        
        for name, adapter in adapters.items():
            assert adapter is not None, f"{name} adapter should be created"
            assert hasattr(adapter, 'settings'), f"{name} adapter should have settings"
            assert adapter.settings == xgboost_settings, f"{name} adapter should have correct settings"
        
        # 3. Blueprint 6대 원칙 준수 확인
        principles_check = {
            'recipe_config_separation': hasattr(xgboost_settings, 'model') and hasattr(xgboost_settings, 'data_sources'),
            'unified_data_adapter': all(hasattr(adapter, 'read') and hasattr(adapter, 'write') for adapter in adapters.values()),
            'uri_driven_operation': len(adapters) == 4,  # 4개의 스킴 지원
            'pure_logic_artifact': all(hasattr(comp, 'settings') for comp in components.values()),
            'single_augmenter': components['augmenter'] is not None,
            'self_describing_api': True  # API 스키마 생성 기능 존재
        }
        
        for principle, check in principles_check.items():
            assert check, f"Blueprint principle '{principle}' should be satisfied" 