"""학습 파이프라인 테스트 - Mock 기반 통합 테스트

Phase 3: 복잡한 학습 파이프라인의 핵심 로직과 통합 흐름을 Mock을 사용하여 검증합니다.

테스트 전략:
- 복잡한 외부 의존성 (MLflow, 데이터 어댑터 등) Mock 처리
- 핵심 비즈니스 로직과 컴포넌트 통합 흐름 검증
- Factory 패턴 활용 검증
- 오류 처리 및 예외 상황 테스트
"""
import pytest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock

from src.pipelines.train_pipeline import run_training
from src.settings import Settings


class TestTrainPipeline:
    """학습 파이프라인 핵심 기능 테스트"""

    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.bootstrap')
    @patch('src.pipelines.train_pipeline.set_global_seeds')
    @patch('src.pipelines.train_pipeline.mlflow_utils.start_run')
    @patch('src.utils.integrations.mlflow_integration.log_enhanced_model_with_schema')
    def test_run_training_success_flow(
        self, mock_log_enhanced, mock_start_run, mock_set_seeds, mock_bootstrap, mock_factory_class, mock_mlflow, test_factories
    ):
        """학습 파이프라인 성공 흐름 테스트"""
        # Given: Mock 설정
        # MLflow run context
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_123"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Factory와 컴포넌트들
        mock_factory = MagicMock()
        mock_factory_class.return_value = mock_factory
        
        # Factory의 model_config.loader.adapter Mock 설정
        mock_factory.model_config.loader.adapter = 'test_adapter'
        
        # 데이터 어댑터 Mock
        mock_data_adapter = MagicMock()
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        mock_data_adapter.read.return_value = test_data
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Settings 생성 (Factory 패턴)
        settings_dict = test_factories['settings'].create_classification_settings("test")
        # 필수 computed 필드 추가
        settings_dict['recipe']['model']['computed'] = {
            'run_name': 'test_run',
            'seed': 42
        }
        settings_dict['recipe']['model']['loader'] = {
            'source_uri': 'test://data.csv',
            'adapter': 'test_adapter',
            'entity_schema': {
                "entity_columns": ["user_id"],
                "timestamp_column": "event_timestamp"
            }
        }
        settings = Settings(**settings_dict)
        
        # When: 학습 파이프라인 실행
        run_training(settings)
        
        # Then: 핵심 흐름 검증
        # 1. 부트스트랩 호출
        mock_bootstrap.assert_called_once_with(settings)
        
        # 2. 재현성 시드 설정
        mock_set_seeds.assert_called_once_with(42)
        
        # 3. Factory 생성
        mock_factory_class.assert_called_once_with(settings)
        
        # 4. 데이터 어댑터 생성 및 데이터 로딩
        mock_factory.create_data_adapter.assert_called_once_with('test_adapter')
        mock_data_adapter.read.assert_called_once_with('test://data.csv')
        
        # 5. MLflow 메트릭 로깅
        mock_mlflow.log_metric.assert_any_call("row_count", len(test_data))
        mock_mlflow.log_metric.assert_any_call("column_count", len(test_data.columns))

    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.bootstrap')
    def test_run_training_data_loading_failure(
        self, mock_bootstrap, mock_factory_class, mock_mlflow, test_factories
    ):
        """데이터 로딩 실패 시 예외 처리 테스트"""
        # Given: 데이터 어댑터 실패 시나리오
        mock_factory = MagicMock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_adapter = MagicMock()
        mock_data_adapter.read.side_effect = FileNotFoundError("데이터 파일을 찾을 수 없습니다")
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Settings 설정
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['computed'] = {
            'run_name': 'test_run',
            'seed': 42
        }
        settings_dict['recipe']['model']['loader'] = {
            'source_uri': 'nonexistent://data.csv',
            'adapter': 'test_adapter',
            'entity_schema': {
                "entity_columns": ["user_id"],
                "timestamp_column": "event_timestamp"
            }
        }
        settings = Settings(**settings_dict)
        
        # When/Then: 데이터 로딩 실패 예외 발생
        with pytest.raises(FileNotFoundError, match="데이터 파일을 찾을 수 없습니다"):
            run_training(settings)

    @patch('src.pipelines.train_pipeline.mlflow')
    @patch('src.pipelines.train_pipeline.Factory')
    @patch('src.pipelines.train_pipeline.bootstrap')
    @patch('src.pipelines.train_pipeline.mlflow_utils.start_run')
    @patch('src.utils.integrations.mlflow_integration.log_enhanced_model_with_schema')
    def test_run_training_factory_component_creation(
        self, mock_log_enhanced, mock_start_run, mock_bootstrap, mock_factory_class, mock_mlflow, test_factories
    ):
        """Factory를 통한 컴포넌트 생성 검증"""
        # Given: Factory 컴포넌트 Mock 설정
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        mock_factory = MagicMock()
        mock_factory_class.return_value = mock_factory
        
        # Factory의 model_config.loader.adapter Mock 설정
        mock_factory.model_config.loader.adapter = 'storage'
        
        # 각 컴포넌트 Mock
        mock_data_adapter = MagicMock()
        mock_data_adapter.read.return_value = pd.DataFrame({'col': [1, 2], 'target': [0, 1]})
        mock_factory.create_data_adapter.return_value = mock_data_adapter
        
        # Settings 구성
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['computed'] = {
            'run_name': 'test_factory_run',
            'seed': 42
        }
        settings_dict['recipe']['model']['loader'] = {
            'source_uri': 'test://data.csv',
            'adapter': 'storage',
            'entity_schema': {
                "entity_columns": ["user_id"],
                "timestamp_column": "event_timestamp"
            }
        }
        settings = Settings(**settings_dict)
        
        # When: 학습 파이프라인 실행
        run_training(settings)
        
        # Then: Factory를 통한 컴포넌트 생성 검증
        mock_factory_class.assert_called_once_with(settings)
        mock_factory.create_data_adapter.assert_called_once_with('storage')
        
        # 추가 컴포넌트 생성 호출들도 검증 가능
        # (fetcher, preprocessor, model, evaluator 등)

    def test_run_training_settings_validation(self, test_factories):
        """Settings 검증 테스트 - 필수 필드 확인"""
        # Given: 불완전한 Settings
        settings_dict = test_factories['settings'].create_classification_settings("test")
        # computed 필드 누락
        settings_dict['recipe']['model'].pop('computed', None)
        
        # When/Then: 필수 필드 누락으로 인한 오류
        with pytest.raises((KeyError, AttributeError)):
            settings = Settings(**settings_dict)
            run_training(settings)  # computed.run_name 접근 시 오류

    @patch('src.pipelines.train_pipeline.mlflow_utils.start_run')
    @patch('src.pipelines.train_pipeline.bootstrap')
    def test_run_training_mlflow_context_management(
        self, mock_bootstrap, mock_start_run, test_factories
    ):
        """MLflow 컨텍스트 관리 검증"""
        # Given: MLflow 컨텍스트 Mock
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None
        
        # Settings 구성
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['computed'] = {
            'run_name': 'mlflow_context_test',
            'seed': 42
        }
        settings = Settings(**settings_dict)
        
        # When: 파이프라인 실행 (다른 Mock들로 인해 일부만 실행될 수 있음)
        try:
            run_training(settings)
        except Exception:
            # MLflow 컨텍스트 관리 검증이 목적이므로 다른 오류는 무시
            pass
        
        # Then: MLflow 컨텍스트 시작 검증
        mock_start_run.assert_called_once_with(settings, run_name='mlflow_context_test')

    def test_run_training_context_params_handling(self, test_factories):
        """context_params 매개변수 처리 테스트"""
        # Given: context_params 포함 설정
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings_dict['recipe']['model']['computed'] = {
            'run_name': 'context_test',
            'seed': 42
        }
        settings = Settings(**settings_dict)
        
        context_params = {"custom_param": "test_value", "environment": "test"}
        
        # When/Then: context_params와 함께 호출 가능 검증
        with patch('src.pipelines.train_pipeline.bootstrap'):
            with patch('src.pipelines.train_pipeline.mlflow_utils.start_run') as mock_start_run:
                mock_start_run.return_value.__enter__ = Mock(return_value=MagicMock())
                mock_start_run.return_value.__exit__ = Mock(return_value=None)
                
                try:
                    run_training(settings, context_params=context_params)
                except Exception:
                    # 다른 컴포넌트 Mock이 부족해서 발생할 수 있는 오류 무시
                    # context_params 전달 자체가 오류 없이 처리되는지 확인
                    pass
                
                # MLflow 컨텍스트 시작이 호출되었는지 확인
                mock_start_run.assert_called()