"""
MLflow integration utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/integrations/mlflow_integration.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import uuid
import datetime

from src.utils.integrations.mlflow_integration import (
    generate_unique_run_name,
    setup_mlflow,
    start_run,
    get_latest_run_id,
    get_model_uri,
    load_pyfunc_model,
    download_artifacts,
    create_model_signature,
    create_enhanced_model_signature_with_schema,
    log_enhanced_model_with_schema,
    log_training_results,
    _infer_pandas_dtype_to_mlflow_type
)


class TestMLflowRunNaming:
    """MLflow run naming 기능 테스트"""

    def test_generate_unique_run_name_format(self, component_test_context):
        """유니크 run name 생성 형식 테스트"""
        with component_test_context.classification_stack() as ctx:
            base_name = "test_run"

            result = generate_unique_run_name(base_name)

            # Format: base_name_YYYYMMDD_HHMMSS_uuid
            parts = result.split('_')
            assert len(parts) >= 4
            assert parts[0] == "test"
            assert parts[1] == "run"
            assert len(parts[-1]) == 8  # UUID 8 characters
            assert len(parts[-2]) == 6   # HHMMSS format

    def test_generate_unique_run_name_uniqueness(self, component_test_context):
        """연속 생성된 run name들의 유니크성 테스트"""
        with component_test_context.classification_stack() as ctx:
            base_name = "uniqueness_test"

            # Generate multiple run names quickly
            names = [generate_unique_run_name(base_name) for _ in range(10)]

            # All should be unique
            assert len(set(names)) == len(names)

            # All should start with base name
            for name in names:
                assert name.startswith(base_name)


class TestMLflowSetup:
    """MLflow 설정 기능 테스트 - Context 클래스 기반"""

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_setup_mlflow_success(self, mock_mlflow, component_test_context):
        """MLflow 설정 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test setup_mlflow function
            setup_mlflow(ctx.settings)

            # Verify MLflow setup calls
            mock_mlflow.set_tracking_uri.assert_called_once()
            mock_mlflow.set_experiment.assert_called_once()

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.Console')
    def test_setup_mlflow_console_output(self, mock_console_class, mock_mlflow, component_test_context):
        """MLflow 설정 시 콘솔 출력 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            setup_mlflow(ctx.settings)

            # Verify console logging
            mock_console.log_milestone.assert_called()
            mock_console.print.assert_called()


class TestMLflowRunManagement:
    """MLflow run 관리 테스트"""

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_start_run_context_manager_success(self, mock_mlflow, component_test_context):
        """start_run 컨텍스트 매니저 성공 시나리오 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock MLflow run
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"

            # Mock context manager
            @contextmanager
            def mock_start_run(*args, **kwargs):
                yield mock_run

            mock_mlflow.start_run = mock_start_run
            mock_mlflow.set_tag = Mock()

            with start_run(ctx.settings, "test_run") as run:
                assert run == mock_run

            # Verify success tags were set
            assert mock_mlflow.set_tag.call_count >= 3  # original_run_name, unique_run_name, status

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_start_run_with_file_tracking_uri(self, mock_mlflow, component_test_context):
        """file:// tracking URI를 사용한 start_run 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Set file:// tracking URI
            ctx.settings.config.mlflow.tracking_uri = f"file://{ctx.temp_dir}/mlruns"

            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"

            @contextmanager
            def mock_start_run(*args, **kwargs):
                yield mock_run

            mock_mlflow.start_run = mock_start_run

            with patch('os.makedirs') as mock_makedirs:
                with start_run(ctx.settings, "test_run") as run:
                    pass

                # Verify directory creation was attempted
                mock_makedirs.assert_called()

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_start_run_exception_handling(self, mock_mlflow, component_test_context):
        """start_run에서 예외 발생 시 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"

            @contextmanager
            def mock_start_run_with_exception(*args, **kwargs):
                try:
                    yield mock_run
                except Exception:
                    pass  # Allow the exception to propagate
                raise Exception("Test exception")

            mock_mlflow.start_run = mock_start_run_with_exception
            mock_mlflow.set_tag = Mock()

            with pytest.raises(Exception):
                with start_run(ctx.settings, "test_run") as run:
                    raise Exception("Test exception")

            # Verify failure tags were set
            mock_mlflow.set_tag.assert_called()

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_start_run_name_collision_retry(self, mock_mlflow, component_test_context):
        """Run name 충돌 시 재시도 로직 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"

            # First call raises collision error, second succeeds
            call_count = 0

            @contextmanager
            def mock_start_run_with_collision(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Run already exists")
                else:
                    yield mock_run

            mock_mlflow.start_run = mock_start_run_with_collision
            mock_mlflow.set_tag = Mock()

            with start_run(ctx.settings, "test_run") as run:
                assert run == mock_run

            # Should have been called twice (original + retry)
            assert call_count == 2


class TestMLflowModelOperations:
    """MLflow 모델 조작 기능 테스트"""

    def test_get_model_uri_format(self, component_test_context):
        """모델 URI 생성 형식 테스트"""
        with component_test_context.classification_stack() as ctx:
            run_id = "test_run_123"
            artifact_path = "model"

            uri = get_model_uri(run_id, artifact_path)

            assert uri == f"runs:/{run_id}/{artifact_path}"

    def test_get_model_uri_default_artifact_path(self, component_test_context):
        """기본 artifact path를 사용한 모델 URI 테스트"""
        with component_test_context.classification_stack() as ctx:
            run_id = "test_run_123"

            uri = get_model_uri(run_id)  # Default artifact_path="model"

            assert uri == f"runs:/{run_id}/model"

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.MlflowClient')
    def test_load_pyfunc_model_runs_uri(self, mock_client_class, mock_mlflow, component_test_context):
        """runs:// URI를 사용한 PyFunc 모델 로드 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_client = Mock()
            mock_client.download_artifacts.return_value = "/local/path/model"
            mock_client_class.return_value = mock_client

            mock_model = Mock()
            mock_mlflow.pyfunc.load_model.return_value = mock_model

            model_uri = "runs:/test_run_123/model"
            result = load_pyfunc_model(ctx.settings, model_uri)

            # Verify MlflowClient was used for runs:// URI
            mock_client_class.assert_called_once_with(tracking_uri=ctx.settings.config.mlflow.tracking_uri)
            mock_client.download_artifacts.assert_called_once_with(run_id="test_run_123", path="model")
            mock_mlflow.pyfunc.load_model.assert_called_once_with(model_uri="/local/path/model")
            assert result == mock_model

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_load_pyfunc_model_local_path(self, mock_mlflow, component_test_context):
        """로컬 경로를 사용한 PyFunc 모델 로드 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_model = Mock()
            mock_mlflow.pyfunc.load_model.return_value = mock_model
            mock_mlflow.set_tracking_uri = Mock()

            model_uri = "/local/path/to/model"
            result = load_pyfunc_model(ctx.settings, model_uri)

            # Verify direct loading for non-runs:// URI
            mock_mlflow.set_tracking_uri.assert_called_once_with(ctx.settings.config.mlflow.tracking_uri)
            mock_mlflow.pyfunc.load_model.assert_called_once_with(model_uri=model_uri)
            assert result == mock_model

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_get_latest_run_id_success(self, mock_mlflow, component_test_context):
        """최신 run ID 조회 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock experiment
            mock_experiment = Mock()
            mock_experiment.experiment_id = "exp_123"
            mock_mlflow.get_experiment_by_name.return_value = mock_experiment

            # Mock search results
            mock_runs_df = pd.DataFrame({
                'run_id': ['run_latest', 'run_older'],
                'start_time': ['2023-01-02', '2023-01-01']
            })
            mock_mlflow.search_runs.return_value = mock_runs_df

            result = get_latest_run_id(ctx.settings, "test_experiment")

            assert result == "run_latest"
            mock_mlflow.search_runs.assert_called_once_with(
                experiment_ids=["exp_123"],
                filter_string="tags.status = 'success'",
                order_by=["start_time DESC"],
                max_results=1
            )

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_get_latest_run_id_no_experiment(self, mock_mlflow, component_test_context):
        """존재하지 않는 experiment 조회 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_mlflow.get_experiment_by_name.return_value = None

            with pytest.raises(ValueError) as exc_info:
                get_latest_run_id(ctx.settings, "nonexistent_experiment")

            assert "찾을 수 없습니다" in str(exc_info.value)

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_download_artifacts_success(self, mock_mlflow, component_test_context):
        """아티팩트 다운로드 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            local_path = "/local/downloaded/artifact"
            mock_mlflow.artifacts.download_artifacts.return_value = local_path

            result = download_artifacts(ctx.settings, "run_123", "model/data.json")

            assert result == local_path
            mock_mlflow.artifacts.download_artifacts.assert_called_once_with(
                run_id="run_123",
                artifact_path="model/data.json",
                dst_path=None
            )


class TestModelSignatureCreation:
    """모델 시그니처 생성 테스트"""

    def test_create_model_signature_success(self, component_test_context):
        """모델 시그니처 생성 성공 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Input/output DataFrames
            input_df = pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0],
                'feature_1': [0.5, 1.5, 2.5]
            })

            output_df = pd.DataFrame({
                'prediction': [0, 1, 0]
            })

            with patch('src.utils.integrations.mlflow_integration.Schema') as mock_schema_class:
                with patch('src.utils.integrations.mlflow_integration.ModelSignature') as mock_signature_class:
                    mock_signature = Mock()
                    mock_signature_class.return_value = mock_signature

                    result = create_model_signature(input_df, output_df)

                    assert result == mock_signature
                    # Should create input and output schemas
                    assert mock_schema_class.call_count == 2

    def test_infer_pandas_dtype_to_mlflow_type_mappings(self, component_test_context):
        """pandas dtype을 MLflow type으로 변환 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test different dtype mappings
            test_cases = [
                (pd.Series([1, 2, 3], dtype='int64').dtype, 'long'),
                (pd.Series([1.0, 2.0, 3.0], dtype='float64').dtype, 'double'),
                (pd.Series([True, False], dtype='bool').dtype, 'boolean'),
                (pd.Series(['a', 'b', 'c'], dtype='object').dtype, 'string'),
                (pd.Series(pd.to_datetime(['2023-01-01'])).dtype, 'datetime'),
            ]

            for pandas_dtype, expected_mlflow_type in test_cases:
                result = _infer_pandas_dtype_to_mlflow_type(pandas_dtype)
                assert result == expected_mlflow_type

    def test_infer_pandas_dtype_unknown_fallback(self, component_test_context):
        """알 수 없는 dtype에 대한 fallback 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock unknown dtype
            unknown_dtype = Mock()
            unknown_dtype.name = 'unknown_type'

            result = _infer_pandas_dtype_to_mlflow_type(unknown_dtype)

            # Should fallback to 'string'
            assert result == 'string'


class TestEnhancedMLflowIntegration:
    """Enhanced MLflow 통합 기능 테스트"""

    @patch('src.utils.integrations.mlflow_integration.generate_training_schema_metadata')
    @patch('src.utils.integrations.mlflow_integration.create_model_signature')
    def test_create_enhanced_model_signature_with_schema(self, mock_create_sig, mock_schema_gen, component_test_context):
        """Enhanced model signature + schema 생성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock dependencies
            mock_signature = Mock()
            mock_create_sig.return_value = mock_signature

            mock_provisional_schema = {
                'feature_columns': ['feature_0', 'feature_1'],
                'schema_version': '2.0'
            }
            mock_schema_gen.return_value = mock_provisional_schema

            training_df = pd.DataFrame({
                'feature_0': [1.0, 2.0],
                'feature_1': [0.5, 1.5],
                'target': [0, 1]
            })

            data_interface_config = {
                'target_column': 'target',
                'entity_columns': []
            }

            signature, data_schema = create_enhanced_model_signature_with_schema(
                training_df, data_interface_config
            )

            # Verify signature and schema
            assert signature == mock_signature
            assert 'schema_version' in data_schema
            assert 'phase_integration' in data_schema
            assert data_schema['phase_integration']['phase_5_enhanced_artifact'] is True

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.Console')
    def test_log_enhanced_model_with_schema(self, mock_console_class, mock_mlflow, component_test_context):
        """Enhanced model + schema 저장 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_console = Mock()
            mock_console.progress_tracker.return_value.__enter__.return_value = Mock()
            mock_console_class.return_value = mock_console

            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"
            mock_mlflow.active_run.return_value = mock_run

            python_model = Mock()
            signature = Mock()
            data_schema = {'schema_version': '2.0'}
            input_example = pd.DataFrame({'feature': [1, 2]})

            log_enhanced_model_with_schema(
                python_model=python_model,
                signature=signature,
                data_schema=data_schema,
                input_example=input_example
            )

            # Verify MLflow logging calls
            mock_mlflow.pyfunc.log_model.assert_called_once()
            mock_mlflow.log_dict.assert_called()  # Multiple calls for different JSON files

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.UnifiedConsole')
    def test_log_training_results_with_hpo(self, mock_console_class, mock_mlflow, component_test_context):
        """HPO 활성화된 훈련 결과 로깅 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            metrics = {'accuracy': 0.95, 'f1': 0.92}
            training_results = {
                'trainer': {
                    'hyperparameter_optimization': {
                        'enabled': True,
                        'best_params': {'n_estimators': 100, 'max_depth': 5},
                        'best_score': 0.95,
                        'total_trials': 50
                    }
                }
            }

            log_training_results(ctx.settings, metrics, training_results)

            # Verify metrics and parameters logged
            mock_mlflow.log_metrics.assert_called_once_with(metrics)
            mock_mlflow.log_params.assert_called_once_with({'n_estimators': 100, 'max_depth': 5})
            mock_mlflow.log_metric.assert_any_call('best_score', 0.95)
            mock_mlflow.log_metric.assert_any_call('total_trials', 50)

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    @patch('src.utils.integrations.mlflow_integration.UnifiedConsole')
    def test_log_training_results_without_hpo(self, mock_console_class, mock_mlflow, component_test_context):
        """HPO 비활성화된 훈련 결과 로깅 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_console = Mock()
            mock_console_class.return_value = mock_console

            metrics = {'rmse': 0.15, 'mae': 0.12}
            training_results = {
                'trainer': {
                    'hyperparameter_optimization': {
                        'enabled': False
                    }
                }
            }

            # Mock fixed hyperparameters
            ctx.settings.recipe.model.hyperparameters.values = {
                'learning_rate': 0.01,
                'n_estimators': 100
            }

            log_training_results(ctx.settings, metrics, training_results)

            # Verify only metrics logged for non-HPO case
            mock_mlflow.log_metrics.assert_called_once_with(metrics)
            mock_mlflow.log_params.assert_called_once_with({
                'learning_rate': 0.01,
                'n_estimators': 100
            })


class TestMLflowIntegrationIntegration:
    """MLflow 통합 시나리오 테스트"""

    @patch('src.utils.integrations.mlflow_integration.mlflow')
    def test_complete_mlflow_workflow(self, mock_mlflow, component_test_context):
        """완전한 MLflow 워크플로우 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock MLflow components
            mock_run = Mock()
            mock_run.info.run_id = "workflow_test_run"

            @contextmanager
            def mock_start_run(*args, **kwargs):
                yield mock_run

            mock_mlflow.start_run = mock_start_run
            mock_mlflow.set_tag = Mock()
            mock_mlflow.log_metrics = Mock()
            mock_mlflow.log_params = Mock()

            # 1. Setup MLflow
            setup_mlflow(ctx.settings)

            # 2. Start run and log results
            with start_run(ctx.settings, "workflow_test") as run:
                # 3. Log training results
                metrics = {'accuracy': 0.90}
                training_results = {'trainer': {'hyperparameter_optimization': {'enabled': False}}}

                log_training_results(ctx.settings, metrics, training_results)

                # 4. Generate model URI
                model_uri = get_model_uri(run.info.run_id)
                assert model_uri == f"runs:/{run.info.run_id}/model"

            # Verify complete workflow
            mock_mlflow.set_tracking_uri.assert_called()
            mock_mlflow.set_experiment.assert_called()
            mock_mlflow.log_metrics.assert_called_with(metrics)
            mock_mlflow.set_tag.assert_called()  # Success tag should be set