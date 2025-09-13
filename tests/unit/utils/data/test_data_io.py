"""
Data I/O utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/data/data_io.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from src.utils.data.data_io import (
    save_output,
    load_data,
    format_predictions,
    load_inference_data,
    process_template_file,
    _format_multiclass_probabilities,
    _detect_probability_predictions
)
from src.settings import Settings
from src.factory import Factory
from src.utils.core.console_manager import RichConsoleManager


class TestDataIOFunctionality:
    """Data I/O 핵심 기능 테스트 - Context 클래스 기반"""

    def test_save_output_storage_adapter(self, component_test_context):
        """Storage adapter를 사용한 출력 저장 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Real data for testing
            test_data = pd.DataFrame({
                'prediction': [0, 1, 0, 1],
                'prob_positive': [0.3, 0.8, 0.2, 0.9],
                'feature_0': [1.0, 2.0, 3.0, 4.0]
            })

            # Mock output configuration
            mock_output_config = Mock()
            mock_output_config.inference = Mock()
            mock_output_config.inference.enabled = True
            mock_output_config.inference.adapter_type = "storage"
            mock_output_config.inference.config = {"base_path": f"{ctx.temp_dir}/output"}

            # Inject mock config
            ctx.settings.config.output = mock_output_config

            with patch('mlflow.log_artifact'):
                # Test save_output function
                save_output(
                    df=test_data,
                    settings=ctx.settings,
                    output_type="inference",
                    factory=ctx.factory,
                    run_id="test_run_123",
                    console=RichConsoleManager()
                )

            # Verify output file exists and has correct structure
            output_files = list(Path(f"{ctx.temp_dir}/output").glob("predictions_*.parquet"))
            assert len(output_files) > 0

            saved_data = pd.read_parquet(output_files[0])
            assert 'inference_run_id' in saved_data.columns
            assert 'inference_timestamp' in saved_data.columns
            assert len(saved_data) == len(test_data)

    def test_load_data_with_real_adapter(self, component_test_context):
        """Real adapter를 사용한 데이터 로드 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test data is already created by context
            console = RichConsoleManager()

            # Load data using real adapter
            loaded_df = load_data(
                data_adapter=ctx.adapter,
                data_source=ctx.data_path,
                context_params=None,
                console=console
            )

            # Validate data structure
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) > 0
            assert 'target' in loaded_df.columns
            assert loaded_df.columns.str.startswith('feature_').any()

    def test_format_predictions_binary_classification(self, component_test_context):
        """Binary classification 예측 결과 포맷팅 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create test predictions
            original_df = pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0],
                'feature_1': [0.5, 1.5, 2.5],
                'entity_id': ['A', 'B', 'C'],
                'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
            })

            # 2D array for binary classification probabilities
            predictions_array = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

            # Data interface configuration
            data_interface = {
                'entity_columns': ['entity_id'],
                'timestamp_column': 'timestamp',
                'feature_columns': ['feature_0', 'feature_1']
            }

            # Test format_predictions
            result_df = format_predictions(
                predictions_result=predictions_array,
                original_df=original_df,
                data_interface=data_interface,
                task_type="classification"
            )

            # Validate results
            assert 'prob_positive' in result_df.columns
            assert 'prob_negative' in result_df.columns
            assert 'entity_id' in result_df.columns
            assert 'timestamp' in result_df.columns
            assert 'feature_0' not in result_df.columns  # Features should be excluded
            assert len(result_df) == len(original_df)

    def test_format_predictions_multiclass(self, component_test_context):
        """Multiclass classification 예측 결과 포맷팅 테스트"""
        with component_test_context.classification_stack() as ctx:
            original_df = pd.DataFrame({
                'feature_0': [1.0, 2.0, 3.0],
                'entity_id': ['A', 'B', 'C']
            })

            # 3-class probabilities
            predictions_array = np.array([
                [0.2, 0.3, 0.5],
                [0.6, 0.2, 0.2],
                [0.1, 0.4, 0.5]
            ])

            data_interface = {
                'entity_columns': ['entity_id'],
                'feature_columns': ['feature_0']
            }

            result_df = format_predictions(
                predictions_result=predictions_array,
                original_df=original_df,
                data_interface=data_interface
            )

            # Validate multiclass probability columns
            assert 'prob_class_0' in result_df.columns
            assert 'prob_class_1' in result_df.columns
            assert 'prob_class_2' in result_df.columns
            assert 'entity_id' in result_df.columns
            assert 'feature_0' not in result_df.columns

    def test_process_template_file_jinja(self, component_test_context):
        """Jinja 템플릿 파일 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Create test template file
            template_content = """
            SELECT * FROM table_{{ table_name }}
            WHERE date >= '{{ start_date }}'
            AND value > {{ threshold }}
            """

            template_path = ctx.temp_dir / "test_template.sql.j2"
            template_path.write_text(template_content)

            context_params = {
                'table_name': 'users',
                'start_date': '2023-01-01',
                'threshold': 100
            }

            # Test template processing
            rendered_sql = process_template_file(
                template_path=str(template_path),
                context_params=context_params,
                console=RichConsoleManager()
            )

            # Validate rendered content
            assert 'table_users' in rendered_sql
            assert "'2023-01-01'" in rendered_sql
            assert '> 100' in rendered_sql

    def test_load_inference_data_with_data_path(self, component_test_context):
        """CLI data_path를 사용한 추론 데이터 로드 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock model (not needed for this path)
            mock_model = Mock()

            # Test load_inference_data with data_path
            loaded_df = load_inference_data(
                data_adapter=ctx.adapter,
                data_path=ctx.data_path,
                model=mock_model,
                run_id="test_run",
                context_params={},
                console=RichConsoleManager()
            )

            # Validate loaded data
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) > 0


class TestDataIOHelperFunctions:
    """Helper functions 단위 테스트"""

    def test_format_multiclass_probabilities_binary(self):
        """Binary classification 확률 포맷팅 헬퍼 테스트"""
        original_df = pd.DataFrame({'dummy': [1, 2, 3]})
        predictions_array = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

        result_df = _format_multiclass_probabilities(predictions_array, original_df)

        assert 'prob_positive' in result_df.columns
        assert 'prob_negative' in result_df.columns
        assert len(result_df) == len(original_df)
        assert result_df.index.equals(original_df.index)

    def test_format_multiclass_probabilities_three_class(self):
        """3-class classification 확률 포맷팅 헬퍼 테스트"""
        original_df = pd.DataFrame({'dummy': [1, 2]})
        predictions_array = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])

        result_df = _format_multiclass_probabilities(predictions_array, original_df)

        assert 'prob_class_0' in result_df.columns
        assert 'prob_class_1' in result_df.columns
        assert 'prob_class_2' in result_df.columns
        assert len(result_df) == len(original_df)

    def test_detect_probability_predictions_positive(self):
        """확률 예측 감지 테스트 - positive case"""
        prob_df = pd.DataFrame({
            'prob_positive': [0.7, 0.8],
            'prob_negative': [0.3, 0.2],
            'other_column': [1, 2]
        })

        assert _detect_probability_predictions(prob_df) is True

    def test_detect_probability_predictions_negative(self):
        """확률 예측 감지 테스트 - negative case"""
        non_prob_df = pd.DataFrame({
            'prediction': [0, 1],
            'score': [0.7, 0.8],
            'other_column': [1, 2]
        })

        assert _detect_probability_predictions(non_prob_df) is False


class TestDataIOErrorHandling:
    """Error handling 및 예외 상황 테스트"""

    def test_save_output_no_config(self, component_test_context):
        """Output 설정이 없는 경우 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_data = pd.DataFrame({'prediction': [0, 1]})

            # Remove output config
            ctx.settings.config.output = None

            # Should not raise exception, just skip save
            save_output(
                df=test_data,
                settings=ctx.settings,
                output_type="inference",
                factory=ctx.factory,
                run_id="test_run"
            )

    def test_process_template_file_not_found(self, component_test_context):
        """존재하지 않는 템플릿 파일 테스트"""
        with component_test_context.classification_stack() as ctx:
            with pytest.raises(FileNotFoundError):
                process_template_file(
                    template_path="nonexistent_template.sql.j2",
                    context_params={'param': 'value'}
                )

    def test_format_predictions_edge_cases(self, component_test_context):
        """format_predictions 엣지 케이스 테스트"""
        with component_test_context.classification_stack() as ctx:
            original_df = pd.DataFrame({'feature': [1, 2, 3]})

            # Test scalar prediction
            scalar_result = format_predictions(
                predictions_result=0.75,
                original_df=original_df
            )
            assert 'prediction' in scalar_result.columns
            assert len(scalar_result) == len(original_df)

            # Test empty DataFrame
            empty_result = format_predictions(
                predictions_result=pd.DataFrame(),
                original_df=original_df
            )
            assert len(empty_result) == 0


class TestDataIOIntegration:
    """Integration 시나리오 테스트"""

    def test_end_to_end_prediction_pipeline(self, component_test_context):
        """End-to-end 예측 파이프라인 테스트"""
        with component_test_context.classification_stack() as ctx:
            # 1. Load data
            raw_data = load_data(
                data_adapter=ctx.adapter,
                data_source=ctx.data_path,
                console=RichConsoleManager()
            )

            # 2. Create mock predictions
            mock_predictions = np.array([[0.3, 0.7], [0.8, 0.2]])[:len(raw_data)]

            # 3. Format predictions
            data_interface = {
                'entity_columns': [],
                'feature_columns': [col for col in raw_data.columns if col.startswith('feature_')]
            }

            formatted_result = format_predictions(
                predictions_result=mock_predictions,
                original_df=raw_data,
                data_interface=data_interface
            )

            # 4. Validate end-to-end flow
            assert len(formatted_result) == len(raw_data)
            assert 'prob_positive' in formatted_result.columns
            assert 'prob_negative' in formatted_result.columns

            # Features should be excluded
            feature_cols_in_result = [col for col in formatted_result.columns if col.startswith('feature_')]
            assert len(feature_cols_in_result) == 0