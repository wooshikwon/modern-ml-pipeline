
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.settings.settings import Settings
from src.pipelines.inference_pipeline import run_batch_inference

@pytest.mark.integration
@patch("src.pipelines.inference_pipeline.mlflow_utils.download_artifact")
@patch("src.pipelines.inference_pipeline.mlflow.pyfunc.load_model")
@patch("src.pipelines.inference_pipeline.joblib.load")
@patch("src.pipelines.inference_pipeline.get_dataset_loader")
@patch("src.pipelines.inference_pipeline.get_augmenter")
@patch("src.pipelines.inference_pipeline.artifact_utils.save_dataset")
def test_run_batch_inference_pipeline_flow(
    mock_save_dataset: MagicMock,
    mock_get_augmenter: MagicMock,
    mock_get_loader: MagicMock,
    mock_joblib_load: MagicMock,
    mock_load_pyfunc: MagicMock,
    mock_download_artifact: MagicMock,
    xgboost_settings: Settings,
):
    """
    배치 추론 파이프라인이 전체적으로 올바르게 실행되는지,
    특히 개별 아티팩트를 다운로드하고 재조립하는 과정을 검증하는 통합 테스트.
    """
    # --- Mock 객체 설정 ---
    # 1. 아티팩트 로드 관련 Mock
    mock_download_artifact.return_value = "mock/path/to/artifact"
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = pd.DataFrame({'feature_a': [1, 2]})
    mock_joblib_load.return_value = mock_preprocessor
    
    mock_raw_model = MagicMock()
    mock_raw_model.predict.return_value = pd.Series([0.1, 0.2])
    mock_load_pyfunc.return_value = mock_raw_model

    # 2. 데이터 처리 관련 Mock
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = pd.DataFrame({'id': [1, 2]})
    mock_get_loader.return_value = mock_loader_instance

    mock_augmenter_instance = MagicMock()
    mock_augmenter_instance.augment.return_value = pd.DataFrame({'id': [1, 2], 'feature_a': [1, 2]})
    mock_get_augmenter.return_value = mock_augmenter_instance

    # --- 테스트 실행 ---
    run_batch_inference(
        settings=xgboost_settings,
        model_name="xgboost_x_learner",
        run_id="test_run_id",
        context_params={}
    )

    # --- 검증 ---
    # 1. 아티팩트 다운로드가 올바르게 호출되었는가?
    mock_download_artifact.assert_called_once_with("test_run_id", "preprocessor", xgboost_settings)
    mock_load_pyfunc.assert_called_once_with("runs:/test_run_id/model")

    # 2. 데이터 로더와 Augmenter가 호출되었는가?
    mock_get_loader.assert_called_once()
    mock_get_augmenter.assert_called_once()

    # 3. 전처리기와 모델이 호출되었는가?
    mock_preprocessor.transform.assert_called_once()
    mock_raw_model.predict.assert_called_once()

    # 4. 최종 결과가 저장되었는가?
    mock_save_dataset.assert_called()
    # 'prediction_results' 저장이 호출되었는지 확인
    assert any(
        call.args[1] == 'prediction_results' for call in mock_save_dataset.call_args_list
    )
