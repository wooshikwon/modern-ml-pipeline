import pytest
from unittest.mock import patch, MagicMock

from src.settings.settings import Settings
from src.pipelines.train_pipeline import run_training

@pytest.mark.integration
@patch("src.pipelines.train_pipeline.get_dataset_loader")
@patch("src.pipelines.train_pipeline.Trainer")
@patch("src.pipelines.train_pipeline.mlflow")
def test_run_training_pipeline_flow(
    mock_mlflow: MagicMock,
    mock_trainer: MagicMock,
    mock_get_loader: MagicMock,
    xgboost_settings: Settings
):
    """
    학습 파이프라인(run_training)이 전체적으로 올바르게 실행되는지,
    주요 함수들이 예상대로 호출되는지 검증하는 통합 테스트.
    (실제 학습이나 데이터 로딩은 Mock 객체로 대체)
    """
    # --- Mock 객체 설정 ---
    # get_dataset_loader가 Mock Loader를 반환하도록 설정
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = MagicMock()  # 빈 DataFrame 모의 객체
    mock_get_loader.return_value = mock_loader_instance

    # Trainer가 Mock 객체들을 반환하도록 설정
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = (MagicMock(), MagicMock(), MagicMock(), {"metrics": {"test_metric": 1.0}})
    mock_trainer.return_value = mock_trainer_instance

    # --- 테스트 실행 ---
    run_training(settings=xgboost_settings)

    # --- 검증 ---
    # 1. MLflow가 올바르게 설정되고 실행되었는가?
    mock_mlflow.set_tracking_uri.assert_called_once_with(xgboost_settings.mlflow.tracking_uri)
    mock_mlflow.set_experiment.assert_called_once_with(xgboost_settings.mlflow.experiment_name)
    mock_mlflow.start_run.assert_called_once()
    
    # 2. 데이터 로더가 호출되었는가?
    mock_get_loader.assert_called_once()
    mock_loader_instance.load.assert_called_once()

    # 3. Trainer가 생성되고 train 메서드가 호출되었는가?
    mock_trainer.assert_called_once_with(settings=xgboost_settings)
    mock_trainer_instance.train.assert_called_once()

    # 4. 결과가 MLflow에 로깅되었는가?
    mock_mlflow.log_params.assert_called()
    mock_mlflow.log_metrics.assert_called_with({"test_metric": 1.0})
    
    # 5. Pyfunc 모델이 올바르게 로깅되었는가?
    mock_mlflow.pyfunc.log_model.assert_called_once()
    log_model_args = mock_mlflow.pyfunc.log_model.call_args[1]
    assert log_model_args['artifact_path'] == xgboost_settings.model.name
    assert log_model_args['registered_model_name'] == xgboost_settings.model.name
    assert 'python_model' in log_model_args

    mock_mlflow.set_tag.assert_called_with("status", "success")
