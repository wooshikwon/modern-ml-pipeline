"""
Inference Pipeline 테스트 (최신 스펙)
- 배치 결과 파일명 규칙 검증: preds_{inference_run_id}.parquet
- 레거시 모듈 경로 제거
"""

import pytest
import mlflow
import shutil
from pathlib import Path
import pandas as pd

from src.settings import Settings
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference


@pytest.fixture(scope="module")
def trained_model_run_id_for_inference(local_test_settings: Settings):
    """배치 추론 테스트용 학습 모델 아티팩트 생성 후 run_id 제공."""
    test_tracking_uri = "./test_mlruns_inference_pipeline"
    mlflow.set_tracking_uri(test_tracking_uri)
    result_artifact = run_training(settings=local_test_settings)
    yield result_artifact.run_id
    shutil.rmtree(test_tracking_uri, ignore_errors=True)
    mlflow.set_tracking_uri("mlruns")


@pytest.mark.e2e
def test_inference_pipeline_saves_pred_file_with_expected_name(
    local_test_settings: Settings,
    trained_model_run_id_for_inference: str
):
    run_id = trained_model_run_id_for_inference

    # 실행
    run_batch_inference(settings=local_test_settings, run_id=run_id)

    # 최신 inference run을 조회하여 저장 경로 확인
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(local_test_settings.mlflow.experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = 'batch_inference_{run_id}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert len(runs) == 1
    inference_run = runs[0]
    inference_run_id = inference_run.info.run_id

    # 예측 결과 파일 저장 규칙 검증: preds_{inference_run_id}.parquet
    pred_dir = Path("./local/artifacts")
    # 저장은 settings.artifact_stores['prediction_results'].base_uri를 따름
    # 기본 경로를 settings에서 읽어와 절대경로로 확인
    base_uri = local_test_settings.artifact_stores['prediction_results'].base_uri
    base_path = Path(base_uri.replace("file://", ""))
    expected_file = base_path / f"preds_{inference_run_id}.parquet"
    assert expected_file.exists(), f"예상 파일이 존재하지 않습니다: {expected_file}"

    # 파일 스키마 최소 검증
    df = pd.read_parquet(expected_file)
    assert 'prediction' in df.columns
    assert 'model_run_id' in df.columns
    assert 'inference_run_id' in df.columns
