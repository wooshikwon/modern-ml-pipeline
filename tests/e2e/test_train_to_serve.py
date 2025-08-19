"""
Training Pipeline 테스트 (최신 스펙)
- LOCAL 환경에서 run_training 실행 후 기본 아티팩트/메타데이터 최소 검증
"""

import pytest
import mlflow
import shutil
from pathlib import Path

from src.settings import Settings
from src.pipelines.train_pipeline import run_training


@pytest.mark.e2e
def test_train_pipeline_minimal_checks_in_local(local_test_settings: Settings):
    test_tracking_uri = "./test_mlruns_train_pipeline"
    mlflow.set_tracking_uri(test_tracking_uri)

    try:
        result_artifact = run_training(settings=local_test_settings)

        # 기본 결과 검증
        assert result_artifact is not None
        assert result_artifact.run_id is not None

        # 모델 아티팩트 저장 확인
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(result_artifact.run_id)
        artifact_path = Path(test_tracking_uri) / run.info.experiment_id / run.info.run_id / "artifacts" / "model"
        assert artifact_path.exists()
        assert (artifact_path / "MLmodel").exists()

        # 최소 메타데이터 확인: Wrapped artifact가 로드되고 training_methodology 존재
        model = mlflow.pyfunc.load_model(f"runs:/{result_artifact.run_id}/model")
        wrapped = model.unwrap_python_model()
        assert hasattr(wrapped, 'training_methodology')
        tm = wrapped.training_methodology
        assert tm.get('preprocessing_fit_scope') == 'train_only'

    finally:
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        mlflow.set_tracking_uri("mlruns")
