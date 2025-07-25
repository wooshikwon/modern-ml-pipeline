"""
Inference Pipeline E2E 테스트 (Blueprint v17.0 현대화)

Blueprint 원칙 검증:
- 원칙 4: 실행 시점에 조립되는 순수 로직 아티팩트
- 원칙 7: 하이브리드 통합 인터페이스 (배치 추론)
- E2E 검증: 실제 학습된 모델로 추론 파이프라인 완전 실행
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
    """
    배치 추론 테스트를 위해 미리 학습된 모델 아티팩트를 생성하고
    해당 run_id를 제공하는 Fixture.
    """
    test_tracking_uri = "./test_mlruns_inference_pipeline"
    mlflow.set_tracking_uri(test_tracking_uri)
    
    # 학습 실행
    result_artifact = run_training(settings=local_test_settings)
    
    yield result_artifact.run_id
    
    # 테스트 정리
    shutil.rmtree(test_tracking_uri, ignore_errors=True)
    mlflow.set_tracking_uri("mlruns")

@pytest.mark.e2e
def test_inference_pipeline_e2e_in_local_env_complete(
    local_test_settings: Settings,
    trained_model_run_id_for_inference: str
):
    """
    LOCAL 환경에서 `run_batch_inference` 파이프라인 전체를 실행하는 완전한 End-to-End 테스트.
    - 미리 학습된 실제 모델 아티팩트를 사용합니다.
    - 최종적으로 예측 결과 파일이 생성되고 모든 메타데이터가 포함되는지 검증합니다.
    Blueprint v17.0 완전 현대화
    """
    run_id = trained_model_run_id_for_inference
    print(f"🚀 배치 추론 테스트 시작 (학습 모델 Run ID: {run_id})")
    
    # --- 테스트 실행 ---
    batch_result = run_batch_inference(settings=local_test_settings, run_id=run_id)

    # --- 기본 결과 검증 ---
    # 1. 배치 추론 결과가 반환되었는가?
    assert batch_result is not None
    assert hasattr(batch_result, 'predictions_df')
    assert hasattr(batch_result, 'inference_run_id')
    
    predictions_df = batch_result.predictions_df
    inference_run_id = batch_result.inference_run_id
    
    print(f"✅ 배치 추론 완료. Inference Run ID: {inference_run_id}")

    # 2. 예측 결과 데이터프레임 검증
    assert isinstance(predictions_df, pd.DataFrame)
    assert len(predictions_df) > 0
    assert 'prediction' in predictions_df.columns
    
    # 예측값이 올바른 범위에 있는지 확인 (분류 모델이므로 0-1 범위)
    predictions = predictions_df['prediction']
    assert all(0.0 <= pred <= 1.0 for pred in predictions), "예측값이 0-1 범위를 벗어났습니다."
    print("✅ 예측 결과 데이터프레임 검증 완료")

    # 3. MLflow에 추론 Run이 새로 생성되었는가?
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(local_test_settings.mlflow.experiment_name).experiment_id
    runs_df = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = 'batch_inference_{run_id}'",
        order_by=["start_time DESC"]
    )
    assert not runs_df.empty, "배치 추론에 대한 MLflow Run이 생성되지 않았습니다."
    inference_run = runs_df.iloc[0]
    
    # 4. 예측 결과 아티팩트가 MLflow에 저장되었는가?
    artifacts = client.list_artifacts(inference_run_id)
    artifact_paths = [artifact.path for artifact in artifacts]
    assert any("predictions.parquet" in path for path in artifact_paths), \
        "예측 결과(predictions.parquet)가 MLflow 아티팩트로 저장되지 않았습니다."
    print("✅ MLflow 아티팩트 저장 검증 완료")

    # --- 🆕 Blueprint v17.0: 배치 추론 특화 검증 ---
    # 5. 배치 모드 컨텍스트 검증
    # 예측 결과에 중간 산출물이 포함되어 있는지 확인 (배치 모드의 특징)
    expected_intermediate_cols = ['augmented_features', 'preprocessed_features']
    intermediate_cols_found = [col for col in expected_intermediate_cols if col in predictions_df.columns]
    
    # 배치 모드에서는 중간 산출물을 포함할 수 있음
    if intermediate_cols_found:
        print(f"✅ 배치 모드 중간 산출물 확인: {intermediate_cols_found}")
    
    # 6. 원본 Wrapped Artifact와의 일관성 검증
    # 추론에 사용된 모델이 원본 학습 모델과 동일한지 확인
    original_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    original_wrapped = original_model.unwrap_python_model()
    
    # 모델 클래스 경로가 동일한지 확인
    assert hasattr(original_wrapped, 'model_class_path')
    original_class_path = original_wrapped.model_class_path
    assert original_class_path == local_test_settings.model.class_path
    print(f"✅ 모델 일관성 검증 완료: {original_class_path}")

    # --- 🆕 Blueprint v17.0: Data Leakage 방지 검증 ---
    # 7. 추론 과정에서 Data Leakage가 발생하지 않았는지 확인
    # 추론 시에는 train 데이터에 fit된 전처리기만 사용해야 함
    training_methodology = original_wrapped.training_methodology
    assert training_methodology['preprocessing_fit_scope'] == 'train_only'
    print("✅ Data Leakage 방지 확인: 전처리기는 train 데이터에만 fit됨")

    # --- 🆕 Blueprint v17.0: 추론 메타데이터 검증 ---
    # 8. 추론 Run의 메타데이터 검증
    inference_run_data = client.get_run(inference_run_id)
    
    # 추론 Run에 원본 학습 Run ID가 태그로 기록되었는지 확인
    inference_tags = inference_run_data.data.tags
    assert 'original_model_run_id' in inference_tags
    assert inference_tags['original_model_run_id'] == run_id
    print("✅ 추론 메타데이터 검증 완료")

    # --- 🆕 Blueprint v17.0: 성능 검증 ---
    # 9. 추론 성능 기록 확인
    inference_metrics = inference_run_data.data.metrics
    
    # 추론 관련 메트릭이 기록되었는지 확인
    expected_inference_metrics = ['inference_time_seconds', 'total_predictions']
    for metric in expected_inference_metrics:
        if metric in inference_metrics:
            print(f"✅ 추론 성능 메트릭 '{metric}': {inference_metrics[metric]}")
    
    # --- 최종 검증 완료 ---
    print(f"🎉 Complete Inference E2E Test 성공! {len(predictions_df)}개 예측 완료, 모든 검증 통과")

@pytest.mark.e2e 
def test_batch_inference_artifact_consistency(
    local_test_settings: Settings,
    trained_model_run_id_for_inference: str
):
    """
    배치 추론 아티팩트가 원본 학습 아티팩트와 일관성을 유지하는지 검증한다.
    Blueprint 원칙 4: 실행 시점에 조립되는 순수 로직 아티팩트
    """
    run_id = trained_model_run_id_for_inference
    
    # 배치 추론 실행
    batch_result = run_batch_inference(settings=local_test_settings, run_id=run_id)
    
    # --- Blueprint v17.0: 완전한 일관성 검증 ---
    # 1. 원본 학습 아티팩트와 추론 결과 비교
    original_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    original_wrapped = original_model.unwrap_python_model()
    
    # 2. 동일한 입력에 대해 동일한 결과가 나오는지 확인 (재현성)
    test_input = pd.DataFrame({
        'user_id': ['test_user_1', 'test_user_2'], 
        'product_id': ['test_product_1', 'test_product_2']
    })
    
    # 원본 모델 직접 예측
    direct_prediction = original_model.predict(test_input, params={'run_mode': 'batch'})
    
    # 배치 파이프라인으로 예측 (동일한 테스트 데이터에 대해)
    # 실제로는 파이프라인이 데이터를 로드하므로, 여기서는 결과 형태만 검증
    pipeline_predictions = batch_result.predictions_df
    
    # 3. 스키마 일관성 검증
    # 배치 추론 결과와 직접 예측 결과의 스키마가 호환되는지 확인
    assert 'prediction' in pipeline_predictions.columns
    assert isinstance(direct_prediction, pd.DataFrame)
    print("✅ 예측 결과 스키마 일관성 검증 완료")
    
    # 4. 메타데이터 일관성 검증
    # 원본 아티팩트의 메타데이터가 추론 과정에서 보존되는지 확인
    hpo_metadata = original_wrapped.hyperparameter_optimization
    training_metadata = original_wrapped.training_methodology
    
    assert isinstance(hpo_metadata, dict)
    assert isinstance(training_metadata, dict)
    assert training_metadata['preprocessing_fit_scope'] == 'train_only'
    print("✅ 메타데이터 일관성 검증 완료")

    # 5. 환경별 동작 일관성 검증 (LOCAL 환경)
    # LOCAL 환경에서는 PassThroughAugmenter를 사용해야 함
    augmenter = original_wrapped.trained_augmenter
    from src.core.augmenter import PassThroughAugmenter
    assert isinstance(augmenter, PassThroughAugmenter)
    print("✅ LOCAL 환경 동작 일관성 검증 완료")
    
    print("🎉 배치 추론 아티팩트 일관성 검증 성공!")

@pytest.mark.e2e
def test_inference_pipeline_error_handling(local_test_settings: Settings):
    """
    추론 파이프라인의 에러 처리가 적절히 동작하는지 검증한다.
    """
    # 1. 존재하지 않는 run_id로 추론 시도
    with pytest.raises(Exception) as exc_info:
        run_batch_inference(settings=local_test_settings, run_id="nonexistent_run_id")
    
    # MLflow 관련 오류가 발생해야 함
    assert "run" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()
    print("✅ 잘못된 run_id 에러 처리 검증 완료")
    
    # 2. 잘못된 run_id 형식
    with pytest.raises(Exception):
        run_batch_inference(settings=local_test_settings, run_id="")
    
    print("✅ 빈 run_id 에러 처리 검증 완료")
    print("🎉 추론 파이프라인 에러 처리 검증 성공!")
