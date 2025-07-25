"""
Training Pipeline E2E 테스트 (Blueprint v17.0 현대화)

Blueprint 원칙 검증:
- 원칙 4: 실행 시점에 조립되는 순수 로직 아티팩트
- 원칙 8: 자동화된 HPO + Data Leakage 완전 방지
- E2E 검증: Mock 없는 완전한 파이프라인 실행
"""

import pytest
import mlflow
import shutil
from pathlib import Path

from src.settings import Settings
from src.pipelines.train_pipeline import run_training

@pytest.mark.e2e
def test_train_pipeline_e2e_in_local_env_complete(local_test_settings: Settings):
    """
    LOCAL 환경에서 `run_training` 파이프라인 전체를 실행하는 완전한 End-to-End 테스트.
    Mock 없이 실제 로직을 실행하여 MLflow 아티팩트가 생성되고 모든 메타데이터가 포함되는지 검증한다.
    Blueprint v17.0 완전 현대화
    """
    # 테스트 격리를 위한 임시 MLflow 경로 설정
    test_tracking_uri = "./test_mlruns_train_pipeline"
    mlflow.set_tracking_uri(test_tracking_uri)
    
    try:
        # --- 테스트 실행 ---
        # LOCAL 환경에서는 실제 학습이 매우 빠르므로 Mock 없이 직접 실행
        result_artifact = run_training(settings=local_test_settings)

        # --- 기본 결과 검증 ---
        # 1. 결과 아티팩트가 반환되었는가?
        assert result_artifact is not None
        assert result_artifact.run_id is not None
        print(f"✅ 학습 완료. Run ID: {result_artifact.run_id}")

        # 2. MLflow에 Run이 실제로 생성되었는가?
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(result_artifact.run_id)
        assert run is not None
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.runName"] == local_test_settings.model.computed["run_name"]
        
        # 3. 모델 아티팩트가 실제로 저장되었는가?
        artifact_path = Path(test_tracking_uri) / run.info.experiment_id / run.info.run_id / "artifacts" / "model"
        assert artifact_path.exists()
        assert (artifact_path / "MLmodel").exists()
        assert (artifact_path / "model.pkl").exists()

        # --- 🆕 Blueprint v17.0: 완전한 메타데이터 검증 ---
        # 4. Wrapped Artifact 메타데이터 검증
        model = mlflow.pyfunc.load_model(f"runs:/{result_artifact.run_id}/model")
        wrapped_model = model.unwrap_python_model()
        
        # 4-1. Data Leakage 방지 메타데이터 검증
        assert hasattr(wrapped_model, 'training_methodology')
        tm = wrapped_model.training_methodology
        assert tm['preprocessing_fit_scope'] == 'train_only'
        assert 'train_test_split_method' in tm
        print("✅ Data Leakage 방지 메타데이터 검증 완료")
        
        # 4-2. 하이퍼파라미터 최적화 메타데이터 검증 (LOCAL에서는 비활성화)
        assert hasattr(wrapped_model, 'hyperparameter_optimization')
        hpo = wrapped_model.hyperparameter_optimization
        # LOCAL 환경에서는 HPO가 비활성화되어 있어야 함
        assert not hpo.get('enabled', False)
        print("✅ 하이퍼파라미터 최적화 메타데이터 검증 완료 (비활성화 확인)")
        
        # 4-3. 로직 스냅샷 검증
        assert hasattr(wrapped_model, 'loader_sql_snapshot')
        assert hasattr(wrapped_model, 'augmenter_config_snapshot')
        assert hasattr(wrapped_model, 'model_class_path')
        
        loader_sql = wrapped_model.loader_sql_snapshot
        assert isinstance(loader_sql, str) and len(loader_sql) > 0
        assert wrapped_model.model_class_path == local_test_settings.model.class_path
        print("✅ 로직 스냅샷 검증 완료")

        # --- 🆕 Blueprint v17.0: 학습된 컴포넌트 검증 ---
        # 5. 학습된 컴포넌트들이 올바르게 포함되었는가?
        assert hasattr(wrapped_model, 'trained_model')
        assert hasattr(wrapped_model, 'trained_preprocessor')
        assert hasattr(wrapped_model, 'trained_augmenter')
        
        # 5-1. 학습된 모델 검증
        trained_model = wrapped_model.trained_model
        assert trained_model is not None
        assert hasattr(trained_model, 'predict')  # 예측 메서드 존재
        print("✅ 학습된 모델 검증 완료")
        
        # 5-2. 학습된 전처리기 검증
        trained_preprocessor = wrapped_model.trained_preprocessor
        assert trained_preprocessor is not None
        assert hasattr(trained_preprocessor, '_is_fitted')
        assert trained_preprocessor._is_fitted()  # fit 상태 확인
        print("✅ 학습된 전처리기 검증 완료")

        # --- 🆕 Blueprint v17.0: MLflow 메트릭 검증 ---
        # 6. 학습 메트릭이 기록되었는가?
        run_data = run.data
        assert len(run_data.metrics) > 0
        
        # 기본 분류 메트릭이 기록되었는지 확인
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        recorded_metrics = list(run_data.metrics.keys())
        
        for metric in expected_metrics:
            assert metric in recorded_metrics, f"메트릭 '{metric}'이 기록되지 않았습니다."
            assert 0.0 <= run_data.metrics[metric] <= 1.0, f"메트릭 '{metric}' 값이 범위를 벗어났습니다."
        
        print("✅ MLflow 메트릭 검증 완료")

        # --- 🆕 Blueprint v17.0: 환경별 동작 검증 ---
        # 7. LOCAL 환경 특성 검증
        assert hasattr(wrapped_model, 'trained_augmenter')
        augmenter = wrapped_model.trained_augmenter
        
        # LOCAL 환경에서는 PassThroughAugmenter를 사용해야 함
        from src.core.augmenter import PassThroughAugmenter
        assert isinstance(augmenter, PassThroughAugmenter)
        print("✅ LOCAL 환경 PassThroughAugmenter 사용 확인")

        # --- 최종 검증 완료 ---
        print(f"🎉 Complete E2E Test 성공! 모든 {len(expected_metrics)}개 메트릭과 메타데이터 검증 완료")

    finally:
        # --- 테스트 정리 ---
        # 테스트 완료 후 임시 MLflow 디렉토리 정리
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        # 전역 MLflow URI를 기본값으로 재설정
        mlflow.set_tracking_uri("mlruns")

@pytest.mark.e2e
def test_train_pipeline_wrapped_artifact_completeness(local_test_settings: Settings):
    """
    생성된 Wrapped Artifact가 Blueprint v17.0의 모든 요구사항을 만족하는지 검증한다.
    원칙 4: 실행 시점에 조립되는 순수 로직 아티팩트
    """
    test_tracking_uri = "./test_mlruns_artifact_test"
    mlflow.set_tracking_uri(test_tracking_uri)
    
    try:
        # 학습 실행
        result_artifact = run_training(settings=local_test_settings)
        
        # Wrapped Artifact 로드
        model = mlflow.pyfunc.load_model(f"runs:/{result_artifact.run_id}/model")
        wrapped_model = model.unwrap_python_model()
        
        # --- Blueprint v17.0: 완전한 자기 완결성 검증 ---
        # 1. 모든 필수 속성 존재 확인
        required_attributes = [
            'trained_model', 'trained_preprocessor', 'trained_augmenter',
            'loader_sql_snapshot', 'augmenter_config_snapshot', 'model_class_path',
            'training_methodology', 'hyperparameter_optimization'
        ]
        
        for attr in required_attributes:
            assert hasattr(wrapped_model, attr), f"Wrapped Artifact에 '{attr}' 속성이 누락되었습니다."
        
        # 2. 인프라 의존성 제거 확인 (순수 로직만 포함)
        # Wrapped Artifact는 특정 DB 연결이나 API 키 등을 포함하면 안 됨
        import json
        
        # 설정 관련 민감 정보가 포함되지 않았는지 확인
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'host', 'port', 'connection']
        
        # loader_sql_snapshot은 순수한 SQL이어야 함 (연결 정보 없음)
        sql_snapshot = wrapped_model.loader_sql_snapshot
        for keyword in sensitive_keywords:
            assert keyword.lower() not in sql_snapshot.lower(), \
                f"SQL 스냅샷에 민감한 정보 '{keyword}'가 포함되어 있습니다."
        
        # 3. 재현성 보장 확인
        # 동일한 Wrapped Artifact로 여러 번 예측 시 동일한 결과가 나와야 함
        import pandas as pd
        test_data = pd.DataFrame({
            'user_id': ['test_user_1', 'test_user_2'],
            'product_id': ['test_product_1', 'test_product_2']
        })
        
        # 첫 번째 예측
        prediction1 = model.predict(test_data, params={'run_mode': 'batch'})
        # 두 번째 예측
        prediction2 = model.predict(test_data, params={'run_mode': 'batch'})
        
        # 결과가 동일한지 확인
        pd.testing.assert_frame_equal(prediction1, prediction2)
        print("✅ Wrapped Artifact 재현성 검증 완료")
        
        print("🎉 Wrapped Artifact 완전성 검증 성공!")
        
    finally:
        shutil.rmtree(test_tracking_uri, ignore_errors=True)
        mlflow.set_tracking_uri("mlruns")
