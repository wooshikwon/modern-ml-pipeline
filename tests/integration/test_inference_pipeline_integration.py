import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.pipelines.inference_pipeline import run_inference_pipeline
from src.pipelines.train_pipeline import run_train_pipeline


class TestInferencePipeline:
    def setup_method(self):
        """각 테스트 메서드 시작 전 MLflow 상태 초기화"""
        # 기존 active run이 있으면 강제 종료 (정책 준수)
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

    def test_inference_happy_path_from_trained_run(self, mlflow_test_context):
        """케이스 A: Happy path - 직전 학습 run의 모델로 배치 추론 성공"""
        with mlflow_test_context.for_classification("infer_happy") as ctx:
            # MLflow tracking URI 명시적 설정 (정책 준수)
            mlflow.set_tracking_uri(ctx.mlflow_uri)

            train_result = run_train_pipeline(ctx.settings)
            assert train_result is not None and hasattr(train_result, "run_id")

            # 훈련 후 MLflow run 종료하여 추론에서 충돌 방지
            mlflow.end_run()

            # 동일 데이터로 배치 추론 (스모크)
            inference_result = run_inference_pipeline(
                run_id=train_result.run_id, data_path=str(ctx.data_path)
            )
            # 예외 없이 완료되면 성공
            assert inference_result is not None
            assert hasattr(inference_result, "run_id")
            assert hasattr(inference_result, "model_uri")
            assert hasattr(inference_result, "prediction_count")

            # MLflow에 추론 run이 기록되었는지 확인
            client = MlflowClient(tracking_uri=ctx.mlflow_uri)
            experiment = client.get_experiment_by_name(ctx.experiment_name)
            if experiment:
                inference_runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.mlflow.runName LIKE 'batch_inference_%'",
                )
                assert len(inference_runs) > 0

    def test_inference_with_nonexistent_run_id_fails_meaningfully(self, mlflow_test_context):
        """케이스 B: 존재하지 않는 run_id - 의미있는 실패"""
        with mlflow_test_context.for_classification("infer_missing_run") as ctx:
            # MLflow tracking URI 명시적 설정 (정책 준수)
            mlflow.set_tracking_uri(ctx.mlflow_uri)

            # 존재하지 않는 run id로 추론 시도 → 의미있는 실패
            with pytest.raises(Exception) as exc_info:
                result = run_inference_pipeline(
                    run_id="nonexistent_run_id_12345", data_path=str(ctx.data_path)
                )

            # 에러 메시지에 Run not found 관련 내용 포함
            error_msg = str(exc_info.value)
            assert any(keyword in error_msg.lower() for keyword in ["run", "not found", "exist"])

    def test_inference_with_schema_mismatch_fails_gracefully(
        self, mlflow_test_context, isolated_temp_directory
    ):
        """케이스 C: 스키마 불일치 - 입력 컬럼 누락/추가 시 방어 동작"""
        with mlflow_test_context.for_classification("infer_schema_test") as ctx:
            # MLflow tracking URI 명시적 설정 (정책 준수)
            mlflow.set_tracking_uri(ctx.mlflow_uri)

            # 정상 모델 학습
            train_result = run_train_pipeline(ctx.settings)
            assert train_result is not None

            # 훈련 후 MLflow run 완전 종료 보장 (정책 준수)
            while mlflow.active_run():
                mlflow.end_run()

            # 다른 스키마의 CSV 파일 생성 (컬럼 누락)
            mismatched_data = pd.DataFrame(
                {"different_column": [1, 2, 3], "another_col": [0.1, 0.2, 0.3]}
            )
            mismatched_path = isolated_temp_directory / "mismatched_data.csv"
            mismatched_data.to_csv(mismatched_path, index=False)

            # 스키마 불일치 데이터로 추론 시도
            with pytest.raises(Exception) as exc_info:
                result = run_inference_pipeline(
                    run_id=train_result.run_id, data_path=str(mismatched_path)
                )

            # 에러가 스키마/컬럼/시그니처 관련인지 확인
            error_msg = str(exc_info.value)
            assert any(
                keyword in error_msg.lower()
                for keyword in ["schema", "column", "signature", "feature", "input"]
            )

    def test_inference_without_data_path_fails_for_csv_models(self, mlflow_test_context):
        """CSV로 학습한 모델에서 데이터 경로 없이 추론 시 적절한 에러 발생"""
        with mlflow_test_context.for_classification("infer_csv_model_no_path") as ctx:
            # MLflow tracking URI 명시적 설정 (정책 준수)
            mlflow.set_tracking_uri(ctx.mlflow_uri)

            train_result = run_train_pipeline(ctx.settings)
            assert train_result is not None

            # 훈련 후 MLflow run 완전 종료 보장 (정책 준수)
            while mlflow.active_run():
                mlflow.end_run()

            # CSV로 학습한 모델에서 data_path 없이 추론 시도 → 의미있는 실패
            with pytest.raises(ValueError) as exc_info:
                result = run_inference_pipeline(run_id=train_result.run_id, data_path=None)

            # 에러 메시지에 data_path 필요성 관련 내용 포함
            error_msg = str(exc_info.value)
            assert any(keyword in error_msg.lower() for keyword in ["data_path", "csv", "필요"])

    def test_inference_with_context_params_on_static_sql_fails_with_security_error(
        self, mlflow_test_context
    ):
        """정적 SQL 모델에 context_params 사용 시 보안 에러"""
        with mlflow_test_context.for_classification("infer_security_test") as ctx:
            # MLflow tracking URI 명시적 설정 (정책 준수)
            mlflow.set_tracking_uri(ctx.mlflow_uri)

            train_result = run_train_pipeline(ctx.settings)
            assert train_result is not None

            # 훈련 후 MLflow run 완전 종료 보장 (정책 준수)
            while mlflow.active_run():
                mlflow.end_run()

            # 정적 SQL 모델에 context_params 전달 시 보안 에러
            context_params = {"date": "2023-01-01"}
            with pytest.raises(ValueError) as exc_info:
                result = run_inference_pipeline(
                    run_id=train_result.run_id, data_path=None, context_params=context_params
                )

            error_msg = str(exc_info.value)
            assert "보안" in error_msg or "security" in error_msg.lower()
