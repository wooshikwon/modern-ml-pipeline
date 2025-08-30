"""서빙 추론 핸들러 테스트 - Mock 기반 통합 테스트

Phase 3.3: _endpoints.py의 핵심 추론 기능들을 Mock을 사용하여 검증합니다.

테스트 전략:
- app_context Mock 처리로 전역 상태 의존성 격리
- MLflow 모델 Mock으로 실제 모델 로딩 없이 추론 로직 검증
- FastAPI HTTP 예외 처리 검증
- Factory 패턴과 일관된 테스트 구조 유지
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from fastapi import HTTPException

from tests.factories.test_data_factory import TestDataFactory


class TestInferenceHandler:
    """추론 요청 처리 핸들러 테스트"""
    
    @patch('src.serving._endpoints.app_context')
    def test_single_prediction_request(self, mock_app_context, test_factories):
        """단일 추론 요청 처리 테스트"""
        # Given: 추론용 데이터와 Mock 모델 설정
        from src.serving._endpoints import predict
        
        # Mock app_context 설정
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({'prediction': [1]})
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs:/test_run/model"
        
        # 테스트 데이터 생성
        inference_data = {"feature1": 1.0, "feature2": 0.5}
        
        # When: 단일 추론 실행
        result = predict(inference_data)
        
        # Then: 추론 결과 검증
        assert result is not None
        assert "prediction" in result
        assert "model_uri" in result
        assert result["model_uri"] == "runs:/test_run/model"
        assert result["prediction"] == 1
        
        # Model.predict 호출 검증
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        assert len(call_args[0][0]) == 1  # 단일 행 DataFrame
        assert call_args[1]["params"]["run_mode"] == "serving"
        assert call_args[1]["params"]["return_intermediate"] is False
    
    @patch('src.serving._endpoints.app_context')
    def test_batch_prediction_request(self, mock_app_context, test_factories):
        """배치 추론 요청 처리 테스트"""
        # Given: 배치 추론 데이터와 Mock 설정
        from src.serving._endpoints import predict_batch
        
        # Mock app_context 설정
        mock_model = MagicMock()
        prediction_results = pd.DataFrame({
            'prediction': [0, 1, 0, 1, 1]
        })
        mock_model.predict.return_value = prediction_results
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs:/batch_test/model"
        
        # Mock BatchPredictionRequest 스키마
        mock_sample = MagicMock()
        mock_sample.model_dump.return_value = {"feature1": 1.0, "feature2": 0.5}
        mock_request = MagicMock()
        mock_request.samples = [mock_sample] * 5
        mock_app_context.BatchPredictionRequest.return_value = mock_request
        
        # 배치 요청 데이터
        batch_request = {
            "samples": [
                {"feature1": 1.0, "feature2": 0.5},
                {"feature1": 2.0, "feature2": 1.0},
                {"feature1": 0.5, "feature2": 0.8},
                {"feature1": 1.5, "feature2": 0.3},
                {"feature1": 2.5, "feature2": 1.2}
            ]
        }
        
        # When: 배치 추론 실행
        result = predict_batch(batch_request)
        
        # Then: 배치 결과 검증
        assert result is not None
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'model_uri') 
        assert hasattr(result, 'sample_count')
        assert result.sample_count == 5
        assert result.model_uri == "runs:/batch_test/model"
        
        # Model.predict 호출 검증
        mock_model.predict.assert_called_once()
        call_args = mock_model.predict.call_args
        assert call_args[1]["params"]["run_mode"] == "serving"
    
    @patch('src.serving._endpoints.app_context')
    def test_prediction_error_handling_empty_samples(self, mock_app_context, test_factories):
        """빈 샘플 입력 시 오류 처리 테스트"""
        # Given: 빈 샘플로 배치 요청
        from src.serving._endpoints import predict_batch
        
        # Mock 설정 - 빈 샘플
        mock_request = MagicMock()
        mock_request.samples = []
        mock_app_context.BatchPredictionRequest.return_value = mock_request
        
        batch_request = {"samples": []}
        
        # When/Then: 빈 샘플 오류 검증
        with pytest.raises(HTTPException) as exc_info:
            predict_batch(batch_request)
        
        assert exc_info.value.status_code == 400
        assert "입력 샘플이 비어있습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_prediction_error_handling_model_failure(self, mock_app_context, test_factories):
        """모델 추론 실패 시 오류 처리 테스트"""
        # Given: 모델 추론 실패 시뮬레이션
        from src.serving._endpoints import predict
        
        # Mock app_context - 모델 추론 실패
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model inference failed")
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs:/failing_model/model"
        
        inference_data = {"feature1": 1.0}
        
        # When/Then: 모델 실패 오류 전파 검증
        with pytest.raises(RuntimeError, match="Model inference failed"):
            predict(inference_data)
    
    @patch('src.serving._endpoints.app_context')
    def test_health_check_healthy_state(self, mock_app_context, test_factories):
        """헬스체크 - 정상 상태 테스트"""
        # Given: 정상 상태의 app_context
        from src.serving._endpoints import health
        
        # Mock app_context - 정상 상태
        mock_model = MagicMock()
        mock_wrapped_model = MagicMock()
        mock_wrapped_model.model_class_path = "sklearn.ensemble.RandomForestClassifier"
        mock_model.unwrap_python_model.return_value = mock_wrapped_model
        
        mock_app_context.model = mock_model
        mock_app_context.settings = MagicMock()  # Settings 객체 존재
        mock_app_context.model_uri = "runs:/healthy_model/model"
        
        # When: 헬스체크 실행
        result = health()
        
        # Then: 정상 응답 검증
        assert result is not None
        assert result.status == "healthy"
        assert result.model_uri == "runs:/healthy_model/model"
        assert result.model_name == "sklearn.ensemble.RandomForestClassifier"
    
    @patch('src.serving._endpoints.app_context')
    def test_health_check_unhealthy_state(self, mock_app_context, test_factories):
        """헬스체크 - 비정상 상태 테스트 (모델 미준비)"""
        # Given: 모델이 준비되지 않은 상태
        from src.serving._endpoints import health
        
        # Mock app_context - 모델 없음
        mock_app_context.model = None
        mock_app_context.settings = MagicMock()
        
        # When/Then: 503 에러 검증
        with pytest.raises(HTTPException) as exc_info:
            health()
        
        assert exc_info.value.status_code == 503
        assert "모델이 준비되지 않았습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_health_check_settings_missing(self, mock_app_context, test_factories):
        """헬스체크 - 설정 누락 상태 테스트"""
        # Given: 설정이 없는 상태
        from src.serving._endpoints import health
        
        # Mock app_context - 설정 없음
        mock_app_context.model = MagicMock()
        mock_app_context.settings = None
        
        # When/Then: 503 에러 검증
        with pytest.raises(HTTPException) as exc_info:
            health()
        
        assert exc_info.value.status_code == 503
        assert "모델이 준비되지 않았습니다" in str(exc_info.value.detail)
    
    @patch('src.serving._endpoints.app_context')
    def test_model_metadata_retrieval(self, mock_app_context, test_factories):
        """모델 메타데이터 조회 테스트"""
        # Given: 완전한 메타데이터를 가진 모델
        from src.serving._endpoints import get_model_metadata
        
        # Mock app_context - 완전한 모델
        mock_model = MagicMock()
        mock_model.model_class_path = "sklearn.ensemble.RandomForestClassifier"
        mock_model.hyperparameter_optimization = {
            "enabled": True,
            "engine": "optuna",
            "best_params": {"n_estimators": 100},
            "best_score": 0.95,
            "total_trials": 50,
            "pruned_trials": 5,
            "optimization_time": "120.5s"
        }
        mock_model.training_methodology = {
            "train_test_split_method": "temporal",
            "train_ratio": 0.8,
            "validation_strategy": "cross_validation",
            "preprocessing_fit_scope": "train_only",
            "random_state": 42
        }
        
        mock_prediction_request = MagicMock()
        mock_prediction_request.model_fields = {"feature1": None, "feature2": None}
        
        mock_app_context.model = mock_model
        mock_app_context.model_uri = "runs:/metadata_test/model"
        mock_app_context.PredictionRequest = mock_prediction_request
        
        # When: 메타데이터 조회
        result = get_model_metadata()
        
        # Then: 메타데이터 검증
        assert result is not None
        assert result.model_uri == "runs:/metadata_test/model"
        assert result.model_class_path == "sklearn.ensemble.RandomForestClassifier"
        
        # HPO 정보 검증
        hpo = result.hyperparameter_optimization
        assert hpo.enabled is True
        assert hpo.engine == "optuna"
        assert hpo.best_params == {"n_estimators": 100}
        assert hpo.best_score == 0.95
        
        # Training 방법론 검증
        tm = result.training_methodology
        assert tm.train_test_split_method == "temporal"
        assert tm.train_ratio == 0.8
        
        # API 스키마 검증
        api_schema = result.api_schema
        assert "input_fields" in api_schema
        assert len(api_schema["input_fields"]) == 2
    
    @patch('src.serving._endpoints.app_context')
    def test_model_metadata_no_model(self, mock_app_context, test_factories):
        """모델 메타데이터 조회 - 모델 없음 테스트"""
        # Given: 모델이 로드되지 않은 상태
        from src.serving._endpoints import get_model_metadata
        
        mock_app_context.model = None
        
        # When/Then: 503 에러 검증
        with pytest.raises(HTTPException) as exc_info:
            get_model_metadata()
        
        assert exc_info.value.status_code == 503
        assert "모델이 로드되지 않았습니다" in str(exc_info.value.detail)