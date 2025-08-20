"""
MLflow Graceful Degradation 테스트

M01-1: MlflowSettings.with_fallback() 구현
M01-2: 환경 변수 기반 자동 설정 구현
서버 연결 테스트 → 파일 모드 fallback 로직 검증
"""

import pytest
from unittest.mock import patch, MagicMock
import requests
import os
from src.settings._config_schema import MlflowSettings


class TestMlflowGracefulDegradation:
    """MLflow Graceful Degradation 테스트"""

    def test_with_fallback_server_available_should_use_server_mode(self):
        """서버가 정상 응답하면 서버 모드 사용"""
        # Given: MLflow 서버가 정상 응답
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # When: fallback 설정 시도
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002",
                experiment_name="test-experiment",
                fallback_uri="./mlruns_fallback"
            )
            
            # Then: 서버 URI 사용
            assert result.tracking_uri == "http://localhost:5002"
            assert result.experiment_name == "test-experiment"
            mock_get.assert_called_once_with("http://localhost:5002/health", timeout=5)

    def test_with_fallback_server_unavailable_should_use_fallback_mode(self):
        """서버가 응답하지 않으면 폴백 모드 사용"""
        # Given: MLflow 서버가 응답하지 않음
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")
            
            # When: fallback 설정 시도
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002",
                experiment_name="test-experiment", 
                fallback_uri="./mlruns_fallback"
            )
            
            # Then: 폴백 URI 사용
            assert result.tracking_uri == "./mlruns_fallback"
            assert result.experiment_name == "test-experiment"

    def test_with_fallback_server_timeout_should_use_fallback_mode(self):
        """서버 타임아웃 시 폴백 모드 사용"""
        # Given: MLflow 서버가 타임아웃
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timeout")
            
            # When: fallback 설정 시도
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002",
                experiment_name="test-experiment",
                fallback_uri="./mlruns_fallback"
            )
            
            # Then: 폴백 URI 사용
            assert result.tracking_uri == "./mlruns_fallback"

    def test_with_fallback_server_error_response_should_use_fallback_mode(self):
        """서버가 에러 응답하면 폴백 모드 사용"""
        # Given: MLflow 서버가 500 에러 응답
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            # When: fallback 설정 시도  
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002",
                experiment_name="test-experiment",
                fallback_uri="./mlruns_fallback"
            )
            
            # Then: 폴백 URI 사용
            assert result.tracking_uri == "./mlruns_fallback"

    def test_with_fallback_without_fallback_uri_should_use_default_fallback(self):
        """폴백 URI 미지정 시 기본 폴백 사용"""
        # Given: 서버 불가능, 폴백 URI 없음
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")
            
            # When: 폴백 URI 없이 설정
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002", 
                experiment_name="test-experiment"
                # fallback_uri 없음
            )
            
            # Then: 기본 폴백 URI 사용
            assert result.tracking_uri == "./mlruns"  # 기본값

    def test_with_fallback_custom_timeout_should_use_custom_timeout(self):
        """커스텀 타임아웃 설정 테스트"""
        # Given: 커스텀 타임아웃 설정
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # When: 커스텀 타임아웃으로 설정
            result = MlflowSettings.with_fallback(
                server_uri="http://localhost:5002",
                experiment_name="test-experiment",
                timeout=10
            )
            
            # Then: 커스텀 타임아웃으로 요청
            mock_get.assert_called_once_with("http://localhost:5002/health", timeout=10)


class TestMlflowEnvironmentAutoDetection:
    """M01-2: 환경 변수 기반 자동 설정 테스트"""

    def test_auto_detect_with_env_var_should_use_server_mode(self):
        """MLFLOW_TRACKING_URI 환경변수 있으면 서버 모드 사용"""
        # Given: MLFLOW_TRACKING_URI 환경변수 설정 + 서버 정상
        with patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://localhost:5002'}):
            with patch('requests.get') as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_get.return_value = mock_response
                
                # When: 자동 감지 시도
                result = MlflowSettings.auto_detect(experiment_name="test-experiment")
                
                # Then: 환경변수의 서버 URI 사용
                assert result.tracking_uri == "http://localhost:5002"
                assert result.experiment_name == "test-experiment"
                mock_get.assert_called_once_with("http://localhost:5002/health", timeout=5)

    def test_auto_detect_with_env_var_server_down_should_use_fallback(self):
        """MLFLOW_TRACKING_URI 있지만 서버 다운시 폴백 모드"""
        # Given: MLFLOW_TRACKING_URI 환경변수 설정 + 서버 다운
        with patch.dict(os.environ, {'MLFLOW_TRACKING_URI': 'http://localhost:5002'}):
            with patch('requests.get') as mock_get:
                mock_get.side_effect = requests.ConnectionError("Connection refused")
                
                # When: 자동 감지 시도
                result = MlflowSettings.auto_detect(experiment_name="test-experiment")
                
                # Then: 폴백 모드로 전환
                assert result.tracking_uri == "./mlruns"
                assert result.experiment_name == "test-experiment"

    def test_auto_detect_without_env_var_should_use_fallback_mode(self):
        """MLFLOW_TRACKING_URI 환경변수 없으면 바로 폴백 모드"""
        # Given: MLFLOW_TRACKING_URI 환경변수 없음
        with patch.dict(os.environ, {}, clear=True):
            with patch('requests.get') as mock_get:
                # When: 자동 감지 시도
                result = MlflowSettings.auto_detect(experiment_name="test-experiment")
                
                # Then: 환경변수 없어서 바로 폴백 모드, 서버 테스트도 안함
                assert result.tracking_uri == "./mlruns"
                assert result.experiment_name == "test-experiment"
                mock_get.assert_not_called()  # 서버 체크 안함

    def test_auto_detect_with_custom_fallback_should_use_custom_fallback(self):
        """환경변수 없고 커스텀 폴백 지정시 커스텀 폴백 사용"""
        # Given: MLFLOW_TRACKING_URI 없음, 커스텀 폴백 지정
        with patch.dict(os.environ, {}, clear=True):
            # When: 커스텀 폴백으로 자동 감지
            result = MlflowSettings.auto_detect(
                experiment_name="test-experiment",
                fallback_uri="./custom_mlruns"
            )
            
            # Then: 커스텀 폴백 URI 사용
            assert result.tracking_uri == "./custom_mlruns"

    def test_auto_detect_with_empty_env_var_should_use_fallback_mode(self):
        """MLFLOW_TRACKING_URI가 빈 문자열이면 폴백 모드"""
        # Given: MLFLOW_TRACKING_URI가 빈 문자열
        with patch.dict(os.environ, {'MLFLOW_TRACKING_URI': ''}):
            with patch('requests.get') as mock_get:
                # When: 자동 감지 시도
                result = MlflowSettings.auto_detect(experiment_name="test-experiment")
                
                # Then: 빈 문자열이므로 폴백 모드
                assert result.tracking_uri == "./mlruns"
                mock_get.assert_not_called()