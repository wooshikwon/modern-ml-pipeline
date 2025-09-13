"""
Logger utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/core/logger.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import os
import logging
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from logging.handlers import TimedRotatingFileHandler

from src.utils.core.logger import setup_logging, logger


class TestLoggerSetupConfiguration:
    """로거 설정 테스트 - Context 클래스 기반"""

    def test_setup_logging_local_environment(self, component_test_context):
        """LOCAL 환경에서 로거 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings object for local environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Mock logging components
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('logging.Formatter') as mock_formatter:

                    mock_handler = Mock()
                    mock_file_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify DEBUG level set
                    mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)

                    # Verify TimedRotatingFileHandler created
                    assert mock_file_handler.called

                    # Verify formatter and handler setup
                    mock_handler.setFormatter.assert_called_once_with(mock_formatter_instance)
                    mock_root_logger.addHandler.assert_called_once_with(mock_handler)

    def test_setup_logging_dev_environment(self, component_test_context):
        """DEV 환경에서 로거 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings object for dev environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "dev"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Mock logging components
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter') as mock_formatter:

                    mock_handler = Mock()
                    mock_stream_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify INFO level set
                    mock_root_logger.setLevel.assert_called_once_with(logging.INFO)

                    # Verify StreamHandler created with stdout
                    mock_stream_handler.assert_called_once_with(sys.stdout)

                    # Verify formatter and handler setup
                    mock_handler.setFormatter.assert_called_once_with(mock_formatter_instance)
                    mock_root_logger.addHandler.assert_called_once_with(mock_handler)

    def test_setup_logging_prod_environment(self, component_test_context):
        """PROD 환경에서 로거 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings object for prod environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "prod"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Mock logging components
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter') as mock_formatter:

                    mock_handler = Mock()
                    mock_stream_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify INFO level set (same as dev)
                    mock_root_logger.setLevel.assert_called_once_with(logging.INFO)

                    # Verify StreamHandler created with stdout
                    mock_stream_handler.assert_called_once_with(sys.stdout)


class TestLoggerHandlerManagement:
    """로거 핸들러 관리 테스트"""

    def test_setup_logging_removes_existing_handlers(self, component_test_context):
        """기존 핸들러 제거 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Mock existing handlers
            mock_handler1 = Mock()
            mock_handler2 = Mock()
            existing_handlers = [mock_handler1, mock_handler2]

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = existing_handlers[:]  # Copy for iteration
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler'), \
                     patch('logging.Formatter'):

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify existing handlers were removed and closed
                    mock_root_logger.removeHandler.assert_any_call(mock_handler1)
                    mock_root_logger.removeHandler.assert_any_call(mock_handler2)
                    mock_handler1.close.assert_called_once()
                    mock_handler2.close.assert_called_once()

    def test_setup_logging_file_handler_configuration(self, component_test_context):
        """파일 핸들러 상세 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings for local environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('pathlib.Path.mkdir') as mock_mkdir:

                    mock_handler = Mock()
                    mock_file_handler.return_value = mock_handler

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify logs directory creation
                    mock_mkdir.assert_called_once_with(exist_ok=True)

                    # Verify TimedRotatingFileHandler configuration
                    assert mock_file_handler.called
                    call_args = mock_file_handler.call_args

                    # Check handler parameters
                    assert 'when' in call_args.kwargs and call_args.kwargs['when'] == "midnight"
                    assert 'backupCount' in call_args.kwargs and call_args.kwargs['backupCount'] == 30
                    assert 'encoding' in call_args.kwargs and call_args.kwargs['encoding'] == "utf-8"


class TestLoggerFormatterConfiguration:
    """로거 포매터 설정 테스트"""

    def test_setup_logging_formatter_pattern(self, component_test_context):
        """로그 포매터 패턴 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler'), \
                     patch('logging.Formatter') as mock_formatter:

                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify formatter created with correct pattern
                    expected_pattern = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    mock_formatter.assert_called_once_with(expected_pattern)

    def test_setup_logging_formatter_applied_to_handler(self, component_test_context):
        """핸들러에 포매터 적용 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "dev"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter') as mock_formatter:

                    mock_handler = Mock()
                    mock_stream_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Verify formatter applied to handler
                    mock_handler.setFormatter.assert_called_once_with(mock_formatter_instance)


class TestSettingsObjectHandling:
    """Settings 객체 처리 테스트"""

    def test_setup_logging_missing_config_attribute(self, component_test_context):
        """config 속성 누락 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings without config attribute
            mock_settings = Mock()
            mock_settings.config = None

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter'):

                    # Test setup_logging - should default to non-local behavior
                    setup_logging(mock_settings)

                    # Should use INFO level (non-local default)
                    mock_root_logger.setLevel.assert_called_once_with(logging.INFO)

                    # Should use StreamHandler
                    assert mock_stream_handler.called

    def test_setup_logging_missing_environment_attribute(self, component_test_context):
        """environment 속성 누락 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with config but no environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_config.environment = None
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter'):

                    # Test setup_logging
                    setup_logging(mock_settings)

                    # Should default to non-local behavior
                    mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
                    assert mock_stream_handler.called

    def test_setup_logging_missing_name_attribute(self, component_test_context):
        """environment.name 속성 누락 시 기본값 사용 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings with environment but no name
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            # Don't set name attribute to test getattr default
            delattr(mock_environment, 'name') if hasattr(mock_environment, 'name') else None
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('logging.Formatter'):

                    # Test setup_logging - should default to "local"
                    setup_logging(mock_settings)

                    # Should use local behavior (DEBUG level, file handler)
                    mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
                    assert mock_file_handler.called


class TestLoggingExceptionHandling:
    """로깅 설정 예외 처리 테스트"""

    def test_setup_logging_handler_creation_error(self, component_test_context):
        """핸들러 생성 오류 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                # Mock TimedRotatingFileHandler to raise exception
                with patch('logging.handlers.TimedRotatingFileHandler',
                          side_effect=Exception("File handler creation failed")):

                    # Should not raise exception (graceful handling expected)
                    try:
                        setup_logging(mock_settings)
                    except Exception as e:
                        pytest.fail(f"setup_logging should handle handler creation errors gracefully: {e}")

    def test_setup_logging_directory_creation_error(self, component_test_context):
        """디렉토리 생성 오류 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings for local environment
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                # Mock mkdir to raise permission error
                with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")), \
                     patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('logging.Formatter'):

                    # Should handle directory creation errors gracefully
                    try:
                        setup_logging(mock_settings)
                        # TimedRotatingFileHandler might still be called despite directory error
                    except PermissionError:
                        pytest.fail("setup_logging should handle directory creation errors gracefully")


class TestGlobalLoggerAccess:
    """전역 로거 접근 테스트"""

    def test_logger_module_level_object(self, component_test_context):
        """모듈 수준 logger 객체 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Verify logger is a logging.Logger instance
            assert isinstance(logger, logging.Logger)

            # Verify logger name
            assert logger.name == "src.utils.core.logger"

    def test_logger_methods_available(self, component_test_context):
        """logger 메소드 사용 가능성 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Test that standard logging methods are available
            assert hasattr(logger, 'debug')
            assert hasattr(logger, 'info')
            assert hasattr(logger, 'warning')
            assert hasattr(logger, 'error')
            assert hasattr(logger, 'critical')

            # Test callable
            assert callable(logger.info)
            assert callable(logger.debug)
            assert callable(logger.error)

    def test_logger_with_setup_logging_integration(self, component_test_context):
        """setup_logging과 전역 logger 통합 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler'), \
                     patch('logging.Formatter'):

                    # Mock the module logger's info method to capture calls
                    with patch.object(logger, 'info') as mock_info:
                        # Test setup_logging
                        setup_logging(mock_settings)

                        # Verify that logger.info was called with setup completion message
                        mock_info.assert_called_once()
                        call_args = mock_info.call_args[0][0]
                        assert "로거 설정 완료" in call_args
                        assert "local" in call_args


class TestLoggingIntegration:
    """로깅 시스템 통합 테스트"""

    def test_complete_logging_workflow_local(self, component_test_context):
        """LOCAL 환경 완전한 로깅 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock complete settings structure
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "local"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Capture all logging operations
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('logging.Formatter') as mock_formatter, \
                     patch('pathlib.Path.mkdir') as mock_mkdir:

                    mock_handler = Mock()
                    mock_file_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test complete workflow
                    setup_logging(mock_settings)

                    # Verify complete setup sequence
                    assert mock_get_logger.called
                    mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
                    mock_mkdir.assert_called_once_with(exist_ok=True)
                    assert mock_file_handler.called
                    mock_formatter.assert_called_once_with(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    mock_handler.setFormatter.assert_called_once_with(mock_formatter_instance)
                    mock_root_logger.addHandler.assert_called_once_with(mock_handler)

    def test_complete_logging_workflow_production(self, component_test_context):
        """PRODUCTION 환경 완전한 로깅 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock production settings
            mock_settings = Mock()
            mock_config = Mock()
            mock_environment = Mock()
            mock_environment.name = "prod"
            mock_config.environment = mock_environment
            mock_settings.config = mock_config

            # Capture production logging setup
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_root_logger.handlers = []
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter') as mock_formatter:

                    mock_handler = Mock()
                    mock_stream_handler.return_value = mock_handler
                    mock_formatter_instance = Mock()
                    mock_formatter.return_value = mock_formatter_instance

                    # Test production workflow
                    setup_logging(mock_settings)

                    # Verify production setup sequence
                    mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
                    mock_stream_handler.assert_called_once_with(sys.stdout)
                    mock_formatter.assert_called_once_with(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    mock_handler.setFormatter.assert_called_once_with(mock_formatter_instance)
                    mock_root_logger.addHandler.assert_called_once_with(mock_handler)

    def test_environment_switching_workflow(self, component_test_context):
        """환경 전환 시 로깅 재설정 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            # First setup: local environment
            local_settings = Mock()
            local_config = Mock()
            local_environment = Mock()
            local_environment.name = "local"
            local_config.environment = local_environment
            local_settings.config = local_config

            # Second setup: production environment
            prod_settings = Mock()
            prod_config = Mock()
            prod_environment = Mock()
            prod_environment.name = "prod"
            prod_config.environment = prod_environment
            prod_settings.config = prod_config

            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = Mock()
                mock_existing_handler = Mock()

                # Initial state with existing handlers
                mock_root_logger.handlers = [mock_existing_handler]
                mock_get_logger.return_value = mock_root_logger

                with patch('logging.handlers.TimedRotatingFileHandler') as mock_file_handler, \
                     patch('logging.StreamHandler') as mock_stream_handler, \
                     patch('logging.Formatter'):

                    # First setup (local)
                    setup_logging(local_settings)

                    # Verify existing handler was removed
                    mock_root_logger.removeHandler.assert_called_with(mock_existing_handler)
                    mock_existing_handler.close.assert_called_once()

                    # Reset for second setup
                    mock_root_logger.reset_mock()
                    mock_root_logger.handlers = [Mock()]  # New handler from first setup

                    # Second setup (production)
                    setup_logging(prod_settings)

                    # Verify handler cleanup happened again
                    assert mock_root_logger.removeHandler.called

                    # Verify different handler types were used
                    assert mock_file_handler.called  # From local setup
                    assert mock_stream_handler.called  # From prod setup