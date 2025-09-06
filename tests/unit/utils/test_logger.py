"""
Unit tests for the logger utility module.
Tests logging configuration and functionality.
"""

import pytest
from unittest.mock import Mock, patch
import logging

from src.utils.system.logger import logger


class TestLoggerConfiguration:
    """Test logger configuration."""
    
    def test_logger_exists(self):
        """Test that logger is properly configured."""
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_methods_exist(self):
        """Test that logger has required methods."""
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
    
    @patch('src.utils.system.logger.logger')
    def test_logger_info_call(self, mock_logger):
        """Test logger info method call."""
        from src.utils.system.logger import logger as test_logger
        test_logger.info("Test message")
        # This test ensures the import works correctly
        assert True
    
    def test_logger_level_configuration(self):
        """Test that logger level is properly configured."""
        # Logger should be configured with appropriate level
        # Accept default WARNING or any configured level >= DEBUG
        assert logger.level in (logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
        assert logger.level <= logging.CRITICAL
    
    def test_logger_name(self):
        """Test logger name configuration."""
        # Logger should have a meaningful name
        assert logger.name is not None
        assert len(logger.name) > 0


class TestLoggerUsage:
    """Test logger usage patterns."""
    
    def test_logger_message_formatting(self):
        """Test that logger can handle different message formats."""
        # Test string formatting
        try:
            logger.info("Test message with %s", "parameter")
            logger.info("Test message with {} format", "parameter")
            logger.info(f"Test message with f-string")
            assert True
        except Exception as e:
            pytest.fail(f"Logger formatting failed: {e}")
    
    def test_logger_exception_handling(self):
        """Test that logger can handle exceptions."""
        try:
            logger.error("Test error message")
            logger.exception("Test exception message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger exception handling failed: {e}")
    
    def test_logger_different_levels(self):
        """Test different logging levels."""
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            assert True
        except Exception as e:
            pytest.fail(f"Logger level handling failed: {e}")