"""
Unit tests for the dependencies utility module.
Tests dependency validation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.utils.system.dependencies import validate_dependencies
from src.settings import Settings
from tests.helpers.builders import ConfigBuilder, RecipeBuilder


class TestValidateDependencies:
    """Test validate_dependencies function."""
    
    def test_validate_dependencies_success(self):
        """Test successful dependency validation."""
        settings = Mock(spec=Settings)
        settings.config = ConfigBuilder.build()
        settings.recipe = RecipeBuilder.build()
        
        # Should not raise any exception
        try:
            validate_dependencies(settings)
            assert True
        except Exception as e:
            pytest.fail(f"Dependencies validation failed: {e}")
    
    def test_validate_dependencies_with_none_settings(self):
        """Test validation with None settings."""
        try:
            validate_dependencies(None)
            assert True
        except Exception as e:
            pytest.fail(f"None settings handling failed: {e}")
    
    def test_validate_dependencies_logging(self):
        """Smoke test: ensure function executes; logging not enforced in module."""
        settings = Mock(spec=Settings)
        settings.config = ConfigBuilder.build()
        settings.recipe = RecipeBuilder.build()
        validate_dependencies(settings)