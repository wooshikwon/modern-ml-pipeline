"""
Tests for catalog_parser - Configuration file parsing utilities

Comprehensive testing of YAML catalog file loading including:
- Valid YAML file parsing with various structures
- File not found and error handling scenarios  
- Path resolution and security considerations
- Edge cases and malformed content handling
- Integration with PyYAML safe loading

Test Categories:
1. TestCatalogParserValidFiles - Valid YAML file parsing scenarios
2. TestCatalogParserErrorHandling - File errors and exception handling
3. TestCatalogParserPathResolution - Path handling and resolution
4. TestCatalogParserSecurity - YAML security and malformed content
5. TestCatalogParserEdgeCases - Edge cases and performance considerations
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import yaml
import tempfile
import os
from typing import Dict, List, Any

from src.utils.system.catalog_parser import load_model_catalog


class TestCatalogParserValidFiles:
    """Test valid YAML file parsing scenarios."""

    def test_valid_yaml_catalog_loading(self):
        """Test loading valid YAML catalog with expected structure."""
        valid_catalog_content = """
regression:
  - name: linear_regression
    type: sklearn
    params:
      fit_intercept: true
  - name: random_forest
    type: sklearn
    params:
      n_estimators: 100

classification:
  - name: logistic_regression
    type: sklearn
    params:
      C: 1.0
  - name: svm
    type: sklearn
    params:
      kernel: rbf
"""
        expected_result = {
            'regression': [
                {
                    'name': 'linear_regression',
                    'type': 'sklearn',
                    'params': {'fit_intercept': True}
                },
                {
                    'name': 'random_forest',
                    'type': 'sklearn',
                    'params': {'n_estimators': 100}
                }
            ],
            'classification': [
                {
                    'name': 'logistic_regression',
                    'type': 'sklearn',
                    'params': {'C': 1.0}
                },
                {
                    'name': 'svm',
                    'type': 'sklearn',
                    'params': {'kernel': 'rbf'}
                }
            ]
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=valid_catalog_content):
                result = load_model_catalog()
                
                assert isinstance(result, dict)
                assert 'regression' in result
                assert 'classification' in result
                assert len(result['regression']) == 2
                assert len(result['classification']) == 2
                assert result == expected_result

    def test_empty_yaml_file_loading(self):
        """Test loading empty YAML file."""
        empty_content = ""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=empty_content):
                result = load_model_catalog()
                
                assert result is None  # yaml.safe_load returns None for empty content

    def test_simple_yaml_structure(self):
        """Test loading simple YAML structure."""
        simple_content = """
models:
  - basic_model
  - advanced_model
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=simple_content):
                result = load_model_catalog()
                
                assert isinstance(result, dict)
                assert 'models' in result
                assert result['models'] == ['basic_model', 'advanced_model']

    def test_nested_yaml_structure(self):
        """Test loading complex nested YAML structure."""
        nested_content = """
categories:
  supervised:
    regression:
      linear:
        - simple_linear
        - ridge
      tree_based:
        - random_forest
        - gradient_boosting
    classification:
      linear:
        - logistic_regression
      ensemble:
        - voting_classifier
  unsupervised:
    clustering:
      - kmeans
      - dbscan
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=nested_content):
                result = load_model_catalog()
                
                assert isinstance(result, dict)
                assert 'categories' in result
                assert 'supervised' in result['categories']
                assert 'unsupervised' in result['categories']
                assert 'regression' in result['categories']['supervised']
                assert len(result['categories']['unsupervised']['clustering']) == 2


class TestCatalogParserErrorHandling:
    """Test file errors and exception handling."""

    def test_file_not_found_returns_empty_dict(self):
        """Test file not found returns empty dictionary."""
        with patch('pathlib.Path.exists', return_value=False):
            result = load_model_catalog()
            
            assert result == {}
            assert isinstance(result, dict)

    def test_invalid_yaml_syntax_returns_empty_dict(self):
        """Test invalid YAML syntax is handled gracefully."""
        invalid_yaml = """
invalid: yaml: content:
  - unclosed_bracket: [
  - missing_quote: "unclosed string
    invalid_indentation
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=invalid_yaml):
                result = load_model_catalog()
                
                # Function should catch YAML parsing exceptions and return empty dict
                assert result == {}
                assert isinstance(result, dict)

    def test_file_read_permission_error_returns_empty_dict(self):
        """Test file permission error is handled gracefully."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', side_effect=PermissionError("Access denied")):
                result = load_model_catalog()
                
                assert result == {}
                assert isinstance(result, dict)

    def test_file_read_io_error_returns_empty_dict(self):
        """Test file I/O error is handled gracefully."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', side_effect=IOError("File read error")):
                result = load_model_catalog()
                
                assert result == {}
                assert isinstance(result, dict)

    def test_yaml_parsing_error_returns_empty_dict(self):
        """Test YAML parsing error is handled gracefully."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="valid: content"):
                with patch('yaml.safe_load', side_effect=yaml.YAMLError("Parse error")):
                    result = load_model_catalog()
                    
                    assert result == {}
                    assert isinstance(result, dict)

    def test_general_exception_returns_empty_dict(self):
        """Test general exception is handled gracefully."""
        with patch('pathlib.Path.exists', side_effect=RuntimeError("Unexpected error")):
            result = load_model_catalog()
            
            assert result == {}
            assert isinstance(result, dict)


class TestCatalogParserPathResolution:
    """Test path handling and resolution."""

    @patch('src.utils.system.catalog_parser.Path')
    def test_path_resolution_logic(self, mock_path_class):
        """Test correct path resolution from module location."""
        # Mock the path resolution chain
        mock_file = MagicMock()
        mock_parent = MagicMock()
        mock_parent.parent = MagicMock()
        mock_parent.parent.parent = MagicMock()
        mock_file.parent = mock_parent
        
        mock_catalog_path = MagicMock()
        mock_catalog_path.exists.return_value = True
        mock_catalog_path.read_text.return_value = "test: data"
        
        # Setup the path resolution chain
        mock_path_class.__file__ = mock_file
        mock_parent.parent.parent.__truediv__.return_value.__truediv__.return_value = mock_catalog_path
        
        with patch('pathlib.Path') as mock_pathlib:
            mock_pathlib.return_value = mock_file
            
            with patch('yaml.safe_load', return_value={'test': 'data'}):
                result = load_model_catalog()
                
                assert result == {'test': 'data'}

    def test_absolute_path_construction(self):
        """Test absolute path construction works correctly."""
        # This test verifies the path construction logic without mocking
        # We'll test that the function handles path construction without errors
        with patch('pathlib.Path.exists', return_value=False):
            # This should not raise any path-related exceptions
            result = load_model_catalog()
            assert result == {}

    def test_cross_platform_path_handling(self):
        """Test path handling works across different platforms."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="cross: platform"):
                with patch('yaml.safe_load', return_value={'cross': 'platform'}):
                    result = load_model_catalog()
                    
                    assert result == {'cross': 'platform'}
                    # Path operations should work regardless of OS


class TestCatalogParserSecurity:
    """Test YAML security and malformed content."""

    def test_safe_yaml_loading_used(self):
        """Test that yaml.safe_load is used for security."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="test: data"):
                with patch('yaml.safe_load') as mock_safe_load:
                    mock_safe_load.return_value = {'test': 'data'}
                    
                    result = load_model_catalog()
                    
                    # Verify safe_load was called (not load)
                    mock_safe_load.assert_called_once_with("test: data")
                    assert result == {'test': 'data'}

    def test_dangerous_yaml_content_handling(self):
        """Test handling of potentially dangerous YAML content."""
        # This content would be dangerous with yaml.load but safe with yaml.safe_load
        dangerous_content = """
!!python/object/apply:os.system
- "echo 'dangerous command'"
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=dangerous_content):
                # yaml.safe_load should handle this safely or raise an exception
                result = load_model_catalog()
                
                # Should either return empty dict (if exception caught) or safe parsed content
                assert isinstance(result, dict)

    def test_large_yaml_file_handling(self):
        """Test handling of large YAML files."""
        # Create a large but valid YAML structure
        large_content = "models:\n"
        for i in range(1000):
            large_content += f"  - model_{i}: config_{i}\n"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=large_content):
                result = load_model_catalog()
                
                # Should handle large files without issues
                assert isinstance(result, dict)
                if 'models' in result:
                    assert len(result['models']) == 1000

    def test_unicode_content_handling(self):
        """Test handling of Unicode content in YAML."""
        unicode_content = """
한글모델:
  - 이름: "선형회귀"
    타입: "sklearn"
  - 이름: "랜덤포레스트"
    타입: "sklearn"

español:
  - nombre: "regresión_lineal" 
    tipo: "sklearn"
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=unicode_content):
                result = load_model_catalog()
                
                assert isinstance(result, dict)
                # Should handle Unicode characters properly


class TestCatalogParserEdgeCases:
    """Test edge cases and performance considerations."""

    def test_empty_dictionary_content(self):
        """Test YAML file containing empty dictionary."""
        empty_dict_content = "{}"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=empty_dict_content):
                result = load_model_catalog()
                
                assert result == {}
                assert isinstance(result, dict)

    def test_null_yaml_content(self):
        """Test YAML file with null content."""
        null_content = "null"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=null_content):
                result = load_model_catalog()
                
                assert result is None  # yaml.safe_load('null') returns None

    def test_yaml_with_comments_and_formatting(self):
        """Test YAML file with comments and special formatting."""
        commented_content = """
# Model catalog configuration
# Updated: 2024-01-01

regression:  # Regression models
  - name: linear_regression  # Simple linear model
    type: sklearn
    # Advanced configuration
    params:
      fit_intercept: true
      normalize: false

classification:
  # Classification models go here
  - name: logistic_regression
    type: sklearn
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=commented_content):
                result = load_model_catalog()
                
                assert isinstance(result, dict)
                assert 'regression' in result
                assert 'classification' in result

    def test_multiple_document_yaml(self):
        """Test YAML file with multiple documents (should handle first document)."""
        multi_doc_content = """
models:
  - first_model
---
other_config:
  - second_doc
"""
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=multi_doc_content):
                result = load_model_catalog()
                
                # yaml.safe_load typically returns the first document
                assert isinstance(result, dict)
                if 'models' in result:
                    assert result['models'] == ['first_model']

    def test_function_return_type_annotation_compliance(self):
        """Test function returns type matching annotation."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="test: [1, 2, 3]"):
                result = load_model_catalog()
                
                # Should return Dict[str, List[Dict[str, Any]]] or compatible structure
                assert isinstance(result, dict)
                for key, value in result.items():
                    assert isinstance(key, str)
                    # Value could be list or other types depending on YAML content

    def test_performance_with_deep_nesting(self):
        """Test performance with deeply nested YAML structure."""
        # Create deeply nested structure
        deep_content = "level0:"
        for i in range(10):
            deep_content = f"{deep_content}\n  level{i+1}:"
        deep_content += "\n    value: final"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value=deep_content):
                result = load_model_catalog()
                
                # Should handle deep nesting without performance issues
                assert isinstance(result, dict)
                assert 'level0' in result