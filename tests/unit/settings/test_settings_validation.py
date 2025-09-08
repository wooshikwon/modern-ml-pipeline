"""
Unit Tests for Settings Validation
Days 3-5: Configuration validation tests using Validator class
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.settings.validator import Validator, ModelCatalog, ModelSpec, HyperparameterSpec
from src.settings import Settings


class TestSettingsValidation:
    """Settings validation tests using Validator class"""
    
    def setup_method(self):
        """Setup validator for testing"""
        self.validator = Validator()
    
    def test_config_validation_success(self, minimal_classification_settings):
        """Test successful config validation"""
        # Should pass validation
        errors = self.validator.validate_config(minimal_classification_settings.config)
        assert errors == []
    
    def test_config_validation_missing_environment_name(self, settings_builder):
        """Test config validation fails with missing environment name"""
        settings = settings_builder.with_environment("").build()
        
        errors = self.validator.validate_config(settings.config)
        assert len(errors) > 0
        assert any("환경 이름이 비어있습니다" in error for error in errors)
    
    def test_config_validation_sql_adapter_missing_connection_uri(self, settings_builder):
        """Test config validation fails for SQL adapter without connection_uri"""
        settings = settings_builder \
            .with_data_source("sql", config={}) \
            .build()
        
        errors = self.validator.validate_config(settings.config)
        assert len(errors) > 0
        assert any("connection_uri가 필요합니다" in error for error in errors)
    
    def test_config_validation_bigquery_adapter_missing_required_fields(self, settings_builder):
        """Test config validation fails for BigQuery adapter without required fields"""
        settings = settings_builder \
            .with_data_source("bigquery", config={"project_id": "test"}) \
            .build()  # Missing dataset_id
        
        errors = self.validator.validate_config(settings.config)
        assert len(errors) > 0
        assert any("BigQuery adapter 필수 필드" in error for error in errors)
        assert any("dataset_id" in error for error in errors)


class TestRecipeValidation:
    """Recipe validation tests"""
    
    def setup_method(self):
        """Setup validator for testing"""  
        self.validator = Validator()
    
    def test_recipe_validation_success(self, minimal_classification_settings):
        """Test successful recipe validation"""
        errors = self.validator.validate_recipe(minimal_classification_settings.recipe)
        assert errors == []
    
    def test_recipe_validation_invalid_metrics_for_task(self, settings_builder, test_data_files):
        """Test recipe validation fails with invalid metrics for task"""
        settings = settings_builder \
            .with_task("classification") \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        # Change metrics to regression metrics (invalid for classification)
        settings.recipe.evaluation.metrics = ["mae", "rmse"]
        
        errors = self.validator.validate_recipe(settings.recipe)
        assert len(errors) > 0
        assert any("지원되지 않습니다" in error for error in errors)
        assert any("mae" in error or "rmse" in error for error in errors)
    
    def test_recipe_validation_tuning_enabled_without_tunable_params(self, settings_builder, test_data_files):
        """Test recipe validation fails when tuning enabled but no tunable parameters"""
        settings = settings_builder \
            .with_hyperparameter_tuning(enabled=True) \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        # Remove tunable parameters
        settings.recipe.model.hyperparameters.tunable = {}
        
        errors = self.validator.validate_recipe(settings.recipe)
        assert len(errors) > 0
        assert any("tunable 파라미터가 없습니다" in error for error in errors)
    
    def test_recipe_validation_invalid_tunable_parameter_structure(self, settings_builder, test_data_files):
        """Test recipe validation fails with malformed tunable parameters"""
        settings = settings_builder \
            .with_hyperparameter_tuning(enabled=True) \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        # Corrupt tunable parameter structure
        settings.recipe.model.hyperparameters.tunable = {
            "corrupted_param": {"range": [1, 10]}  # Missing 'type'
        }
        
        errors = self.validator.validate_recipe(settings.recipe)
        assert len(errors) > 0
        assert any("'type'이 없습니다" in error for error in errors)


class TestModelCatalogValidation:
    """Model catalog integration with validation"""
    
    @patch('src.settings.validator.ModelCatalog.load_from_directory')
    def test_recipe_validation_with_model_catalog_invalid_model(self, mock_load_catalog, 
                                                               settings_builder, test_data_files):
        """Test recipe validation with model catalog detects invalid model"""
        # Setup mock catalog
        mock_catalog = MagicMock()
        mock_catalog.get_model_spec.return_value = None  # Model not found
        mock_catalog.list_models_for_task.return_value = ["RandomForestClassifier", "LogisticRegression"]
        mock_load_catalog.return_value = mock_catalog
        
        # Create validator with catalog
        catalog_dir = Path("/fake/catalog")
        validator = Validator(catalog_dir)
        validator.catalog = mock_catalog
        
        # Create settings with invalid model
        settings = settings_builder \
            .with_model("sklearn.ensemble.InvalidClassifier") \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        errors = validator.validate_recipe(settings.recipe)
        assert len(errors) > 0
        assert any("카탈로그에 없습니다" in error for error in errors)
        assert any("InvalidClassifier" in error for error in errors)
    
    @patch('src.settings.validator.ModelCatalog.load_from_directory')
    def test_recipe_validation_with_model_catalog_incompatible_task(self, mock_load_catalog,
                                                                   settings_builder, test_data_files):
        """Test recipe validation with model catalog detects task incompatibility"""
        # Setup mock catalog with model spec
        mock_spec = MagicMock()
        mock_spec.is_compatible_with_task.return_value = False
        
        mock_catalog = MagicMock()
        mock_catalog.get_model_spec.return_value = mock_spec
        mock_load_catalog.return_value = mock_catalog
        
        # Create validator with catalog
        catalog_dir = Path("/fake/catalog")  
        validator = Validator(catalog_dir)
        validator.catalog = mock_catalog
        
        # Create settings
        settings = settings_builder \
            .with_model("sklearn.cluster.KMeans") \
            .with_task("classification") \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        errors = validator.validate_recipe(settings.recipe)
        assert len(errors) > 0
        assert any("호환되지 않습니다" in error for error in errors)


class TestSettingsFullValidation:
    """Complete Settings validation including compatibility checks"""
    
    def setup_method(self):
        """Setup validator for testing"""
        self.validator = Validator()
    
    def test_settings_validation_success(self, minimal_classification_settings):
        """Test successful full settings validation"""
        errors = self.validator.validate_settings(minimal_classification_settings)
        assert errors == []
    
    def test_settings_validation_feature_store_compatibility_error(self, settings_builder, test_data_files):
        """Test settings validation detects feature store compatibility issues"""
        settings = settings_builder \
            .with_feature_store(enabled=False) \
            .with_data_path(str(test_data_files["classification"])) \
            .build()
        
        # Force recipe to use feature_store fetcher but config has no feast
        settings.recipe.data.fetcher.type = "feature_store"
        
        errors = self.validator.validate_settings(settings)
        assert len(errors) > 0
        assert any("Compatibility" in error for error in errors)
        assert any("feature_store" in error and "Feast" in error for error in errors)
    
    def test_settings_validation_aggregates_all_errors(self, settings_builder):
        """Test settings validation collects errors from all validation layers"""
        settings = settings_builder \
            .with_environment("") \
            .with_data_source("sql", config={}) \
            .build()
        
        # Force invalid metrics
        settings.recipe.evaluation.metrics = ["invalid_metric"]
        
        errors = self.validator.validate_settings(settings)
        
        # Should have errors from both Config and Recipe validation
        config_errors = [e for e in errors if "[Config]" in e]
        recipe_errors = [e for e in errors if "[Recipe]" in e]
        
        assert len(config_errors) > 0
        assert len(recipe_errors) > 0
        assert len(errors) >= len(config_errors) + len(recipe_errors)
    
    def test_validator_validate_method_raises_exception(self, settings_builder):
        """Test legacy validate method raises ValueError on validation failure"""
        settings = settings_builder \
            .with_environment("") \
            .build()
        
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate(settings)
        
        assert "Settings 검증 실패" in str(exc_info.value)
        assert "환경 이름이 비어있습니다" in str(exc_info.value)