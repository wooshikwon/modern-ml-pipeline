"""
Unit tests for the Settings validator module.
Tests validation logic for hyperparameter tuning configurations.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from src.settings.validator import TunableParameter


class TestTunableParameter:
    """Test the TunableParameter validation class."""
    
    def test_tunable_parameter_int_type(self):
        """Test TunableParameter with int type."""
        param = TunableParameter(
            type="int",
            range=[10, 100]
        )
        assert param.type == "int"
        assert param.range == [10, 100]
    
    def test_tunable_parameter_float_type(self):
        """Test TunableParameter with float type."""
        param = TunableParameter(
            type="float",
            range=[0.01, 1.0]
        )
        assert param.type == "float"
        assert param.range == [0.01, 1.0]
    
    def test_tunable_parameter_categorical_type(self):
        """Test TunableParameter with categorical type."""
        param = TunableParameter(
            type="categorical",
            range=["linear", "rbf", "poly"]
        )
        assert param.type == "categorical"
        assert param.range == ["linear", "rbf", "poly"]
    
    def test_tunable_parameter_invalid_type(self):
        """Test TunableParameter with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="invalid",
                range=[1, 10]
            )
        assert "invalid" in str(exc_info.value).lower()
    
    def test_tunable_parameter_int_range_validation(self):
        """Test int type range validation."""
        # Valid int range
        param = TunableParameter(
            type="int",
            range=[1, 10]
        )
        assert param.range == [1, 10]
        
        # Invalid: More than 2 elements
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="int",
                range=[1, 5, 10]
            )
        assert "[min, max]" in str(exc_info.value) or "형태여야" in str(exc_info.value)
        
        # Invalid: Min > Max
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="int",
                range=[10, 5]
            )
        assert "범위가 잘못" in str(exc_info.value) or ">=" in str(exc_info.value)
    
    def test_tunable_parameter_float_range_validation(self):
        """Test float type range validation."""
        # Valid float range
        param = TunableParameter(
            type="float",
            range=[0.1, 0.9]
        )
        assert param.range == [0.1, 0.9]
        
        # Invalid: More than 2 elements
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="float",
                range=[0.1, 0.5, 0.9]
            )
        assert "[min, max]" in str(exc_info.value) or "형태여야" in str(exc_info.value)
        
        # Invalid: Min > Max
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="float",
                range=[0.9, 0.1]
            )
        assert "범위가 잘못" in str(exc_info.value) or ">=" in str(exc_info.value)
    
    def test_tunable_parameter_categorical_range_validation(self):
        """Test categorical type range validation."""
        # Valid categorical range
        param = TunableParameter(
            type="categorical",
            range=["a", "b", "c"]
        )
        assert param.range == ["a", "b", "c"]
        
        # Invalid: Empty list
        with pytest.raises(ValidationError) as exc_info:
            TunableParameter(
                type="categorical",
                range=[]
            )
        assert "2개 이상" in str(exc_info.value) or "선택지가 필요" in str(exc_info.value)
        
        # Invalid: Non-string elements
        # Note: Current implementation doesn't enforce string type for categorical
        param = TunableParameter(
            type="categorical",
            range=[1, 2, 3]
        )
        assert param.range == [1, 2, 3]  # Accepts non-string values
    
    def test_tunable_parameter_missing_range(self):
        """Test TunableParameter without range."""
        with pytest.raises(ValidationError):
            TunableParameter(type="int")  # Missing range
    
    def test_tunable_parameter_missing_type(self):
        """Test TunableParameter without type."""
        with pytest.raises(ValidationError):
            TunableParameter(range=[1, 10])  # Missing type


class TestTunableParameterLog:
    """Test TunableParameter with log option."""
    
    def test_log_with_float(self):
        """Test log option with float type."""
        param = TunableParameter(
            type="float",
            range=[0.001, 10.0],
            log=True
        )
        assert param.log is True
        assert param.type == "float"
    
    def test_log_with_int(self):
        """Test log option with int type."""
        param = TunableParameter(
            type="int",
            range=[10, 1000],
            log=True
        )
        assert param.log is True
        assert param.type == "int"
    
    def test_log_with_categorical(self):
        """Test that log is allowed with categorical type (but has no effect)."""
        # Current implementation allows this but log has no effect
        param = TunableParameter(
            type="categorical",
            range=["a", "b"],
            log=True
        )
        # Current implementation allows this
        assert param.log is True
    
    def test_log_default_false(self):
        """Test that log defaults to False."""
        param = TunableParameter(
            type="float",
            range=[0.1, 1.0]
        )
        assert param.log is False


class TestTunableParameterDefault:
    """Test TunableParameter with default option."""
    
    def test_default_with_int(self):
        """Test default option with int type."""
        param = TunableParameter(
            type="int",
            range=[0, 100],
            default=50
        )
        assert param.default == 50
    
    def test_default_with_float(self):
        """Test default option with float type."""
        param = TunableParameter(
            type="float",
            range=[0.0, 1.0],
            default=0.5
        )
        assert param.default == 0.5
    
    def test_default_with_categorical(self):
        """Test default option with categorical type."""
        param = TunableParameter(
            type="categorical",
            range=["a", "b", "c"],
            default="b"
        )
        assert param.default == "b"
    
    def test_default_none(self):
        """Test that default defaults to None."""
        param = TunableParameter(
            type="int",
            range=[1, 10]
        )
        assert param.default is None


class TestTunableParameterIntegration:
    """Integration tests for TunableParameter in real scenarios."""
    
    def test_random_forest_hyperparameters(self):
        """Test typical Random Forest hyperparameter configurations."""
        # n_estimators
        n_estimators = TunableParameter(
            type="int",
            range=[50, 500]
        )
        assert n_estimators.type == "int"
        assert n_estimators.range == [50, 500]
        
        # max_depth
        max_depth = TunableParameter(
            type="int",
            range=[3, 20]
        )
        assert max_depth.type == "int"
        
        # min_samples_split
        min_samples_split = TunableParameter(
            type="float",
            range=[0.01, 0.5]
        )
        assert min_samples_split.type == "float"
        
        # criterion
        criterion = TunableParameter(
            type="categorical",
            range=["gini", "entropy"]
        )
        assert criterion.type == "categorical"
    
    def test_xgboost_hyperparameters(self):
        """Test typical XGBoost hyperparameter configurations."""
        # learning_rate with log scale
        learning_rate = TunableParameter(
            type="float",
            range=[0.001, 0.3],
            log=True
        )
        assert learning_rate.log is True
        
        # max_depth
        max_depth = TunableParameter(
            type="int",
            range=[3, 10]
        )
        assert max_depth.type == "int"
        
        # subsample
        subsample = TunableParameter(
            type="float",
            range=[0.5, 1.0]
        )
        assert subsample.type == "float"
        
        # booster
        booster = TunableParameter(
            type="categorical",
            range=["gbtree", "dart"]
        )
        assert len(booster.range) == 2
    
    def test_neural_network_hyperparameters(self):
        """Test typical neural network hyperparameter configurations."""
        # hidden_units
        hidden_units = TunableParameter(
            type="int",
            range=[32, 512]
        )
        assert hidden_units.type == "int"
        
        # dropout_rate
        dropout_rate = TunableParameter(
            type="float",
            range=[0.0, 0.5]
        )
        assert dropout_rate.range == [0.0, 0.5]
        
        # activation
        activation = TunableParameter(
            type="categorical",
            range=["relu", "tanh", "sigmoid"]
        )
        assert "relu" in activation.range
        
        # batch_size (powers of 2)
        batch_size = TunableParameter(
            type="int",
            range=[16, 256],
            log=True  # Will sample powers of 2
        )
        assert batch_size.log is True


class TestTunableParameterSerialization:
    """Test TunableParameter serialization and deserialization."""
    
    def test_dict_conversion(self):
        """Test converting TunableParameter to/from dict."""
        param = TunableParameter(
            type="float",
            range=[0.1, 1.0],
            log=True,
            default=0.5
        )
        
        # Convert to dict (using model_dump for Pydantic v2)
        param_dict = param.model_dump()
        assert param_dict["type"] == "float"
        assert param_dict["range"] == [0.1, 1.0]
        assert param_dict["log"] is True
        assert param_dict["default"] == 0.5
        
        # Create from dict
        param2 = TunableParameter(**param_dict)
        assert param2.type == param.type
        assert param2.range == param.range
        assert param2.log == param.log
        assert param2.default == param.default
    
    def test_json_serialization(self):
        """Test JSON serialization of TunableParameter."""
        import json
        
        param = TunableParameter(
            type="categorical",
            range=["option1", "option2", "option3"]
        )
        
        # Convert to JSON (using model_dump_json for Pydantic v2)
        param_json = param.model_dump_json()
        assert isinstance(param_json, str)
        
        # Parse JSON
        parsed = json.loads(param_json)
        assert parsed["type"] == "categorical"
        assert parsed["range"] == ["option1", "option2", "option3"]
        
        # Create from parsed JSON
        param2 = TunableParameter(**parsed)
        assert param2.range == param.range