"""
Unit tests for inference pipeline format_predictions function.
"""

import numpy as np
import pandas as pd

from src.utils.data.data_io import format_predictions


class TestFormatPredictions:
    """Test format_predictions function with DataInterface"""

    def test_format_with_datainterface(self):
        """Test formatting predictions with DataInterface config"""
        predictions = [100.0, 200.0, 150.0]

        df = pd.DataFrame(
            {
                "product_id": ["P1", "P2", "P3"],
                "brand": ["A", "B", "A"],
                "category": ["X", "Y", "X"],
            }
        )

        data_interface = {"entity_columns": ["product_id"]}

        result = format_predictions(predictions, df, data_interface)

        # Should include entity columns and predictions
        assert "product_id" in result.columns
        assert "prediction" in result.columns
        assert len(result) == 3
        assert list(result["prediction"]) == predictions
        assert list(result["product_id"]) == ["P1", "P2", "P3"]

    def test_format_with_multiple_entities(self):
        """Test formatting with multiple entity columns"""
        predictions = pd.DataFrame({"prediction": [0.8, 0.3], "probability": [0.8, 0.3]})

        df = pd.DataFrame({"user_id": ["U1", "U2"], "item_id": ["I1", "I2"], "feature": [1, 2]})

        data_interface = {"entity_columns": ["user_id", "item_id"]}

        result = format_predictions(predictions, df, data_interface)

        # Should include all entity columns
        assert "user_id" in result.columns
        assert "item_id" in result.columns
        assert "prediction" in result.columns
        assert "probability" in result.columns

    def test_format_without_datainterface(self):
        """Test formatting without DataInterface (fallback)"""
        predictions = [1, 0, 1]

        df = pd.DataFrame({"id": [1, 2, 3], "feature": ["a", "b", "c"]})

        # No data_interface provided
        result = format_predictions(predictions, df, {})

        # Should still create valid output
        assert "prediction" in result.columns
        assert len(result) == 3

    def test_format_numpy_array_predictions(self):
        """Test formatting numpy array predictions"""
        predictions = np.array([0.1, 0.2, 0.3])

        df = pd.DataFrame({"id": [1, 2, 3]})

        data_interface = {"entity_columns": ["id"]}

        result = format_predictions(predictions, df, data_interface)

        assert "prediction" in result.columns
        assert list(result["prediction"]) == [0.1, 0.2, 0.3]

    def test_format_multiclass_predictions(self):
        """Test formatting multiclass probability predictions"""
        # Multiclass probabilities (3 samples, 3 classes)
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])

        df = pd.DataFrame({"id": [1, 2, 3]})

        data_interface = {"entity_columns": ["id"]}

        result = format_predictions(predictions, df, data_interface)

        # Should create probability columns for multiclass predictions
        assert "prob_class_0" in result.columns
        assert "prob_class_1" in result.columns
        assert "prob_class_2" in result.columns
        assert "id" in result.columns
        assert result.shape == (3, 4)  # 3 prob columns + 1 entity column
