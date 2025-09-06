from typing import Dict, Any, Optional
from .dataframe_builder import DataFrameBuilder

class MockBuilder:
    @staticmethod
    def build_mock_model(model_type: str = 'classifier', predict_value: Any = None):
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        if model_type == 'classifier':
            if predict_value is None:
                predict_value = [0, 1, 0, 1, 0]
            mock_model.predict.return_value = predict_value
            mock_model.predict_proba.return_value = [
                [0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7], [0.8, 0.2]
            ]
            mock_model.classes_ = [0, 1]
        elif model_type == 'regressor':
            if predict_value is None:
                predict_value = [1.5, 2.3, 0.8, 3.2, 1.9]
            mock_model.predict.return_value = predict_value
        mock_model.fit.return_value = mock_model
        return mock_model

    @staticmethod
    def build_mock_adapter():
        from unittest.mock import MagicMock
        mock_adapter = MagicMock()
        mock_adapter.read.return_value = DataFrameBuilder.build_classification_data()
        mock_adapter.write.return_value = None
        return mock_adapter

    @staticmethod
    def build_mock_fetcher():
        from unittest.mock import MagicMock
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = DataFrameBuilder.build_classification_data()
        return mock_fetcher

    @staticmethod
    def build_mock_evaluator(metrics: Optional[Dict[str, float]] = None):
        from unittest.mock import MagicMock
        mock_evaluator = MagicMock()
        if metrics is None:
            metrics = {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1': 0.85}
        mock_evaluator.evaluate.return_value = metrics
        return mock_evaluator
