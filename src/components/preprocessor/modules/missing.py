# src/components/preprocessor/modules/missing.py
"""
Comprehensive Missing Value Handling Strategies
DataFrame-First approach with multiple strategies for handling missing data
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.components.preprocessor.base import BasePreprocessor

from ..registry import PreprocessorStepRegistry


class DropMissingWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: Drop rows or columns with missing values
    Provides control over missing value threshold and axis (rows/columns)
    """

    def __init__(self, axis: str = "rows", threshold: float = 0.0, columns: List[str] = None):
        """
        Args:
            axis: 'rows' to drop rows, 'columns' to drop columns
            threshold: Fraction of non-missing values required to keep (0.0 = any missing drops, 1.0 = all missing drops)
            columns: Specific columns to consider (None = all columns)
        """
        self.axis = axis
        self.threshold = threshold
        self.columns = columns
        self._dropped_columns = []  # Track dropped columns for consistency

    def fit(self, X: pd.DataFrame, y=None):
        """Learn which rows/columns to drop based on missing patterns"""
        if self.axis == "columns":
            # Determine which columns to drop
            if self.columns is not None:
                # Only consider specified columns
                target_cols = [col for col in self.columns if col in X.columns]
            else:
                # Consider all columns
                target_cols = list(X.columns)

            # Calculate missing fraction for each column
            missing_fractions = X[target_cols].isnull().mean()
            # Drop columns that exceed missing threshold
            self._dropped_columns = missing_fractions[
                missing_fractions > self.threshold
            ].index.tolist()
        else:
            # For row dropping, no pre-computation needed
            self._dropped_columns = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop missing values according to the configured strategy"""
        result = X.copy()

        if self.axis == "columns":
            # Drop columns with too many missing values
            columns_to_drop = [col for col in self._dropped_columns if col in result.columns]
            if columns_to_drop:
                result = result.drop(columns=columns_to_drop)
        else:
            # Drop rows with missing values
            if self.columns is not None:
                # Only consider specified columns for row dropping
                target_cols = [col for col in self.columns if col in result.columns]
                if target_cols:
                    # Calculate missing fraction per row for target columns
                    missing_per_row = result[target_cols].isnull().mean(axis=1)
                    # Drop rows that exceed the threshold
                    rows_to_drop = missing_per_row > self.threshold
                    result = result[~rows_to_drop]
            else:
                # Consider all columns
                missing_per_row = result.isnull().mean(axis=1)
                rows_to_drop = missing_per_row > self.threshold
                result = result[~rows_to_drop]

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """Return expected output columns after dropping"""
        if self.axis == "columns":
            return [col for col in input_columns if col not in self._dropped_columns]
        else:
            return input_columns  # Row dropping doesn't change columns

    def preserves_column_names(self) -> bool:
        """Column names preserved for row dropping, not for column dropping"""
        return self.axis == "rows"

    def get_application_type(self) -> str:
        """Targeted application - can specify which columns to consider"""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """All columns are potentially applicable for missing value analysis"""
        return list(X.columns)


class ForwardFillWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: Forward fill missing values (propagate last valid observation)
    Useful for time series data where missing values can be filled with previous values
    """

    def __init__(self, limit: Optional[int] = None, columns: List[str] = None):
        """
        Args:
            limit: Maximum number of consecutive NaN values to fill
            columns: Specific columns to apply forward fill (None = all applicable)
        """
        self.limit = limit
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """No learning needed for forward fill"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply forward fill to missing values"""
        result = X.copy()

        # Determine target columns
        if self.columns is not None:
            target_cols = [col for col in self.columns if col in result.columns]
        else:
            # Apply to all columns with missing values
            target_cols = [col for col in result.columns if result[col].isnull().any()]

        # Apply forward fill to target columns
        for col in target_cols:
            if self.limit is not None:
                result[col] = result[col].ffill(limit=self.limit)
            else:
                result[col] = result[col].ffill()

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """Forward fill preserves column names"""
        return input_columns

    def preserves_column_names(self) -> bool:
        """Forward fill preserves column names"""
        return True

    def get_application_type(self) -> str:
        """Targeted application"""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """Columns with missing values are applicable"""
        return [col for col in X.columns if X[col].isnull().any()]


class BackwardFillWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: Backward fill missing values (propagate next valid observation)
    Useful for time series data where missing values can be filled with future values
    """

    def __init__(self, limit: Optional[int] = None, columns: List[str] = None):
        """
        Args:
            limit: Maximum number of consecutive NaN values to fill
            columns: Specific columns to apply backward fill (None = all applicable)
        """
        self.limit = limit
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """No learning needed for backward fill"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply backward fill to missing values"""
        result = X.copy()

        # Determine target columns
        if self.columns is not None:
            target_cols = [col for col in self.columns if col in result.columns]
        else:
            # Apply to all columns with missing values
            target_cols = [col for col in result.columns if result[col].isnull().any()]

        # Apply backward fill to target columns
        for col in target_cols:
            if self.limit is not None:
                result[col] = result[col].bfill(limit=self.limit)
            else:
                result[col] = result[col].bfill()

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """Backward fill preserves column names"""
        return input_columns

    def preserves_column_names(self) -> bool:
        """Backward fill preserves column names"""
        return True

    def get_application_type(self) -> str:
        """Targeted application"""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """Columns with missing values are applicable"""
        return [col for col in X.columns if X[col].isnull().any()]


class ConstantFillWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: Fill missing values with constant values
    Allows different constants for different columns
    """

    def __init__(self, fill_value: Union[Any, Dict[str, Any]] = 0, columns: List[str] = None):
        """
        Args:
            fill_value: Constant value(s) to fill. Can be:
                - Single value: applied to all columns
                - Dict: {column_name: fill_value} for column-specific values
            columns: Specific columns to apply constant fill (None = all applicable)
        """
        self.fill_value = fill_value
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """No learning needed for constant fill"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply constant fill to missing values"""
        result = X.copy()

        # Determine target columns
        if self.columns is not None:
            target_cols = [col for col in self.columns if col in result.columns]
        else:
            # Apply to all columns with missing values
            target_cols = [col for col in result.columns if result[col].isnull().any()]

        # Apply constant fill
        if isinstance(self.fill_value, dict):
            # Column-specific fill values
            for col in target_cols:
                if col in self.fill_value:
                    result[col] = result[col].fillna(self.fill_value[col])
                # Columns not in dict are left unchanged
        else:
            # Single fill value for all columns
            for col in target_cols:
                result[col] = result[col].fillna(self.fill_value)

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """Constant fill preserves column names"""
        return input_columns

    def preserves_column_names(self) -> bool:
        """Constant fill preserves column names"""
        return True

    def get_application_type(self) -> str:
        """Targeted application"""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """Columns with missing values are applicable"""
        return [col for col in X.columns if X[col].isnull().any()]


class InterpolationWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: Interpolate missing values using various methods
    Supports linear, polynomial, and other interpolation strategies
    """

    def __init__(
        self,
        method: str = "linear",
        order: Optional[int] = None,
        limit: Optional[int] = None,
        columns: List[str] = None,
    ):
        """
        Args:
            method: Interpolation method ('linear', 'polynomial', 'spline', etc.)
            order: Order for polynomial/spline interpolation
            limit: Maximum number of consecutive NaN values to interpolate
            columns: Specific columns to apply interpolation (None = all numeric)
        """
        self.method = method
        self.order = order
        self.limit = limit
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """No learning needed for interpolation"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply interpolation to missing values"""
        result = X.copy()

        # Determine target columns
        if self.columns is not None:
            target_cols = [col for col in self.columns if col in result.columns]
        else:
            # Apply to numeric columns with missing values
            target_cols = [
                col
                for col in result.columns
                if result[col].dtype in ["int64", "float64"] and result[col].isnull().any()
            ]

        # Apply interpolation to target columns
        for col in target_cols:
            try:
                if self.method in ["polynomial", "spline"] and self.order is not None:
                    result[col] = result[col].interpolate(
                        method=self.method, order=self.order, limit=self.limit
                    )
                else:
                    result[col] = result[col].interpolate(method=self.method, limit=self.limit)
            except (ValueError, TypeError):
                # Fallback to linear interpolation if method fails
                result[col] = result[col].interpolate(method="linear", limit=self.limit)

        return result

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """Interpolation preserves column names"""
        return input_columns

    def preserves_column_names(self) -> bool:
        """Interpolation preserves column names"""
        return True

    def get_application_type(self) -> str:
        """Targeted application"""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """Numeric columns with missing values are applicable"""
        return [
            col
            for col in X.columns
            if X[col].dtype in ["int64", "float64"] and X[col].isnull().any()
        ]


# Register all missing value handlers
PreprocessorStepRegistry.register("drop_missing", DropMissingWrapper)
PreprocessorStepRegistry.register("forward_fill", ForwardFillWrapper)
PreprocessorStepRegistry.register("backward_fill", BackwardFillWrapper)
PreprocessorStepRegistry.register("constant_fill", ConstantFillWrapper)
PreprocessorStepRegistry.register("interpolation", InterpolationWrapper)
