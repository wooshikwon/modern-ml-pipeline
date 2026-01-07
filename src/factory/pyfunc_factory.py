"""PyfuncWrapper 생성 전담 팩토리"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from src.utils.core.logger import logger

if TYPE_CHECKING:
    from src.components.preprocessor.base import BasePreprocessor
    from src.settings import Settings
    from src.settings.schemas.recipe import Recipe
    from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper


class PyfuncFactory:
    """MLflow PyfuncWrapper 생성 전담"""

    def __init__(self, settings: "Settings", recipe: "Recipe"):
        self.settings = settings
        self.recipe = recipe

    def create(
        self,
        trained_model: Any,
        trained_datahandler: Any,
        trained_preprocessor: Optional["BasePreprocessor"],
        trained_fetcher: Optional[Any],
        trained_calibrator: Optional[Any] = None,
        training_df: Optional[pd.DataFrame] = None,
        training_results: Optional[Dict[str, Any]] = None,
    ) -> "PyfuncWrapper":
        """PyfuncWrapper 생성"""
        from src.utils.integrations.pyfunc_wrapper import PyfuncWrapper

        logger.debug("[FACT] PyfuncWrapper artifact 생성 중")

        signature, data_schema = self._create_signature_schema(
            training_df, trained_datahandler, trained_preprocessor
        )

        data_interface_schema = self._create_data_interface_schema(
            training_df, trained_datahandler, trained_preprocessor
        )

        return PyfuncWrapper(
            settings=self.settings,
            trained_model=trained_model,
            trained_datahandler=trained_datahandler,
            trained_preprocessor=trained_preprocessor,
            trained_fetcher=trained_fetcher,
            trained_calibrator=trained_calibrator,
            training_results=training_results,
            signature=signature,
            data_schema=data_schema,
            data_interface_schema=data_interface_schema,
        )

    def _create_signature_schema(
        self,
        training_df: Optional[pd.DataFrame],
        datahandler: Any,
        preprocessor: Optional["BasePreprocessor"],
    ):
        """MLflow signature 및 schema 생성"""
        if training_df is None:
            return None, None

        from src.utils.integrations.mlflow_integration import (
            create_enhanced_model_signature_with_schema,
        )

        training_df = self._ensure_timestamp_dtype(training_df)

        input_features = datahandler.get_feature_columns()
        model_features = (
            preprocessor.get_output_columns() if preprocessor else input_features
        )

        data_interface_config = self._build_data_interface_config(
            input_features, model_features
        )

        signature, data_schema = create_enhanced_model_signature_with_schema(
            training_df, data_interface_config
        )
        logger.debug("[FACT] Signature/Schema 생성 완료")

        return signature, data_schema

    def _create_data_interface_schema(
        self,
        training_df: Optional[pd.DataFrame],
        datahandler: Any,
        preprocessor: Optional["BasePreprocessor"],
    ) -> Optional[Dict[str, Any]]:
        """DataInterface 검증용 스키마 생성"""
        if training_df is None:
            return None

        from src.utils.data.validation import create_data_interface_schema_for_storage

        input_features = datahandler.get_feature_columns()
        model_features = (
            preprocessor.get_output_columns() if preprocessor else input_features
        )

        data_interface_schema = create_data_interface_schema_for_storage(
            data_interface=self.recipe.data.data_interface,
            df=training_df,
            task_choice=self.recipe.task_choice,
            input_feature_columns=input_features,
            model_feature_columns=model_features,
        )

        required_cols = len(data_interface_schema.get("required_columns", []))
        logger.debug(f"[FACT] DataInterface schema 생성 완료: {required_cols}개 필수 컬럼")

        return data_interface_schema

    def _ensure_timestamp_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """timestamp 컬럼 datetime 변환"""
        ts_col = self._get_timestamp_column()

        if ts_col and ts_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df = df.copy()
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        return df

    def _get_timestamp_column(self) -> Optional[str]:
        """timestamp 컬럼명 추출 (fetcher → data_interface 순으로 폴백)"""
        fetcher_conf = self.recipe.data.fetcher
        data_interface = self.recipe.data.data_interface

        if fetcher_conf and getattr(fetcher_conf, "timestamp_column", None):
            return fetcher_conf.timestamp_column
        if getattr(data_interface, "timestamp_column", None):
            return data_interface.timestamp_column

        return None

    def _build_data_interface_config(
        self, input_features: list, model_features: list
    ) -> Dict[str, Any]:
        """data_interface_config 구성"""
        data_interface = self.recipe.data.data_interface
        ts_col = self._get_timestamp_column()

        return {
            "entity_columns": data_interface.entity_columns,
            "timestamp_column": ts_col,
            "task_type": self.recipe.task_choice,
            "target_column": data_interface.target_column,
            "treatment_column": getattr(data_interface, "treatment_column", None),
            "input_feature_columns": input_features,
            "model_feature_columns": model_features,
            "feature_columns": input_features,  # 하위 호환
        }
