from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

import mlflow
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.interface import BasePreprocessor, BaseFetcher, BaseDataHandler
    from src.settings import Settings

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 모든 설정 정보를 캡슐화하는 MLflow PythonModel 구현체.
    """
    def __init__(
        self,
        settings: Settings,
        trained_model: Any,
        trained_datahandler: Optional[BaseDataHandler],
        trained_preprocessor: Optional[BasePreprocessor],
        trained_fetcher: Optional[BaseFetcher],
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None, # mlflow.models.ModelSignature
        data_schema: Optional[Any] = None, # mlflow.types.Schema
    ):
        self.settings = settings
        self.trained_model = trained_model
        self.trained_datahandler = trained_datahandler
        self.trained_preprocessor = trained_preprocessor
        self.trained_fetcher = trained_fetcher
        self.training_results = training_results or {}
        self.signature = signature
        self.data_schema = data_schema
        
        # Task type별 추론 파이프라인 결정
        self._task_type = settings.recipe.data.data_interface.task_type
        self._requires_datahandler = self._task_type in ["timeseries"]  # 향후 deeplearning 추가 가능

    def _validate_input_schema(self, df: pd.DataFrame):
        """입력 데이터프레임의 스키마를 검증합니다."""
        if self.data_schema:
            try:
                # Timestamp 컬럼이 문자열로 들어오는 단순 배치 입력을 대비해 사전 변환 시도
                ts_col = self.data_schema.get('timestamp_column') if isinstance(self.data_schema, dict) else None
                if ts_col and ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
                from src.utils.system.schema_utils import SchemaConsistencyValidator
                validator = SchemaConsistencyValidator(self.data_schema)
                validator.validate_inference_consistency(df)
                logger.info("✅ PyfuncWrapper: 입력 스키마 검증 완료.")
            except ValueError as e:
                logger.error(f"🚨 PyfuncWrapper: 스키마 검증 실패 (Schema Drift 감지): {e}")
                raise

    @property
    def model_class_path(self) -> str:
        return self.settings.recipe.model.class_path

    @property
    def loader_sql_snapshot(self) -> str:
        return self.settings.recipe.data.loader.source_uri

    @property
    def fetcher_config_snapshot(self) -> Dict[str, Any]:
        if self.settings.recipe.data.fetcher:
            return self.settings.recipe.data.fetcher.model_dump()
        return {}

    @property
    def recipe_yaml_snapshot(self) -> str:
        # ToDo: Implement a robust way to get original yaml text
        # For now, we can dump the pydantic model back to yaml string
        import yaml
        return yaml.dump(self.settings.recipe.model_dump())

    @property
    def hyperparameter_optimization(self) -> Dict[str, Any]:
        return self.training_results.get('hyperparameter_optimization', {})

    @property
    def training_methodology(self) -> Dict[str, Any]:
        return self.training_results.get('training_methodology', {})

    def predict(self, context, model_input, params=None):
        run_mode = params.get("run_mode", "batch") if params else "batch"

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
            
        # 1. 자동 스키마 검증
        self._validate_input_schema(model_input)

        # 2. 올바른 파이프라인 순서: Fetcher → DataHandler → Preprocessor → Model
        if self._requires_datahandler and self.trained_datahandler:
            return self._predict_with_datahandler(model_input, run_mode)
        else:
            return self._predict_traditional(model_input, run_mode)
    
    def _predict_with_datahandler(self, model_input: pd.DataFrame, run_mode: str) -> pd.DataFrame:
        """DataHandler가 필요한 task (timeseries 등)의 추론 파이프라인"""
        logger.info(f"🔄 DataHandler 파이프라인 실행 (task_type: {self._task_type})")
        
        # 1. Fetcher: 피처 증강
        fetched_df = self.trained_fetcher.fetch(model_input, run_mode=run_mode)
        
        # 2. DataHandler: 특성 생성/변환 (재현성 보장)
        X, _, additional_data = self.trained_datahandler.prepare_data(fetched_df)
        
        # 3. Preprocessor: 스케일링/인코딩
        if self.trained_preprocessor:
            X = self.trained_preprocessor.transform(X)
        
        # 4. Model: 예측
        predictions = self.trained_model.predict(X)
        
        # 5. 결과 구성 (timeseries의 경우 timestamp 정보 추가)
        result_df = pd.DataFrame(predictions, columns=['prediction'], index=model_input.index)
        
        if self._task_type == "timeseries" and additional_data.get('timestamp') is not None:
            result_df['timestamp'] = additional_data['timestamp']
        
        logger.info(f"✅ DataHandler 파이프라인 완료. 예측 결과: {len(result_df)}개")
        return result_df
    
    def _predict_traditional(self, model_input: pd.DataFrame, run_mode: str) -> pd.DataFrame:
        """기존 방식 (tabular 등)의 추론 파이프라인"""
        logger.info(f"🔄 기존 파이프라인 실행 (task_type: {self._task_type})")
        
        # 기존 로직: Fetcher → Preprocessor → Model
        fetched_df = self.trained_fetcher.fetch(model_input, run_mode=run_mode)
        preprocessed_df = self.trained_preprocessor.transform(fetched_df) if self.trained_preprocessor else fetched_df
        predictions = self.trained_model.predict(preprocessed_df)
        
        result_df = pd.DataFrame(predictions, columns=['prediction'], index=model_input.index)
        return result_df
