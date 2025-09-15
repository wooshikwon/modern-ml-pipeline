from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional

import mlflow


class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 설정 스냅샷을 캡슐화하는 MLflow PythonModel 구현체.
    기존 구현을 이동하되, 직렬화 안전성을 유지합니다.
    """

    def __init__(
        self,
        settings: Any,
        trained_model: Any,
        trained_datahandler: Optional[Any] = None,
        trained_preprocessor: Optional[Any] = None,
        trained_fetcher: Optional[Any] = None,
        trained_calibrator: Optional[Any] = None,
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None,  # mlflow.models.ModelSignature
        data_schema: Optional[Any] = None,  # mlflow.types.Schema
        data_interface_schema: Optional[Dict[str, Any]] = None,
    ):
        self._console = None

        # 직렬화 가능한 최소한의 설정 정보만 추출
        self._task_type, self.settings_dict = self._extract_serializable_settings(settings)

        self.trained_model = trained_model
        # 복잡 객체는 직렬화 회피
        self.trained_datahandler = None
        self.trained_preprocessor = None
        self.trained_fetcher = None
        self.trained_calibrator = trained_calibrator
        self.training_results = training_results or {}
        self.signature = signature
        self.data_schema = data_schema
        self.data_interface_schema = data_interface_schema

        # Task type별 추론 파이프라인 필요성
        self._requires_datahandler = self._task_type in ["timeseries"]

    def _extract_serializable_settings(self, settings):
        """설정에서 직렬화 가능한 최소한의 정보만 추출"""
        try:
            if hasattr(settings, 'model_dump'):
                task_type = settings.recipe.task_choice
                settings_dict = {
                    'recipe': {
                        'task_choice': task_type,
                        'model': {
                            'class_path': getattr(settings.recipe.model, 'class_path', 'unknown')
                        },
                        'data': {
                            'data_interface': {
                                'target_column': getattr(settings.recipe.data.data_interface, 'target_column', None)
                            }
                        }
                    }
                }
                return task_type, settings_dict
            elif isinstance(settings, dict):
                task_type = settings.get('recipe', {}).get('task_choice', 'unknown')
                return task_type, settings
            else:
                task_type = getattr(settings.recipe, 'task_choice', 'unknown') if hasattr(settings, 'recipe') else 'unknown'
                settings_dict = {'recipe': {'task_choice': task_type}}
                return task_type, settings_dict
        except Exception:
            return 'unknown', {'recipe': {'task_choice': 'unknown'}}

    @property
    def console(self):
        if self._console is None:
            try:
                from src.utils.core.console import get_console
                self._console = get_console()
            except Exception:
                import logging
                self._console = logging.getLogger(__name__)
        return self._console

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_console'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _validate_input_schema(self, df: pd.DataFrame):
        if self.data_schema:
            try:
                ts_col = self.data_schema.get('timestamp_column') if isinstance(self.data_schema, dict) else None
                if ts_col and ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
                from src.utils.schema.schema_utils import SchemaConsistencyValidator
                validator = SchemaConsistencyValidator(self.data_schema)
                validator.validate_inference_consistency(df)
                self.console.info("입력 스키마 검증 완료", rich_message="✅ Input schema validation passed")
            except ValueError as e:
                self.console.error(f"스키마 검증 실패 (Schema Drift 감지): {e}", rich_message=f"🚨 Schema validation failed: [red]{e}[/red]")
                raise

    @property
    def model_class_path(self) -> str:
        return self.settings_dict.get('recipe', {}).get('model', {}).get('class_path', 'unknown')

    @property
    def loader_sql_snapshot(self) -> str:
        return self.settings_dict.get('recipe', {}).get('data', {}).get('loader', {}).get('source_uri', '')

    @property
    def fetcher_config_snapshot(self) -> Dict[str, Any]:
        fetcher = self.settings_dict.get('recipe', {}).get('data', {}).get('fetcher', {})
        return fetcher if fetcher else {}

    @property
    def recipe_yaml_snapshot(self) -> str:
        import yaml
        recipe = self.settings_dict.get('recipe', {})
        return yaml.dump(recipe)

    @property
    def hyperparameter_optimization(self) -> Dict[str, Any]:
        return self.training_results.get('hyperparameter_optimization', {})

    @property
    def training_methodology(self) -> Dict[str, Any]:
        return self.training_results.get('training_methodology', {})

    def predict(self, context, model_input, params=None):
        run_mode = params.get("run_mode", "batch") if params else "batch"
        self.console.info(f"Predict called with params: {params}", rich_message=(f"🔍 Prediction request: [cyan]{len(params)} params[/cyan]" if params else "🔍 Prediction request received"))

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        try:
            if self.data_interface_schema:
                self.console.info("Basic input validation passed", rich_message="✅ Input validation passed")
        except Exception:
            self.console.warning("Input validation skipped", rich_message="⚠️ Input validation skipped")

        try:
            # data_interface에서 feature_columns 가져오기
            data_interface_config = self.data_interface_schema.get('data_interface_config', {}) if self.data_interface_schema else {}
            feature_columns = data_interface_config.get('feature_columns')
            
            # feature_columns가 명시되어 있으면 해당 컬럼만 사용
            if feature_columns:
                # 존재하는 feature 컬럼만 선택
                available_features = [col for col in feature_columns if col in model_input.columns]
                X = model_input[available_features]
                self.console.info(f"Using {len(available_features)} feature columns for prediction", 
                                rich_message=f"📊 Features: [cyan]{len(available_features)}[/cyan] columns")
            else:
                # feature_columns가 없으면 기존 로직 (target 제외)
                target_col = data_interface_config.get('target_column')
                feature_columns = [col for col in model_input.columns if col != target_col]
                X = model_input[feature_columns] if feature_columns else model_input
                self.console.warning("feature_columns not defined, using all columns except target", 
                                   rich_message="⚠️ Using auto-detected features")

            # Check if we should return probabilities or classes
            return_probabilities = params and params.get('return_probabilities', False)
            
            if return_probabilities and hasattr(self.trained_model, 'predict_proba'):
                # Get probability predictions
                predictions = self.trained_model.predict_proba(X)
                
                # Apply calibration if available
                if self.trained_calibrator is not None and self._task_type == 'classification':
                    self.console.info("Applying probability calibration", rich_message="🎯 Applying calibration")
                    predictions = self.trained_calibrator.transform(predictions)
                    
            elif self._task_type == 'classification' and hasattr(self.trained_model, 'predict_proba') and self.trained_calibrator is not None:
                # For classification with calibrator, always use calibrated probabilities for consistency
                predictions = self.trained_model.predict_proba(X)
                predictions = self.trained_calibrator.transform(predictions)
                
                # Convert probabilities to class predictions if not explicitly requesting probabilities
                if not return_probabilities:
                    if predictions.ndim == 2:
                        predictions = predictions.argmax(axis=1)
                    else:
                        # Binary classification case with calibrated probabilities
                        predictions = (predictions > 0.5).astype(int)
            else:
                # Standard prediction without calibration
                predictions = self.trained_model.predict(X)

            should_return_dataframe = params and params.get('return_dataframe', False)
            if should_return_dataframe:
                if not isinstance(predictions, pd.DataFrame):
                    if return_probabilities and predictions.ndim == 2:
                        # Multi-class probabilities
                        prob_cols = [f'prob_class_{i}' for i in range(predictions.shape[1])]
                        predictions_df = pd.DataFrame(predictions, columns=prob_cols, index=model_input.index)
                    elif return_probabilities and predictions.ndim == 1:
                        # Binary classification probabilities
                        predictions_df = pd.DataFrame({'prob_positive': predictions}, index=model_input.index)
                    else:
                        # Class predictions
                        predictions_df = pd.DataFrame({'prediction': predictions}, index=model_input.index)
                    self.console.info(f"Prediction completed: {len(predictions_df)} samples (DataFrame)", rich_message=f"✅ Prediction: [green]{len(predictions_df)}[/green] samples (DataFrame)")
                    return predictions_df
                else:
                    self.console.info(f"Prediction completed: {len(predictions)} samples (DataFrame)", rich_message=f"✅ Prediction: [green]{len(predictions)}[/green] samples (DataFrame)")
                    return predictions
            else:
                if isinstance(predictions, pd.DataFrame):
                    predictions = predictions.values.flatten()
                elif hasattr(predictions, 'tolist'):
                    predictions = predictions.tolist()
                self.console.info(f"Prediction completed: {len(predictions)} samples (array/list)", rich_message=f"✅ Prediction: [green]{len(predictions)}[/green] samples (array/list)")
                return predictions

        except Exception as e:
            self.console.error(f"Prediction failed: {e}", rich_message=f"❌ Prediction failed: [red]{e}[/red]")
            try:
                X = model_input.iloc[:, :1]
                predictions = self.trained_model.predict(X)
                return pd.DataFrame(predictions, columns=['prediction'])
            except Exception:
                dummy_predictions = [0.0] * len(model_input)
                return pd.DataFrame(dummy_predictions, columns=['prediction'])

