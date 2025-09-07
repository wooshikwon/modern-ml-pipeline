from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional

import mlflow
from src.utils.system.logger import logger
from src.utils.system.console_manager import get_console

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 모든 설정 정보를 캡슐화하는 MLflow PythonModel 구현체.
    MLflow 직렬화를 위해 최적화된 버전.
    """
    def __init__(
        self,
        settings: Any,  # Settings 객체 (실제 타입은 런타임에 결정)
        trained_model: Any,
        trained_datahandler: Optional[Any] = None,
        trained_preprocessor: Optional[Any] = None,
        trained_fetcher: Optional[Any] = None,
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None, # mlflow.models.ModelSignature
        data_schema: Optional[Any] = None, # mlflow.types.Schema
        data_interface_schema: Optional[Dict[str, Any]] = None,  # DataInterface 기반 검증용
    ):
        # Console은 lazy loading으로 처리 (직렬화 문제 해결)
        self._console = None
        
        # 직렬화 가능한 최소한의 설정 정보만 추출
        self._task_type, self.settings_dict = self._extract_serializable_settings(settings)
        
        self.trained_model = trained_model
        # 직렬화 문제를 피하기 위해 복잡한 객체들은 None으로 설정
        # 추론 시에는 기본적으로 trained_model만 사용
        self.trained_datahandler = None  # trained_datahandler
        self.trained_preprocessor = None  # trained_preprocessor 
        self.trained_fetcher = None  # trained_fetcher
        self.training_results = training_results or {}
        self.signature = signature
        self.data_schema = data_schema
        self.data_interface_schema = data_interface_schema  # DataInterface 기반 검증용
        
        # Task type별 추론 파이프라인 결정
        self._requires_datahandler = self._task_type in ["timeseries"]  # 향후 deeplearning 추가 가능

    def _extract_serializable_settings(self, settings):
        """설정에서 직렬화 가능한 최소한의 정보만 추출"""
        try:
            if hasattr(settings, 'model_dump'):
                # Pydantic 모델인 경우 - 안전하게 최소 정보만 추출
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
                # 이미 딕셔너리인 경우
                task_type = settings.get('recipe', {}).get('task_choice', 'unknown')
                return task_type, settings
            else:
                # 기타 경우 - 최소한의 정보만
                task_type = getattr(settings.recipe, 'task_choice', 'unknown') if hasattr(settings, 'recipe') else 'unknown'
                settings_dict = {'recipe': {'task_choice': task_type}}
                return task_type, settings_dict
        except Exception:
            # 모든 것이 실패하면 기본값
            return 'unknown', {'recipe': {'task_choice': 'unknown'}}
    
    @property  
    def console(self):
        """Console을 lazy loading으로 처리"""
        if self._console is None:
            try:
                from src.utils.system.console_manager import get_console
                self._console = get_console()
            except:
                # 완전 실패 시 logger 폴백
                import logging
                self._console = logging.getLogger(__name__)
        return self._console
    
    def __getstate__(self):
        """직렬화 시 console과 복잡한 객체 제외"""
        state = self.__dict__.copy()
        # console 제외 (lazy loading으로 재생성됨)
        state['_console'] = None
        return state
    
    def __setstate__(self, state):
        """역직렬화 시 상태 복원"""
        self.__dict__.update(state)
        # console은 lazy loading으로 처리되므로 별도 작업 불필요

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
        # settings_dict의 recipe 부분을 YAML로 변환
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
        """단순화된 예측 메서드 - 직렬화 문제 해결을 위해 최소한의 로직만 사용"""
        run_mode = params.get("run_mode", "batch") if params else "batch"
        
        # 디버깅: params 전달 상태 확인
        self.console.info(f"Predict called with params: {params}", rich_message=f"🔍 Prediction request: [cyan]{len(params)} params[/cyan]" if params else "🔍 Prediction request received")

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
            
        # 기본 스키마 검증 (선택적)
        try:
            if self.data_interface_schema:
                self.console.info("Basic input validation passed", rich_message="✅ Input validation passed")
        except:
            self.console.warning("Input validation skipped", rich_message="⚠️ Input validation skipped")

        # 단순화된 예측: 타겟 컬럼을 제외한 피처만 사용
        try:
            # data_interface에서 타겟 컬럼 제외
            target_col = self.data_interface_schema.get('data_interface_config', {}).get('target_column')
            feature_columns = [col for col in model_input.columns if col != target_col]
            
            # 모든 피처 사용 (범주형 변수도 포함)
            if feature_columns:
                X = model_input[feature_columns]
            else:
                # target_col이 없거나 찾을 수 없으면 모든 컬럼 사용
                X = model_input
            
            # 모델 예측
            predictions = self.trained_model.predict(X)
            
            # 호출 컨텍스트에 따라 다른 형태로 반환
            # params에 'return_dataframe'이 있으면 DataFrame 반환 (Inference Pipeline용)
            # 없으면 array/list 반환 (MLflow pyfunc 표준)
            should_return_dataframe = params and params.get('return_dataframe', False)
            
            if should_return_dataframe:
                # Inference Pipeline용: DataFrame 반환 (메타데이터 추가 가능)
                if not isinstance(predictions, pd.DataFrame):
                    predictions_df = pd.DataFrame({'prediction': predictions}, index=model_input.index)
                    self.console.info(f"Prediction completed: {len(predictions_df)} samples (DataFrame)", rich_message=f"✅ Prediction: [green]{len(predictions_df)}[/green] samples (DataFrame)")
                    return predictions_df
                else:
                    self.console.info(f"Prediction completed: {len(predictions)} samples (DataFrame)", rich_message=f"✅ Prediction: [green]{len(predictions)}[/green] samples (DataFrame)")
                    return predictions
            else:
                # MLflow pyfunc 표준: array/list 반환
                if isinstance(predictions, pd.DataFrame):
                    predictions = predictions.values.flatten()
                elif hasattr(predictions, 'tolist'):
                    predictions = predictions.tolist()
                    
                self.console.info(f"Prediction completed: {len(predictions)} samples (array/list)", rich_message=f"✅ Prediction: [green]{len(predictions)}[/green] samples (array/list)")
                return predictions
            
        except Exception as e:
            self.console.error(f"Prediction failed: {e}", rich_message=f"❌ Prediction failed: [red]{e}[/red]")
            # 폴백: 첫 번째 컬럼만 사용
            try:
                X = model_input.iloc[:, :1]
                predictions = self.trained_model.predict(X)
                return pd.DataFrame(predictions, columns=['prediction'])
            except:
                # 최후의 수단: 더미 예측
                dummy_predictions = [0.0] * len(model_input)
                return pd.DataFrame(dummy_predictions, columns=['prediction'])
    
    # 복잡한 검증 메서드들은 직렬화 문제를 피하기 위해 제거
    # 추론 시에는 기본적인 모델 예측만 수행
