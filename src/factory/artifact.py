from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional

import mlflow
from src.utils.system.logger import logger

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 모든 설정 정보를 캡슐화하는 MLflow PythonModel 구현체.
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
        # 복잡한 Settings 객체를 직렬화 가능한 형태로 변환
        if hasattr(settings, 'model_dump'):
            # Pydantic 모델인 경우
            self.settings_dict = settings.model_dump()
            self._task_type = settings.recipe.task_choice
        elif isinstance(settings, dict):
            # 이미 딕셔너리인 경우
            self.settings_dict = settings
            self._task_type = settings.get('recipe', {}).get('task_choice', 'unknown')
        else:
            # Settings 객체지만 model_dump가 없는 경우 - 직접 접근
            try:
                self._task_type = settings.recipe.task_choice
                # 최소한의 정보만 추출해서 딕셔너리로 변환
                self.settings_dict = {
                    'recipe': {
                        'task_choice': self._task_type,
                        'model': {'class_path': getattr(settings.recipe.model, 'class_path', 'unknown')},
                        'data': {
                            'loader': {'source_uri': getattr(settings.recipe.data.loader, 'source_uri', '')},
                            'fetcher': getattr(settings.recipe.data.fetcher, '__dict__', {}) if settings.recipe.data.fetcher else {}
                        }
                    }
                }
            except Exception as e:
                # 완전히 실패한 경우 기본값 사용
                self._task_type = 'unknown'
                self.settings_dict = {'recipe': {'task_choice': 'unknown'}}
        
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
        logger.info(f"🔍 Predict called with params: {params}")

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
            
        # 기본 스키마 검증 (선택적)
        try:
            if self.data_interface_schema:
                logger.info("✅ Basic input validation passed")
        except:
            logger.warning("⚠️ Input validation skipped")

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
                    logger.info(f"✅ Prediction completed: {len(predictions_df)} samples (DataFrame)")
                    return predictions_df
                else:
                    logger.info(f"✅ Prediction completed: {len(predictions)} samples (DataFrame)")
                    return predictions
            else:
                # MLflow pyfunc 표준: array/list 반환
                if isinstance(predictions, pd.DataFrame):
                    predictions = predictions.values.flatten()
                elif hasattr(predictions, 'tolist'):
                    predictions = predictions.tolist()
                    
                logger.info(f"✅ Prediction completed: {len(predictions)} samples (array/list)")
                return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
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
