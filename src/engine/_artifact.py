from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

import mlflow
from src.utils.system.logger import logger

if TYPE_CHECKING:
    from src.interface import BasePreprocessor, BaseAugmenter
    from src.settings import Settings

class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 모든 설정 정보를 캡슐화하는 MLflow PythonModel 구현체.
    """
    def __init__(
        self,
        settings: Settings,
        trained_model: Any,
        trained_preprocessor: Optional[BasePreprocessor],
        trained_augmenter: Optional[BaseAugmenter],
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None, # mlflow.models.ModelSignature
        data_schema: Optional[Any] = None, # mlflow.types.Schema
    ):
        self.settings = settings
        self.trained_model = trained_model
        self.trained_preprocessor = trained_preprocessor
        self.trained_augmenter = trained_augmenter
        self.training_results = training_results or {}
        self.signature = signature
        self.data_schema = data_schema

    @property
    def model_class_path(self) -> str:
        return self.settings.recipe.model.class_path

    @property
    def loader_sql_snapshot(self) -> str:
        return self.settings.recipe.model.loader.source_uri

    @property
    def augmenter_config_snapshot(self) -> Dict[str, Any]:
        if self.settings.recipe.model.augmenter:
            return self.settings.recipe.model.augmenter.model_dump()
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

        # 0. 🆕 Phase 4: 자동 스키마 일관성 검증
        if run_mode == "batch" and self.data_schema:
            try:
                self.data_schema.validate_inference_consistency(model_input)
                logger.info("✅ PyfuncWrapper 자동 스키마 검증 완료")
            except ValueError as e:
                # Schema Drift 감지 → 상세한 진단 메시지
                raise ValueError(f"🚨 PyfuncWrapper Schema Drift 감지: {e}")
        elif run_mode != "batch":
            # API 서빙 모드에서도 검증 (성능상 간단한 검증만)
            if self.data_schema and 'inference_columns' in self.data_schema:
                missing_cols = set(self.data_schema['inference_columns']) - set(model_input.columns)
                if missing_cols:
                    raise ValueError(f"🚨 API 요청 스키마 불일치: 필수 컬럼 누락 {missing_cols}")

        # 1. 피처 증강 (Augmenter)
        augmented_df = self.trained_augmenter.augment(model_input, run_mode=run_mode)

        # 2. 데이터 전처리 (Preprocessor)
        preprocessed_df = self.trained_preprocessor.transform(augmented_df) if self.trained_preprocessor else augmented_df

        # 3. 예측 (Model)
        # 전처리기가 모델이 사용할 피처만 반환한다고 가정합니다.
        predictions = self.trained_model.predict(preprocessed_df)

        # 4. 결과 포맷팅
        # 입력 데이터에 예측 결과를 추가하여 반환합니다.
        result_df = pd.DataFrame(predictions, columns=['prediction'], index=model_input.index)
        return result_df
