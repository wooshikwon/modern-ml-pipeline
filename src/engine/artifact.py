from __future__ import annotations
import mlflow
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.components.augmenter import BaseAugmenter
    from src.components.preprocessor import BasePreprocessor


class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    Blueprint 원칙에 따라 학습의 모든 논리를 캡슐화하는 자기 완결적 아티팩트.
    이 Wrapper는 MLflow에 저장되어, 어떤 환경에서든 완전한 재현성을 보장합니다.
    """
    def __init__(self,
                 trained_model: Any,
                 trained_preprocessor: Optional[BasePreprocessor],
                 trained_augmenter: Optional[BaseAugmenter],
                 loader_sql_snapshot: str,
                 augmenter_config_snapshot: Dict[str, Any],
                 recipe_yaml_snapshot: str,
                 model_class_path: str,
                 hyperparameter_optimization: Optional[Dict[str, Any]],
                 training_methodology: Dict[str, Any],
                 data_schema: Optional[Dict[str, Any]] = None,
                 schema_validator: Optional[Any] = None,
                 signature: Optional[Any] = None):
        self.trained_model = trained_model
        self.trained_preprocessor = trained_preprocessor
        self.trained_augmenter = trained_augmenter
        self.loader_sql_snapshot = loader_sql_snapshot
        self.augmenter_config_snapshot = augmenter_config_snapshot
        self.recipe_yaml_snapshot = recipe_yaml_snapshot
        self.model_class_path = model_class_path
        self.hyperparameter_optimization = hyperparameter_optimization
        self.training_methodology = training_methodology
        # 🆕 Phase 4: 스키마 일관성 검증을 위한 메타데이터
        self.data_schema = data_schema
        self.schema_validator = schema_validator
        # 🆕 Phase 5: Enhanced Model Signature
        self.signature = signature

    def predict(self, context, model_input: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        배치 추론 및 API 서빙을 위한 통합 예측 인터페이스.
        실행 흐름: [🆕 스키마 검증 -> 피처 증강 -> 전처리 -> 예측]
        """
        params = params or {}
        run_mode = params.get("run_mode", "batch")

        # 0. 🆕 Phase 4: 자동 스키마 일관성 검증
        if run_mode == "batch" and self.schema_validator:
            try:
                self.schema_validator.validate_inference_consistency(model_input)
                from src.utils.system.logger import logger
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
        df_augmented = model_input.copy()
        if self.trained_augmenter:
            df_augmented = self.trained_augmenter.augment(df_augmented, run_mode=run_mode)

        # 2. 데이터 전처리 (Preprocessor)
        df_processed = df_augmented
        if self.trained_preprocessor:
            df_processed = self.trained_preprocessor.transform(df_processed)

        # 3. 예측 (Model)
        # 전처리기가 모델이 사용할 피처만 반환한다고 가정합니다.
        predictions = self.trained_model.predict(df_processed)

        # 4. 결과 포맷팅
        # 입력 데이터에 예측 결과를 추가하여 반환합니다.
        output_df = model_input.copy()
        output_df['prediction'] = predictions
        
        return output_df
