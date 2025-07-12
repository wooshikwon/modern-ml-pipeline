import pandas as pd
import mlflow
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split

from src.settings.settings import Settings
from src.utils.logger import logger
from src.core.factory import Factory
from src.core.augmenter import get_augmenter
# ... (기존 import)

class Trainer:
    # ... (기존 __init__)

    def train(self, df: pd.DataFrame, context_params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Preprocessor], BaseModel, Dict[str, Any]]:
        logger.info("모델 학습 프로세스 시작...")
        context_params = context_params or {}

        # 1. 데이터 분할
        train_df, test_df = self._split_data(df)

        # 2. 피처 증강 (선택적)
        train_df_aug, test_df_aug = train_df, test_df
        augmenter = get_augmenter(self.settings)
        if augmenter:
            logger.info("피처 증강을 시작합니다.")
            train_df_aug = augmenter.augment(train_df, context_params)
            test_df_aug = augmenter.augment(test_df, context_params)
            logger.info("피처 증강 완료.")
        else:
            logger.info("Augmenter 설정이 없어 피처 증강을 건너뜁니다.")
        
        # ... (이후 로직은 동일)


    def _validate_schema(self, df: pd.DataFrame):
        """입력 데이터프레임이 data_interface에 정의된 스키마와 일치하는지 검증합니다."""
        expected_schema = self.settings.model.data_interface.features
        errors = []
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                errors.append(f"- 필수 컬럼 누락: '{col}'")
                continue
            
            actual_type = str(df[col].dtype)
            is_valid = False
            if expected_type == "numeric" and pd.api.types.is_numeric_dtype(df[col]):
                is_valid = True
            elif expected_type == "category" and (pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                is_valid = True
            
            if not is_valid:
                errors.append(f"- 컬럼 '{col}' 타입 불일치: 예상='{expected_type}', 실제='{actual_type}'")

        if errors:
            error_message = "모델 입력 데이터 스키마 검증 실패:\n" + "\n".join(errors)
            error_message += "\n\n'preprocessor' 또는 'augmenter' 설정을 확인하여 스키마를 맞추세요."
            raise TypeError(error_message)
        
        logger.info("스키마 검증 성공.")

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터를 학습/테스트 세트로 분할합니다."""
        treatment_col = self.settings.model.data_interface.treatment_col
        test_size = 0.2
        logger.info(f"데이터 분할 (테스트 사이즈: {test_size}, 기준: {treatment_col})")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df.get(treatment_col))
        logger.info(f"분할 완료: 학습셋 {len(train_df)} 행, 테스트셋 {len(test_df)} 행")
        return train_df, test_df

    def _evaluate(self, model: BaseModel, train_orig: pd.DataFrame, X_train: pd.DataFrame, test_orig: pd.DataFrame, X_test: pd.DataFrame) -> Dict[str, Any]:
        """학습된 모델의 성능을 종합적으로 평가합니다."""
        logger.info("모델 성능 평가 시작...")
        treatment_col = self.settings.model.data_interface.treatment_col
        target_col = self.settings.model.data_interface.target_col
        treatment_value = self.settings.model.data_interface.treatment_value
        metrics = {}
        train_uplift = model.predict(X_train)
        test_uplift = model.predict(X_test)
        for prefix, df, uplift in [('train', train_orig, train_uplift), ('test', test_orig, test_uplift)]:
            treatment_mask = df[treatment_col] == treatment_value
            control_mask = ~treatment_mask
            if treatment_mask.sum() == 0 or control_mask.sum() == 0:
                actual_ate = float('nan')
            else:
                actual_ate = df.loc[treatment_mask, target_col].mean() - df.loc[control_mask, target_col].mean()
            metrics[f'{prefix}_actual_ate'] = actual_ate
            metrics[f'{prefix}_predicted_ate'] = uplift.mean()
        logger.info(f"모델 성능 평가 완료: {metrics}")
        return metrics