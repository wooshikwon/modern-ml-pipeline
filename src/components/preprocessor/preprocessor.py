from __future__ import annotations
import pandas as pd
from typing import Optional, TYPE_CHECKING

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.interface import BasePreprocessor
from src.utils.system.logger import logger
from .registry import PreprocessorStepRegistry

if TYPE_CHECKING:
    from src.settings import Settings

class Preprocessor(BasePreprocessor):
    """
    Recipe에 정의된 여러 전처리 단계를 동적으로 조립하고 실행하는
    Pipeline Builder 클래스입니다.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.recipe.preprocessor  # Recipe 루트의 preprocessor 참조
        self.pipeline: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        logger.info("선언적 전처리 파이프라인 빌드를 시작합니다...")
        
        column_transformers = []
        if self.config and self.config.steps:
            for step in self.config.steps:
                # PreprocessorStep 객체에서 파라미터 추출 (type과 columns 제외)
                step_params = step.model_dump(exclude={'type', 'columns'})
                # None 값 제거
                step_params = {k: v for k, v in step_params.items() if v is not None}
                
                transformer = PreprocessorStepRegistry.create(step.type, **step_params)
                # 고유한 이름 생성 (type + columns hash)
                step_name = f"{step.type}_{hash(tuple(step.columns)) % 10000}"
                column_transformers.append((step_name, transformer, step.columns))
        
        preprocessor_stage = ColumnTransformer(transformers=column_transformers, remainder='passthrough')

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_stage)
        ])
        
        self.pipeline.fit(X, y)
        logger.info("전처리 파이프라인 빌드 및 학습 완료.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor가 아직 학습되지 않았습니다. 'fit'을 먼저 호출하세요.")
        
        # 구성에 정의된 컬럼이 입력에 없으면 기본값(0)으로 생성하여 변환 파이프라인이 실패하지 않도록 함
        try:
            required_columns = []
            if self.config and self.config.steps:
                for step in self.config.steps:
                    required_columns.extend(list(step.columns or []))
            for col in set(required_columns):
                if col not in X.columns:
                    X[col] = 0
        except Exception:
            pass

        X_transformed = self.pipeline.transform(X)
        
        try:
            feature_names = self.pipeline.get_feature_names_out()
        except Exception:
            feature_names = None

        return pd.DataFrame(X_transformed, index=X.index, columns=feature_names)

    def save(self, file_path: str):
        pass

    @classmethod
    def load(cls, file_path: str) -> 'Preprocessor':
        pass
        
