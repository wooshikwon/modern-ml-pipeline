from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import importlib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.interface import BasePreprocessor
from src.utils.system.logger import logger
from ._registry import PreprocessorStepRegistry

if TYPE_CHECKING:
    from src.settings import Settings

class Preprocessor(BasePreprocessor):
    """
    Recipe에 정의된 여러 전처리 단계를 동적으로 조립하고 실행하는
    Pipeline Builder 클래스입니다.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.recipe.model.preprocessor
        self.pipeline: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        logger.info("선언적 전처리 파이프라인 빌드를 시작합니다...")
        
        column_transformers = []
        for name, conf in self.config.column_transforms.items():
            # class_path 대신 type을 사용하여 Registry에서 생성
            transformer = PreprocessorStepRegistry.create(conf.type, **conf.params)
            column_transformers.append((name, transformer, conf.columns))
        
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
            for name, conf in self.config.column_transforms.items():
                required_columns.extend(list(conf.columns or []))
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
        
