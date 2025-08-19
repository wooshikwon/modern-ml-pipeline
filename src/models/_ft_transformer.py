# src/models/_ft_transformer.py
import pandas as pd
from typing import Any
from rtdl_revisiting_models import FTTransformer

from src.interface import BaseModel
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class _FTTransformerWrapperBase(BaseModel):
    handles_own_preprocessing = True

    def __init__(self, **hyperparams: Any):
        self.model: FTTransformer | None = None
        self.hyperparams = hyperparams
        self._internal_preprocessor: ColumnTransformer | None = None

    def _initialize_and_fit(self, X: pd.DataFrame, y: pd.Series, d_out: int):
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = X.select_dtypes(include='number').columns.tolist()

        # 내부 전처리기 정의
        self._internal_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
            ],
            remainder='passthrough'
        )

        # 내부 전처리기 학습 및 데이터 변환
        X_transformed = self._internal_preprocessor.fit_transform(X)
        
        # Cardinality를 전처리기 학습 *후*에 계산하여 "unknown" 경우를 포함시킵니다.
        cat_cardinalities = []
        if categorical_features:
            # 'cat' 파이프라인에서 OrdinalEncoder를 가져옵니다.
            ordinal_encoder = self._internal_preprocessor.named_transformers_['cat']
            for categories in ordinal_encoder.categories_:
                # (실제 카테고리 개수 + 1)을 하여 unknown 값을 위한 공간 확보
                cat_cardinalities.append(len(categories) + 1)

        # FT-Transformer 모델 '지연 초기화' 및 학습
        self.model = FTTransformer(
            n_num_features=len(numerical_features),
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            **self.hyperparams
        )
        self.model.fit(X_transformed, y.to_numpy())
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self._internal_preprocessor is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        
        X_transformed = self._internal_preprocessor.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X_transformed)
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
            else:
                predictions = predictions.argmax(axis=1)
        else:
            predictions = self.model.predict(X_transformed)

        return pd.DataFrame(predictions, index=X.index, columns=['prediction'])

class FTTransformerClassifier(_FTTransformerWrapperBase):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'FTTransformerClassifier':
        # 이진 분류든 다중 분류든 항상 y.nunique()를 사용
        d_out = y.nunique()
        super()._initialize_and_fit(X, y, d_out)
        return self

class FTTransformerRegressor(_FTTransformerWrapperBase):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'FTTransformerRegressor':
        d_out = 1
        super()._initialize_and_fit(X, y, d_out)
        return self