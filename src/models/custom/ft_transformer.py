# src/models/_ft_transformer.py
import pandas as pd
from typing import Any
from rtdl_revisiting_models import FTTransformer

from src.interface import BaseModel
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class FTTransformerWrapperBase(BaseModel):
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
        # rtdl-revisiting-models API에 맞는 기본값 설정
        ft_params = {
            'n_cont_features': len(numerical_features),
            'cat_cardinalities': cat_cardinalities,
            'd_out': d_out,
            # 필수 매개변수들 기본값 설정
            'd_block': self.hyperparams.get('d_block', 32),
            'n_blocks': self.hyperparams.get('n_blocks', 2),
            'attention_n_heads': self.hyperparams.get('attention_n_heads', 
                                                    self.hyperparams.get('n_heads', 2)),
            'attention_dropout': self.hyperparams.get('attention_dropout', 0.1),
            'ffn_d_hidden_multiplier': self.hyperparams.get('ffn_d_hidden_multiplier', 4),
            'ffn_dropout': self.hyperparams.get('ffn_dropout', 0.1),
            'residual_dropout': self.hyperparams.get('residual_dropout', 0.0)
        }
        
        # 추가 하이퍼파라미터가 있다면 포함 (위 필수 매개변수들과 충돌하지 않는 것만)
        extra_params = {k: v for k, v in self.hyperparams.items() 
                       if k not in ['d_block', 'n_blocks', 'n_heads', 'attention_n_heads', 
                                  'attention_dropout', 'ffn_d_hidden_multiplier', 
                                  'ffn_dropout', 'residual_dropout']}
        ft_params.update(extra_params)
        
        self.model = FTTransformer(**ft_params)
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

class FTTransformerClassifier(FTTransformerWrapperBase):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'FTTransformerClassifier':
        # 이진 분류든 다중 분류든 항상 y.nunique()를 사용
        d_out = y.nunique()
        super()._initialize_and_fit(X, y, d_out)
        return self

class FTTransformerRegressor(FTTransformerWrapperBase):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> 'FTTransformerRegressor':
        d_out = 1
        super()._initialize_and_fit(X, y, d_out)
        return self