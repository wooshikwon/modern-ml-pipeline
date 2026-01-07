# src/models/custom/ft_transformer.py
"""
FT-Transformer 모델 wrapper - BaseModel 직접 상속

rtdl_revisiting_models의 FTTransformer를 sklearn 인터페이스로 래핑합니다.
pytorch_utils의 공통 학습/예측 로직을 활용합니다.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.models.base import BaseModel
from src.utils.core.logger import logger

from .pytorch_utils import (
    create_dataloader,
    get_device,
    predict_with_pytorch_model,
    set_seed,
    train_pytorch_model,
)


class FTTransformerModule(nn.Module):
    """
    FTTransformer를 단일 입력 텐서로 사용할 수 있도록 래핑하는 nn.Module

    rtdl_revisiting_models의 FTTransformer는 forward(x_cont, x_cat)를 요구하지만,
    pytorch_utils의 공통 학습 루프는 단일 텐서를 전달합니다.
    이 래퍼가 입력을 분리하여 FTTransformer에 전달합니다.
    """

    def __init__(self, ft_transformer: FTTransformer, n_cont_features: int, n_cat_features: int):
        super().__init__()
        self.ft_transformer = ft_transformer
        self.n_cont_features = n_cont_features
        self.n_cat_features = n_cat_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        단일 입력 텐서를 x_cont, x_cat으로 분리하여 FTTransformer에 전달

        Args:
            x: (batch_size, n_cont_features + n_cat_features) 형태의 텐서

        Returns:
            FTTransformer 출력
        """
        if self.n_cat_features > 0:
            x_cont = x[:, : self.n_cont_features]
            x_cat = x[:, self.n_cont_features :].long()
        else:
            x_cont = x
            x_cat = None

        return self.ft_transformer(x_cont, x_cat)


class FTTransformerWrapperBase(BaseModel):
    """FT-Transformer 기반 모델의 공통 베이스 클래스"""

    handles_own_preprocessing = True

    def __init__(
        self,
        d_block: int = 64,
        n_blocks: int = 2,
        attention_n_heads: int = 4,
        attention_dropout: float = 0.1,
        ffn_d_hidden_multiplier: float = 4.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        """
        FT-Transformer 모델 초기화

        Args:
            d_block: 블록 차원 수
            n_blocks: Transformer 블록 수
            attention_n_heads: 어텐션 헤드 수
            attention_dropout: 어텐션 드롭아웃 비율
            ffn_d_hidden_multiplier: FFN 은닉층 크기 배수
            ffn_dropout: FFN 드롭아웃 비율
            residual_dropout: 잔차 연결 드롭아웃 비율
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            early_stopping_patience: 조기 종료 patience
            seed: 랜덤 시드
        """
        # 모델 구조 하이퍼파라미터
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        # 학습 하이퍼파라미터
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        # 하이퍼파라미터 저장 (테스트 호환성 및 요약용)
        self.hyperparams = {
            "d_block": d_block,
            "n_blocks": n_blocks,
            "attention_n_heads": attention_n_heads,
            "attention_dropout": attention_dropout,
            "ffn_d_hidden_multiplier": ffn_d_hidden_multiplier,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            **kwargs,
        }

        # 내부 상태
        self.device = get_device()
        self.model: Optional[nn.Module] = None
        self._internal_preprocessor: Optional[ColumnTransformer] = None
        self._numerical_features: list = []
        self._categorical_features: list = []
        self.is_fitted = False

        logger.info("[INIT:FTTransformer] 초기화 완료")
        logger.info(f"  구조: d_block={d_block}, n_blocks={n_blocks}, heads={attention_n_heads}")
        logger.info(f"  학습: epochs={epochs}, batch={batch_size}, lr={learning_rate}")

    def _build_preprocessor(self, X: pd.DataFrame) -> np.ndarray:
        """
        내부 전처리기 구축 및 데이터 변환

        Args:
            X: 입력 DataFrame

        Returns:
            전처리된 numpy array
        """
        self._categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self._numerical_features = X.select_dtypes(include="number").columns.tolist()

        # NaN 처리: 수치형은 0으로, 카테고리는 'missing'으로
        X_clean = X.copy()
        for col in self._numerical_features:
            X_clean[col] = X_clean[col].fillna(0)
        for col in self._categorical_features:
            X_clean[col] = X_clean[col].fillna("missing")

        nan_count = X[self._numerical_features].isna().sum().sum()
        if nan_count > 0:
            logger.info(f"[PREP:FTTransformer] NaN 처리: {nan_count}개 값을 0으로 대체")

        self._internal_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self._numerical_features),
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    self._categorical_features,
                ),
            ],
            remainder="passthrough",
        )

        X_transformed = self._internal_preprocessor.fit_transform(X_clean)
        logger.info(
            f"[PREP:FTTransformer] 전처리 완료 - num: {len(self._numerical_features)}, cat: {len(self._categorical_features)}"
        )

        return X_transformed.astype(np.float32)

    def _get_cat_cardinalities(self) -> list:
        """카테고리 변수의 cardinality 계산"""
        cat_cardinalities = []
        if self._categorical_features and self._internal_preprocessor is not None:
            ordinal_encoder = self._internal_preprocessor.named_transformers_["cat"]
            for categories in ordinal_encoder.categories_:
                cat_cardinalities.append(len(categories) + 1)  # unknown 포함
        return cat_cardinalities

    def _build_model(self, d_out: int) -> nn.Module:
        """
        FT-Transformer 모델 아키텍처 생성

        Args:
            d_out: 출력 차원 (classification: 클래스 수, regression: 1)

        Returns:
            FTTransformerModule 인스턴스 (pytorch_utils 호환)
        """
        cat_cardinalities = self._get_cat_cardinalities()

        # 최소 1개 이상의 피처 필요
        n_num = len(self._numerical_features)
        n_cat = len(cat_cardinalities) if cat_cardinalities else 0
        if n_num == 0 and n_cat == 0:
            raise ValueError(
                "FTTransformer는 최소 1개 이상의 numerical 또는 categorical 피처가 필요합니다. "
                f"현재: numerical={n_num}, categorical={len(self._categorical_features)}"
            )

        ft_transformer = FTTransformer(
            n_cont_features=len(self._numerical_features),
            cat_cardinalities=cat_cardinalities if cat_cardinalities else None,
            d_out=d_out,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )

        # pytorch_utils 호환을 위해 FTTransformerModule로 래핑
        model = FTTransformerModule(
            ft_transformer=ft_transformer,
            n_cont_features=len(self._numerical_features),
            n_cat_features=len(self._categorical_features),
        )

        logger.info(
            f"[BUILD:FTTransformer] 모델 생성 완료 - d_out={d_out}, cont={len(self._numerical_features)}, cat={len(self._categorical_features)}"
        )
        return model

    def _prepare_data_loaders(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2):
        """
        학습/검증 데이터 로더 생성

        Args:
            X: 전처리된 입력 데이터
            y: 타겟 데이터
            val_ratio: 검증 데이터 비율

        Returns:
            train_loader, val_loader 튜플
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - val_ratio))

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_loader = create_dataloader(X_train, y_train, self.batch_size, shuffle=True)
        val_loader = (
            create_dataloader(X_val, y_val, self.batch_size, shuffle=False)
            if len(X_val) > 0
            else None
        )

        logger.info(
            f"[DATA:FTTransformer] Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}"
        )

        return train_loader, val_loader

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """예측 시 NaN 처리"""
        X_clean = X.copy()
        for col in self._numerical_features:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(0)
        for col in self._categorical_features:
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna("missing")
        return X_clean

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        예측 수행

        Args:
            X: 입력 DataFrame

        Returns:
            예측 결과 DataFrame
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_clean = self._clean_data(X)
        X_transformed = self._internal_preprocessor.transform(X_clean).astype(np.float32)
        test_loader = create_dataloader(
            X_transformed, y=None, batch_size=self.batch_size, shuffle=False
        )

        predictions = predict_with_pytorch_model(self.model, test_loader, self.device)

        return pd.DataFrame(predictions, index=X.index, columns=["prediction"])


class FTTransformerClassifier(FTTransformerWrapperBase):
    """FT-Transformer 분류 모델"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_classes = None
        self._label_encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "FTTransformerClassifier":
        """
        분류 모델 학습

        Args:
            X: 입력 DataFrame
            y: 타겟 Series

        Returns:
            학습된 모델 인스턴스
        """
        logger.info("[TRAIN:FTTransformer] Classification 학습 시작")
        set_seed(self.seed)

        # 1. 전처리
        X_transformed = self._build_preprocessor(X)

        # 2. 라벨 인코딩
        self._n_classes = y.nunique()
        y_encoded = y.values
        if y.dtype == "object" or y.dtype.name == "category":
            from sklearn.preprocessing import LabelEncoder

            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)

        # 3. 모델 생성
        self.model = self._build_model(d_out=self._n_classes).to(self.device)

        # 4. 데이터 로더
        train_loader, val_loader = self._prepare_data_loaders(X_transformed, y_encoded)

        # 5. 학습 (CrossEntropyLoss 사용)
        criterion = nn.CrossEntropyLoss()
        history = train_pytorch_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            criterion=criterion,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
            log_interval=max(1, self.epochs // 10),
        )

        self.is_fitted = True
        logger.info(f"[TRAIN:FTTransformer] 학습 완료 - Best Epoch: {history.get('best_epoch', 0)}")

        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        분류 예측 수행 (클래스 라벨 반환)

        Args:
            X: 입력 DataFrame

        Returns:
            예측 클래스 라벨 DataFrame
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_clean = self._clean_data(X)
        X_transformed = self._internal_preprocessor.transform(X_clean).astype(np.float32)
        test_loader = create_dataloader(
            X_transformed, y=None, batch_size=self.batch_size, shuffle=False
        )

        # 예측
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_data in test_loader:
                batch_x = batch_data[0].to(self.device)
                outputs = self.model(batch_x)
                # argmax로 클래스 라벨 반환
                pred_labels = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(pred_labels)

        return pd.DataFrame(predictions, index=X.index, columns=["prediction"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        분류 확률 예측 수행

        Args:
            X: 입력 DataFrame

        Returns:
            예측 확률 배열 - shape: (n_samples, n_classes)
            sklearn 호환 형식으로 모든 클래스의 확률 반환
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_clean = self._clean_data(X)
        X_transformed = self._internal_preprocessor.transform(X_clean).astype(np.float32)
        test_loader = create_dataloader(
            X_transformed, y=None, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        probs_list = []

        with torch.no_grad():
            for batch_data in test_loader:
                batch_x = batch_data[0].to(self.device)
                outputs = self.model(batch_x)
                probs = torch.softmax(outputs, dim=1)
                probs_list.append(probs.cpu().numpy())

        # (n_samples, n_classes) 형태로 반환 - sklearn 호환
        return np.vstack(probs_list)


class FTTransformerRegressor(FTTransformerWrapperBase):
    """FT-Transformer 회귀 모델"""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "FTTransformerRegressor":
        """
        회귀 모델 학습

        Args:
            X: 입력 DataFrame
            y: 타겟 Series

        Returns:
            학습된 모델 인스턴스
        """
        logger.info("[TRAIN:FTTransformer] Regression 학습 시작")
        set_seed(self.seed)

        # 1. 전처리
        X_transformed = self._build_preprocessor(X)
        y_array = y.values.astype(np.float32)

        # 2. 모델 생성 (d_out=1 for regression)
        self.model = self._build_model(d_out=1).to(self.device)

        # 3. 데이터 로더
        train_loader, val_loader = self._prepare_data_loaders(X_transformed, y_array)

        # 4. 학습 (MSELoss 사용)
        criterion = nn.MSELoss()
        history = train_pytorch_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            criterion=criterion,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
            log_interval=max(1, self.epochs // 10),
        )

        self.is_fitted = True
        logger.info(f"[TRAIN:FTTransformer] 학습 완료 - Best Epoch: {history.get('best_epoch', 0)}")

        return self


__all__ = ["FTTransformerClassifier", "FTTransformerRegressor", "FTTransformerWrapperBase"]
