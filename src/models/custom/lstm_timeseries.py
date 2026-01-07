# src/models/custom/lstm_timeseries.py
"""
LSTM TimeSeries 모델 - BaseModel 직접 상속

PyTorch 기반 LSTM을 사용한 시계열 예측 모델입니다.
DeepLearning DataHandler와 완전 통합되어 3D 시퀀스 데이터를 처리합니다.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch.nn as nn

from src.models.base import BaseModel
from src.utils.core.logger import logger

from .pytorch_utils import (
    create_dataloader,
    get_device,
    predict_with_pytorch_model,
    set_seed,
    train_pytorch_model,
)


class LSTMTimeSeries(BaseModel):
    """LSTM 기반 시계열 예측 모델 - BaseModel 직접 상속"""

    handles_own_preprocessing = True

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        bidirectional: bool = False,
        **kwargs,
    ):
        """
        LSTM TimeSeries 모델 초기화

        Args:
            hidden_dim: LSTM 은닉층 차원 수
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            early_stopping_patience: 조기 종료 patience
            bidirectional: 양방향 LSTM 사용 여부
        """
        # 하이퍼파라미터 저장
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.bidirectional = bidirectional

        # 내부 상태
        self.device = get_device()
        self.model = None
        self.is_fitted = False
        self.sequence_info = None  # 시퀀스 메타데이터 저장

        logger.info("[INIT:LSTM] 초기화 완료")
        logger.info(f"  구조: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
        logger.info(f"  학습: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
        logger.info(f"  설정: bidirectional={bidirectional}, device={self.device}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "LSTMTimeSeries":
        """
        LSTM 모델 학습

        Args:
            X: 시퀀스 데이터가 flatten된 DataFrame
            y: 타겟 Series
            **kwargs: 메타데이터 (original_sequence_shape 등)

        Returns:
            학습된 모델 인스턴스
        """
        logger.info("[TRAIN:LSTM] 학습 시작")

        # 시드 설정 (재현성)
        set_seed(42)

        # 1. 메타데이터에서 시퀀스 정보 복원
        self.sequence_info = self._extract_sequence_info(X, kwargs)
        X_3d = self._reconstruct_3d_sequences(X, self.sequence_info)

        logger.info(f"[PREP:LSTM] 시퀀스 복원: {X.shape} -> {X_3d.shape}")

        # 2. 데이터 검증
        self._validate_sequence_data(X_3d, y)

        # 3. Train/Validation 분할 (시계열이므로 시간 순서 유지)
        X_train, X_val, y_train, y_val = self._time_based_split(X_3d, y)

        # 4. 모델 아키텍처 구축
        n_samples, seq_len, n_features = X_3d.shape
        self.model = self._build_lstm_model(n_features).to(self.device)

        logger.info(
            f"[BUILD:LSTM] 모델 구축 완료 - features={n_features}, hidden={self.hidden_dim}"
        )

        # 5. DataLoader 생성
        train_loader = create_dataloader(
            X_train, y_train, self.batch_size, shuffle=False
        )  # 시계열은 순서 유지
        val_loader = (
            create_dataloader(X_val, y_val, self.batch_size, shuffle=False)
            if X_val is not None
            else None
        )

        # 6. LSTM 학습
        history = train_pytorch_model(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self.device,
            early_stopping_patience=self.early_stopping_patience,
            log_interval=max(1, self.epochs // 10),  # 10번 정도 로그 출력
        )

        logger.info(f"[TRAIN:LSTM] 학습 완료 - Best Epoch: {history.get('best_epoch', 0)}")
        logger.info(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        if history["val_loss"]:
            logger.info(f"  Val Loss: {history['val_loss'][-1]:.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        LSTM 예측 수행

        Args:
            X: 예측할 시퀀스 데이터 (DataFrame)

        Returns:
            예측 결과 DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        logger.info(f"[INFER:LSTM] 예측 시작 - {X.shape}")

        # 1. 3D 시퀀스로 복원
        X_3d = self._reconstruct_3d_sequences(X, self.sequence_info)
        logger.info(f"[PREP:LSTM] 시퀀스 복원: {X.shape} -> {X_3d.shape}")

        # 2. DataLoader 생성 (y 없이)
        test_loader = create_dataloader(X_3d, y=None, batch_size=self.batch_size, shuffle=False)

        # 3. 예측 수행
        predictions = predict_with_pytorch_model(self.model, test_loader, self.device)

        # 4. DataFrame으로 반환 (BaseModel 인터페이스 준수)
        result = pd.DataFrame(predictions, index=X.index, columns=["prediction"])

        logger.info(f"[INFER:LSTM] 예측 완료 - {len(predictions)}개")

        return result

    def _extract_sequence_info(self, X: pd.DataFrame, kwargs: Dict) -> Dict:
        """메타데이터에서 시퀀스 정보 추출"""
        if "original_sequence_shape" in kwargs:
            # DeepLearning DataHandler에서 전달된 메타데이터 사용
            original_shape = kwargs["original_sequence_shape"]
            return {
                "original_shape": original_shape,
                "sequence_length": original_shape[1],
                "n_features": original_shape[2],
                "from_datahandler": True,
            }
        else:
            # Fallback: DataFrame에서 직접 추론 (테스트나 다른 경로에서 사용)
            logger.warning("[PREP:LSTM] 메타데이터 없음 - DataFrame에서 시퀀스 정보 추론")
            return self._infer_sequence_info_from_dataframe(X)

    def _infer_sequence_info_from_dataframe(self, X: pd.DataFrame) -> Dict:
        """DataFrame에서 시퀀스 정보 추론 (Fallback)"""
        # 컬럼명 패턴에서 추론: seq0_feat0, seq0_feat1, ..., seq1_feat0, ...
        if X.columns[0].startswith("seq") and "_feat" in X.columns[0]:
            # 시퀀스 길이와 특성 개수 추론
            max_seq = max(int(col.split("_")[0][3:]) for col in X.columns) + 1  # seq0, seq1, ...
            max_feat = max(int(col.split("_")[1][4:]) for col in X.columns) + 1  # feat0, feat1, ...

            return {
                "original_shape": (len(X), max_seq, max_feat),
                "sequence_length": max_seq,
                "n_features": max_feat,
                "from_datahandler": False,
            }
        else:
            # 마지막 수단: 기본값 사용
            logger.warning("[PREP:LSTM] 시퀀스 정보 추론 불가 - 기본값 사용")
            n_cols = len(X.columns)
            # 가정: sequence_length=10, 나머지는 features
            seq_len = 10
            n_feat = n_cols // seq_len
            return {
                "original_shape": (len(X), seq_len, n_feat),
                "sequence_length": seq_len,
                "n_features": n_feat,
                "from_datahandler": False,
            }

    def _reconstruct_3d_sequences(self, X: pd.DataFrame, seq_info: Dict) -> np.ndarray:
        """DataFrame을 3D 시퀀스 데이터로 복원"""
        original_shape = seq_info["original_shape"]
        stored_n_samples, seq_len, n_features = original_shape

        # 실제 입력 데이터의 크기 사용 (train과 predict에서 샘플 수가 다를 수 있음)
        actual_n_samples = len(X)
        expected_features = seq_len * n_features

        # DataFrame shape 검증
        if X.shape[1] != expected_features:
            raise ValueError(
                f"입력 데이터의 특성 수가 맞지 않습니다. "
                f"기대값: {expected_features} (seq_len={seq_len} × n_features={n_features}), "
                f"실제값: {X.shape[1]}"
            )

        # DataFrame을 numpy array로 변환 후 reshape
        X_flat = X.values  # (actual_n_samples, seq_len * n_features)
        X_3d = X_flat.reshape(actual_n_samples, seq_len, n_features)

        return X_3d.astype(np.float32)

    def _validate_sequence_data(self, X: np.ndarray, y: pd.Series):
        """시퀀스 데이터 검증"""
        if len(X.shape) != 3:
            raise ValueError(f"3D 시퀀스 데이터가 필요합니다. 현재: {X.shape}")

        if len(X) != len(y):
            raise ValueError(f"X와 y의 길이가 다릅니다: {len(X)} vs {len(y)}")

        if X.shape[0] < 10:
            logger.warning(f"[DATA:LSTM] 학습 데이터 부족: {X.shape[0]}개 (최소 10개 권장)")

    def _time_based_split(self, X: np.ndarray, y: pd.Series, val_ratio: float = 0.2):
        """시계열 데이터의 시간 기준 분할"""
        n_samples = len(X)
        split_idx = int(n_samples * (1 - val_ratio))

        if split_idx < 5:  # 너무 적으면 validation 생략
            logger.warning(f"[DATA:LSTM] 데이터 부족으로 validation 생략: {n_samples}개")
            return X, None, y.values, None

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y.iloc[:split_idx].values, y.iloc[split_idx:].values

        logger.info(f"[DATA:LSTM] 시간 분할 - Train: {len(X_train)}, Val: {len(X_val)}")

        return X_train, X_val, y_train, y_val

    def _build_lstm_model(self, input_size: int) -> nn.Module:
        """LSTM 아키텍처 정의"""

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_dim, num_layers, dropout, bidirectional):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,  # Single layer면 dropout=0
                    batch_first=True,
                    bidirectional=bidirectional,
                )

                # Linear layer input size
                lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
                self.fc = nn.Linear(lstm_output_size, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # LSTM forward
                lstm_out, (hidden, cell) = self.lstm(x)
                # 마지막 시점의 출력 사용
                last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
                # Dropout + Linear
                output = self.dropout(last_output)
                output = self.fc(output)
                return output

        return LSTMNet(
            input_size=input_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.is_fitted:
            return {"status": "not_fitted"}

        from .pytorch_utils import count_parameters

        param_info = count_parameters(self.model)

        return {
            "status": "fitted",
            "architecture": "LSTM",
            "hyperparameters": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "learning_rate": self.learning_rate,
            },
            "sequence_info": self.sequence_info,
            "model_parameters": param_info,
            "device": str(self.device),
        }
