# src/models/custom/pytorch_utils.py
"""
PyTorch 공통 유틸리티 함수들

BaseModel을 직접 상속하는 PyTorch 모델들이 공통으로 사용할 수 있는
헬퍼 함수들을 제공합니다. 중복 코드 제거와 일관성 유지가 목적입니다.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.core.logger import log_sys_debug, log_train, log_train_debug


def get_device() -> torch.device:
    """
    GPU/CPU 자동 선택 유틸리티

    Returns:
        사용 가능한 최적의 디바이스 (cuda 또는 cpu)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_train_debug(f"GPU 사용: {torch.cuda.get_device_name()}")
    else:
        log_train_debug("CPU 사용")
    return device


def create_dataloader(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    PyTorch DataLoader 생성 헬퍼

    Args:
        X: 입력 데이터 (numpy array 또는 pandas DataFrame)
        y: 타겟 데이터 (선택사항, numpy array 또는 pandas Series)
        batch_size: 배치 크기
        shuffle: 데이터 섞기 여부

    Returns:
        PyTorch DataLoader 인스턴스
    """
    # DataFrame을 numpy array로 변환
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Tensor 생성
    X_tensor = torch.FloatTensor(X_array)

    if y is not None:
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        y_tensor = torch.FloatTensor(y_array)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
    log_interval: int = 10,
) -> dict:
    """
    공통 PyTorch 학습 루프

    Args:
        model: 학습할 PyTorch 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더 (선택사항)
        epochs: 학습 에포크 수
        learning_rate: 학습률
        criterion: 손실 함수 (기본값: MSELoss)
        device: 디바이스 (기본값: 자동 선택)
        early_stopping_patience: 조기 종료 patience
        log_interval: 로그 출력 간격

    Returns:
        학습 히스토리 딕셔너리
    """
    if device is None:
        device = get_device()

    if criterion is None:
        criterion = nn.MSELoss()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 히스토리 추적
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_data in train_loader:
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                batch_x = batch_data[0]
                batch_y = None

            batch_x = batch_x.to(device)
            if batch_y is not None:
                batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            if batch_y is not None:
                # CrossEntropyLoss는 Long 타입 라벨 필요, MSELoss는 Float 필요
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs.squeeze(), batch_y.float())
            else:
                # Unsupervised learning case (예: autoencoder)
                loss = criterion(outputs, batch_x)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        val_loss = None
        if val_loader is not None:
            val_loss = validate_pytorch_model(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                history["best_epoch"] = epoch
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    log_train(f"Early stopping at epoch {epoch+1}")
                    break

        # 로그 출력
        if epoch % log_interval == 0 or epoch == epochs - 1:
            if val_loss is not None:
                log_train(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                log_train(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    return history


def validate_pytorch_model(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """
    PyTorch 모델 검증

    Args:
        model: 검증할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 디바이스

    Returns:
        평균 검증 손실
    """
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 2:
                batch_x, batch_y = batch_data
            else:
                batch_x = batch_data[0]
                batch_y = None

            batch_x = batch_x.to(device)
            if batch_y is not None:
                batch_y = batch_y.to(device)

            outputs = model(batch_x)

            if batch_y is not None:
                # CrossEntropyLoss는 Long 타입 라벨 필요, MSELoss는 Float 필요
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs.squeeze(), batch_y.float())
            else:
                loss = criterion(outputs, batch_x)

            val_loss += loss.item()
            val_batches += 1

    return val_loss / val_batches


def predict_with_pytorch_model(
    model: nn.Module, data_loader: DataLoader, device: Optional[torch.device] = None
) -> np.ndarray:
    """
    PyTorch 모델로 예측 수행

    Args:
        model: 예측에 사용할 모델
        data_loader: 예측할 데이터 로더
        device: 디바이스 (기본값: 자동 선택)

    Returns:
        예측 결과 numpy array
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_x = batch_data[0].to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())

    return np.array(predictions)


def get_model_device(model: nn.Module) -> torch.device:
    """
    모델이 현재 어떤 디바이스에 있는지 확인

    Args:
        model: 확인할 PyTorch 모델

    Returns:
        모델이 위치한 디바이스
    """
    return next(model.parameters()).device


def count_parameters(model: nn.Module) -> dict:
    """
    모델의 파라미터 개수 계산

    Args:
        model: 분석할 PyTorch 모델

    Returns:
        파라미터 정보 딕셔너리
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def set_seed(seed: int = 42):
    """
    PyTorch 재현성을 위한 시드 설정

    Args:
        seed: 설정할 시드 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log_sys_debug(f"PyTorch seed: {seed}")
