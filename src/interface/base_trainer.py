from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any, TYPE_CHECKING


if TYPE_CHECKING:
    from src.components.preprocessor import Preprocessor
# BaseModel import 제거: 외부 라이브러리 직접 사용으로 전환


class BaseTrainer(ABC):
    """
    모델 학습 및 평가 전체 과정을 관장하는 클래스의 추상 기본 클래스(ABC).
    """

    @abstractmethod
    def train(self, df: pd.DataFrame) -> Tuple[Preprocessor, Any, Dict[str, Any]]:
        """
        데이터 분할, 피처 증강, 피처 전처리, 모델 학습 및 평가, MLflow 로깅의
        전체 파이프라인을 실행하고 결과를 반환하는 추상 메서드입니다.
        """
        raise NotImplementedError
