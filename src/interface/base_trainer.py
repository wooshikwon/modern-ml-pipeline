from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any

# 순환 참조를 피하기 위해 타입 힌트는 문자열로 처리하거나, 별도의 파일로 분리할 수 있습니다.
# 여기서는 간단하게 필요한 클래스만 임포트합니다.
from src.core.preprocessor import Preprocessor
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
