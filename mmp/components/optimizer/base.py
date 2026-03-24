"""
BaseOptimizer - 하이퍼파라미터 최적화기 기본 인터페이스
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import pandas as pd


class BaseOptimizer(ABC):
    """
    모든 Optimizer가 따라야 할 표준 계약.
    하이퍼파라미터 최적화를 위한 Strategy Pattern의 기반.
    """

    def __init__(self, settings: Any, **kwargs: Any):
        """
        Optimizer 초기화

        Args:
            settings: 프로젝트 설정 객체
            **kwargs: 추가 초기화 인자 (구현체별 확장용)
        """
        self.settings = settings

    @abstractmethod
    def optimize(
        self, train_df: pd.DataFrame, training_callback: Callable
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 수행

        Args:
            train_df: 학습 데이터프레임
            training_callback: 학습 콜백 함수 (train_df, params, trial_number) -> result_dict

        Returns:
            Dict[str, Any]: 최적화 결과
                - best_params: 최적 하이퍼파라미터
                - best_score: 최적 점수
                - 기타 구현체별 추가 정보
        """
        pass
