"""
BaseDataHandler - 데이터 핸들러 기본 인터페이스

각 task type별로 특화된 데이터 처리를 위한 추상 클래스입니다.
- TabularHandler: classification, regression, clustering, causal (2D 유지)
- TimeseriesHandler: timeseries 통계 모델용 (2D + 피처 자동 생성)
- SequenceHandler: LSTM 등 시퀀스 모델용 (2D→3D sliding window 변환)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.core.logger import logger

if TYPE_CHECKING:
    from src.settings import Settings


class BaseDataHandler(ABC):
    """데이터 핸들러 기본 인터페이스"""

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.data_interface = settings.recipe.data.data_interface
        self._is_fitted: bool = False

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        데이터 준비 - 각 task type에 특화된 데이터 처리
        """
        pass

    @abstractmethod
    def get_feature_columns(self) -> list[str]:
        """
        실제 학습에 사용된 피처 컬럼 목록 반환.
        자동 선택 로직이 적용된 경우 실제 선택된 컬럼들을 반환해야 함.
        """
        pass

    @abstractmethod
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        데이터 분할 - 각 handler에서 구현 필수

        Args:
            df: 원본 데이터프레임

        Returns:
            Dict with keys: 'train', 'validation', 'test', 'calibration'
            calibration은 None일 수 있음
        """
        pass

    def fit(self, df: pd.DataFrame) -> "BaseDataHandler":
        """
        변환에 필요한 상태 학습.
        train 데이터로 1회만 호출되어야 함.

        Args:
            df: 학습 데이터프레임

        Returns:
            self (메서드 체이닝 지원)
        """
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 상태로 피처 변환 수행.
        추론 시 사용됨.

        Args:
            df: 변환할 데이터프레임

        Returns:
            변환된 피처 DataFrame (y 미포함)
        """
        if not self._is_fitted:
            raise ValueError("fit()을 먼저 호출하세요")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fit + transform 한번에 수행.

        Args:
            df: 데이터프레임

        Returns:
            변환된 피처 DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def _get_split_config(self) -> Dict[str, float]:
        """
        설정에서 분할 비율 추출 (공통 로직)

        Returns:
            Dict with keys: train, validation, test, calibration
        """
        split_config = getattr(self.settings.recipe.data, "split", None)
        if not split_config:
            raise ValueError(
                "데이터 분할 설정(data.split)이 없습니다. "
                "Recipe에서 data.split 섹션을 정의하세요."
            )

        ratios = {
            "train": float(split_config.train),
            "validation": float(split_config.validation),
            "test": float(split_config.test),
            "calibration": float(getattr(split_config, "calibration", 0.0) or 0.0),
        }

        # 비율 합 검증
        total = sum(ratios.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total}")

        return ratios

    def _get_random_state(self) -> int:
        """
        설정에서 random_state 추출 (공통 로직)

        우선순위:
        1. evaluation.random_state (명시적 설정)
        2. model.computed.seed (파이프라인에서 설정)
        3. 기본값 42
        """
        # 1순위: evaluation.random_state
        evaluation = getattr(self.settings.recipe, "evaluation", None)
        if evaluation and getattr(evaluation, "random_state", None) is not None:
            return evaluation.random_state

        # 2순위: model.computed.seed
        model = getattr(self.settings.recipe, "model", None)
        if model:
            computed = getattr(model, "computed", None)
            if computed and isinstance(computed, dict) and "seed" in computed:
                return computed["seed"]

        # 기본값
        return 42

    def split_and_prepare(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        Any,
        Dict[str, Any],  # train
        pd.DataFrame,
        Any,
        Dict[str, Any],  # validation
        pd.DataFrame,
        Any,
        Dict[str, Any],  # test
        Optional[Tuple[pd.DataFrame, Any, Dict[str, Any]]],  # calibration
    ]:
        """
        표준화된 4-way interface: 데이터 분할 + 각 분할에 대해 prepare_data 수행

        모든 DataHandler가 동일한 형식으로 반환하여 Pipeline에서 일관되게 처리 가능

        Returns:
            (X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data)
            calibration_data는 (X_calib, y_calib, add_calib) 또는 None
        """
        split_results = self.split_data(df)

        # Train
        X_train, y_train, add_train = self.prepare_data(split_results["train"])

        # Validation
        val_df = split_results.get("validation")
        if val_df is not None and not val_df.empty:
            X_val, y_val, add_val = self.prepare_data(val_df)
        else:
            X_val, y_val, add_val = pd.DataFrame(), None, {}

        # Test
        test_df = split_results.get("test")
        if test_df is not None and not test_df.empty:
            X_test, y_test, add_test = self.prepare_data(test_df)
        else:
            X_test, y_test, add_test = pd.DataFrame(), None, {}

        # Calibration
        calibration_data = None
        calib_df = split_results.get("calibration")
        if calib_df is not None and not calib_df.empty:
            X_calib, y_calib, add_calib = self.prepare_data(calib_df)
            calibration_data = (X_calib, y_calib, add_calib)

        return (
            X_train,
            y_train,
            add_train,
            X_val,
            y_val,
            add_val,
            X_test,
            y_test,
            add_test,
            calibration_data,
        )

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        데이터 검증 (각 handler별로 구체화)

        Args:
            df: 검증할 데이터프레임

        Returns:
            검증 통과 여부
        """
        if df.empty:
            raise ValueError("데이터프레임이 비어있습니다")
        return True

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 메타 정보 반환

        Args:
            df: 분석할 데이터프레임

        Returns:
            데이터 메타 정보 딕셔너리
        """
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_ratio": (df.isnull().sum() / len(df)).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    def _get_exclude_columns(self, df: pd.DataFrame) -> List[str]:
        """
        학습에서 제외할 컬럼 목록 반환 (공통 로직)

        제외 대상:
        - entity_columns: 항상 제외
        - fetcher.timestamp_column: feature_store 모드에서 제외

        Returns:
            실제로 df에 존재하는 제외 대상 컬럼 목록
        """
        exclude_columns = []

        # Entity columns
        if self.data_interface.entity_columns:
            exclude_columns.extend(self.data_interface.entity_columns)

        # Feature Store timestamp column
        fetcher_conf = getattr(self.settings.recipe.data, "fetcher", None)
        if fetcher_conf:
            if getattr(fetcher_conf, "type", None) == "feature_store":
                ts_col = getattr(fetcher_conf, "timestamp_column", None)
                if ts_col:
                    exclude_columns.append(ts_col)

        return [col for col in exclude_columns if col in df.columns]

    def _check_missing_values_warning(self, X: pd.DataFrame, threshold: float = 0.05) -> None:
        """
        결측치 비율이 임계값 이상인 컬럼 경고 (공통 로직)

        Args:
            X: 피처 데이터프레임
            threshold: 결측치 비율 임계값 (기본 5%)
        """
        if X.empty:
            return

        missing_info = []
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_ratio = missing_count / len(X)
            if missing_ratio >= threshold:
                missing_info.append({
                    "column": col,
                    "missing_count": missing_count,
                    "missing_ratio": missing_ratio,
                })

        if missing_info:
            handler_name = self.__class__.__name__
            logger.warning(
                f"[DATA:{handler_name}] 결측치 {threshold:.0%} 이상: {len(missing_info)}개 컬럼"
            )
            for info in missing_info:
                logger.warning(
                    f"[DATA:{handler_name}]   {info['column']}: "
                    f"{info['missing_count']:,}개 ({info['missing_ratio']:.1%})"
                )
