# src/components/_preprocessor/_steps/_discretizer.py
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

from src.components.preprocessor.base import BasePreprocessor
from src.utils.core.logger import logger

from ..registry import PreprocessorStepRegistry


class KBinsDiscretizerWrapper(BasePreprocessor, BaseEstimator, TransformerMixin):
    """
    DataFrame-First: scikit-learn의 KBinsDiscretizer를 위한 래퍼
    연속적인 수치형 변수를 여러 구간(bin)으로 나누어 범주형 변수처럼 만듭니다.
    이를 통해 모델이 비선형 관계를 더 쉽게 학습하도록 돕습니다.
    """

    def __init__(
        self,
        n_bins: int = 5,
        encode: str = "ordinal",
        strategy: str = "quantile",
        columns: List[str] = None,
    ):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.columns = columns
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode=self.encode, strategy=self.strategy, subsample=None
        )

    def fit(self, X: pd.DataFrame, y=None):
        """KBinsDiscretizer를 학습시킵니다."""
        logger.info(
            f"[KBinsDiscretizer] 이산화 학습을 시작합니다 - n_bins: {self.n_bins}, encode: {self.encode}, strategy: {self.strategy}"
        )
        try:
            self.discretizer.fit(X)
        except ValueError as e:
            error_msg = str(e)
            if "strategy" in error_msg.lower():
                logger.error(f"[KBinsDiscretizer] strategy 설정 오류: {self.strategy}")
                raise ValueError(
                    f"KBinsDiscretizer strategy '{self.strategy}' 설정에 문제가 있습니다.\n"
                    f"사용 가능한 strategy:\n"
                    f"- 'uniform': 동일한 폭의 구간으로 분할\n"
                    f"- 'quantile': 동일한 빈도의 구간으로 분할\n"
                    f"- 'kmeans': K-means 클러스터링으로 분할\n"
                    f"원본 오류: {error_msg}"
                ) from e
            elif "encode" in error_msg.lower():
                logger.error(f"[KBinsDiscretizer] encode 설정 오류: {self.encode}")
                raise ValueError(
                    f"KBinsDiscretizer encode '{self.encode}' 설정에 문제가 있습니다.\n"
                    f"사용 가능한 encode:\n"
                    f"- 'ordinal': 순서형 정수로 인코딩\n"
                    f"- 'onehot': 원-핫 벡터로 인코딩\n"
                    f"- 'onehot-dense': 원-핫 벡터 (dense array)\n"
                    f"원본 오류: {error_msg}"
                ) from e
            else:
                raise
        logger.info(
            f"[KBinsDiscretizer] 학습 완료 - n_bins: {self.n_bins}, 입력 컬럼: {len(X.columns)}개"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 Discretizer를 사용하여 데이터를 변환하고 DataFrame으로 반환합니다."""
        logger.info(
            f"[KBinsDiscretizer] 이산화 변환을 시작합니다 - 입력: {X.shape}, encode: {self.encode}"
        )
        result_array = self.discretizer.transform(X)

        # sparse matrix 처리 (onehot encoding 시 sklearn이 sparse matrix 반환)
        if hasattr(result_array, "toarray"):
            result_array = result_array.toarray()

        # 컬럼명 생성 정책: discretize_ 접두사 사용
        if self.encode == "ordinal":
            # ordinal encoding의 경우 원본 컬럼명에 접두사 추가
            new_columns = [f"discretized_{col}" for col in X.columns]
        else:
            # onehot encoding의 경우 bin별 컬럼 생성
            new_columns = []
            for col in X.columns:
                for bin_idx in range(self.n_bins):
                    new_columns.append(f"discretized_{col}_bin{bin_idx}")

        result_df = pd.DataFrame(
            result_array, index=X.index, columns=new_columns[: result_array.shape[1]]
        )
        logger.info(
            f"[KBinsDiscretizer] 이산화 완료 - 출력: {result_df.shape}, 생성된 컬럼: {len(new_columns[:result_array.shape[1]])}개"
        )
        return result_df

    def get_output_column_names(self, input_columns: List[str]) -> List[str]:
        """변환 후 예상되는 출력 컬럼명을 반환합니다."""
        if self.encode == "ordinal":
            return [f"discretized_{col}" for col in input_columns]
        else:
            # onehot의 경우 bin별 컬럼들
            new_columns = []
            for col in input_columns:
                for bin_idx in range(self.n_bins):
                    new_columns.append(f"discretized_{col}_bin{bin_idx}")
            return new_columns

    def preserves_column_names(self) -> bool:
        """이 전처리기는 원본 컬럼명을 보존하지 않습니다."""
        return False

    def get_application_type(self) -> str:
        """Discretizer는 특정 수치형 컬럼에 적용됩니다."""
        return "targeted"

    def get_applicable_columns(self, X: pd.DataFrame) -> List[str]:
        """숫자형 컬럼만 대상으로 합니다."""
        return [col for col in X.columns if X[col].dtype in ["int64", "float64"]]


PreprocessorStepRegistry.register("kbins_discretizer", KBinsDiscretizerWrapper)
