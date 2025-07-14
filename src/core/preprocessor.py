import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# GCS utils 함수들은 gcs_adapter로 통합됨 - 임시 비활성화
# from src.utils.gcs_utils import upload_object_to_gcs, download_object_from_gcs, generate_model_path
from src.settings import Settings, PreprocessorSettings
from src.interface.base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class Preprocessor(BasePreprocessor):
    """
    설정에 따라 범주형 변수 인코딩과 수치형 변수 정규화를 수행하는 전처리기.
    """

    def __init__(self, config: PreprocessorSettings, settings: Settings):
        self.config = config
        self.settings = settings
        self.criterion_col = config.params.criterion_col

        special_cols = {
            self.settings.model.data_interface.target_col,
            self.settings.model.data_interface.treatment_col,
        }
        # set.union()을 사용하여 가독성 개선
        self.exclude_cols = sorted(list(set(self.config.params.exclude_cols).union(special_cols)))
        
        logger.info(f"Preprocessor 제외 컬럼: {self.exclude_cols}")

        self.categorical_cols_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []

        self._scaler = StandardScaler()
        self._rank_mappings: Dict[str, Dict[str, float]] = {}
        self._unseen_value_filler: int = -1

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        """
        입력 데이터(X)로부터 변환 규칙(인코딩, 스케일링)을 학습합니다.
        """
        logger.info("Preprocessor 학습을 시작합니다.")
        X_fit = X.copy()

        self.categorical_cols_ = sorted(
            col for col in X_fit.select_dtypes(include=['object', 'category']).columns
            if col not in self.exclude_cols
        )
        self.numerical_cols_ = sorted(
            col for col in X_fit.select_dtypes(include=['number']).columns
            if col not in self.exclude_cols
        )

        if self.categorical_cols_:
            self._fit_categorical(X_fit)
        else:
            logger.info("범주형 변수가 없어 인코딩을 건너뜁니다.")

        if self.numerical_cols_:
            logger.info("수치형 변수 정규화를 시작합니다.")
            self._scaler.fit(X_fit[self.numerical_cols_])
            logger.info("수치형 변수 정규화 완료.")
        else:
            logger.info("수치형 변수가 없어 정규화를 건너뜁니다.")

        self.feature_names_out_ = sorted(self.categorical_cols_ + self.numerical_cols_)
        logger.info(f"학습 완료. 최종 출력 변수 {len(self.feature_names_out_)}개: {self.feature_names_out_}")
        return self

    def _fit_categorical(self, X: pd.DataFrame):
        """범주형 변수에 대한 인코딩 규칙을 학습합니다."""
        if self.criterion_col:
            logger.info(f"'{self.criterion_col}' 기준 중앙값 순위 인코딩을 사용합니다.")
            if self.criterion_col not in X.columns:
                raise ValueError(f"기준 컬럼 '{self.criterion_col}'이 데이터에 없습니다.")
            
            for col in self.categorical_cols_:
                mapping = X.groupby(col)[self.criterion_col].median().rank(method='first').astype(int)
                self._rank_mappings[col] = mapping.to_dict()
        else:
            logger.info("빈도(Frequency) 인코딩을 사용합니다.")
            for col in self.categorical_cols_:
                mapping = X[col].value_counts().to_dict()
                self._rank_mappings[col] = mapping

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 규칙을 사용하여 데이터를 변환합니다."""
        if not self._is_fitted():
            raise NotFittedError("Preprocessor가 아직 학습되지 않았습니다. 'fit'을 먼저 호출하세요.")

        X_transformed = X.copy()

        for col in self.categorical_cols_:
            mapping = self._rank_mappings.get(col)
            if mapping:
                X_transformed[col] = X_transformed[col].map(mapping).fillna(self._unseen_value_filler)

        if self.numerical_cols_:
            X_transformed[self.numerical_cols_] = self._scaler.transform(X_transformed[self.numerical_cols_])

        return X_transformed[self.feature_names_out_]

    def _is_fitted(self) -> bool:
        # categorical_cols_가 있는데 매핑 정보가 없거나,
        # numerical_cols_가 있는데 스케일러가 학습되지 않은 경우 False
        cat_fitted = not self.categorical_cols_ or bool(self._rank_mappings)
        num_fitted = not self.numerical_cols_ or hasattr(self._scaler, 'mean_')
        return cat_fitted and num_fitted

    def save(self, file_name: str = "preprocessor.joblib", version: Optional[str] = None) -> str:
        """학습된 Preprocessor 객체를 저장합니다."""
        output_type = self.config.output.type
        if output_type == "gcs":
            # TODO: GCS 저장 기능 재구현 필요 (gcs_adapter 통합 후)
            logger.warning("GCS 저장 기능이 임시 비활성화됨. 로컬 저장으로 대체합니다.")
            output_type = "local"
        
        if output_type == "local" or True:  # 임시로 모든 저장을 로컬로
            local_dir = Path("./local/artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            version_str = version or 'latest'
            file_path = Path(file_name)
            local_path = local_dir / f"{file_path.stem}-{version_str}{file_path.suffix}"
            
            logger.info(f"학습된 Preprocessor를 로컬에 저장합니다: {local_path}")
            joblib.dump(self, local_path)
            return str(local_path)

    @classmethod
    def load(cls, path: str, settings: Settings) -> 'Preprocessor':
        """경로에서 Preprocessor 객체를 로드합니다."""
        logger.info(f"경로에서 Preprocessor를 로드합니다: {path}")
        if path.startswith("gs://"):
            # TODO: GCS 로딩 기능 재구현 필요 (gcs_adapter 통합 후)
            raise NotImplementedError("GCS 로딩 기능이 임시 비활성화됨")
        else:
            return joblib.load(path)
        
