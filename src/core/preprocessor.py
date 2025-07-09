import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Dict, Any, Optional

# 유틸리티 모듈 임포트
from src.utils.gcs_utils import upload_objet_to_gcs, download_object_from_gcs, generate_model_path
from config.settings import Settings, TransformerSettings

# library import
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# interface import
from src.interface.base_preprocessor import BasePreprocessor


class Preprocessor(BasePreprocessor):
    """
    설정에 따라 두 가지 방식으로 범주형 변수를 변환하는 지능형 Preprocessor.
        - criterion_col이 있으면: 해당 컬럼 기준 '중앙값 순위(Median Rank)' 인코딩
        - criterion_col이 없으면: '빈도(Frequency)' 인코딩
    """

    def __init__(self, config: TransformerSettings, settings: Settings):
        # config.yaml에 설정한 **Preprocessor 설정**을 config 인스턴스로 저장
        self.config = config
        self.settings = settings
        self.criterion_col = config.params.criterion_col

        # y값과 uplift mocel에서 '처치' 컬럼의 경우, 학습하는 컬럼이 아니므로 특수 컬럼으로 정의
        special_cols = {
            self.settings.data_interface.target_col, # model_config.yaml에 설정한 target_col
            self.settings.data_interface.treatment_col, # model_config.yaml에 설정한 treatment_col
        }

        # config.yaml에 설정한 학습 예외 컬럼 (e.g. PK용 컬럼 등)
        # config.yaml에서 설정한 예외 컬럼과 model_config.yaml에서 설정한 특수 컬럼 모두 '예외 컬럼'이 된다.
        self.exclude_cols = sorted(list(set(self.config.params.exclude_cols + special_cols)))
        
        logger.info(f"Transformer 제외 컬럼: {self.exclude_cols}")

        # fit 시 사용하는 변수들
        self.categorical_cols_: List[str] = [] # 범주형 변수 리스트
        self.numerical_cols_: List[str] = [] # 수치형 변수 리스트
        self.feature_names_out_: List[str] = [] # 학습에 활용할 최종 출력 변수 리스트

        # 최종적으로 정규화 스케일러와 중앙값 순위 인코딩 매핑 정보를 저장하는 인스턴스들
        self._scaler = StandardScaler()
        self._rank_mappings: Dict[str, Dict[str, float]] = {}
        self._unseen_value_filter: int = -1 # 범주형 변수에서 인코딩(fit)시 미등장 값이 transform 시 등장하는 경우. -1로 대체한다.
        
    # 데이터 프레임을 입력으로 받아 범주형 변수를 수치형 변환하는 기준, 표준화 스케일러를 적합시켜 Transformer 인스턴스로 반환한다
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Transformer':
        """
        전처리가 완료된 데이터셋을 입력으로 받아, Preprocessor 인스턴스를 학습시킨다.
            - 범주형 변수 인코딩 도구 학습
            - 수치형 변수 정규화 도구 학습
            - feature_names_out_에 학습에 활용할 최종 변수 리스트르 출력.
            - 최종적으로 self._rank_mappings 변수와 self._scaler가 Preprocessor 인스턴스로 활용된다.
        """
        logger.info(f"Transformer 학습을 시작합니다.")

        self.categorical_cols_ = sorted(
            col for col in X.select_dtypes(include=['object', 'category']).columns.tolist()
            if col not in self.exclude_cols
        )
        self.numerical_cols_ = sorted(
            col for col in X.select_dtypes(include=['number']).columns.tolist()
            if col not in self.exclude_cols
        )

        # 범주형 변수가 없으면 바로 수치형 변수 정규화로 넘어간다
        if not self.categorical_cols_:
            logger.info(f"범주형 변수가 없으므로 범주형 변수 인코딩을 건너뜁니다.")

        # 범주형 변수가 있고 config.yaml에 설정한 criterion_col이 있으면 중앙값 순위 인코딩을 사용한다.
        elif self.criterion_col:
            logger.info(f"{self.criterion_col}을 기준으로 중앙값 순위(Median Rank) 인코딩을 사용합니다.")
            if self.criterion_col not in X.columns:
                raise ValueError(f"criterion_col이 데이터 셋에 존재하지 않습니다.")

            # 개별 범주형 변수 컬럼에 대해 group by 했을 때. '기준 컬럼' 중앙값이 낮은 값부터 1,2,3... 정수로 변환된다/
            temp_df = X.copy()
            for col in self.categorical_cols_:
                mapping = temp_df.groupby(col)[self.criterion_col].median().rank(method='first').astype(int)
                self._rank_mappings[col] = mapping.to_dict() # e.g. {'region1': {'Busan': 1, 'Seoul': 2, 'Daegu': 3, 'Incheon': 4}}

        # 범주형 변수가 있고 config.yaml에 설정한 criterion_col이 없으면 빈도 인코딩을 사용한다.
        # 트리 기반 모델 사용을 염두해 Fequency Encoding을 사용한다 (Frequency Rank Encoding 대신, 실제 '빈도값' 자체가 중요한 정보를 담고 있음)
        else:
            logger.info(f"criterion_col이 설정되지 않았습니다. 빈도(Frequency) 인코딩을 사용합니다.")
            for col in self.categorical_cols_:
                mapping = X[col].value_counts().to_dict() # e.g. {'region1': {'Busan': 100, 'Seoul': 200, 'Daegu': 300, 'Incheon': 400}}
                self._rank_mappings[col] = mapping

        # 범주형 변수 인코딩 완료 후, 수치형 변수 정규화를 진행한다.
        if self.numerical_cols_:
            logger.info(f"수치형 변수 정규화를 시작합니다.")
            self._scaler.fit(X[self.numerical_cols_])
            logger.info(f"수치형 변수 정규화 완료.")
        else 
            logger.info(f"수치형 변수가 없으므로 수치형 변수 정규화를 건너뜁니다.")

        # 최종 학습에 사용할 컬럼 리스트를 생성한다.
        self.feature_names_out_ = self.categorical_cols_ + self.numerical_cols_
        logger.info(f"최종 출력 변수 리스트: {self.feature_names_out_}")

        # 최종적으로 self._rank_mappings 변수와 self._scaler가 Transformer 인스턴스로 활용된다.
        return self

    def _is_fitted(self) -> bool:
        if self.categorical_cols_ and not self._rank_mappings:
            return False
        if self.numerical_cols_ and not hasattr(self._scaler, 'mean_'):
            return False
        return True
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        fit()을 통해 학습된 변환 규칙을 데이터프레임에 일괄 적용한다.
        이 메서드는 내부에 저장된 인코딩 맵(_rank_mappings)과 스케일러(_scaler)를 데이터에 적용하는 역할을 함.

        # 파이프라인에서 작동하는 방식
            - 학습 시: fit()으로 규칙을 학습시킨 직후, 학습 데이터를 변환하기 위해 호출
            - 추론 시: save()로 저장된 Transformer 객체를 load()한 뒤, 새로운 데이터를 예측 전에 변환하기 위해 호출
        """
        if not self._is_fitted():
            raise NotFittedError("Preprocessor 인스턴스가 학습되지 않았습니다.")

        X_transformed = X.copy()

        # fit된 Preprocessor 인스턴스에 저장된 모든 범주형 컬럼에 대해 하나씩 인코딩 적용
        for col in self.categorical_cols_:
            # 현재 컬럼에 해당하는 인코딩 규칙 값을 불러옴 (e.g. {'Busan': 1, 'Seoul': 2, 'Daegu': 3, 'Incheon': 4})
            mapping = self._rank_mappings.get(col)
            if mapping is not None:
                X_transformed[col] = X_transformed[col].map(mapping).fillna(self._unseen_value_filter)

        # 수치형 변수 전체에 fit된 정규화 스케일러 적용
        if self.numerical_cols_:
            X_transformed[self.numerical_cols_] = self._scaler.transform(X_transformed[self.numerical_cols_])

        return X_transformed[self.feature_names_out_]

    def save(self, file_name: str = "transformer.jobilb", version: Optional[str] = None) -> str:
        """
        학습된 Preprocessor 인스턴스를 저장하고, 저장된 파일 경로를 반환한다.
        """

        output_type = self.config.output.type

        if output_type == "gcs":
            bucket_name = self.config.output.gcs.bucket_name
            if not bucket_name:
                raise ValueError("GCS 저장을 위해 'bucket_name' 설정이 필요합니다.")

            # 버전명을 지정하지 않으면 'latest' 파일을 덮어쓴다.
            gcs_path = generate_model_path(
                model_name=file_name.split(".")[0],
                version=version or "latest",
                settings=self.settings
            )
            full_gcs_path = f"gs://{bucket_name}/{gcs_path}"

            logger.info(f"학습된 Preprocessor를 GCS에 저장합니다: {full_gcs_path}")
            upload_objet_to_gcs(bucket_name, gcs_path, serialization_format='joblib', settings=self.settings)

            return full_gcs_path
        else: # local -- 추후 output_type="S3" 추가 시, S3 저장 옵션 추가
            local_dir = Path("./local/artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / f"{file_name}.{version or 'latest'}.joblib"

            logger.info(f"학습된 Preprocessor를 로컬에 저장합니다: {local_path}")
            joblib.dump(self, local_path)

            return str(local_path)

    @classmethod
    # 이 메서드를 클래스 메서드로 정의. Preprocessor 인스턴스 없이 `Preprocessor.load(...)` 형태로 호출할 수 있음.
    def load(cls, path: str, settings: Settings) -> 'Preprocessor':
        """
        파일 경로로부터 학습된 Preprocessor 객체를 생성하여 반환하는 팩토리 메서드.
        
        Args:
            cls: Preprocessor 클래스 자체를 의미하는 인자입니다.
            path (str): 불러올 파일의 경로. GCS 경로('gs://...') 또는 로컬 경로를 지원합니다.
            settings (Settings): GCS 클라이언트 인증 등에 필요한 전체 설정 객체입니다.

        Returns:
            Preprocessor: 파일로부터 모든 학습 정보가 복원된 Preprocessor 객체 인스턴스.
        """
        # 입력된 경로가 'gs://'로 시작하는지 확인하여 GCS 경로인지 로컬 경로인지 구분합니다.
        if path.startswith("gs://"):
            # GCS에서 객체를 불러오는 중임을 로그로 남깁니다.
            logger.info(f"GCS에서 Preprocessor를 로드합니다: {path}")
            
            # GCS 경로('gs://버킷명/객체경로')에서 세 번째 요소인 버킷 이름을 추출합니다.
            bucket_name = path.split('/')[2]
            
            # 버킷 이름을 제외한 나머지 경로(e.g., 'models/v1/transformer.joblib')를 객체 이름(blob_name)으로 재구성합니다.
            blob_name = '/'.join(path.split('/')[3:])
            
            # GCS 유틸리티 함수를 호출하여 지정된 버킷과 객체 경로에서 파일을 다운로드하고, 
            # 역직렬화(joblib)하여 객체로 복원한 뒤 반환합니다.
            return download_object_from_gcs(
                bucket_name, 
                blob_name, 
                serialization_format='joblib', 
                settings=settings
            )
        else:
            # 'gs://'로 시작하지 않는 모든 경로는 로컬 파일 시스템의 경로로 간주합니다.
            logger.info(f"로컬에서 Preprocessor를 로드합니다: {path}")
            
            # joblib 라이브러리를 사용해 지정된 로컬 경로의 파일을 읽어 객체로 복원한 뒤 반환합니다.
            return joblib.load(path)
        
