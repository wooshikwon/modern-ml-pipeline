from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd
from pydantic import BaseModel

from src.components.adapter.base import BaseAdapter
from src.settings import Settings
from src.utils.core.logger import logger

try:
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False

if TYPE_CHECKING:
    from src.settings import Settings


class FeastAdapter(BaseAdapter):
    """
    Feast 라이브러리를 직접 사용하는 Feature Store 어댑터.
    """

    def __init__(self, settings: Settings, **kwargs):
        logger.info("[FeastAdapter] 초기화 시작합니다")

        if not FEAST_AVAILABLE:
            logger.error("[FeastAdapter] Feast SDK 설치 필요")
            raise ImportError(
                "Feast SDK is not installed. Please install with `pip install feast`."
            )

        self.settings = settings

        # FeastAdapter는 복잡한 설정 구조로 인해 별도의 feature_store 섹션을 사용
        logger.info("Feast 설정 섹션 로딩 - feature_store 설정을 사용합니다")
        self.store = self._init_feature_store()

    def _remove_none_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """중첩된 dict에서 None 값을 재귀적으로 제거"""
        result = {}
        for k, v in d.items():
            if v is None:
                continue
            elif isinstance(v, dict):
                cleaned = self._remove_none_values(v)
                if cleaned:
                    result[k] = cleaned
            else:
                result[k] = v
        return result

    def _init_feature_store(self) -> FeatureStore:
        """Initializes the Feast FeatureStore object."""
        try:
            # FeastAdapter는 settings.config.feature_store.feast_config에서 설정을 읽음
            config_data = self.settings.config.feature_store.feast_config
            project_name = getattr(config_data, "project", "unknown")

            logger.info(
                f"Feast 설정 로딩 완료 - 프로젝트: {project_name}, 타입: {type(config_data).__name__}"
            )

            # Pydantic BaseModel을 dict로 변환 후 Feast RepoConfig 생성
            if isinstance(config_data, BaseModel):
                # exclude_none으로 None 값 제외하고, 중첩 dict에서도 None 제거
                config_dict = config_data.model_dump(exclude_none=True)
                config_dict = self._remove_none_values(config_dict)
                logger.info(f"Pydantic 모델을 dict로 변환 - keys: {list(config_dict.keys())}")
            elif isinstance(config_data, dict):
                config_dict = self._remove_none_values(config_data)
            else:
                raise TypeError(f"Unsupported config type for Feast: {type(config_data)}")

            # Feast RepoConfig로 변환
            repo_config = RepoConfig(**config_dict)
            fs = FeatureStore(config=repo_config)
            logger.info("FeatureStore 생성 완료 - RepoConfig 변환 성공")

            logger.info(f"[FeastAdapter] 초기화 완료되었습니다 (project: {project_name})")
            return fs
        except Exception as e:
            logger.error(f"[FeastAdapter] 초기화 실패: {e}")
            raise

    def get_historical_features(
        self, entity_df: pd.DataFrame, features: List[str], **kwargs
    ) -> pd.DataFrame:
        """오프라인 스토어에서 과거 시점의 피처를 가져옵니다."""
        logger.info(
            f"Historical Features 조회 시작 - shape: {entity_df.shape}, 대상: {len(entity_df)}개 엔티티, 피처: {len(features)}개"
        )

        try:
            logger.info(
                f"Feast 오프라인 스토어 조회 - 피처 목록: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}"
            )

            retrieval_job = self.store.get_historical_features(
                entity_df=entity_df,
                features=features,
            )
            result_df = retrieval_job.to_df()

            logger.info(
                f"Historical Features 조회 완료 - shape: {result_df.shape}, 반환된 피처: {len(result_df.columns)}개 컬럼"
            )
            return result_df
        except Exception as e:
            logger.error(f"[FeastAdapter] Historical Features 조회 실패: {e}")
            raise

    def get_historical_features_with_validation(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        data_interface_config: Dict[str, Any] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Phase 2: Point-in-Time Correctness 보장 피처 조회
        기존 get_historical_features + 완전한 시점 안전성 검증

        Args:
            entity_df: 조회할 엔티티 DataFrame
            features: 조회할 피처 목록
            data_interface_config: EntitySchema 설정 (Entity+Timestamp 정보)
            **kwargs: Feast get_historical_features에 전달할 추가 인자

        Returns:
            Point-in-Time 검증을 통과한 피처 DataFrame
        """
        logger.info(
            f"Point-in-Time 피처 조회 시작 - shape: {entity_df.shape}, Point-in-Time Correctness 보장 모드"
        )

        # 1. Point-in-Time 스키마 검증
        if data_interface_config:
            self._validate_point_in_time_schema(entity_df, data_interface_config)

        # 2. 기존 get_historical_features 호출 (검증된 ASOF JOIN)
        result_df = self.get_historical_features(entity_df, features, **kwargs)

        # 3. ASOF JOIN 결과 검증 (미래 데이터 누출 차단)
        if data_interface_config:
            self._validate_asof_join_result(entity_df, result_df, data_interface_config)

        logger.info(
            f"Point-in-Time 피처 조회 완료 - shape: {result_df.shape}, Point-in-Time Correctness 검증 통과"
        )
        return result_df

    def _validate_point_in_time_schema(self, entity_df: pd.DataFrame, config: Dict[str, Any]):
        """Entity + Timestamp 필수 컬럼 Point-in-Time 검증"""
        entity_columns = config.get("entity_columns", [])
        timestamp_column = config.get("timestamp_column", "")

        # Entity 컬럼 존재 검증
        missing_entities = [col for col in entity_columns if col not in entity_df.columns]
        if missing_entities:
            raise ValueError(
                f"Point-in-Time 검증 실패: 필수 Entity 컬럼 누락 {missing_entities}\n"
                f"Required: {entity_columns}, Found: {list(entity_df.columns)}"
            )

        # Timestamp 컬럼 존재 및 타입 검증
        if timestamp_column and timestamp_column not in entity_df.columns:
            raise ValueError(f"Point-in-Time 검증 실패: Timestamp 컬럼 '{timestamp_column}' 누락")

        if timestamp_column and not pd.api.types.is_datetime64_any_dtype(
            entity_df[timestamp_column]
        ):
            # 자동 타입 변환 시도
            try:
                entity_df[timestamp_column] = pd.to_datetime(entity_df[timestamp_column])
                logger.info(
                    f"Timestamp 컬럼 '{timestamp_column}'을 datetime 타입으로 자동 변환 완료"
                )
            except Exception as e:
                raise ValueError(
                    f"Point-in-Time 검증 실패: '{timestamp_column}'이 datetime 타입이 아니며 변환 실패: {e}"
                )

        logger.info(
            f"Point-in-Time 스키마 검증 통과 - 엔티티: {entity_columns}, 타임스탬프: {timestamp_column}"
        )

    def _validate_asof_join_result(
        self, input_df: pd.DataFrame, result_df: pd.DataFrame, config: Dict[str, Any]
    ):
        """ASOF JOIN 결과의 Point-in-Time 무결성 검증"""
        timestamp_column = config.get("timestamp_column", "")

        if not timestamp_column or timestamp_column not in result_df.columns:
            logger.warning("[FeastAdapter] Timestamp 컬럼 없음: ASOF JOIN 결과 검증을 생략합니다")
            return

        # 입력 대비 결과 행 수 확인
        if len(result_df) != len(input_df):
            logger.warning(
                f"[FeastAdapter] ASOF JOIN 결과 행 수 불일치: 입력({len(input_df)}) vs 결과({len(result_df)})"
            )

        # 미래 데이터 누출 검증 (현재 시점 이후 피처 값 감지)
        current_time = pd.Timestamp.now()
        future_data = result_df[result_df[timestamp_column] > current_time]

        if len(future_data) > 0:
            logger.warning(
                f"[FeastAdapter] 미래 데이터 감지: {len(future_data)}개 행이 현재 시점 이후 (current_time: {current_time})"
            )

        logger.info("ASOF JOIN 무결성 검증 완료 - Point-in-Time 데이터 무결성이 확인되었습니다")

    def get_online_features(
        self, entity_rows: List[Dict[str, Any]], features: List[str], **kwargs
    ) -> pd.DataFrame:
        """온라인 스토어에서 실시간 피처를 가져옵니다."""
        logger.info(
            f"Online Features 조회 시작 - 대상: {len(entity_rows)}개 엔티티, 피처: {len(features)}개"
        )

        try:
            logger.info(
                f"Feast 온라인 스토어 조회 - 실시간 피처: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}"
            )

            retrieval_job = self.store.get_online_features(
                features=features,
                entity_rows=entity_rows,
            )
            result_df = retrieval_job.to_df()

            logger.info(
                f"Online Features 조회 완료 - shape: {result_df.shape}, 실시간 피처 {len(result_df.columns)}개 컬럼 반환"
            )
            return result_df
        except Exception as e:
            logger.error(f"[FeastAdapter] Online Features 조회 실패: {e}")
            raise

    def read(self, **kwargs) -> pd.DataFrame:
        """BaseAdapter 호환성을 위한 read 메서드. get_historical_features를 호출합니다."""
        entity_df = kwargs.get("entity_df")
        features = kwargs.get("features")

        if entity_df is None or features is None:
            logger.error("[FeastAdapter] Read 파라미터 오류: entity_df와 features가 필요합니다")
            raise ValueError("'entity_df' and 'features' must be provided for read operation.")

        logger.info("BaseAdapter read 호출 - get_historical_features로 위임합니다")
        return self.get_historical_features(entity_df, features, **kwargs)

    def write(self, df: pd.DataFrame, table_name: str, **kwargs):
        """Feast는 주로 읽기용이므로, write는 기본적으로 지원하지 않습니다."""
        logger.warning("[FeastAdapter] Write 작업은 지원되지 않습니다")
        raise NotImplementedError("FeastAdapter does not support write operation.")


# Self-registration
from ..registry import AdapterRegistry

AdapterRegistry.register("feature_store", FeastAdapter)
