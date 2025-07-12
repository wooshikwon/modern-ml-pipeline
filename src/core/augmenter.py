import pandas as pd
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from pathlib import Path

from src.interface.base_augmenter import BaseAugmenter
from src.utils.logger import logger
from src.settings.settings import Settings
from src.utils import sql_utils
from src.core.factory import Factory # Adapter 생성을 위해 Factory 사용

class LocalFileAugmenter(BaseAugmenter):
    """로컬 피처 파일과 조인하여 데이터를 증강하는 클래스. (개발용)"""
    def __init__(self, uri: str):
        parsed_uri = urlparse(uri)
        self.feature_path = Path(parsed_uri.path.lstrip('/'))
        if not self.feature_path.is_absolute():
            self.feature_path = Path(__file__).resolve().parent.parent.parent / self.feature_path

    def augment(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logger.info(f"로컬 피처 파일로 증강을 시작합니다: {self.feature_path}")
        if not self.feature_path.exists():
            raise FileNotFoundError(f"로컬 피처 파일을 찾을 수 없습니다: {self.feature_path}")
        feature_df = pd.read_parquet(self.feature_path)
        return pd.merge(data, feature_df, on="member_id", how="left")

class Augmenter(BaseAugmenter):
    """
    실행 컨텍스트(run_mode)에 따라 배치 또는 실시간으로 동작하는 단일 증강기 클래스.
    """
    def __init__(self, source_uri: str, settings: Settings):
        self.source_uri = source_uri
        self.settings = settings # Factory에서 Adapter를 만들기 위해 필요
        self.sql_template_str = self._load_sql_template()
        self.realtime_features_list = sql_utils.get_selected_columns(self.sql_template_str)

    def _load_sql_template(self) -> str:
        parsed_uri = urlparse(self.source_uri)
        path = Path(parsed_uri.path.lstrip('/'))
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Augmenter SQL 템플릿 파일을 찾을 수 없습니다: {path}")
        return path.read_text(encoding="utf-8")

    def augment(
        self,
        data: pd.DataFrame,
        run_mode: str,
        context_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if run_mode == "batch":
            return self._augment_batch(data, context_params)
        elif run_mode == "serving":
            # serving 모드에서는 feature_store_config가 kwargs로 전달될 것으로 기대
            return self._augment_realtime(data, kwargs.get("feature_store_config"))
        else:
            raise ValueError(f"지원하지 않는 Augmenter 실행 모드입니다: {run_mode}")

    def _augment_batch(
        self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        logger.info(f"배치 모드 피처 증강을 시작합니다. (URI: {self.source_uri})")
        
        factory = Factory(self.settings)
        # Augmenter의 source_uri는 항상 bq라고 가정
        adapter = factory.create_data_adapter('bq')
        
        # Augmenter는 대상 테이블을 직접 알 필요가 없음.
        # 임시 테이블 생성 및 삭제는 파이프라인 레벨에서 처리하는 것이 더 나은 설계일 수 있음.
        # 현재 구조 유지를 위해 임시 테이블 로직은 여기에 둠.
        # TODO: 임시 테이블 관리 로직을 파이프라인으로 이전 고려
        
        feature_df = adapter.read(self.source_uri, params=context_params)
        return pd.merge(data, feature_df, on="member_id", how="left")

    def _augment_realtime(
        self, data: pd.DataFrame, feature_store_config: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        if not feature_store_config:
            raise ValueError("실시간 증강을 위해서는 feature_store_config가 필요합니다.")
        if "member_id" not in data.columns:
            raise ValueError("'member_id' 컬럼이 없어 실시간 피처 조회를 할 수 없습니다.")

        user_ids = data["member_id"].tolist()
        logger.info(f"{len(user_ids)}명의 사용자에 대한 실시간 피처 조회를 시작합니다.")

        store_type = feature_store_config.get("store_type")
        factory = Factory(self.settings) # 임시 settings로 factory 생성
        
        if store_type == "redis":
            redis_adapter = factory.create_redis_adapter(feature_store_config)
            feature_map = redis_adapter.get_features(user_ids, self.realtime_features_list)
        else:
            raise NotImplementedError(f"지원하지 않는 실시간 스토어 타입입니다: {store_type}")

        if not feature_map:
            logger.warning("조회된 실시간 피처가 없습니다.")
            feature_df = pd.DataFrame(columns=["member_id"] + self.realtime_features_list)
        else:
            feature_df = pd.DataFrame.from_records(list(feature_map.values()))
            feature_df["member_id"] = list(feature_map.keys())

        for col in self.realtime_features_list:
            if col not in feature_df.columns:
                feature_df[col] = None
        
        return pd.merge(data, feature_df, on="member_id", how="left")
