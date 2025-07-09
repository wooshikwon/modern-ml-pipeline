import pandas as pd
from typing import Dict, Union

from src.interface.base_augmenter import BaseAugmenter
from src.utils.logger import logger
from config.settings import Settings, AugmenterBatchSettings, AugmenterRealtimeSettings
from src.utils.bigquery_utils import execute_query
from src.utils.feature_store_utils import get_features_from_redis


class BatchAugmenter(BaseAugmenter):
    """배치 환경에서 BigQuery를 통해 피처를 증강하는 클래스."""
    def __init__(self, config: AugmenterBatchSettings, settings: Settings):
        self.config = config
        self.settings = settings
        self.feature_sql_path = config.feature_store_sql_path

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"배치 증강을 시작합니다. SQL: {self.feature_sql_path}")
        
        # 피처 스토어(BigQuery)에서 피처 데이터 로드
        feature_df = execute_query(f"SELECT * FROM `{self.feature_sql_path}`", settings=self.settings)
        
        # 입력 데이터와 피처 데이터를 'member_id' 기준으로 조인
        augmented_df = pd.merge(data, feature_df, on="member_id", how="left")
        logger.info(f"배치 증강 완료. {len(data)}행 -> {len(augmented_df)}행")
        return augmented_df


class RealtimeAugmenter(BaseAugmenter):
    """실시간 환경에서 Redis를 통해 피처를 증강하는 클래스."""
    def __init__(self, config: AugmenterRealtimeSettings):
        self.config = config

    def augment(self, data: pd.DataFrame) -> pd.DataFrame:
        user_ids = data['member_id'].tolist()
        logger.info(f"실시간 증강을 시작합니다. 사용자 수: {len(user_ids)}")
        
        # 피처 스토어(Redis)에서 피처 데이터 조회
        features = get_features_from_redis(user_ids, self.config)
        if not features:
            logger.warning("Redis에서 조회된 피처가 없습니다.")
            return data

        feature_df = pd.DataFrame.from_dict(features, orient='index')
        
        # 입력 데이터와 피처 데이터를 'member_id' 기준으로 조인
        augmented_df = pd.merge(data, feature_df, left_on="member_id", right_index=True, how="left")
        logger.info(f"실시간 증강 완료. {len(data)}행 -> {len(augmented_df)}행")
        return augmented_df


def create_augmenter(settings: Settings) -> BaseAugmenter:
    """
    실행 환경('run_mode')에 따라 적절한 Augmenter 인스턴스를 생성하여 반환합니다.
    """
    run_mode = settings.environment.run_mode
    
    if run_mode == "serving":
        logger.info("실시간 서빙용 RealtimeAugmenter를 생성합니다.")
        return RealtimeAugmenter(config=settings.augmenter.realtime)
    else:  # "local", "batch", "train" 등 나머지 모든 경우
        logger.info("배치 처리용 BatchAugmenter를 생성합니다.")
        return BatchAugmenter(config=settings.augmenter.batch, settings=settings)
