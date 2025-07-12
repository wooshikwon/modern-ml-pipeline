from typing import Dict, Any, List, Optional
import redis
import json

from src.interface.base_adapter import BaseAdapter
from src.settings.settings import RealtimeFeatureStoreSettings
from src.utils.logger import logger

class RedisAdapter(BaseAdapter):
    """
    Redis (온라인 피처 스토어)와의 상호작용을 처리하는 어댑터.
    읽기 전용(get_features)이며, BaseDataAdapter를 상속하지 않는다.
    """
    def __init__(self, settings: RealtimeFeatureStoreSettings):
        super().__init__(settings)

    def _get_client(self) -> redis.Redis:
        """Redis 클라이언트를 초기화하고 반환합니다."""
        try:
            client = redis.Redis(
                host=self.settings.connection.host,
                port=self.settings.connection.port,
                db=self.settings.connection.db,
                decode_responses=True
            )
            client.ping()
            logger.info(f"Redis 클라이언트 연결 성공: {self.settings.connection.host}:{self.settings.connection.port}")
            return client
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis 연결 실패: {e}", exc_info=True)
            raise

    def get_features(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Redis에서 여러 사용자의 피처를 한 번에 조회합니다.
        """
        if not self.client or not user_ids:
            return {}

        try:
            redis_keys = [f"user_features:{uid}" for uid in user_ids]
            feature_json_list = self.client.mget(redis_keys)

            results: Dict[str, Dict[str, Any]] = {}
            for user_id, feature_json in zip(user_ids, feature_json_list):
                if feature_json:
                    try:
                        results[user_id] = json.loads(feature_json)
                    except json.JSONDecodeError:
                        logger.warning(f"Redis에서 '{user_id}'의 피처 JSON 파싱 실패.")
            
            logger.info(f"{len(user_ids)}개 ID 중 {len(results)}개의 피처를 Redis에서 조회했습니다.")
            return results

        except Exception as e:
            logger.error(f"Redis에서 피처 조회 중 오류 발생: {e}", exc_info=True)
            return {}
