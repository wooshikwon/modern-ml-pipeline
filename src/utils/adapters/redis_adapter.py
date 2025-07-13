from typing import Dict, Any, List, Optional
import redis
import json
import pandas as pd

from src.interface.base_adapter import BaseAdapter
from src.settings.settings import RealtimeFeatureStoreSettings
from src.utils.system.logger import logger

class RedisAdapter(BaseAdapter):
    """
    Redis (온라인 피처 스토어)와의 상호작용을 처리하는 어댑터.
    BaseAdapter를 상속하여 표준 read/write 인터페이스를 제공하면서도
    Redis 전용 get_features 메서드를 추가로 제공합니다.
    """
    def __init__(self, settings: RealtimeFeatureStoreSettings):
        # BaseAdapter 초기화
        super().__init__(settings)
        self.client = self._get_client()

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

    def read(
        self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Redis에서 데이터를 읽어 DataFrame으로 반환합니다.
        
        Args:
            source: 조회할 키 패턴 또는 사용자 ID 리스트
            params: 추가 파라미터 (feature_columns, user_ids 등)
        """
        params = params or {}
        user_ids = params.get("user_ids", [])
        
        if not user_ids:
            logger.warning("Redis read: user_ids가 제공되지 않았습니다.")
            return pd.DataFrame()
        
        features_dict = self.get_features(user_ids)
        
        # Dict를 DataFrame으로 변환
        if features_dict:
            df = pd.DataFrame.from_dict(features_dict, orient='index')
            df.index.name = 'user_id'
            df = df.reset_index()
            return df
        else:
            return pd.DataFrame()

    def write(
        self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        DataFrame을 Redis에 저장합니다.
        
        Args:
            df: 저장할 DataFrame
            target: 저장할 키 패턴
            options: 저장 옵션 (user_id_column, ttl 등)
        """
        options = options or {}
        user_id_column = options.get("user_id_column", "user_id")
        ttl = options.get("ttl", None)
        
        if user_id_column not in df.columns:
            raise ValueError(f"DataFrame에 '{user_id_column}' 컬럼이 없습니다.")
        
        logger.info(f"Redis에 {len(df)}개 사용자 피처 저장 시작")
        
        for _, row in df.iterrows():
            user_id = row[user_id_column]
            features = row.drop(user_id_column).to_dict()
            
            # JSON으로 직렬화하여 저장
            key = f"user_features:{user_id}"
            self.client.set(key, json.dumps(features), ex=ttl)
        
        logger.info(f"Redis에 {len(df)}개 사용자 피처 저장 완료")

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