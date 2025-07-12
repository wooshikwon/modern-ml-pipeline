import redis
import json
from typing import List, Dict, Any, Optional

from src.settings.settings import AugmenterRealtimeSettings
from src.utils.logger import logger

# 이 파일은 실시간 피처 스토어(예: Redis)와의 상호작용을 담당합니다.

def get_redis_client(settings: AugmenterRealtimeSettings) -> Optional[redis.Redis]:
    """
    설정 정보를 바탕으로 Redis 클라이언트 연결을 생성하고 반환합니다.
    연결 실패 시 None을 반환하고 에러를 로깅합니다.
    """
    try:
        client = redis.Redis(
            host=settings.host,
            port=settings.port,
            decode_responses=True  # 응답을 자동으로 UTF-8로 디코딩
        )
        client.ping()  # 연결 테스트
        logger.info(f"Redis 클라이언트 연결 성공: {settings.host}:{settings.port}")
        return client
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis 연결 실패: {e}", exc_info=True)
        return None

def get_features_from_redis(
    user_ids: List[str],
    settings: AugmenterRealtimeSettings
) -> Dict[str, Dict[str, Any]]:
    """
    Redis에서 여러 사용자의 피처를 한 번에 조회합니다.
    
    Args:
        user_ids: 피처를 조회할 사용자 ID 리스트.
        settings: 실시간 Augmenter 설정.

    Returns:
        사용자 ID를 키로, 피처 딕셔너리를 값으로 하는 딕셔너리.
        피처가 없는 사용자는 결과에 포함되지 않습니다.
    """
    client = get_redis_client(settings)
    if not client or not user_ids:
        return {}

    try:
        # Redis의 MGET (Multiple Get)을 사용하여 여러 키를 한 번에 조회
        # 각 키는 'user_features:<user_id>'와 같은 형식을 가정
        redis_keys = [f"user_features:{uid}" for uid in user_ids]
        feature_json_list = client.mget(redis_keys)

        results: Dict[str, Dict[str, Any]] = {}
        for user_id, feature_json in zip(user_ids, feature_json_list):
            if feature_json:
                try:
                    # 저장된 JSON 문자열을 파싱하여 딕셔너리로 변환
                    results[user_id] = json.loads(feature_json)
                except json.JSONDecodeError:
                    logger.warning(f"Redis에서 '{user_id}'의 피처 JSON 파싱 실패.")
        
        logger.info(f"{len(user_ids)}개 ID 중 {len(results)}개의 피처를 Redis에서 조회했습니다.")
        return results

    except Exception as e:
        logger.error(f"Redis에서 피처 조회 중 오류 발생: {e}", exc_info=True)
        return {}
