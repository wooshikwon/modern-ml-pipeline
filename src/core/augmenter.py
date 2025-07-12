import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from src.interface.base_augmenter import BaseAugmenter
from src.utils.logger import logger
from src.settings.settings import Settings, AugmenterSettings
from src.utils import bigquery_utils, sql_utils, redis_utils

class SQLTemplateAugmenter(BaseAugmenter):
    """
    SQL 템플릿을 사용하여 배치 환경에서 피처를 증강하는 클래스.
    Loader의 결과물을 임시 테이블로 업로드하고, 이를 SQL 템플릿과 조인하여
    시점 일관성을 보장하는 동적 피처를 생성합니다.
    """
    def __init__(self, config: AugmenterSettings, settings: Settings):
        self.config = config
        self.settings = settings

    def augment(self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if self.config.type != "sql_template":
            raise ValueError("SQLTemplateAugmenter는 'sql_template' 타입의 설정만 지원합니다.")

        logger.info(f"SQL 템플릿 기반 피처 증강을 시작합니다. (템플릿: {self.config.template_path})")
        
        # 1. Loader의 결과물을 임시 테이블로 업로드
        temp_table_id = self._upload_targets_to_temp_table(data)
        
        # 2. SQL 템플릿 렌더링
        params = context_params or {}
        params['temp_target_table_id'] = temp_table_id # 임시 테이블 ID를 파라미터에 추가
        
        rendered_sql = sql_utils.render_sql(self.config.template_path, params)
        
        # 3. 렌더링된 SQL을 실행하여 피처 데이터 가져오기
        logger.info("증강용 피처 데이터를 조회합니다.")
        feature_df = bigquery_utils.execute_query(rendered_sql, self.settings)
        
        # 4. 원본 데이터와 피처 데이터를 조인하여 증강된 데이터 생성
        # 피처 테이블에는 'member_id'와 같은 조인 키가 반드시 포함되어야 함
        augmented_df = pd.merge(data, feature_df, on="member_id", how="left")
        
        # 5. 임시 테이블 삭제
        bigquery_utils.delete_table(temp_table_id, self.settings)
        
        logger.info(f"피처 증강 완료. {len(data)}행 -> {len(augmented_df)}행")
        return augmented_df

    def _upload_targets_to_temp_table(self, df: pd.DataFrame) -> str:
        """DataFrame을 BigQuery의 임시 테이블로 업로드하고 테이블 ID를 반환합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 임시 데이터셋은 미리 만들어져 있어야 함 (e.g., `temp_data`)
        temp_table_id = f"{self.settings.environment.gcp_project_id}.temp_data.targets_{timestamp}"
        
        logger.info(f"추론 대상을 임시 테이블에 업로드합니다: {temp_table_id}")
        bigquery_utils.upload_df_to_bigquery(df, temp_table_id, self.settings, write_disposition="WRITE_TRUNCATE")
        
        # (선택) 임시 테이블에 만료 시간 설정
        bigquery_utils.set_table_expiration(temp_table_id, hours=24, settings=self.settings)
        
        return temp_table_id

class RealtimeAugmenter(BaseAugmenter):
    """실시간 환경에서 Redis를 통해 피처를 증강하는 클래스."""
    def __init__(self, config: AugmenterSettings):
        self.config = config

    def augment(self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        # context_params는 현재 사용되지 않지만, 인터페이스 일관성을 위해 유지
        user_ids = data['member_id'].tolist()
        logger.info(f"실시간 증강을 시작합니다. 사용자 수: {len(user_ids)}")
        
        features = redis_utils.get_features_from_redis(user_ids, self.config)
        if not features:
            logger.warning("Redis에서 조회된 피처가 없습니다.")
            return data

        feature_df = pd.DataFrame.from_dict(features, orient='index')
        
        augmented_df = pd.merge(data, feature_df, left_on="member_id", right_index=True, how="left")
        logger.info(f"실시간 증강 완료. {len(data)}행 -> {len(augmented_df)}행")
        return augmented_df



