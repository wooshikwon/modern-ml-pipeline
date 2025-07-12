import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.interface.base_augmenter import BaseAugmenter
from src.utils.logger import logger
from src.settings.settings import Settings, AugmenterSettings
from src.utils import bigquery_utils, sql_utils

class LocalFileAugmenter(BaseAugmenter):
    """로컬 피처 파일과 조인하여 데이터를 증강하는 클래스."""
    def __init__(self, path: str):
        self.feature_path = Path(path)
        if not self.feature_path.is_absolute():
            self.feature_path = Path(__file__).resolve().parent.parent.parent / self.feature_path

    def augment(self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"로컬 피처 파일로 증강을 시작합니다: {self.feature_path}")
        feature_df = pd.read_parquet(self.feature_path)
        return pd.merge(data, feature_df, on="member_id", how="left")

class SQLTemplateAugmenter(BaseAugmenter):
    """SQL 템플릿을 사용하여 배치 환경에서 피처를 증강하는 클래스."""
    def __init__(self, template_path: str, settings: Settings):
        self.template_path = template_path
        self.settings = settings

    def augment(self, data: pd.DataFrame, context_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        logger.info(f"SQL 템플릿 기반 피처 증강을 시작합니다. (템플릿: {self.template_path})")
        temp_table_id = self._upload_targets_to_temp_table(data)
        params = context_params or {}
        params['temp_target_table_id'] = temp_table_id
        rendered_sql = sql_utils.render_sql(self.template_path, params)
        feature_df = bigquery_utils.execute_query(rendered_sql, self.settings)
        augmented_df = pd.merge(data, feature_df, on="member_id", how="left")
        bigquery_utils.delete_table(temp_table_id, self.settings)
        return augmented_df

    def _upload_targets_to_temp_table(self, df: pd.DataFrame) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_table_id = f"{self.settings.environment.gcp_project_id}.temp_data.targets_{timestamp}"
        logger.info(f"추론 대상을 임시 테이블에 업로드합니다: {temp_table_id}")
        bigquery_utils.upload_df_to_bigquery(df, temp_table_id, self.settings)
        bigquery_utils.set_table_expiration(temp_table_id, hours=24, settings=self.settings)
        return temp_table_id

def get_augmenter(settings: Settings) -> Optional[BaseAugmenter]:
    """
    "환경 인지" 팩토리 함수.
    레시피의 augmenter 설정과 현재 실행 환경(app_env)에 따라
    적절한 증강기 인스턴스를 생성하여 반환합니다.
    """
    augmenter_config = settings.model.augmenter
    if not augmenter_config:
        return None

    logger.info(f"'{augmenter_config.name}' 증강기 생성을 시작합니다. (환경: {settings.environment.app_env})")
    is_local = settings.environment.app_env == "local"
    local_path = augmenter_config.local_override_path

    if is_local and local_path:
        logger.info(f"로컬 재정의 경로를 사용하여 LocalFileAugmenter를 생성합니다: {local_path}")
        return LocalFileAugmenter(path=local_path)
    
    if augmenter_config.source_template_path:
        logger.info(f"SQL 템플릿을 사용하여 SQLTemplateAugmenter를 생성합니다: {augmenter_config.source_template_path}")
        return SQLTemplateAugmenter(template_path=augmenter_config.source_template_path, settings=settings)
        
    raise ValueError(f"'{augmenter_config.name}' 증강기에 대한 유효한 소스 경로를 찾을 수 없습니다.")