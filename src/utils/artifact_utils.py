from urllib.parse import urlparse
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from src.settings.settings import Settings
from src.core.factory import Factory
from src.utils.logger import logger

def save_dataset(
    df: pd.DataFrame,
    store_name: str,
    settings: Settings,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Factory를 통해 적절한 데이터 어댑터를 생성하고, DataFrame을 저장합니다.

    Args:
        df: 저장할 DataFrame.
        store_name: 사용할 아티팩트 스토어의 이름 (config.yaml에 정의됨).
        settings: 전체 프로젝트 설정 객체.
        options: 저장 시 사용할 추가 옵션 (e.g., write_mode).
    """
    if df.empty:
        logger.warning(f"DataFrame이 비어있어, '{store_name}' 아티팩트 저장을 건너뜁니다.")
        return

    try:
        store_config = settings.artifact_stores[store_name]
    except KeyError:
        logger.error(f"'{store_name}'에 해당하는 아티팩트 스토어 설정을 찾을 수 없습니다.")
        raise

    if not store_config.enabled:
        logger.info(f"'{store_name}' 아티팩트 스토어가 비활성화���어 있어 저장을 건너뜁니다.")
        return

    base_uri = store_config.base_uri
    parsed_uri = urlparse(base_uri)
    scheme = parsed_uri.scheme

    factory = Factory(settings)
    adapter = factory.create_data_adapter(scheme)

    # 저장될 최종 경로(테이블명 또는 파일명) 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"{settings.model.name}_{timestamp}"
    
    # BigQuery의 경우 project.dataset.table, GCS/S3/File의 경우 bucket/path/file
    if scheme == 'bq':
        target_path = f"{parsed_uri.netloc}.{artifact_name}"
    else:
        # path가 '/'로 시작하면 중복 슬래시 제거
        path_prefix = parsed_uri.path.lstrip('/')
        target_path = f"{base_uri.rstrip('/')}/{artifact_name}"

    logger.info(f"'{store_name}'에 데이터 저장 시작. 어댑터: {type(adapter).__name__}, 경로: {target_path}")
    
    adapter.write(df, target_path, options)
