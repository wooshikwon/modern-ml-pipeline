# src/utils/artifact_utils.py

import pandas as pd
from urllib.parse import urlparse
from datetime import datetime

from src.settings.settings import Settings
from src.utils.logger import logger
from src.utils import bigquery_utils, gcs_utils

def save_dataset(df: pd.DataFrame, store_name: str, settings: Settings):
    """
    DataFrame을 config에 정의된 아티팩트 스토어에 저장합니다.
    URI의 scheme에 따라 적절한 저장 유틸리티를 동적으로 호출합니다.

    Args:
        df: 저장할 DataFrame.
        store_name: 사용할 아티팩트 스토어의 이름 (config.yaml에 정의됨).
        settings: 프로젝트 설정 객체.
    """
    if df.empty:
        logger.warning(f"DataFrame이 비어있어, '{store_name}' 아티팩트 저장을 건너뜁니다.")
        return

    try:
        config = settings.artifact_stores[store_name]
    except KeyError:
        logger.error(f"'{store_name}'에 해당하는 아티팩트 스토어 설정을 찾을 수 없습니다.")
        raise

    if not config.enabled:
        logger.info(f"'{store_name}' 아티팩트 스토어가 비활성화되어 있어 저장을 건너뜁니다.")
        return

    base_uri = config.base_uri
    parsed_uri = urlparse(base_uri)
    
    # 모델 이름과 타임스탬프를 포함하는 표준 아티팩트 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"{settings.model.name}_{timestamp}"

    logger.info(f"'{store_name}' 아티팩트 스토어에 데이터 저장을 시작합니다. (URI: {base_uri})")

    try:
        if parsed_uri.scheme == 'bq':
            # bq://project-id.dataset-id
            project_id, dataset_id = parsed_uri.netloc.split('.', 1)
            table_id = f"{dataset_id}.{artifact_name}"
            logger.info(f"BigQuery 테이블에 저장: {project_id}.{table_id}")
            bigquery_utils.upload_df_to_bigquery(df, f"{project_id}.{table_id}", settings)

        from pathlib import Path
# ... (기존 import)

# ... (save_dataset 함수 내부)
        elif parsed_uri.scheme == 'gs':
            # ... (기존 로직)
        
        elif parsed_uri.scheme == 'file':
            # file:///path/to/dir 또는 file://path/to/dir
            dir_path = Path(parsed_uri.path)
            if not dir_path.is_absolute():
                # 프로젝트 루트 기준 상대 경로로 처리
                dir_path = Path(__file__).resolve().parent.parent.parent / dir_path
            
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{artifact_name}.parquet"
            logger.info(f"로컬 파일 시스템에 Parquet 파일로 저장: {file_path}")
            df.to_parquet(file_path, index=False)

        # elif parsed_uri.scheme == 's3':
        #     # 향후 S3 지원 확장 지점
        #     logger.info("S3 저장은 아직 지원되지 않습니다.")
        #     pass
            
        else:
            raise ValueError(f"지원하지 않는 아티팩트 스토어 URI scheme입니다: '{parsed_uri.scheme}'")

        logger.info(f"'{store_name}' 아티팩트 저장을 성공적으로 완료했습니다.")

    except Exception as e:
        logger.error(f"'{store_name}' 아티팩트 저장 중 오류 발생: {e}", exc_info=True)
        raise
