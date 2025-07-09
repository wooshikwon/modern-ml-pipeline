# uplift-virtual-coupon/utils/gcs_utils.py

import io
import json
import pickle
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Union, Optional, Dict, List
from datetime import datetime

from google.cloud import storage
from google.api_core import exceptions
from google.oauth2 import service_account

from config.settings import Settings
from src.utils.logger import logger


def get_gcs_client(settings: Settings) -> storage.Client:
    """GCS 클라이언트를 초기화하고 반환합니다."""
    try:
        project_id = settings.environment.gcp_project_id
        credential_path = settings.environment.gcp_credential_path

        if not project_id:
            raise ValueError("GCP project ID가 설정되지 않았습니다.")

        if not credential_path:
            logger.warning("GCP credential path가 설정되지 않았습니다. 기본 인증을 사용합니다.")
            return storage.Client(project=project_id)

        credential_file = Path(credential_path)
        if not credential_file.exists():
            raise FileNotFoundError(f"인증 파일을 찾을 수 없습니다: {credential_path}")

        credentials = service_account.Credentials.from_service_account_file(credential_path)
        client = storage.Client(credentials=credentials, project=project_id)

        logger.info(f"GCS 클라이언트 초기화 완료 (Project: {project_id})")
        return client

    except Exception as e:
        logger.error(f"GCS 클라이언트 초기화 실패: {e}")
        raise


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """GCS URI (gs://bucket/blob)를 파싱하여 버킷 이름과 블롭 경로를 반환합니다."""
    if not isinstance(gcs_uri, str) or not gcs_uri.strip() or not gcs_uri.startswith("gs://"):
        raise ValueError("GCS URI는 'gs://'로 시작하는 비어있지 않은 문자열이어야 합니다.")
    
    path_part = gcs_uri[5:]
    if not path_part or "/" not in path_part:
        raise ValueError("GCS URI 형식이 잘못되었습니다. 'gs://<bucket_name>/<blob_path>' 형식이어야 합니다.")
    
    bucket_name, blob_name = path_part.split("/", 1)
    if not bucket_name or not blob_name:
        raise ValueError("버킷 이름과 블롭 경로가 모두 필요합니다.")
        
    return bucket_name, blob_name


def _get_serialization_format(gcs_uri: str) -> str:
    """파일 확장자를 기반으로 직렬화 형식을 결정합니다."""
    extension = Path(gcs_uri).suffix.lower()
    return {
        '.pkl': 'pickle', '.pickle': 'pickle', '.joblib': 'joblib',
        '.json': 'json', '.csv': 'csv', '.parquet': 'parquet'
    }.get(extension, 'joblib')


def upload_object_to_gcs(
    obj: Any,
    gcs_uri: str,
    settings: Settings,
    serialization_format: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> bool:
    """Python 객체를 GCS에 업로드합니다."""
    try:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        fmt = serialization_format or _get_serialization_format(gcs_uri)
        logger.info(f"GCS 업로드 시작: {gcs_uri} (형식: {fmt})")

        client = get_gcs_client(settings)
        blob = client.bucket(bucket_name).blob(blob_name)
        
        blob.metadata = metadata or {}
        blob.metadata.update({'uploaded_at': datetime.now().isoformat(), 'serialization_format': fmt})

        with io.BytesIO() as buffer:
            if fmt == 'pickle':
                pickle.dump(obj, buffer)
                content_type = "application/octet-stream"
            elif fmt == 'joblib':
                joblib.dump(obj, buffer)
                content_type = "application/octet-stream"
            elif fmt == 'json':
                buffer.write(json.dumps(obj, ensure_ascii=False, indent=2).encode('utf-8'))
                content_type = "application/json"
            elif isinstance(obj, pd.DataFrame) and fmt == 'csv':
                obj.to_csv(buffer, index=False)
                content_type = "text/csv"
            elif isinstance(obj, pd.DataFrame) and fmt == 'parquet':
                obj.to_parquet(buffer, index=False)
                content_type = "application/octet-stream"
            else:
                raise ValueError(f"지원되지 않는 직렬화 형식: {fmt}")
            
            buffer.seek(0)
            blob.upload_from_file(buffer, content_type=content_type)

        logger.info(f"GCS 업로드 완료: {gcs_uri} (크기: {blob.size:,} bytes)")
        return True
    except (ValueError, exceptions.GoogleAPICallError) as e:
        logger.error(f"GCS 업로드 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"GCS 업로드 중 예상치 못한 오류: {e}")
        return False


def download_object_from_gcs(gcs_uri: str, settings: Settings, serialization_format: Optional[str] = None) -> Any:
    """GCS에서 객체를 다운로드합니다."""
    try:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        fmt = serialization_format or _get_serialization_format(gcs_uri)
        logger.info(f"GCS 다운로드 시작: {gcs_uri} (형식: {fmt})")

        client = get_gcs_client(settings)
        blob = client.bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS에서 파일을 찾을 수 없습니다: {gcs_uri}")

        with io.BytesIO(blob.download_as_bytes()) as buffer:
            if fmt == 'pickle': return pickle.load(buffer)
            if fmt == 'joblib': return joblib.load(buffer)
            if fmt == 'json': return json.load(buffer)
            if fmt == 'csv': return pd.read_csv(buffer)
            if fmt == 'parquet': return pd.read_parquet(buffer)
            raise ValueError(f"지원되지 않는 직렬화 형식: {fmt}")

    except (FileNotFoundError, ValueError, exceptions.GoogleAPICallError) as e:
        logger.error(f"GCS 다운로드 실패: {e}")
        raise
    except Exception as e:
        logger.error(f"GCS 다운로드 중 예상치 못한 오류: {e}")
        raise

def generate_model_path(model_name: str, settings: Settings, version: Optional[str] = None) -> str:
    """모델 저장을 위한 GCS 경로를 생성합니다."""
    try:
        bucket_name = settings.pipeline.transformer.output.bucket_name
        if not bucket_name:
            raise ValueError("pipeline.transformer.output.bucket_name이 설정되지 않았습니다.")
        
        version_str = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{model_name}/v{version_str}/{model_name}.joblib"
        gcs_uri = f"gs://{bucket_name}/{model_path}"
        
        logger.info(f"모델 경로 생성: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"모델 경로 생성 실패: {e}")
        raise
