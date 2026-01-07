from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.components.adapter.base import BaseAdapter
from src.settings import Settings
from src.utils.core.logger import log_data_debug, logger


class StorageAdapter(BaseAdapter):
    """
    fsspec 라이브러리를 기반으로 하는 통합 스토리지 어댑터.
    로컬 파일 시스템, GCS, S3 등 다양한 스토리지를 지원합니다.
    """

    def __init__(self, settings: Settings, **kwargs):
        super().__init__(settings, **kwargs)
        self.base_path = None
        self.storage_options = {}

        # 새로운 settings 스키마에서 data_source 설정을 사용
        try:
            config = settings.config.data_source.config
            # base_path 추출 (상대 경로 지원용)
            if hasattr(config, "base_path"):
                self.base_path = config.base_path
            elif isinstance(config, dict):
                self.base_path = config.get("base_path")

            # storage_options 추출 (pydantic 모델 → dict 변환)
            if hasattr(config, "storage_options"):
                opts = config.storage_options
                if hasattr(opts, "model_dump"):
                    raw_opts = opts.model_dump(exclude_none=True)
                elif hasattr(opts, "dict"):
                    raw_opts = opts.dict(exclude_none=True)
                elif isinstance(opts, dict):
                    raw_opts = opts
                else:
                    raw_opts = {}
                self.storage_options = self._convert_storage_options(raw_opts)
            elif isinstance(config, dict):
                raw_opts = config.get("storage_options", {})
                self.storage_options = self._convert_storage_options(raw_opts)

            log_data_debug(f"초기화 완료 - base_path: {self.base_path}", "StorageAdapter")
        except Exception as e:
            logger.warning(f"[DATA:StorageAdapter] 설정 로드 실패, 기본값 사용: {e}")

    def _convert_storage_options(self, opts: dict) -> dict:
        """스키마 필드명을 fsspec/s3fs가 기대하는 형식으로 변환합니다."""
        if not opts:
            return {}

        result = {}

        # S3 옵션 매핑: aws_access_key_id → key, aws_secret_access_key → secret
        if "aws_access_key_id" in opts:
            result["key"] = opts["aws_access_key_id"]
        if "aws_secret_access_key" in opts:
            result["secret"] = opts["aws_secret_access_key"]

        # region_name → client_kwargs
        if "region_name" in opts:
            result["client_kwargs"] = {"region_name": opts["region_name"]}

        # 이미 s3fs 형식이면 그대로 사용
        if "key" in opts and "key" not in result:
            result["key"] = opts["key"]
        if "secret" in opts and "secret" not in result:
            result["secret"] = opts["secret"]
        if "client_kwargs" in opts and "client_kwargs" not in result:
            result["client_kwargs"] = opts["client_kwargs"]

        # GCS 등 다른 스토리지 옵션은 그대로 전달
        for key, value in opts.items():
            if key not in result and key not in [
                "aws_access_key_id",
                "aws_secret_access_key",
                "region_name",
            ]:
                result[key] = value

        return result

    def _resolve_path(self, uri: str) -> str:
        """
        경로를 해석합니다.

        - 절대 경로(/로 시작) 또는 URL(://포함): 그대로 사용
        - 상대 경로: 프로젝트 루트 기준으로 직접 사용 (data/file.csv)
        - 클라우드 base_path(s3://, gs://): base_path와 결합
        """
        uri_str = str(uri)

        # 이미 절대 경로이거나 URL인 경우 그대로 반환
        if uri_str.startswith("/") or "://" in uri_str:
            return uri_str

        # 클라우드 스토리지인 경우에만 base_path와 결합
        if self.base_path:
            base = str(self.base_path)
            if "://" in base:
                resolved = base.rstrip("/") + "/" + uri_str
                log_data_debug(f"클라우드 경로 결합: {uri_str} -> {resolved}", "StorageAdapter")
                return resolved

        # 로컬 경로는 그대로 사용 (프로젝트 루트 기준 상대 경로)
        return uri_str

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI로부터 데이터를 읽어 DataFrame으로 반환합니다."""
        # 상대 경로를 절대 경로로 변환
        uri_str = self._resolve_path(uri)
        file_name = Path(uri_str).name
        file_ext = Path(uri_str).suffix.lower()

        # pandas에서 지원하지 않는 인자 필터링
        pandas_kwargs = {k: v for k, v in kwargs.items() if k != "params"}

        log_data_debug(f"파일 읽기 시작: {file_name}", "StorageAdapter")

        try:
            lower = uri_str.lower()
            if lower.endswith(".csv"):
                log_data_debug(f"CSV 파싱 - 옵션: {len(pandas_kwargs)}개", "StorageAdapter")
                result = pd.read_csv(uri_str, storage_options=self.storage_options, **pandas_kwargs)

            elif lower.endswith(".parquet"):
                log_data_debug(f"Parquet 파싱 - 옵션: {len(pandas_kwargs)}개", "StorageAdapter")
                result = pd.read_parquet(
                    uri_str, storage_options=self.storage_options, **pandas_kwargs
                )

            else:
                # 기타 파일 형식도 지원
                if lower.endswith(".json"):
                    result = pd.read_json(uri_str, **pandas_kwargs)
                else:
                    raise ValueError(f"지원되지 않는 파일 형식: {file_ext}")

            # 데이터 크기 및 품질 정보
            data_size_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
            null_count = result.isnull().sum().sum()

            log_data_debug(
                f"파일 읽기 완료: {file_name}, {len(result)}행 × {len(result.columns)}열, {data_size_mb:.1f}MB",
                "StorageAdapter",
            )

            if null_count > 0:
                log_data_debug(f"결측값 {null_count:,}개 발견", "StorageAdapter")

            return result

        except Exception as e:
            logger.error(f"[DATA:StorageAdapter] 파일 읽기 실패: {file_name}, 오류: {e}")
            raise

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 저장합니다."""
        # Convert Path object to string if needed
        uri_str = str(uri)
        file_name = Path(uri_str).name
        file_ext = Path(uri_str).suffix.lower()
        data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        log_data_debug(
            f"파일 저장 시작: {file_name}, {len(df)}행, {data_size_mb:.1f}MB", "StorageAdapter"
        )

        # 로컬 파일 시스템의 경우, 쓰기 전에 디렉토리가 존재하는지 확인하고 생성합니다.
        if "://" not in uri_str or uri_str.startswith("file://"):
            path = Path(uri_str.replace("file://", ""))
            if not path.parent.exists():
                log_data_debug(f"디렉토리 생성: {path.parent}", "StorageAdapter")
                path.parent.mkdir(parents=True, exist_ok=True)

        try:
            lower = uri_str.lower()
            if lower.endswith(".csv"):
                df.to_csv(uri_str, index=False, **kwargs)

            elif lower.endswith(".parquet"):
                df.to_parquet(uri_str, storage_options=self.storage_options, **kwargs)

            elif lower.endswith(".json"):
                df.to_json(uri_str, **kwargs)

            else:
                # 기본적으로 Parquet 사용
                df.to_parquet(uri_str, storage_options=self.storage_options, **kwargs)

            log_data_debug(f"파일 저장 완료: {file_name}, {len(df):,}행", "StorageAdapter")

        except Exception as e:
            logger.error(f"[DATA:StorageAdapter] 파일 저장 실패: {file_name}, 오류: {e}")
            raise


# Self-registration
from ..registry import AdapterRegistry

AdapterRegistry.register("storage", StorageAdapter)
