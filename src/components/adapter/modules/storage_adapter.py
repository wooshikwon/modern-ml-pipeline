from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.interface.base_adapter import BaseAdapter
from src.settings import Settings
from src.utils.core.console import get_console

class StorageAdapter(BaseAdapter):
    """
    fsspec 라이브러리를 기반으로 하는 통합 스토리지 어댑터.
    로컬 파일 시스템, GCS, S3 등 다양한 스토리지를 지원합니다.
    """
    def __init__(self, settings: Settings, **kwargs):
        console = get_console(settings)
        console.info("[StorageAdapter] 초기화 시작합니다")

        super().__init__(settings, **kwargs)
        # 새로운 settings 스키마에서 data_source 설정을 사용
        try:
            # data_source.config에서 storage_options 추출
            config = settings.config.data_source.config
            if hasattr(config, 'storage_options'):
                # Pydantic model의 경우 (LocalFilesConfig)
                self.storage_options = config.storage_options
            elif isinstance(config, dict):
                # dict의 경우
                self.storage_options = config.get('storage_options', {})
            else:
                self.storage_options = {}
            console.info(f"[StorageAdapter] 설정 로드 완료: storage_options={len(self.storage_options)} items")
        except Exception as e:
            console.warning(f"[StorageAdapter] 설정을 찾을 수 없습니다: {e}. 기본값 사용합니다")
            self.storage_options = {}

        console.info("[StorageAdapter] 초기화 완료되었습니다",
                    rich_message="✅ [StorageAdapter] initialized")

    def read(self, uri: str, **kwargs) -> pd.DataFrame:
        """URI로부터 데이터를 읽어 DataFrame으로 반환합니다."""
        console = get_console()
        # Convert Path object to string if needed
        uri_str = str(uri)
        file_name = Path(uri_str).name
        file_ext = Path(uri_str).suffix.lower()

        # 파일 정보 및 옵션 체크
        console.log_processing_step(
            f"Storage 파일 읽기 시작: {file_name}",
            f"유형: {file_ext}, Storage 옵션: {len(self.storage_options)}개"
        )

        # 지원 형식 및 옵션 검증
        if self.storage_options:
            console.log_processing_step(
                "Storage 옵션 적용",
                f"설정된 옵션: {', '.join(self.storage_options.keys())}"
            )

        try:
            lower = uri_str.lower()
            if lower.endswith('.csv'):
                console.log_processing_step("CSV 파일 파싱 시작", f"속성: {len(kwargs)}개 옵션 적용")
                result = pd.read_csv(uri_str, storage_options=self.storage_options, **kwargs)

            elif lower.endswith('.parquet'):
                console.log_processing_step("Parquet 파일 파싱 시작", f"속성: {len(kwargs)}개 옵션 적용")
                result = pd.read_parquet(uri_str, storage_options=self.storage_options, **kwargs)

            else:
                # 기타 파일 형식도 지원
                if lower.endswith('.json'):
                    result = pd.read_json(uri_str, **kwargs)
                else:
                    raise ValueError(f"지원되지 않는 파일 형식: {file_ext}")

            # 데이터 크기 및 품질 정보
            data_size_mb = result.memory_usage(deep=True).sum() / (1024 * 1024)
            null_count = result.isnull().sum().sum()

            console.log_data_operation(
                f"Storage 파일 읽기 완료",
                shape=(len(result), len(result.columns)),
                details=f"파일: {file_name}, 메모리: {data_size_mb:.1f} MB"
            )

            # 데이터 품질 체크
            if null_count > 0:
                console.log_processing_step(
                    "데이터 품질 감지",
                    f"결측값 {null_count:,}개 발견"
                )

            return result

        except Exception as e:
            console.log_error_with_context(
                f"Storage 파일 읽기 실패: {e}",
                context={
                    "file_path": uri_str,
                    "file_type": file_ext,
                    "storage_options_count": len(self.storage_options),
                    "kwargs_count": len(kwargs)
                },
                suggestion="파일 경로와 액세스 권한을 확인하세요"
            )
            raise

    def write(self, df: pd.DataFrame, uri: str, **kwargs):
        """DataFrame을 지정된 URI에 저장합니다."""
        console = get_console()
        # Convert Path object to string if needed
        uri_str = str(uri)
        file_name = Path(uri_str).name
        file_ext = Path(uri_str).suffix.lower()
        data_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        console.log_processing_step(
            f"Storage 파일 저장 시작: {file_name}",
            f"크기: {data_size_mb:.1f} MB, 형식: {file_ext}"
        )

        # 로컬 파일 시스템의 경우, 쓰기 전에 디렉토리가 존재하는지 확인하고 생성합니다.
        if "://" not in uri_str or uri_str.startswith("file://"):
            path = Path(uri_str.replace("file://", ""))
            if not path.parent.exists():
                console.log_processing_step(
                    "디렉토리 생성",
                    f"경로 생성: {path.parent}"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                console.log_processing_step("디렉토리 확인", "기존 경로 사용")

        try:
            lower = uri_str.lower()
            if lower.endswith('.csv'):
                console.log_processing_step("CSV 형식 저장", f"{len(df):,} rows × {len(df.columns)} columns")
                df.to_csv(uri_str, index=False, **kwargs)

            elif lower.endswith('.parquet'):
                console.log_processing_step("Parquet 형식 저장", f"{len(df):,} rows × {len(df.columns)} columns")
                if self.storage_options:
                    console.log_processing_step(
                        "Storage 옵션 적용",
                        f"{len(self.storage_options)}개 옵션 사용"
                    )
                df.to_parquet(uri_str, storage_options=self.storage_options, **kwargs)

            elif lower.endswith('.json'):
                console.log_processing_step("JSON 형식 저장", f"{len(df):,} rows × {len(df.columns)} columns")
                df.to_json(uri_str, **kwargs)

            else:
                # 기본적으로 Parquet 사용
                console.log_processing_step("Parquet 형식으로 자동 저장", f"확장자 미지원: {file_ext}")
                df.to_parquet(uri_str, storage_options=self.storage_options, **kwargs)

            # 저장 완료 확인 및 파일 크기 체크
            if "://" not in uri_str or uri_str.startswith("file://"):
                saved_path = Path(uri_str.replace("file://", ""))
                if saved_path.exists():
                    file_size_mb = saved_path.stat().st_size / (1024 * 1024)
                    console.log_processing_step(
                        f"Storage 파일 저장 완료: {file_name}",
                        f"디스크 크기: {file_size_mb:.1f} MB, 압축비: {data_size_mb/file_size_mb:.1f}x"
                    )
                else:
                    console.log_processing_step(
                        f"Storage 파일 저장 완료: {file_name}",
                        f"{len(df):,} rows 저장"
                    )
            else:
                console.log_processing_step(
                    f"Remote storage 저장 완료: {file_name}",
                    f"{len(df):,} rows 전송"
                )

        except Exception as e:
            console.log_error_with_context(
                f"Storage 파일 저장 실패: {e}",
                context={
                    "file_path": uri_str,
                    "file_type": file_ext,
                    "data_shape": f"{len(df)} × {len(df.columns)}",
                    "storage_options_count": len(self.storage_options)
                },
                suggestion="디렉토리 권한과 디스크 용량을 확인하세요"
            )
            raise

# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("storage", StorageAdapter)