from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Optional

from src.interface.base_adapter import BaseAdapter
from src.utils.system.console_manager import get_console


class BigQueryAdapter(BaseAdapter):
    """
    BigQuery Adapter
    - pandas-gbq 기반 단순 쓰기 구현 (append 모드)
    - 인증은 GOOGLE_APPLICATION_CREDENTIALS 또는 ADC에 의존
    """
    def __init__(self, settings: Any, **kwargs):
        super().__init__(settings)
        # 설정은 settings 또는 호출시 kwargs로 전달됨
        self._project_id: Optional[str] = None
        self._location: Optional[str] = None
        try:
            # data_source 또는 output.config에서 project_id/location을 가져올 수 있음
            self._project_id = kwargs.get("project_id")
            self._location = kwargs.get("location")
        except Exception:
            pass

    def read(self, source: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> pd.DataFrame:
        try:
            import pandas_gbq  # noqa: F401
        except Exception as e:
            raise ImportError("pandas-gbq가 필요합니다. 'pip install pandas-gbq pyarrow'를 설치하세요.") from e
        # 간단 구현: 완전한 read는 현재 요구사항 범위 밖. 필요 시 확장
        raise NotImplementedError("BigQueryAdapter.read는 현재 구현되지 않았습니다.")

    def write(self, df: pd.DataFrame, target: str, options: Optional[Dict[str, Any]] = None, **kwargs):
        """
        target: 'dataset.table' 형태 문자열을 기대
        options: {'if_exists': 'append'|'replace', 'project_id': str, 'location': str}
        또는 kwargs로 project_id/location 전달 가능
        """
        try:
            from pandas_gbq import to_gbq
        except Exception as e:
            raise ImportError("pandas-gbq가 필요합니다. 'pip install pandas-gbq pyarrow'를 설치하세요.") from e

        options = options or {}
        project_id = options.get("project_id") or kwargs.get("project_id") or self._project_id
        location = options.get("location") or kwargs.get("location") or self._location
        if_exists = options.get("if_exists", "append")

        if not project_id:
            raise ValueError("BigQuery project_id가 필요합니다. config 또는 환경변수를 확인하세요.")

        console = get_console()
        console.info(f"BigQuery에 쓰기 시작: target={target}, if_exists={if_exists}, project_id={project_id}, location={location}",
                    rich_message=f"📝 BigQuery write: [cyan]{target}[/cyan] mode=[yellow]{if_exists}[/yellow]")
        to_gbq(
            dataframe=df,
            destination_table=target,
            project_id=project_id,
            if_exists=if_exists,
            location=location,
        )
        console.info(f"BigQuery 쓰기 완료: rows={len(df)}",
                    rich_message=f"✅ BigQuery write complete: [green]{len(df)}[/green] rows")


# Self-registration
from ..registry import AdapterRegistry
AdapterRegistry.register("bigquery", BigQueryAdapter) 