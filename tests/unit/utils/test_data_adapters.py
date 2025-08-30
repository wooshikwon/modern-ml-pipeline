"""
데이터 어댑터 테스트 (독립적인 시나리오 기반)
- SqlAdapter: 가드 정책 검증(SELECT * / DDL 금지, 파일 경로 오류)
- StorageAdapter: 로컬 parquet R/W 기본 동작 검증
- FeastAdapter: 존재/미존재에 따른 import 가드만 확인(통합 테스트는 별도)
"""

import pytest
import os
import tempfile
import pandas as pd

from src.settings import Settings
from src.utils.adapters.sql_adapter import SqlAdapter
from src.utils.adapters.storage_adapter import StorageAdapter


class TestSqlAdapterGuards:
    def test_select_star_blocked(self, test_factories):
        """SELECT * 쿼리 차단 테스트 - Factory 패턴으로 파일 의존성 제거"""
        # SettingsFactory로 설정 생성
        settings_dict = test_factories['settings'].create_regression_settings("test")
        settings = Settings(**settings_dict)
        
        adapter = SqlAdapter(settings)
        with pytest.raises(ValueError):
            adapter.read("SELECT * FROM some_table LIMIT 1")

    def test_ddl_blocked(self, test_factories):
        """DDL 쿼리 차단 테스트 - Factory 패턴으로 파일 의존성 제거"""
        # SettingsFactory로 설정 생성
        settings_dict = test_factories['settings'].create_regression_settings("test")
        settings = Settings(**settings_dict)
        
        adapter = SqlAdapter(settings)
        with pytest.raises(ValueError):
            adapter.read("DROP TABLE dangerous")

    def test_sql_file_not_found(self, test_factories):
        """존재하지 않는 SQL 파일 오류 테스트 - Factory 패턴으로 파일 의존성 제거"""
        # SettingsFactory로 설정 생성
        settings_dict = test_factories['settings'].create_regression_settings("test")
        settings = Settings(**settings_dict)
        
        adapter = SqlAdapter(settings)
        with pytest.raises(FileNotFoundError):
            adapter.read("tests/fixtures/sql/does_not_exist.sql")


class TestStorageAdapterBasics:
    def test_parquet_rw_local(self, test_factories):
        """로컬 parquet 파일 읽기/쓰기 테스트 - Factory 패턴으로 파일 의존성 제거"""
        pytest.importorskip("pyarrow")
        pytest.importorskip("fsspec")
        
        # SettingsFactory로 설정 생성
        settings_dict = test_factories['settings'].create_classification_settings("test")
        settings = Settings(**settings_dict)
        
        adapter = StorageAdapter(settings)
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
            uri = f"file://{os.path.join(tmp, 'data.parquet')}"
            adapter.write(df, uri)
            assert os.path.exists(uri.replace("file://", ""))
            df2 = adapter.read(uri)
            pd.testing.assert_frame_equal(df2, df)


class TestFeastAdapterImport:
    def test_optional_import_guard(self):
        try:
            from src.utils.adapters.feast_adapter import FeastAdapter  # noqa
            assert True  # 임포트 성공 시 OK (동작 검증은 통합 테스트에서)
        except ImportError:
            pytest.skip("Feast SDK 미설치 환경: import만 스킵") 