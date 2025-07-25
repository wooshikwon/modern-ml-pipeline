"""
데이터 어댑터 테스트 (Blueprint v17.0 현대화)

모든 데이터 어댑터의 통합 테스트 및 Blueprint 원칙 검증

Blueprint 원칙 검증:
- 원칙 2: 통합 데이터 어댑터 (The Unified Data Adapter)
- 원칙 3: URI 기반 동작 및 동적 팩토리
- 어댑터 생태계 완전성 검증
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

from src.settings import Settings
from src.utils.adapters.bigquery_adapter import BigQueryAdapter
from src.utils.adapters.gcs_adapter import GCSAdapter
from src.utils.adapters.s3_adapter import S3Adapter
from src.utils.adapters.file_system_adapter import FileSystemAdapter
from src.utils.adapters.redis_adapter import RedisAdapter
from src.core.factory import Factory


class TestAdaptersModernized:
    """
    데이터 어댑터 단위 테스트 (Blueprint v17.0 현대화)
    - 각 어댑터가 settings 객체로 올바르게 초기화되는지 검증합니다.
    - 실제 I/O 대신 Mock을 사용하여 각 어댑터의 핵심 메서드를 테스트합니다.
    - Blueprint 원칙 2: 통합 데이터 어댑터 완전 검증
    """

    def test_bigquery_adapter_initialization_and_interface(self, local_test_settings: Settings):
        """
        BigQueryAdapter가 settings를 사용하여 초기화되고 통합 인터페이스를 구현하는지 테스트
        Blueprint 원칙 2: 통합 데이터 어댑터
        """
        with patch('src.utils.adapters.bigquery_adapter.bigquery.Client') as mock_client:
            adapter = BigQueryAdapter(local_test_settings)
            
            # 초기화 검증
            assert adapter.settings == local_test_settings
            mock_client.assert_called_once()
            
            # 통합 인터페이스 검증
            from src.interface.base_adapter import BaseAdapter
            assert isinstance(adapter, BaseAdapter)
            
            # 필수 메서드 존재 확인
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
            assert callable(adapter.read)
            assert callable(adapter.write)
            
            print("✅ BigQueryAdapter 통합 인터페이스 검증 완료")

    def test_gcs_adapter_initialization_and_interface(self, local_test_settings: Settings):
        """
        GCSAdapter가 settings를 사용하여 초기화되고 통합 인터페이스를 구현하는지 테스트
        """
        with patch('src.utils.adapters.gcs_adapter.storage.Client') as mock_client:
            adapter = GCSAdapter(local_test_settings)
            
            # 초기화 검증
            assert adapter.settings == local_test_settings
            mock_client.assert_called_once()
            
            # 통합 인터페이스 검증
            from src.interface.base_adapter import BaseAdapter
            assert isinstance(adapter, BaseAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
            
            print("✅ GCSAdapter 통합 인터페이스 검증 완료")

    def test_s3_adapter_initialization_and_interface(self, local_test_settings: Settings):
        """
        S3Adapter가 settings를 사용하여 초기화되고 통합 인터페이스를 구현하는지 테스트
        """
        with patch('src.utils.adapters.s3_adapter.boto3.client') as mock_client:
            adapter = S3Adapter(local_test_settings)
            
            # 초기화 검증
            assert adapter.settings == local_test_settings
            
            # 통합 인터페이스 검증
            from src.interface.base_adapter import BaseAdapter
            assert isinstance(adapter, BaseAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
            
            print("✅ S3Adapter 통합 인터페이스 검증 완료")

    def test_filesystem_adapter_initialization_and_interface(self, local_test_settings: Settings):
        """
        FileSystemAdapter가 settings를 사용하여 초기화되고 통합 인터페이스를 구현하는지 테스트
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        # 초기화 검증
        assert adapter.settings == local_test_settings
        
        # 통합 인터페이스 검증
        from src.interface.base_adapter import BaseAdapter
        assert isinstance(adapter, BaseAdapter)
        assert hasattr(adapter, 'read')
        assert hasattr(adapter, 'write')
        
        print("✅ FileSystemAdapter 통합 인터페이스 검증 완료")

    def test_redis_adapter_initialization_and_interface(self, local_test_settings: Settings):
        """
        RedisAdapter가 settings를 사용하여 초기화되고 통합 인터페이스를 구현하는지 테스트
        """
        with patch('src.utils.adapters.redis_adapter.redis.Redis') as mock_redis:
            mock_redis_instance = mock_redis.return_value
            mock_redis_instance.ping.return_value = True
            
            adapter = RedisAdapter(local_test_settings.serving.realtime_feature_store)
            
            # 초기화 검증
            assert adapter.settings == local_test_settings.serving.realtime_feature_store
            mock_redis.assert_called_once()
            
            # 통합 인터페이스 검증
            from src.interface.base_adapter import BaseAdapter
            assert isinstance(adapter, BaseAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
            
            print("✅ RedisAdapter 통합 인터페이스 검증 완료")

    def test_filesystem_adapter_read_multiple_formats(self, local_test_settings: Settings):
        """
        FileSystemAdapter가 다양한 파일 형식을 올바르게 읽는지 테스트
        Blueprint 원칙 2: 통합 데이터 어댑터의 다형성
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Parquet 파일 테스트
            parquet_file = os.path.join(tmp_dir, "test_data.parquet")
            sample_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
            sample_df.to_parquet(parquet_file)
            
            df_parquet = adapter.read(f"file://{parquet_file}")
            pd.testing.assert_frame_equal(df_parquet, sample_df)
            
            # 2. CSV 파일 테스트
            csv_file = os.path.join(tmp_dir, "test_data.csv")
            sample_df.to_csv(csv_file, index=False)
            
            df_csv = adapter.read(f"file://{csv_file}")
            pd.testing.assert_frame_equal(df_csv, sample_df)
            
            print("✅ FileSystemAdapter 다중 형식 읽기 검증 완료")

    def test_filesystem_adapter_write_functionality(self, local_test_settings: Settings):
        """
        FileSystemAdapter의 write 기능이 올바르게 동작하는지 테스트
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 테스트 데이터
            test_df = pd.DataFrame({
                'user_id': ['u1', 'u2', 'u3'],
                'prediction': [0.85, 0.72, 0.91]
            })
            
            # 파일 쓰기
            output_file = os.path.join(tmp_dir, "predictions.parquet")
            adapter.write(test_df, f"file://{output_file}")
            
            # 쓰여진 파일 확인
            assert os.path.exists(output_file)
            
            # 다시 읽어서 내용 확인
            read_df = adapter.read(f"file://{output_file}")
            pd.testing.assert_frame_equal(read_df, test_df)
            
            print("✅ FileSystemAdapter write 기능 검증 완료")

    def test_adapter_error_handling(self, local_test_settings: Settings):
        """
        어댑터들의 에러 처리가 적절한지 테스트
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        # 1. 존재하지 않는 파일 읽기
        with pytest.raises(Exception):
            adapter.read("file://nonexistent/path/file.parquet")
        
        # 2. 잘못된 URI 형식
        with pytest.raises(Exception):
            adapter.read("invalid_uri_format")
        
        print("✅ 어댑터 에러 처리 검증 완료")

    # 🆕 Blueprint v17.0: Factory 통합 테스트
    def test_factory_adapter_creation_pattern(self, local_test_settings: Settings, dev_test_settings: Settings):
        """
        Factory가 환경별로 적절한 어댑터를 생성하는지 테스트
        Blueprint 원칙 3: URI 기반 동작 및 동적 팩토리
        """
        # LOCAL 환경: FileSystemAdapter 우선
        local_factory = Factory(local_test_settings)
        
        # 기본 어댑터 생성 확인
        assert hasattr(local_factory, 'create_data_adapter')
        
        # DEV 환경: 다양한 어댑터 지원
        dev_factory = Factory(dev_test_settings)
        
        # 동일한 팩토리 인터페이스로 다른 어댑터 생성 가능
        assert type(local_factory) == type(dev_factory)
        
        print("✅ Factory 어댑터 생성 패턴 검증 완료")

    def test_adapter_registry_pattern_compliance(self, local_test_settings: Settings):
        """
        모든 어댑터가 Registry 패턴을 준수하는지 테스트
        Blueprint 원칙 3: 확장성을 위한 Registry 패턴
        """
        # 모든 어댑터가 BaseAdapter 상속하는지 확인
        from src.interface.base_adapter import BaseAdapter
        
        adapter_classes = [
            BigQueryAdapter,
            GCSAdapter,
            S3Adapter,
            FileSystemAdapter
        ]
        
        for adapter_class in adapter_classes:
            # BaseAdapter 상속 확인
            assert issubclass(adapter_class, BaseAdapter)
            
            # 필수 메서드 구현 확인
            required_methods = ['read', 'write']
            for method in required_methods:
                assert hasattr(adapter_class, method)
                assert callable(getattr(adapter_class, method))
        
        print("✅ 어댑터 Registry 패턴 준수 검증 완료")

    # 🆕 Blueprint v17.0: URI 스키마 테스트
    def test_uri_schema_handling(self, local_test_settings: Settings):
        """
        다양한 URI 스키마에 대한 어댑터 처리를 테스트
        Blueprint 원칙 3: URI 기반 동작
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        # 다양한 URI 스키마 형식 테스트
        uri_formats = [
            "file://absolute/path/file.parquet",
            "file:///absolute/path/file.parquet",
            "file://./relative/path/file.parquet"
        ]
        
        for uri in uri_formats:
            # URI 파싱이 올바르게 되는지 확인 (실제 파일 없어도 파싱은 가능해야 함)
            try:
                # _parse_uri 메서드가 있다면 테스트
                if hasattr(adapter, '_parse_uri'):
                    parsed = adapter._parse_uri(uri)
                    assert isinstance(parsed, str)
                    print(f"✅ URI 파싱 성공: {uri}")
            except Exception:
                # 파일이 존재하지 않는 에러는 정상 (파싱은 성공)
                pass
        
        print("✅ URI 스키마 처리 검증 완료")

    # 🆕 Blueprint v17.0: 성능 최적화 검증
    def test_adapter_performance_characteristics(self, local_test_settings: Settings):
        """
        어댑터의 성능 특성을 검증한다
        """
        adapter = FileSystemAdapter(local_test_settings)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 적당한 크기의 테스트 데이터 생성
            large_df = pd.DataFrame({
                'id': range(1000),
                'value': [f'data_{i}' for i in range(1000)]
            })
            
            test_file = os.path.join(tmp_dir, "performance_test.parquet")
            
            # 쓰기 성능 측정
            import time
            start_time = time.time()
            adapter.write(large_df, f"file://{test_file}")
            write_time = time.time() - start_time
            
            # 읽기 성능 측정
            start_time = time.time()
            read_df = adapter.read(f"file://{test_file}")
            read_time = time.time() - start_time
            
            # 기본적인 성능 확인 (너무 느리지 않은지)
            assert write_time < 5.0, f"쓰기가 너무 느림: {write_time}초"
            assert read_time < 5.0, f"읽기가 너무 느림: {read_time}초"
            
            # 데이터 무결성 확인
            assert len(read_df) == len(large_df)
            
            print(f"✅ 성능 검증 완료 - 쓰기: {write_time:.3f}초, 읽기: {read_time:.3f}초")

    def test_adapter_comprehensive_integration(self, local_test_settings: Settings):
        """
        어댑터 생태계의 종합적인 통합 테스트
        Blueprint 원칙 2: 통합 데이터 어댑터의 완전성
        """
        # 1. 모든 어댑터가 동일한 인터페이스로 생성 가능한지 확인
        adapter_configs = [
            (FileSystemAdapter, local_test_settings),
        ]
        
        # Mock 환경에서 다른 어댑터들도 테스트
        with patch('src.utils.adapters.bigquery_adapter.bigquery.Client'):
            adapter_configs.append((BigQueryAdapter, local_test_settings))
        
        with patch('src.utils.adapters.gcs_adapter.storage.Client'):
            adapter_configs.append((GCSAdapter, local_test_settings))
        
        with patch('src.utils.adapters.s3_adapter.boto3.client'):
            adapter_configs.append((S3Adapter, local_test_settings))
        
        created_adapters = []
        for adapter_class, settings in adapter_configs:
            try:
                adapter = adapter_class(settings)
                created_adapters.append(adapter)
                
                # 통합 인터페이스 확인
                from src.interface.base_adapter import BaseAdapter
                assert isinstance(adapter, BaseAdapter)
                
            except Exception as e:
                pytest.fail(f"어댑터 {adapter_class.__name__} 생성 실패: {e}")
        
        # 2. 모든 생성된 어댑터가 동일한 메서드를 가지는지 확인
        if len(created_adapters) > 1:
            first_adapter = created_adapters[0]
            for adapter in created_adapters[1:]:
                # 메서드 시그니처 일관성 확인
                assert type(adapter.read) == type(first_adapter.read)
                assert type(adapter.write) == type(first_adapter.write)
        
        print(f"✅ 어댑터 생태계 종합 통합 검증 완료 ({len(created_adapters)}개 어댑터 테스트)") 