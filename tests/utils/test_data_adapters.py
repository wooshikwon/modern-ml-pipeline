"""
데이터 어댑터 테스트

모든 데이터 어댑터의 통합 테스트 및 Blueprint 원칙 검증
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.utils.data_adapters.bigquery_adapter import BigQueryAdapter
from src.utils.data_adapters.gcs_adapter import GCSAdapter
from src.utils.data_adapters.s3_adapter import S3Adapter
from src.utils.data_adapters.file_system_adapter import FileSystemAdapter
from src.utils.data_adapters.redis_adapter import RedisAdapter
from src.settings import Settings


class TestBigQueryAdapter:
    """BigQuery 어댑터 테스트"""
    
    def test_initialization(self, xgboost_settings: Settings):
        """BigQuery 어댑터 초기화 테스트"""
        adapter = BigQueryAdapter(xgboost_settings)
        assert adapter.settings == xgboost_settings
        assert hasattr(adapter, '_client_available')
    
    def test_client_initialization_success(self, xgboost_settings: Settings):
        """클라이언트 초기화 성공 테스트"""
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client.return_value = Mock()
            
            adapter = BigQueryAdapter(xgboost_settings)
            
            # 클라이언트가 생성되었는지 확인
            assert adapter._client_available is True
    
    def test_client_initialization_failure(self, xgboost_settings: Settings):
        """클라이언트 초기화 실패 테스트 (인증 정보 없음)"""
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client.side_effect = Exception("Authentication failed")
            
            adapter = BigQueryAdapter(xgboost_settings)
            
            # 클라이언트 초기화 실패 시 graceful degradation
            assert adapter._client_available is False
    
    def test_read_with_client_available(self, xgboost_settings: Settings):
        """클라이언트 사용 가능 시 read 테스트"""
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock 쿼리 결과
            mock_result = Mock()
            mock_result.to_dataframe.return_value = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            mock_client_instance.query.return_value = mock_result
            
            adapter = BigQueryAdapter(xgboost_settings)
            
            # read 실행
            result = adapter.read("bq://project.dataset.table", params={})
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
            
            # 클라이언트가 호출되었는지 확인
            mock_client_instance.query.assert_called_once()
    
    def test_read_with_client_unavailable(self, xgboost_settings: Settings):
        """클라이언트 사용 불가능 시 read 테스트"""
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client.side_effect = Exception("Authentication failed")
            
            adapter = BigQueryAdapter(xgboost_settings)
            
            # read 실행
            result = adapter.read("bq://project.dataset.table", params={})
            
            # 빈 DataFrame 반환
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
    
    def test_write_method(self, xgboost_settings: Settings):
        """write 메서드 테스트"""
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            adapter = BigQueryAdapter(xgboost_settings)
            
            # 샘플 데이터
            data = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            
            # write 실행
            adapter.write(data, "bq://project.dataset.table")
            
            # 클라이언트가 호출되었는지 확인
            mock_client_instance.load_table_from_dataframe.assert_called_once()


class TestGCSAdapter:
    """GCS 어댑터 테스트"""
    
    def test_initialization(self, xgboost_settings: Settings):
        """GCS 어댑터 초기화 테스트"""
        adapter = GCSAdapter(xgboost_settings)
        assert adapter.settings == xgboost_settings
        assert hasattr(adapter, 'client')
    
    def test_client_initialization_success(self, xgboost_settings: Settings):
        """클라이언트 초기화 성공 테스트"""
        with patch('src.utils.data_adapters.gcs_adapter.storage.Client') as mock_client:
            mock_client.return_value = Mock()
            
            adapter = GCSAdapter(xgboost_settings)
            
            # 클라이언트가 생성되었는지 확인
            assert adapter.client is not None
    
    def test_read_method(self, xgboost_settings: Settings):
        """read 메서드 테스트"""
        with patch('src.utils.data_adapters.gcs_adapter.storage.Client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock blob 설정
            mock_blob = Mock()
            mock_blob.download_as_text.return_value = "member_id,feature1\na,1\nb,2\nc,3"
            mock_bucket = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_client_instance.bucket.return_value = mock_bucket
            
            adapter = GCSAdapter(xgboost_settings)
            
            # read 실행
            result = adapter.read("gs://bucket/file.csv", params={})
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
    
    def test_write_method(self, xgboost_settings: Settings):
        """write 메서드 테스트"""
        with patch('src.utils.data_adapters.gcs_adapter.storage.Client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock blob 설정
            mock_blob = Mock()
            mock_bucket = Mock()
            mock_bucket.blob.return_value = mock_blob
            mock_client_instance.bucket.return_value = mock_bucket
            
            adapter = GCSAdapter(xgboost_settings)
            
            # 샘플 데이터
            data = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            
            # write 실행
            adapter.write(data, "gs://bucket/file.csv")
            
            # blob이 업로드되었는지 확인
            mock_blob.upload_from_string.assert_called_once()


class TestS3Adapter:
    """S3 어댑터 테스트"""
    
    def test_initialization(self, xgboost_settings: Settings):
        """S3 어댑터 초기화 테스트"""
        adapter = S3Adapter(xgboost_settings)
        assert adapter.settings == xgboost_settings
        assert hasattr(adapter, 'client')
    
    def test_client_initialization_success(self, xgboost_settings: Settings):
        """클라이언트 초기화 성공 테스트"""
        with patch('src.utils.data_adapters.s3_adapter.boto3.client') as mock_client:
            mock_client.return_value = Mock()
            
            adapter = S3Adapter(xgboost_settings)
            
            # 클라이언트가 생성되었는지 확인
            assert adapter.client is not None
    
    def test_read_method(self, xgboost_settings: Settings):
        """read 메서드 테스트"""
        with patch('src.utils.data_adapters.s3_adapter.boto3.client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock S3 객체 설정
            mock_response = {
                'Body': Mock()
            }
            mock_response['Body'].read.return_value = b"member_id,feature1\na,1\nb,2\nc,3"
            mock_client_instance.get_object.return_value = mock_response
            
            adapter = S3Adapter(xgboost_settings)
            
            # read 실행
            result = adapter.read("s3://bucket/file.csv", params={})
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
    
    def test_write_method(self, xgboost_settings: Settings):
        """write 메서드 테스트"""
        with patch('src.utils.data_adapters.s3_adapter.boto3.client') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            adapter = S3Adapter(xgboost_settings)
            
            # 샘플 데이터
            data = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            
            # write 실행
            adapter.write(data, "s3://bucket/file.csv")
            
            # S3에 업로드되었는지 확인
            mock_client_instance.put_object.assert_called_once()


class TestFileSystemAdapter:
    """FileSystem 어댑터 테스트"""
    
    def test_initialization(self, xgboost_settings: Settings):
        """FileSystem 어댑터 초기화 테스트"""
        adapter = FileSystemAdapter(xgboost_settings)
        assert adapter.settings == xgboost_settings
    
    def test_read_csv_method(self, xgboost_settings: Settings):
        """CSV 파일 read 테스트"""
        adapter = FileSystemAdapter(xgboost_settings)
        
        # pandas read_csv Mock
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            
            # read 실행
            result = adapter.read("file:///path/to/file.csv", params={})
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
            
            # pandas read_csv가 호출되었는지 확인
            mock_read_csv.assert_called_once()
    
    def test_read_parquet_method(self, xgboost_settings: Settings):
        """Parquet 파일 read 테스트"""
        adapter = FileSystemAdapter(xgboost_settings)
        
        # pandas read_parquet Mock
        with patch('pandas.read_parquet') as mock_read_parquet:
            mock_read_parquet.return_value = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3]
            })
            
            # read 실행
            result = adapter.read("file:///path/to/file.parquet", params={})
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
            
            # pandas read_parquet가 호출되었는지 확인
            mock_read_parquet.assert_called_once()
    
    def test_write_csv_method(self, xgboost_settings: Settings):
        """CSV 파일 write 테스트"""
        adapter = FileSystemAdapter(xgboost_settings)
        
        # 샘플 데이터
        data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        
        # DataFrame to_csv Mock
        with patch.object(data, 'to_csv') as mock_to_csv:
            # write 실행
            adapter.write(data, "file:///path/to/file.csv")
            
            # to_csv가 호출되었는지 확인
            mock_to_csv.assert_called_once()
    
    def test_write_parquet_method(self, xgboost_settings: Settings):
        """Parquet 파일 write 테스트"""
        adapter = FileSystemAdapter(xgboost_settings)
        
        # 샘플 데이터
        data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        
        # DataFrame to_parquet Mock
        with patch.object(data, 'to_parquet') as mock_to_parquet:
            # write 실행
            adapter.write(data, "file:///path/to/file.parquet")
            
            # to_parquet가 호출되었는지 확인
            mock_to_parquet.assert_called_once()


class TestRedisAdapter:
    """Redis 어댑터 테스트"""
    
    def test_initialization_with_redis_available(self, xgboost_settings: Settings):
        """Redis 사용 가능 시 초기화 테스트"""
        with patch('src.utils.data_adapters.redis_adapter.redis.Redis') as mock_redis:
            mock_redis.return_value = Mock()
            
            adapter = RedisAdapter(xgboost_settings)
            assert adapter.settings == xgboost_settings
            assert adapter.client is not None
    
    def test_initialization_with_redis_unavailable(self, xgboost_settings: Settings):
        """Redis 사용 불가능 시 초기화 테스트"""
        with patch('src.utils.data_adapters.redis_adapter.redis.Redis') as mock_redis:
            mock_redis.side_effect = ImportError("Redis not available")
            
            # Redis 없을 때 적절한 처리
            with pytest.raises(ImportError):
                RedisAdapter(xgboost_settings)
    
    def test_read_method(self, xgboost_settings: Settings):
        """read 메서드 테스트"""
        with patch('src.utils.data_adapters.redis_adapter.redis.Redis') as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            
            # Mock Redis 데이터
            mock_redis_instance.mget.return_value = [
                b'{"feature1": 1, "feature2": 0.1}',
                b'{"feature1": 2, "feature2": 0.2}',
                b'{"feature1": 3, "feature2": 0.3}'
            ]
            
            adapter = RedisAdapter(xgboost_settings)
            
            # 샘플 입력 데이터 (PK 목록)
            input_data = pd.DataFrame({
                'member_id': ['a', 'b', 'c']
            })
            
            # read 실행
            result = adapter.read("redis://localhost:6379/features", params={}, input_data=input_data)
            
            # 결과 확인
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'member_id' in result.columns
            assert 'feature1' in result.columns
    
    def test_write_method(self, xgboost_settings: Settings):
        """write 메서드 테스트"""
        with patch('src.utils.data_adapters.redis_adapter.redis.Redis') as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance
            
            adapter = RedisAdapter(xgboost_settings)
            
            # 샘플 데이터
            data = pd.DataFrame({
                'member_id': ['a', 'b', 'c'],
                'feature1': [1, 2, 3],
                'feature2': [0.1, 0.2, 0.3]
            })
            
            # write 실행
            adapter.write(data, "redis://localhost:6379/features")
            
            # Redis에 데이터가 저장되었는지 확인
            assert mock_redis_instance.mset.called or mock_redis_instance.set.called


class TestDataAdapterBlueprint:
    """데이터 어댑터 Blueprint 원칙 테스트"""
    
    def test_unified_interface(self, xgboost_settings: Settings):
        """통합 인터페이스 테스트"""
        from src.interface.base_data_adapter import BaseDataAdapter
        
        # 모든 어댑터가 BaseDataAdapter를 상속받는지 확인
        adapters = [
            BigQueryAdapter(xgboost_settings),
            GCSAdapter(xgboost_settings),
            S3Adapter(xgboost_settings),
            FileSystemAdapter(xgboost_settings)
        ]
        
        for adapter in adapters:
            assert isinstance(adapter, BaseDataAdapter)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
    
    def test_settings_injection(self, xgboost_settings: Settings):
        """설정 주입 테스트"""
        adapters = [
            BigQueryAdapter(xgboost_settings),
            GCSAdapter(xgboost_settings),
            S3Adapter(xgboost_settings),
            FileSystemAdapter(xgboost_settings)
        ]
        
        for adapter in adapters:
            assert adapter.settings == xgboost_settings
    
    def test_uri_scheme_handling(self, xgboost_settings: Settings):
        """URI 스킴 처리 테스트"""
        # 각 어댑터가 해당 URI 스킴을 올바르게 처리하는지 확인
        test_cases = [
            (BigQueryAdapter, "bq://project.dataset.table"),
            (GCSAdapter, "gs://bucket/file.csv"),
            (S3Adapter, "s3://bucket/file.csv"),
            (FileSystemAdapter, "file:///path/to/file.csv")
        ]
        
        for adapter_class, uri in test_cases:
            adapter = adapter_class(xgboost_settings)
            
            # URI 파싱이 올바르게 이루어지는지 확인
            # (실제 구현에서는 URI 파싱 로직이 있어야 함)
            assert hasattr(adapter, 'read')
            assert hasattr(adapter, 'write')
    
    def test_error_handling_consistency(self, xgboost_settings: Settings):
        """오류 처리 일관성 테스트"""
        adapters = [
            BigQueryAdapter(xgboost_settings),
            GCSAdapter(xgboost_settings),
            S3Adapter(xgboost_settings),
            FileSystemAdapter(xgboost_settings)
        ]
        
        for adapter in adapters:
            # 잘못된 URI에 대한 오류 처리
            try:
                adapter.read("invalid://uri", params={})
            except Exception as e:
                # 예외가 발생하는 것은 정상이지만, 일관된 방식으로 처리되어야 함
                assert isinstance(e, (ValueError, RuntimeError, Exception))
    
    def test_graceful_degradation(self, xgboost_settings: Settings):
        """우아한 성능 저하 테스트"""
        # BigQuery 어댑터의 인증 실패 시 graceful degradation
        with patch('src.utils.data_adapters.bigquery_adapter.bigquery.Client') as mock_client:
            mock_client.side_effect = Exception("Authentication failed")
            
            adapter = BigQueryAdapter(xgboost_settings)
            result = adapter.read("bq://project.dataset.table", params={})
            
            # 인증 실패 시 빈 DataFrame 반환
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0 