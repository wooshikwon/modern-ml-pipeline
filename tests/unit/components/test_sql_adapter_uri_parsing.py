"""
Unit tests for SqlAdapter URI parsing functionality.
Tests the automatic database engine selection based on URI scheme.

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Tuple

from src.components.adapter._modules.sql_adapter import SqlAdapter
from src.settings import Settings


class TestSqlAdapterUriParsing:
    """SqlAdapter URI 파싱 기능 테스트."""
    
    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Mock Settings 객체 생성."""
        settings = Mock(spec=Settings)
        settings.data_adapters = Mock()
        settings.data_adapters.adapters = {}
        return settings
    
    def create_sql_config(self, connection_uri: str, **kwargs) -> Dict[str, Any]:
        """SQL 어댑터 설정 생성 헬퍼."""
        config = {'connection_uri': connection_uri}
        config.update(kwargs)
        return Mock(config=config)
    
    def test_parse_bigquery_uri(self, mock_settings):
        """BigQuery URI 파싱 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'bigquery://my-project/my-dataset'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                # URI 파싱 메서드 직접 테스트
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'bigquery://my-project/my-dataset'
                )
                
                assert db_type == 'bigquery'
                assert processed_uri == 'bigquery://my-project/my-dataset'
                assert 'pool_pre_ping' in engine_kwargs
                assert engine_kwargs['pool_pre_ping'] is True
                assert engine_kwargs['pool_size'] == 5
    
    def test_parse_postgresql_uri(self, mock_settings):
        """PostgreSQL URI 파싱 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'postgresql://user:pass@localhost/dbname'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                # URI 파싱 메서드 직접 테스트
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'postgresql://user:pass@localhost/dbname'
                )
                
                assert db_type == 'postgresql'
                assert processed_uri == 'postgresql://user:pass@localhost/dbname'
                assert engine_kwargs['pool_size'] == 10
                assert engine_kwargs['max_overflow'] == 20
                assert engine_kwargs['pool_pre_ping'] is True
                assert 'connect_args' in engine_kwargs
                assert engine_kwargs['connect_args']['connect_timeout'] == 10
    
    def test_parse_postgres_uri_variant(self, mock_settings):
        """postgres:// 변형 URI 파싱 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'postgres://user:pass@localhost/dbname'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'postgres://user:pass@localhost/dbname'
                )
                
                assert db_type == 'postgresql'
    
    def test_parse_mysql_uri(self, mock_settings):
        """MySQL URI 파싱 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'mysql://user:pass@localhost/dbname'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'mysql://user:pass@localhost/dbname'
                )
                
                assert db_type == 'mysql'
                assert engine_kwargs['pool_size'] == 10
                assert engine_kwargs['pool_recycle'] == 3600
                assert engine_kwargs['pool_pre_ping'] is True
    
    def test_parse_sqlite_uri(self, mock_settings):
        """SQLite URI 파싱 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'sqlite:///path/to/database.db'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'sqlite:///path/to/database.db'
                )
                
                assert db_type == 'sqlite'
                assert 'poolclass' in engine_kwargs
                assert 'connect_args' in engine_kwargs
                assert engine_kwargs['connect_args']['check_same_thread'] is False
    
    def test_parse_unknown_uri(self, mock_settings):
        """알 수 없는 URI 스키마 처리 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'oracle://user:pass@localhost/dbname'
        )
        
        with patch('sqlalchemy.create_engine') as mock_create_engine:
            with patch.object(SqlAdapter, '_create_engine') as mock_create:
                mock_create.return_value = MagicMock()
                adapter = SqlAdapter(mock_settings)
                
                db_type, processed_uri, engine_kwargs = adapter._parse_connection_uri(
                    'oracle://user:pass@localhost/dbname'
                )
                
                assert db_type == 'generic'
                assert processed_uri == 'oracle://user:pass@localhost/dbname'
                assert engine_kwargs == {}  # 기본 설정
    
    @patch('sqlalchemy.create_engine')
    def test_bigquery_with_credentials(self, mock_create_engine, mock_settings):
        """BigQuery 인증 파일 설정 테스트."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'bigquery://my-project/my-dataset',
            credentials_path='/path/to/credentials.json'
        )
        
        # Mock 엔진과 연결
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        with patch('os.environ', {}) as mock_environ:
            adapter = SqlAdapter(mock_settings)
            
            # 환경 변수가 설정되었는지 확인
            assert 'GOOGLE_APPLICATION_CREDENTIALS' in mock_environ
            assert mock_environ['GOOGLE_APPLICATION_CREDENTIALS'] == '/path/to/credentials.json'
    
    @patch('sqlalchemy.create_engine')
    def test_connection_test_on_creation(self, mock_create_engine, mock_settings):
        """엔진 생성 시 연결 테스트 수행 확인."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'postgresql://user:pass@localhost/dbname'
        )
        
        # Mock 엔진과 연결
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        adapter = SqlAdapter(mock_settings)
        
        # 연결 테스트가 수행되었는지 확인
        mock_engine.connect.assert_called_once()
    
    def test_missing_sql_adapter_config(self, mock_settings):
        """SQL 어댑터 설정이 없을 때 에러 처리."""
        mock_settings.data_adapters.adapters = {}  # SQL 설정 없음
        
        with pytest.raises(ValueError, match="SQL 어댑터 설정이 누락되었습니다"):
            SqlAdapter(mock_settings)
    
    @patch('sqlalchemy.create_engine')
    def test_engine_creation_failure(self, mock_create_engine, mock_settings):
        """엔진 생성 실패 시 에러 처리."""
        mock_settings.data_adapters.adapters['sql'] = self.create_sql_config(
            'postgresql://user:pass@localhost/dbname'
        )
        
        # 엔진 생성 시 예외 발생
        mock_create_engine.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            SqlAdapter(mock_settings)