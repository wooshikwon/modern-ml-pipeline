"""
Augmenter 컴포넌트 테스트

Blueprint 원칙 검증:
- 단일 Augmenter, 컨텍스트 주입 원칙
- 통합 데이터 어댑터 원칙
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.core.augmenter import Augmenter
from src.settings.settings import Settings


class TestAugmenter:
    """Augmenter 컴포넌트 테스트"""
    
    def test_augmenter_initialization(self, xgboost_settings: Settings):
        """Augmenter가 올바른 설정으로 초기화되는지 테스트"""
        augmenter = Augmenter(xgboost_settings)
        assert augmenter.settings == xgboost_settings
        assert augmenter.settings.model.name == "xgboost_x_learner"
    
    def test_augmenter_initialization_with_adapters(self, xgboost_settings: Settings):
        """Augmenter가 데이터 어댑터들과 함께 초기화되는지 테스트"""
        augmenter = Augmenter(xgboost_settings)
        
        # 배치 어댑터가 생성되었는지 확인
        assert hasattr(augmenter, 'batch_adapter')
        assert augmenter.batch_adapter is not None
        
        # Redis 어댑터가 선택적으로 생성되었는지 확인
        assert hasattr(augmenter, 'redis_adapter')
        # Redis가 없는 경우 None일 수 있음
    
    @patch('src.core.augmenter.Factory')
    def test_augmenter_batch_adapter_creation(self, mock_factory, xgboost_settings: Settings):
        """배치 어댑터 생성 로직 테스트"""
        mock_factory_instance = Mock()
        mock_batch_adapter = Mock()
        mock_factory_instance.create_data_adapter.return_value = mock_batch_adapter
        mock_factory.return_value = mock_factory_instance
        
        augmenter = Augmenter(xgboost_settings)
        
        # Factory가 올바르게 생성되었는지 확인
        mock_factory.assert_called_once_with(xgboost_settings)
        # 배치 어댑터가 생성되었는지 확인
        mock_factory_instance.create_data_adapter.assert_called_with("bq")
    
    def test_augment_batch_mode(self, xgboost_settings: Settings):
        """배치 모드 augment 테스트"""
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        
        with patch.object(Augmenter, '_augment_batch') as mock_augment_batch:
            mock_augment_batch.return_value = sample_data
            
            augmenter = Augmenter(xgboost_settings)
            result = augmenter.augment(sample_data, run_mode="batch")
            
            # 배치 모드 메서드가 호출되었는지 확인
            mock_augment_batch.assert_called_once_with(sample_data, {})
            assert result.equals(sample_data)
    
    def test_augment_realtime_mode(self, xgboost_settings: Settings):
        """실시간 모드 augment 테스트"""
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        
        feature_store_config = {"host": "localhost", "port": 6379}
        
        with patch.object(Augmenter, '_augment_realtime') as mock_augment_realtime:
            mock_augment_realtime.return_value = sample_data
            
            augmenter = Augmenter(xgboost_settings)
            result = augmenter.augment(
                sample_data, 
                run_mode="realtime",
                feature_store_config=feature_store_config
            )
            
            # 실시간 모드 메서드가 호출되었는지 확인
            mock_augment_realtime.assert_called_once_with(sample_data, feature_store_config)
            assert result.equals(sample_data)
    
    def test_augment_invalid_mode(self, xgboost_settings: Settings):
        """잘못된 모드에 대한 오류 처리 테스트"""
        augmenter = Augmenter(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        with pytest.raises(ValueError, match="Unknown run_mode"):
            augmenter.augment(sample_data, run_mode="invalid")
    
    @patch('src.core.augmenter.sql_utils.render_sql_template')
    def test_augment_batch_sql_rendering(self, mock_render_sql, xgboost_settings: Settings):
        """배치 모드에서 SQL 템플릿 렌더링 테스트"""
        # Mock 설정
        mock_render_sql.return_value = "SELECT * FROM table"
        
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        
        augmenter = Augmenter(xgboost_settings)
        
        # batch_adapter Mock 설정
        mock_batch_adapter = Mock()
        mock_batch_adapter.read.return_value = sample_data
        augmenter.batch_adapter = mock_batch_adapter
        
        result = augmenter._augment_batch(sample_data, {})
        
        # SQL 템플릿이 렌더링되었는지 확인
        mock_render_sql.assert_called_once()
        # 배치 어댑터가 호출되었는지 확인
        mock_batch_adapter.read.assert_called_once()
    
    def test_augment_realtime_redis_available(self, xgboost_settings: Settings):
        """실시간 모드에서 Redis 사용 가능한 경우 테스트"""
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c']
        })
        
        feature_store_config = {"host": "localhost", "port": 6379}
        
        augmenter = Augmenter(xgboost_settings)
        
        # Redis 어댑터 Mock 설정
        mock_redis_adapter = Mock()
        mock_redis_adapter.read.return_value = pd.DataFrame({
            'member_id': ['a', 'b', 'c'],
            'feature1': [1, 2, 3]
        })
        augmenter.redis_adapter = mock_redis_adapter
        
        result = augmenter._augment_realtime(sample_data, feature_store_config)
        
        # Redis 어댑터가 호출되었는지 확인
        mock_redis_adapter.read.assert_called_once()
        assert not result.empty
    
    def test_augment_realtime_redis_unavailable(self, xgboost_settings: Settings):
        """실시간 모드에서 Redis 사용 불가능한 경우 테스트"""
        sample_data = pd.DataFrame({
            'member_id': ['a', 'b', 'c']
        })
        
        feature_store_config = {"host": "localhost", "port": 6379}
        
        augmenter = Augmenter(xgboost_settings)
        augmenter.redis_adapter = None  # Redis 없음
        
        with patch('src.core.augmenter.logger') as mock_logger:
            result = augmenter._augment_realtime(sample_data, feature_store_config)
            
            # 경고 로그가 출력되었는지 확인
            mock_logger.warning.assert_called_once()
            # 원본 데이터가 그대로 반환되었는지 확인
            assert result.equals(sample_data)
    
    def test_blueprint_principle_single_augmenter_context_injection(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: 단일 Augmenter, 컨텍스트 주입"""
        augmenter = Augmenter(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        # 동일한 Augmenter 인스턴스가 다른 컨텍스트로 동작하는지 확인
        with patch.object(augmenter, '_augment_batch') as mock_batch:
            mock_batch.return_value = sample_data
            result_batch = augmenter.augment(sample_data, run_mode="batch")
            
        with patch.object(augmenter, '_augment_realtime') as mock_realtime:
            mock_realtime.return_value = sample_data
            result_realtime = augmenter.augment(
                sample_data, 
                run_mode="realtime",
                feature_store_config={}
            )
        
        # 각 모드별 메서드가 호출되었는지 확인
        mock_batch.assert_called_once()
        mock_realtime.assert_called_once()
        
        # 동일한 인스턴스가 사용되었는지 확인
        assert id(augmenter) == id(augmenter)
    
    def test_blueprint_principle_responsibility_separation(self, xgboost_settings: Settings):
        """Blueprint 원칙 검증: 책임 분리 (Augmenter는 더 이상 Factory를 실행 중에 생성하지 않음)"""
        augmenter = Augmenter(xgboost_settings)
        
        # Augmenter가 초기화 시에만 어댑터를 생성하고, 실행 중에는 생성하지 않는지 확인
        assert hasattr(augmenter, 'batch_adapter')
        assert hasattr(augmenter, 'redis_adapter')
        
        # 내부 메서드들이 Factory를 사용하지 않는지 확인
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        with patch('src.core.augmenter.Factory') as mock_factory:
            # 배치 모드 실행
            with patch.object(augmenter, 'batch_adapter') as mock_batch_adapter:
                mock_batch_adapter.read.return_value = sample_data
                augmenter._augment_batch(sample_data, {})
                
                # 실행 중에 Factory가 생성되지 않았는지 확인
                mock_factory.assert_not_called()
    
    def test_context_params_handling(self, xgboost_settings: Settings):
        """컨텍스트 파라미터 처리 테스트"""
        augmenter = Augmenter(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        context_params = {
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'additional_filter': 'active_users'
        }
        
        with patch.object(augmenter, '_augment_batch') as mock_batch:
            mock_batch.return_value = sample_data
            
            augmenter.augment(sample_data, run_mode="batch", context_params=context_params)
            
            # 컨텍스트 파라미터가 올바르게 전달되었는지 확인
            mock_batch.assert_called_once_with(sample_data, context_params)
    
    def test_empty_dataframe_handling(self, xgboost_settings: Settings):
        """빈 데이터프레임 처리 테스트"""
        augmenter = Augmenter(xgboost_settings)
        empty_data = pd.DataFrame()
        
        with patch.object(augmenter, '_augment_batch') as mock_batch:
            mock_batch.return_value = empty_data
            
            result = augmenter.augment(empty_data, run_mode="batch")
            
            # 빈 데이터프레임이 올바르게 처리되었는지 확인
            mock_batch.assert_called_once()
            assert result.empty
    
    def test_error_handling_in_batch_mode(self, xgboost_settings: Settings):
        """배치 모드에서 오류 처리 테스트"""
        augmenter = Augmenter(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        with patch.object(augmenter, '_augment_batch') as mock_batch:
            mock_batch.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                augmenter.augment(sample_data, run_mode="batch")
    
    def test_error_handling_in_realtime_mode(self, xgboost_settings: Settings):
        """실시간 모드에서 오류 처리 테스트"""
        augmenter = Augmenter(xgboost_settings)
        sample_data = pd.DataFrame({'member_id': ['a']})
        
        with patch.object(augmenter, '_augment_realtime') as mock_realtime:
            mock_realtime.side_effect = Exception("Redis connection failed")
            
            with pytest.raises(Exception, match="Redis connection failed"):
                augmenter.augment(sample_data, run_mode="realtime", feature_store_config={}) 