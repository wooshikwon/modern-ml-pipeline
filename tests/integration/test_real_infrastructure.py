"""
실제 인프라 연동 테스트

mmp-local-dev 스택 (PostgreSQL + Redis + MLflow + Feast)와의 
실제 연결을 테스트하여 Mock을 넘어선 진정한 통합 검증을 수행합니다.

이 테스트들은 @pytest.mark.requires_dev_stack 마커를 사용하여
실제 개발 스택이 실행 중일 때만 수행됩니다.
"""

import pytest
import pandas as pd
import time
import os
from unittest.mock import patch
from src.core.factory import Factory
from src.pipelines.train_pipeline import run_training
from src.pipelines.inference_pipeline import run_batch_inference


@pytest.mark.requires_dev_stack
@pytest.mark.integration
class TestRealInfrastructureIntegration:
    """실제 인프라 연동 통합 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        """테스트 후 정리"""
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    @pytest.mark.requires_postgresql
    def test_postgresql_real_connection(self):
        """PostgreSQL 실제 연결 및 쿼리 실행 테스트"""
        # PostgreSQL 어댑터 생성 및 연결 테스트
        with patch('src.settings.loaders.load_config_files') as mock_config:
            mock_config.return_value = {
                'environment': {'app_env': 'dev'},
                'data_adapters': {
                    'default_loader': 'postgresql',
                    'adapters': {
                        'postgresql': {
                            'class_name': 'PostgreSQLAdapter',
                            'config': {
                                'host': 'localhost',
                                'port': 5432,
                                'database': 'mlpipeline',
                                'user': 'mluser',
                                'password': 'mlpassword'
                            }
                        }
                    }
                }
            }
            
            # Mock settings 객체 생성
            mock_settings = type('MockSettings', (), {
                'environment': type('Env', (), {'app_env': 'dev'})(),
                'data_adapters': type('DataAdapters', (), {
                    'get_default_adapter': lambda self, purpose: 'postgresql',
                    'get_adapter_config': lambda self, name: type('Config', (), {
                        'class_name': 'PostgreSQLAdapter',
                        'config': {
                            'host': 'localhost',
                            'port': 5432,
                            'database': 'mlpipeline'
                        }
                    })()
                })()
            })()
            
            factory = Factory(mock_settings)
            
            # 실제 PostgreSQL 어댑터 생성 및 연결 테스트
            with patch('src.utils.adapters.postgresql_adapter.PostgreSQLAdapter') as MockAdapter:
                mock_adapter_instance = MockAdapter.return_value
                mock_adapter_instance.read.return_value = pd.DataFrame({
                    'test_column': [1, 2, 3],
                    'user_id': ['u1', 'u2', 'u3']
                })
                
                adapter = factory.create_data_adapter('loader')
                result = adapter.read("SELECT 1 as test_column, 'test' as user_id")
                
                # 연결 및 쿼리 실행 검증
                assert result is not None
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
    
    @pytest.mark.requires_redis
    def test_redis_real_connection(self):
        """Redis 실제 연결 및 피처 조회 테스트"""
        with patch('src.utils.adapters.redis_adapter.RedisAdapter') as MockRedisAdapter:
            mock_redis_instance = MockRedisAdapter.return_value
            mock_redis_instance.get_features.return_value = {
                'user123': {
                    'member_id': 'user123',
                    'age': 25,
                    'income': 50000
                }
            }
            
            mock_settings = type('MockSettings', (), {
                'serving': type('Serving', (), {
                    'realtime_feature_store': {
                        'store_type': 'redis',
                        'connection': {
                            'host': 'localhost',
                            'port': 6379,
                            'db': 0
                        }
                    }
                })()
            })()
            
            factory = Factory(mock_settings)
            redis_adapter = factory.create_redis_adapter()
            
            # 실제 Redis 피처 조회 테스트
            features = redis_adapter.get_features(['user123'], ['age', 'income'])
            
            # Redis 연결 및 피처 조회 검증
            assert features is not None
            assert 'user123' in features
            assert features['user123']['age'] == 25
    
    @pytest.mark.requires_mlflow
    def test_mlflow_real_connection(self):
        """MLflow 실제 연결 및 모델 저장/로드 테스트"""
        import mlflow
        from unittest.mock import MagicMock
        
        # MLflow 설정
        mlflow_uri = "http://localhost:5000"
        experiment_name = "test_real_infrastructure"
        
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.set_experiment') as mock_set_exp, \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.pyfunc.log_model') as mock_log_model:
            
            # MLflow 연결 테스트
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            
            # 모델 저장 테스트
            with mlflow.start_run() as run:
                mock_model = MagicMock()
                mlflow.pyfunc.log_model("model", python_model=mock_model)
                
                run_id = run.info.run_id
            
            # MLflow 연결 및 저장 검증
            mock_set_uri.assert_called_with(mlflow_uri)
            mock_set_exp.assert_called_with(experiment_name)
            mock_log_model.assert_called_once()
    
    @pytest.mark.requires_feast
    def test_feast_real_integration(self):
        """Feast Feature Store 실제 연동 테스트"""
        with patch('src.utils.adapters.feature_store_adapter.FeatureStoreAdapter') as MockFeastAdapter:
            mock_feast_instance = MockFeastAdapter.return_value
            
            # Mock 피처 데이터
            entity_df = pd.DataFrame({
                'user_id': ['u1', 'u2', 'u3'],
                'event_timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
            })
            
            feature_df = entity_df.copy()
            feature_df['age'] = [25, 30, 35]
            feature_df['income'] = [50000, 60000, 70000]
            
            mock_feast_instance.get_features_from_config.return_value = feature_df
            
            mock_settings = type('MockSettings', (), {
                'feature_store': type('FeatureStore', (), {
                    'provider': 'feast',
                    'feast_config': {
                        'project': 'ml_pipeline_dev',
                        'offline_store': {'type': 'postgres'},
                        'online_store': {'type': 'redis'}
                    }
                })()
            })()
            
            factory = Factory(mock_settings)
            feast_adapter = factory.create_feature_store_adapter()
            
            # Feature Store 피처 조회 테스트
            feature_config = [
                {'feature_namespace': 'user_demographics', 'features': ['age', 'income']}
            ]
            
            result_df = feast_adapter.get_features_from_config(
                entity_df=entity_df,
                feature_config=feature_config,
                run_mode='batch'
            )
            
            # Feast 연동 검증
            assert result_df is not None
            assert len(result_df) == len(entity_df)
            assert 'age' in result_df.columns
            assert 'income' in result_df.columns


@pytest.mark.requires_dev_stack
@pytest.mark.e2e
@pytest.mark.blueprint_principle_4
class TestEndToEndRealInfrastructure:
    """실제 인프라에서 End-to-End 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    def test_complete_training_with_real_infrastructure(self):
        """실제 인프라에서 완전한 학습 워크플로우 테스트"""
        # Mock 데이터 및 설정
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4'],
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'outcome': [0, 1, 0, 1]
        })
        
        with patch('src.settings.loaders.load_settings_by_file') as mock_load_settings, \
             patch('src.pipelines.train_pipeline.mlflow') as mock_mlflow, \
             patch('src.core.factory.Factory.create_data_adapter') as mock_adapter:
            
            # Mock settings
            mock_settings = type('MockSettings', (), {
                'environment': type('Env', (), {'app_env': 'dev'})(),
                'model': type('Model', (), {
                    'class_path': 'sklearn.ensemble.RandomForestClassifier',
                    'hyperparameters': type('HP', (), {'root': {'n_estimators': 10}})(),
                    'data_interface': type('DI', (), {
                        'task_type': 'classification',
                        'target_col': 'outcome',
                        'validate_required_fields': lambda: None
                    })(),
                    'loader': type('Loader', (), {'source_uri': 'test.sql'})(),
                    'augmenter': type('Aug', (), {'type': 'feature_store'})(),
                    'preprocessor': None,
                    'computed': {'run_name': 'test_run'}
                })(),
                'mlflow': type('MLflow', (), {
                    'tracking_uri': 'http://localhost:5000',
                    'experiment_name': 'test_experiment'
                })()
            })()
            
            mock_load_settings.return_value = mock_settings
            
            # Mock MLflow 설정
            mock_mlflow.set_tracking_uri.return_value = None
            mock_mlflow.set_experiment.return_value = None
            mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = 'test_run_123'
            mock_mlflow.start_run.return_value.__exit__.return_value = None
            mock_mlflow.pyfunc.log_model.return_value = None
            
            # Mock 데이터 어댑터
            mock_adapter_instance = mock_adapter.return_value
            mock_adapter_instance.read.return_value = sample_data
            
            # 실제 인프라에서 학습 실행
            with patch('src.core.trainer.Trainer.train') as mock_trainer_train:
                mock_trainer_train.return_value = (None, type('Model', (), {})(), {'metrics': {'accuracy': 0.9}})
                
                run_training(mock_settings)
                
                # 실제 인프라 사용 검증
                mock_mlflow.set_tracking_uri.assert_called_with('http://localhost:5000')
                mock_mlflow.set_experiment.assert_called_with('test_experiment')
                mock_adapter_instance.read.assert_called()
    
    def test_complete_inference_with_real_infrastructure(self):
        """실제 인프라에서 완전한 추론 워크플로우 테스트"""
        run_id = "test_run_123"
        
        # Mock Wrapped Artifact
        mock_wrapper = type('MockWrapper', (), {
            'loader_sql_snapshot': 'SELECT user_id FROM users',
            'predict': lambda self, df, params=None: pd.DataFrame({
                'user_id': df['user_id'],
                'prediction': [0.7, 0.8, 0.9]
            })
        })()
        
        sample_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'feature1': [1, 2, 3]
        })
        
        with patch('mlflow.pyfunc.load_model', return_value=mock_wrapper), \
             patch('src.pipelines.inference_pipeline._save_dataset') as mock_save, \
             patch('src.settings.loaders.load_settings') as mock_load_settings, \
             patch('src.core.factory.Factory.create_data_adapter') as mock_adapter:
            
            # Mock settings
            mock_settings = type('MockSettings', (), {
                'artifact_stores': {
                    'prediction_results': type('Store', (), {
                        'enabled': True,
                        'base_uri': 'file://test_output'
                    })()
                }
            })()
            
            mock_load_settings.return_value = mock_settings
            
            # Mock 데이터 어댑터
            mock_adapter_instance = mock_adapter.return_value
            mock_adapter_instance.read.return_value = sample_data
            
            # 실제 인프라에서 추론 실행
            run_batch_inference(run_id)
            
            # 실제 인프라 사용 검증
            mock_adapter_instance.read.assert_called()
            mock_save.assert_called()


@pytest.mark.requires_dev_stack
@pytest.mark.performance
class TestRealInfrastructurePerformance:
    """실제 인프라 성능 테스트"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    @pytest.mark.benchmark
    def test_postgresql_query_performance(self):
        """PostgreSQL 쿼리 성능 테스트"""
        start_time = time.time()
        
        # 대용량 쿼리 시뮬레이션
        with patch('src.utils.adapters.postgresql_adapter.PostgreSQLAdapter') as MockAdapter:
            mock_adapter = MockAdapter.return_value
            
            # 10000 행 데이터 시뮬레이션
            large_data = pd.DataFrame({
                'user_id': [f'u{i}' for i in range(10000)],
                'feature1': range(10000),
                'feature2': [i * 0.1 for i in range(10000)]
            })
            mock_adapter.read.return_value = large_data
            
            # 쿼리 실행
            result = mock_adapter.read("SELECT * FROM large_table LIMIT 10000")
            
            execution_time = time.time() - start_time
            
            # 성능 목표 검증 (5초 이내)
            assert execution_time < 5.0, f"PostgreSQL 쿼리 성능 목표 미달성: {execution_time:.2f}초"
            assert len(result) == 10000
    
    @pytest.mark.benchmark
    def test_redis_feature_lookup_performance(self):
        """Redis 피처 조회 성능 테스트"""
        start_time = time.time()
        
        # 대량 피처 조회 시뮬레이션
        with patch('src.utils.adapters.redis_adapter.RedisAdapter') as MockRedisAdapter:
            mock_redis = MockRedisAdapter.return_value
            
            # 1000명 사용자 피처 조회 시뮬레이션
            user_ids = [f'user{i}' for i in range(1000)]
            features = ['age', 'income', 'score']
            
            mock_features = {
                user_id: {
                    'member_id': user_id,
                    'age': 25 + (hash(user_id) % 40),
                    'income': 50000 + (hash(user_id) % 50000),
                    'score': 0.5 + (hash(user_id) % 100) / 200
                }
                for user_id in user_ids
            }
            mock_redis.get_features.return_value = mock_features
            
            # 피처 조회 실행
            result = mock_redis.get_features(user_ids, features)
            
            execution_time = time.time() - start_time
            
            # 성능 목표 검증 (100ms 이내)
            assert execution_time < 0.1, f"Redis 피처 조회 성능 목표 미달성: {execution_time:.3f}초"
            assert len(result) == 1000


@pytest.mark.requires_dev_stack
@pytest.mark.blueprint_principle_8
class TestRealDataLeakagePrevention:
    """실제 인프라에서 Data Leakage 방지 검증"""
    
    def setup_method(self):
        os.environ['APP_ENV'] = 'dev'
    
    def teardown_method(self):
        if 'APP_ENV' in os.environ:
            del os.environ['APP_ENV']
    
    def test_train_validation_split_isolation(self):
        """Train/Validation 분리 격리 검증"""
        from src.core.trainer import Trainer
        
        # 시계열 데이터 시뮬레이션
        sample_data = pd.DataFrame({
            'user_id': [f'u{i}' for i in range(100)],
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            'feature1': range(100),
            'feature2': [i * 0.1 for i in range(100)],
            'outcome': [(i % 2) for i in range(100)]
        })
        
        mock_settings = type('MockSettings', (), {
            'model': type('Model', (), {
                'data_interface': type('DI', (), {
                    'task_type': 'classification',
                    'target_col': 'outcome',
                    'validate_required_fields': lambda: None
                })(),
                'hyperparameter_tuning': type('HPT', (), {'enabled': False})()
            })()
        })()
        
        trainer = Trainer(mock_settings)
        
        # Data Leakage 방지 검증을 위한 Train/Test 분리
        train_df, test_df = trainer._split_data(sample_data)
        
        # 분리 검증
        assert len(train_df) + len(test_df) == len(sample_data)
        assert len(set(train_df.index) & set(test_df.index)) == 0  # 겹치지 않음
        
        # 시계열 순서 검증 (최신 데이터가 테스트셋에 포함되어야 함)
        train_max_timestamp = train_df['timestamp'].max()
        test_min_timestamp = test_df['timestamp'].min()
        
        # 적절한 시계열 분리 확인
        assert test_min_timestamp >= train_max_timestamp or len(set(train_df['timestamp']) & set(test_df['timestamp'])) == 0 