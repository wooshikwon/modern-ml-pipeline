"""Mock 컴포넌트 중앙 관리 레지스트리

테스트에서 사용되는 모든 컴포넌트의 Mock을 중앙에서 관리
- 일관된 Mock 인터페이스 제공
- 캐싱을 통한 성능 최적화
- Contract 준수 보장
"""
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional, List
import hashlib
import time
from collections import OrderedDict
import sys


class MockComponentRegistry:
    """Mock 컴포넌트 중앙 관리 레지스트리 - 고도화된 LRU 캐싱"""
    
    _instances: OrderedDict = OrderedDict()  # LRU 캐싱
    _cache_stats = {
        'hits': 0,
        'misses': 0,
        'total_requests': 0,
        'memory_usage_kb': 0
    }
    _max_cache_size = 100  # 최대 캐시 항목 수
    _access_times = {}  # 접근 시간 추적
    
    @classmethod
    def get_fetcher(cls, fetcher_type: str = "pass_through") -> Mock:
        """fetcher Mock 제공 - Blueprint 계약 준수"""
        key = f"fetcher_{fetcher_type}"
        cls._cache_stats['total_requests'] += 1
        
        if key in cls._instances:
            # LRU 캐시 hit - 최근 사용된 아이템으로 이동
            cls._instances.move_to_end(key)
            cls._cache_stats['hits'] += 1
            cls._access_times[key] = time.time()
            return cls._instances[key]
        
        # 캐시 miss - 새 인스턴스 생성
        cls._cache_stats['misses'] += 1
        mock = Mock(spec=['fetch'])
        
        def mock_fetch(data, run_mode="train"):
            """실제 fetcher와 동일한 인터페이스 Mock"""
            # Blueprint 데이터 계약: entity + timestamp 보존
            if isinstance(data, pd.DataFrame):
                # 실제 PassThroughfetcher처럼 입력 데이터 그대로 반환
                result = data.copy()
                
                # Entity 스키마 보존 확인
                if 'user_id' not in result.columns:
                    result['user_id'] = ['mock_user'] * len(result)
                if 'event_timestamp' not in result.columns and 'event_ts' not in result.columns:
                    result['event_timestamp'] = pd.date_range('2024-01-01', periods=len(result), freq='h')
                
                return result
            else:
                # 최소한의 데이터프레임 반환
                return pd.DataFrame({
                    'user_id': ['mock_user'],
                    'event_timestamp': [pd.Timestamp('2024-01-01')],
                    'feature1': [1.0]
                })
        
        mock.fetch.side_effect = mock_fetch
        cls._add_to_cache(key, mock)
            
        return cls._instances[key]
    
    @classmethod
    def get_preprocessor(cls, preprocessor_type: str = "simple_scaler") -> Mock:
        """Preprocessor Mock 제공 - sklearn 호환 인터페이스"""
        key = f"preprocessor_{preprocessor_type}"
        
        if key not in cls._instances:
            mock = Mock(spec=['fit', 'transform', 'fit_transform'])
            
            def mock_fit_transform(X, y=None):
                """sklearn 호환 fit_transform Mock"""
                if isinstance(X, pd.DataFrame):
                    # 숫자형 컬럼만 선택하여 처리
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # 표준화 시뮬레이션 (평균=0, 표준편차=1)
                        result = X.copy()
                        for col in numeric_cols:
                            result[col] = (result[col] - result[col].mean()) / result[col].std()
                        return result
                    else:
                        # 숫자형 컬럼이 없으면 더미 데이터 반환
                        return np.array([[1.0, 2.0], [3.0, 4.0]])[:len(X)]
                else:
                    # numpy array 처리
                    if hasattr(X, 'shape') and len(X.shape) == 2:
                        return np.random.standard_normal(X.shape)
                    else:
                        return np.array([[1.0, 2.0], [3.0, 4.0]])
            
            def mock_transform(X):
                """sklearn 호환 transform Mock"""
                return mock_fit_transform(X, y=None)
            
            def mock_fit(X, y=None):
                """sklearn 호환 fit Mock"""
                return mock  # self 반환
            
            mock.fit_transform.side_effect = mock_fit_transform
            mock.transform.side_effect = mock_transform
            mock.fit.side_effect = mock_fit
            
            cls._instances[key] = mock
            
        return cls._instances[key]
    
    @classmethod
    def get_model(cls, model_type: str = "classifier") -> Mock:
        """모델 Mock 제공 - sklearn 호환 인터페이스"""
        key = f"model_{model_type}"
        
        if key not in cls._instances:
            mock = Mock(spec=['fit', 'predict', 'predict_proba'])
            
            def mock_fit(X, y):
                """sklearn 호환 fit Mock"""
                return mock
            
            def mock_predict(X):
                """sklearn 호환 predict Mock"""
                if hasattr(X, '__len__'):
                    n_samples = len(X)
                    if model_type == "classifier":
                        return np.random.choice([0, 1], n_samples)
                    else:  # regressor
                        return np.random.normal(0, 1, n_samples)
                else:
                    return np.array([1] if model_type == "classifier" else [0.5])
            
            def mock_predict_proba(X):
                """분류기 전용 predict_proba Mock"""
                if hasattr(X, '__len__'):
                    n_samples = len(X)
                    # 2클래스 확률 반환
                    proba = np.random.uniform(0, 1, (n_samples, 2))
                    proba = proba / proba.sum(axis=1, keepdims=True)  # 정규화
                    return proba
                else:
                    return np.array([[0.3, 0.7]])
            
            mock.fit.side_effect = mock_fit
            mock.predict.side_effect = mock_predict
            if model_type == "classifier":
                mock.predict_proba.side_effect = mock_predict_proba
            
            cls._instances[key] = mock
            
        return cls._instances[key]
    
    @classmethod  
    def get_factory(cls, settings_dict: Dict) -> Mock:
        """Factory Mock 제공 - 설정 기반 캐싱"""
        # 설정 해시 생성 (캐싱 키)
        settings_str = str(sorted(settings_dict.items()))
        settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]
        key = f"factory_{settings_hash}"
        
        if key not in cls._instances:
            mock = Mock(spec=[
                'create_fetcher', 'create_preprocessor', 'create_model', 
                'create_evaluator', 'create_data_adapter'
            ])
            
            # settings 속성 설정 (factory.settings 접근을 위해)
            mock.settings = cls._create_settings_mock(settings_dict)
            
            # 컴포넌트 생성 메서드 Mock 설정
            def create_fetcher(run_mode=None):
                fetcher_type = "pass_through"  # 기본값
                if hasattr(mock.settings, 'recipe') and hasattr(mock.settings.recipe, 'model'):
                    if hasattr(mock.settings.recipe.model, 'fetcher'):
                        fetcher_type = getattr(mock.settings.recipe.model.fetcher, 'type', 'pass_through')
                
                # Serving 모드 제약 검증 (Blueprint 정책)
                if run_mode == "serving" and fetcher_type == "pass_through":
                    raise TypeError("Serving에서는 pass_through 사용이 금지됩니다. Feature Store 연결이 필요합니다.")
                
                return cls.get_fetcher(fetcher_type)
            
            def create_preprocessor():
                preprocessor_name = "simple_scaler"  # 기본값
                if hasattr(mock.settings, 'recipe') and hasattr(mock.settings.recipe, 'model'):
                    if hasattr(mock.settings.recipe.model, 'preprocessor'):
                        preprocessor_name = getattr(mock.settings.recipe.model.preprocessor, 'name', 'simple_scaler')
                
                return cls.get_preprocessor(preprocessor_name)
            
            def create_model():
                task_type = "classification"  # 기본값
                if hasattr(mock.settings, 'recipe') and hasattr(mock.settings.recipe, 'model'):
                    if hasattr(mock.settings.recipe.model, 'data_interface'):
                        task_type = getattr(mock.settings.recipe.model.data_interface, 'task_type', 'classification')
                
                model_type = "classifier" if task_type == "classification" else "regressor"
                return cls.get_model(model_type)
            
            def create_evaluator():
                return Mock(spec=['evaluate'])
            
            def create_data_adapter(adapter_type=None):
                return Mock(spec=['read', 'write'])
            
            mock.create_fetcher.side_effect = create_fetcher
            mock.create_preprocessor.side_effect = create_preprocessor
            mock.create_model.side_effect = create_model
            mock.create_evaluator.side_effect = create_evaluator
            mock.create_data_adapter.side_effect = create_data_adapter
            
            # model_config property Mock
            if 'recipe' in settings_dict and 'model' in settings_dict['recipe']:
                mock.model_config = cls._dict_to_mock(settings_dict['recipe']['model'])
            else:
                mock.model_config = Mock()
            
            cls._instances[key] = mock
            
        return cls._instances[key]
    
    @classmethod
    def get_settings(cls, settings_dict: Dict) -> Mock:
        """Settings Mock 제공 - 중첩 딕셔너리를 Mock 속성으로 변환"""
        settings_str = str(sorted(settings_dict.items()))
        settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:8]
        key = f"settings_{settings_hash}"
        
        if key not in cls._instances:
            mock = cls._create_settings_mock(settings_dict)
            cls._instances[key] = mock
            
        return cls._instances[key]
    
    @classmethod
    def _create_settings_mock(cls, settings_dict: Dict) -> Mock:
        """중첩 딕셔너리를 Mock 객체로 변환"""
        return cls._dict_to_mock(settings_dict)
    
    @classmethod
    def _dict_to_mock(cls, d: Any) -> Any:
        """딕셔너리를 Mock 객체로 재귀적 변환"""
        if isinstance(d, dict):
            mock_obj = Mock()
            for key, value in d.items():
                setattr(mock_obj, key, cls._dict_to_mock(value))
            return mock_obj
        elif isinstance(d, list):
            return [cls._dict_to_mock(item) for item in d]
        else:
            return d
    
    @classmethod
    def get_data_adapter(cls, adapter_type: str = "storage") -> Mock:
        """데이터 어댑터 Mock 제공"""
        key = f"adapter_{adapter_type}"
        
        if key not in cls._instances:
            mock = Mock(spec=['read', 'write'])
            
            def mock_read(source):
                """어댑터 read Mock - 표준 데이터프레임 반환"""
                return pd.DataFrame({
                    'user_id': ['user_001', 'user_002'],
                    'event_timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
                    'feature1': [1.0, 2.0],
                    'target': [0, 1]
                })
            
            def mock_write(data, destination):
                """어댑터 write Mock"""
                return {"status": "success", "records": len(data) if hasattr(data, '__len__') else 1}
            
            mock.read.side_effect = mock_read
            mock.write.side_effect = mock_write
            
            cls._instances[key] = mock
            
        return cls._instances[key]
    
    @classmethod
    def _add_to_cache(cls, key: str, mock_obj: Mock):
        """캐시에 객체 추가 (LRU 정책 적용)"""
        # 캐시 크기 제한 확인
        if len(cls._instances) >= cls._max_cache_size:
            # LRU 정책: 가장 오래된 항목 제거
            oldest_key = next(iter(cls._instances))
            del cls._instances[oldest_key]
            if oldest_key in cls._access_times:
                del cls._access_times[oldest_key]
        
        cls._instances[key] = mock_obj
        cls._access_times[key] = time.time()
        cls._update_memory_usage()
    
    @classmethod
    def _update_memory_usage(cls):
        """메모리 사용량 업데이트"""
        total_size = sys.getsizeof(cls._instances)
        for key, obj in cls._instances.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(obj)
        cls._cache_stats['memory_usage_kb'] = total_size / 1024
    
    @classmethod
    def warm_cache(cls):
        """자주 사용되는 Mock 객체들을 미리 캐시에 로드"""
        # 기본 컴포넌트들 미리 로드
        cls.get_fetcher("pass_through")
        cls.get_preprocessor("simple_scaler") 
        cls.get_model("classifier")
        cls.get_model("regressor")
        cls.get_data_adapter("storage")
    
    @classmethod
    def reset_all(cls):
        """모든 Mock 인스턴스 및 통계 초기화"""
        cls._instances.clear()
        cls._access_times.clear()
        cls._cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'memory_usage_kb': 0
        }
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """고도화된 캐시 통계 제공"""
        hit_rate = (cls._cache_stats['hits'] / cls._cache_stats['total_requests'] * 100) if cls._cache_stats['total_requests'] > 0 else 0
        
        return {
            "total_cached_instances": len(cls._instances),
            "cached_types": list(set(key.split('_')[0] for key in cls._instances.keys())),
            "cache_keys": list(cls._instances.keys()),
            "hit_rate_percent": round(hit_rate, 2),
            "hits": cls._cache_stats['hits'],
            "misses": cls._cache_stats['misses'],
            "total_requests": cls._cache_stats['total_requests'],
            "memory_usage_kb": round(cls._cache_stats['memory_usage_kb'], 2),
            "avg_access_age_seconds": cls._get_avg_access_age()
        }
    
    @classmethod
    def _get_avg_access_age(cls) -> float:
        """평균 접근 연령 계산"""
        if not cls._access_times:
            return 0.0
        
        current_time = time.time()
        ages = [current_time - access_time for access_time in cls._access_times.values()]
        return round(sum(ages) / len(ages), 2)