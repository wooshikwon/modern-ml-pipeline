"""테스트 데이터 생성 팩토리 - 하드코딩 제거 및 표준화

현재 테스트에서 사용되는 데이터 패턴 분석 후 표준화:
- Entity 스키마: user_id + timestamp 기반
- 다양한 feature 타입: 숫자형, 범주형, 결측치 포함
- Task별 표준 데이터: classification, regression, clustering
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class TestDataFactory:
    """표준 테스트 데이터 생성 팩토리"""
    
    @staticmethod
    def create_classification_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
        """분류 작업용 표준 테스트 데이터
        
        현재 test_fetcher.py, test_preprocessor.py에서 사용하는 패턴을 표준화
        - Entity 스키마 준수: user_id, event_timestamp
        - 숫자형/범주형 피처 포함
        - 분류 타겟 변수
        """
        np.random.seed(seed)
        
        return pd.DataFrame({
            # Entity 스키마 (Blueprint 계약)
            'user_id': [f'user_{i:03d}' for i in range(n_samples)],
            'event_timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            
            # 숫자형 피처 (결측치 포함)
            'feature1': np.concatenate([
                np.random.normal(0, 1, n_samples-int(n_samples*0.1)), 
                [np.nan]*int(n_samples*0.1)
            ]),
            'feature2': np.random.normal(5, 2, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            
            # 범주형 피처  
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            
            # 타겟 변수 (분류)
            'target': np.random.choice([0, 1], n_samples)
        })
    
    @staticmethod
    def create_regression_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
        """회귀 작업용 표준 테스트 데이터"""
        np.random.seed(seed)
        
        # 실제적인 선형 관계 생성
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples) 
        X3 = np.random.normal(0, 1, n_samples)
        
        # y = 1.5*X1 - 2.0*X2 + 0.5*X3 + noise
        y = 1.5 * X1 - 2.0 * X2 + 0.5 * X3 + np.random.normal(0, 0.1, n_samples)
        
        return pd.DataFrame({
            # Entity 스키마
            'user_id': [f'user_{i:03d}' for i in range(n_samples)],
            'event_timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            
            # 피처 변수들
            'feature1': X1,
            'feature2': X2,
            'feature3': X3,
            'additional_numeric': np.random.uniform(0, 100, n_samples),
            
            # 타겟 변수 (회귀)
            'target': y
        })
    
    @staticmethod  
    def create_entity_data(entity_ids: List[str], start_date: str = '2024-01-01') -> pd.DataFrame:
        """Entity 기반 기본 데이터 프레임 (현재 test_fetcher.py 패턴)"""
        return pd.DataFrame({
            'user_id': entity_ids,
            'event_timestamp': pd.date_range(start_date, periods=len(entity_ids), freq='h')
        })
    
    @staticmethod
    def create_comprehensive_training_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
        """포괄적 학습 데이터 - test_preprocessor.py의 comprehensive_training_data 패턴"""
        np.random.seed(seed)
        
        return pd.DataFrame({
            # Entity 스키마
            'user_id': range(n_samples),
            'event_timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
            
            # 숫자형 피처 (결측치 포함)
            'age': np.concatenate([
                np.random.normal(35, 10, n_samples-20), 
                [np.nan]*20
            ]),
            'income': np.random.lognormal(10, 1, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            
            # 범주형 피처
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_type': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            
            # Boolean 피처
            'is_premium': np.random.choice([True, False], n_samples),
            'has_history': np.random.choice([True, False], n_samples),
            
            # 타겟 변수 (분류 예시)
            'approved': np.random.choice([0, 1], n_samples)
        })
    
    @staticmethod
    def create_minimal_entity_data(entity_count: int = 3, seed: int = 42) -> pd.DataFrame:
        """최소한의 Entity 데이터 - 빠른 테스트용 (sample_entity_data 패턴)"""
        np.random.seed(seed)
        
        return pd.DataFrame({
            'user_id': [f'u{i+1}' for i in range(entity_count)],
            'event_ts': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-03'
            ][:entity_count]),
            'label': np.random.choice([0, 1], entity_count)
        })
    
    @staticmethod
    def create_time_series_data(n_samples: int = 100, 
                              start_date: str = '2024-01-01',
                              freq: str = 'D',
                              seed: int = 42) -> pd.DataFrame:
        """시계열 데이터 생성"""
        np.random.seed(seed)
        
        dates = pd.date_range(start_date, periods=n_samples, freq=freq)
        trend = np.linspace(0, 10, n_samples)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
        noise = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame({
            'user_id': ['time_series_user'] * n_samples,
            'event_timestamp': dates,
            'value': trend + seasonal + noise,
            'feature1': np.random.normal(0, 1, n_samples),
            'target': trend + seasonal + 0.5 * noise
        })
    
    @classmethod
    def get_standard_exclude_columns(cls) -> List[str]:
        """표준 제외 컬럼 리스트 - preprocessor 설정에서 사용"""
        return ['user_id', 'event_timestamp', 'event_ts', 'target', 'approved', 'label']
    
    @classmethod
    def get_entity_columns(cls) -> List[str]:
        """Entity 컬럼 리스트 - settings에서 사용"""
        return ['user_id']
    
    @classmethod
    def get_timestamp_column(cls) -> str:
        """Timestamp 컬럼명 - settings에서 사용"""  
        return 'event_timestamp'