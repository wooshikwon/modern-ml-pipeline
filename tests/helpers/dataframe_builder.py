from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataFrameBuilder:
    @staticmethod
    def build_classification_data(n_samples: int = 100, n_features: int = 5, n_classes: int = 2, add_entity_column: bool = True, add_timestamp: bool = False, random_state: int = 42) -> pd.DataFrame:
        random.seed(random_state)
        data: Dict[str, Any] = {}
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [base_time + timedelta(hours=i) for i in range(n_samples)]
        for i in range(n_features):
            data[f'feature_{i}'] = [random.gauss(0, 1) for _ in range(n_samples)]
        data['target'] = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
        return pd.DataFrame(data)

    @staticmethod
    def build_regression_data(n_samples: int = 100, n_features: int = 5, add_entity_column: bool = True, add_timestamp: bool = False, random_state: int = 42) -> pd.DataFrame:
        random.seed(random_state)
        data: Dict[str, Any] = {}
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [base_time + timedelta(hours=i) for i in range(n_samples)]
        for i in range(n_features):
            data[f'feature_{i}'] = [random.gauss(0, 1) for _ in range(n_samples)]
        data['target'] = [sum(data[f'feature_{i}'][j] for i in range(n_features)) + random.gauss(0, 0.1) for j in range(n_samples)]
        return pd.DataFrame(data)

    @staticmethod
    def build_time_series_data(n_samples: int = 100, n_features: int = 3, freq: str = 'H', random_state: int = 42) -> pd.DataFrame:
        random.seed(random_state)
        time_index = pd.date_range(start=datetime.now(), periods=n_samples, freq=freq)
        data: Dict[str, Any] = {'timestamp': time_index}
        for i in range(n_features):
            values = []
            current = random.gauss(0, 1)
            for _ in range(n_samples):
                current = 0.8 * current + 0.2 * random.gauss(0, 1)
                values.append(current)
            data[f'feature_{i}'] = values
        data['target'] = [sum(data[f'feature_{i}'][j] for i in range(n_features)) + 0.1 * j + random.gauss(0, 0.1) for j in range(n_samples)]
        return pd.DataFrame(data)

    @staticmethod
    def build_clustering_data(n_samples: int = 100, n_clusters: int = 3, separation: str = 'well_separated', n_features: int = 2, add_entity_column: bool = True, add_timestamp: bool = False, random_state: int = 42) -> pd.DataFrame:
        random.seed(random_state)
        np.random.seed(random_state)
        data: Dict[str, Any] = {}
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [base_time + timedelta(hours=i) for i in range(n_samples)]
        samples_per_cluster = n_samples // n_clusters
        extra_samples = n_samples % n_clusters
        feature_data = {f'feature_{i}': [] for i in range(n_features)}
        if separation == 'well_separated':
            cluster_centers = [[c * 10 + random.gauss(0, 0.5) for _ in range(n_features)] for c in range(n_clusters)]
            for c in range(n_clusters):
                n_cluster_samples = samples_per_cluster + (1 if c < extra_samples else 0)
                for _ in range(n_cluster_samples):
                    for f in range(n_features):
                        feature_data[f'feature_{f}'].append(cluster_centers[c][f] + random.gauss(0, 1.5))
        elif separation == 'overlapping':
            cluster_centers = [[c * 3 + random.gauss(0, 0.5) for _ in range(n_features)] for c in range(n_clusters)]
            for c in range(n_clusters):
                n_cluster_samples = samples_per_cluster + (1 if c < extra_samples else 0)
                for _ in range(n_cluster_samples):
                    for f in range(n_features):
                        feature_data[f'feature_{f}'].append(cluster_centers[c][f] + random.gauss(0, 2.5))
        else:
            for _ in range(n_samples):
                for f in range(n_features):
                    feature_data[f'feature_{f}'].append(random.gauss(0, 5))
        for key, values in feature_data.items():
            data[key] = values
        if separation != 'random':
            cluster_labels = []
            for c in range(n_clusters):
                n_cluster_samples = samples_per_cluster + (1 if c < extra_samples else 0)
                cluster_labels.extend([c] * n_cluster_samples)
            data['cluster_label'] = cluster_labels
        return pd.DataFrame(data)

    @staticmethod
    def build_causal_data(n_samples: int = 100, treatment_effect: float = 2.0, confounding_strength: float = 0.5, noise_level: float = 0.1, n_features: int = 3, add_entity_column: bool = True, add_timestamp: bool = False, random_state: int = 42) -> pd.DataFrame:
        random.seed(random_state)
        np.random.seed(random_state)
        data: Dict[str, Any] = {}
        if add_entity_column:
            data['user_id'] = list(range(n_samples))
        if add_timestamp:
            base_time = datetime.now()
            data['timestamp'] = [base_time + timedelta(hours=i) for i in range(n_samples)]
        confounders = {f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}
        for i in range(n_features):
            data[f'feature_{i}'] = confounders[f'feature_{i}'].tolist()
        treatment_propensity = np.zeros(n_samples)
        for i in range(n_features):
            treatment_propensity += confounding_strength * confounders[f'feature_{i}']
        treatment_propensity += np.random.randn(n_samples) * 0.5
        treatment_probs = 1 / (1 + np.exp(-treatment_propensity))
        treatment = (np.random.rand(n_samples) < treatment_probs).astype(int)
        data['treatment'] = treatment.tolist()
        outcome = np.zeros(n_samples)
        for i in range(n_features):
            outcome += 0.5 * confounders[f'feature_{i}']
        outcome += treatment * treatment_effect
        outcome += np.random.randn(n_samples) * noise_level
        data['outcome'] = outcome.tolist()
        outcome_binary = (outcome > np.median(outcome)).astype(int)
        data['outcome_binary'] = outcome_binary.tolist()
        return pd.DataFrame(data)

    @staticmethod
    def build_numeric_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """스케일링용 숫자형 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            'feature_1': np.random.normal(100, 15, n_samples),  # 평균 100, 표준편차 15
            'feature_2': np.random.uniform(0, 1000, n_samples),  # 0-1000 균등분포
            'feature_3': np.random.exponential(5, n_samples),   # 지수분포 (이상치 포함)
            'feature_4': np.random.normal(0, 1, n_samples)      # 표준정규분포
        })
    
    @staticmethod
    def build_extreme_values_data(n_samples: int = 50, random_state: int = 42) -> pd.DataFrame:
        """이상치가 포함된 극단값 데이터프레임 생성"""
        np.random.seed(random_state)
        data = pd.DataFrame({
            'normal_feature': np.random.normal(10, 2, n_samples),
            'extreme_feature': np.concatenate([
                np.random.normal(5, 1, n_samples-10),  # 정상값
                np.array([100, -50, 200, -80, 150, 300, -100, 250, -90, 180])  # 극단값
            ])
        })
        return data

    @staticmethod
    def build_categorical_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """인코딩용 범주형 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 기본 범주형 피처들
            'category_low_card': np.random.choice(['A', 'B', 'C'], n_samples),  # 낮은 카디널리티
            'category_medium_card': np.random.choice(['Red', 'Green', 'Blue', 'Yellow', 'Purple'], n_samples),  # 중간 카디널리티
            'category_high_card': np.random.choice([f'Item_{i}' for i in range(20)], n_samples),  # 높은 카디널리티
            
            # 순서가 있는 범주형 피처
            'ordinal_feature': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            
            # 타겟 변수 (supervised encoding용)
            'target': np.random.choice([0, 1], n_samples)
        })
    
    @staticmethod
    def build_mixed_categorical_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """숫자형 + 범주형 혼합 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 숫자형 피처
            'numeric_1': np.random.normal(0, 1, n_samples),
            'numeric_2': np.random.uniform(-1, 1, n_samples),
            
            # 범주형 피처  
            'category_1': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
            'category_2': np.random.choice(['Small', 'Large'], n_samples),
            
            # 타겟 변수
            'target': np.random.randint(0, 3, n_samples)  # 다중 클래스
        })
    
    @staticmethod
    def build_mixed_preprocessor_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """전처리용 숫자형/범주형/결측값이 혼합된 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 숫자형 피처 (스케일링 대상)
            'num_feature_1': np.random.normal(100, 15, n_samples),
            'num_feature_2': np.random.uniform(0, 1, n_samples),  
            'num_feature_3': np.random.exponential(2, n_samples),
            
            # 범주형 피처 (인코딩 대상)
            'cat_feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat_feature_2': np.random.choice(['X', 'Y'], n_samples),
            
            # 결측값 포함 피처 (임퓨터 대상)
            'missing_feature': np.where(
                np.random.random(n_samples) > 0.8, 
                np.nan, 
                np.random.normal(50, 10, n_samples)
            ),
            
            # 타겟 변수
            'target': np.random.choice([0, 1], n_samples)
        })
    
    @staticmethod
    def build_missing_values_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """결측값이 포함된 임퓨터 테스트용 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 결측값이 있는 숫자형 피처들 (다양한 결측 패턴)
            'numeric_few_missing': np.where(
                np.random.random(n_samples) > 0.9,  # 10% 결측
                np.nan, 
                np.random.normal(10, 3, n_samples)
            ),
            'numeric_many_missing': np.where(
                np.random.random(n_samples) > 0.6,  # 40% 결측
                np.nan, 
                np.random.uniform(0, 100, n_samples)
            ),
            'numeric_extreme_missing': np.where(
                np.random.random(n_samples) > 0.2,  # 80% 결측 (극단적)
                np.nan, 
                np.random.exponential(5, n_samples)
            ),
            
            # 결측값이 있는 범주형 피처들
            'category_missing': np.where(
                np.random.random(n_samples) > 0.85,  # 15% 결측
                None,  # None을 사용하여 dtype 이슈 해결
                np.random.choice(['A', 'B', 'C', 'D'], n_samples)
            ),
            
            # 결측값이 없는 피처 (비교용)
            'numeric_complete': np.random.normal(50, 10, n_samples),
            'category_complete': np.random.choice(['X', 'Y', 'Z'], n_samples)
        })
    
    @staticmethod
    def build_feature_generation_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """피처 생성용 데이터프레임 생성 (TreeBased, Polynomial 테스트용)"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 피처 생성용 숫자형 피처들
            'feature_1': np.random.uniform(-2, 2, n_samples),
            'feature_2': np.random.uniform(-1, 3, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            
            # 비선형 관계를 가진 타겟 (TreeBased 테스트용)
            'target': np.where(
                (np.random.uniform(-2, 2, n_samples) * np.random.uniform(-1, 3, n_samples)) > 0,
                1, 0
            )
        })
    
    @staticmethod
    def build_discretization_data(n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """구간화(Discretization) 테스트용 연속형 데이터프레임 생성"""
        np.random.seed(random_state)
        return pd.DataFrame({
            # 다양한 분포의 연속형 변수들
            'uniform_dist': np.random.uniform(0, 100, n_samples),
            'normal_dist': np.random.normal(50, 15, n_samples),
            'exponential_dist': np.random.exponential(2, n_samples),
            'bimodal_dist': np.concatenate([
                np.random.normal(20, 5, n_samples//2),
                np.random.normal(80, 8, n_samples - n_samples//2)
            ])
        })
