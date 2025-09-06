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
