"""
환경별 기본 설정 정의
Recipe 생성 시 환경에 따라 다른 설정을 적용하기 위한 설정 파일
"""

from typing import Dict, Any

# 환경별 기본 설정
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'local': {
        'data_config': {
            'loader_name': 'local_data_loader',
            'source_uri': 'data/train.csv',
            'adapter': 'storage',
            'target_column': 'target',
            'entity_columns': None,
            'timestamp_column': None
        },
        'feature_store_enabled': False,
        'hyperparameter_tuning': False,
        'n_trials': 10,
        'validation_config': {
            'method': 'train_test_split',
            'test_size': 0.2,
            'random_state': 42
        }
    },
    'dev': {
        'data_config': {
            'loader_name': 'sql_data_loader', 
            'source_uri': '${MMP_DEV_DATA_URI}',
            'adapter': 'sql',
            'target_column': 'target',
            'entity_columns': ['entity_id'],
            'timestamp_column': 'event_timestamp'
        },
        'feature_store_enabled': True,
        'feature_namespace': 'dev_features',
        'feature_list': ['feature_1', 'feature_2', 'feature_3'],
        'hyperparameter_tuning': True,
        'n_trials': 50,
        'validation_config': {
            'method': 'time_series_split',
            'test_size': 0.2,
            'random_state': 42
        }
    },
    'prod': {
        'data_config': {
            'loader_name': 'sql_data_loader',
            'source_uri': '${MMP_PROD_DATA_URI}', 
            'adapter': 'sql',
            'target_column': 'target',
            'entity_columns': ['entity_id'],
            'timestamp_column': 'event_timestamp'
        },
        'feature_store_enabled': True,
        'feature_namespace': 'prod_features',
        'feature_list': ['feature_1', 'feature_2', 'feature_3'],
        'hyperparameter_tuning': True,
        'n_trials': 100,
        'validation_config': {
            'method': 'time_series_split',
            'test_size': 0.15,
            'random_state': 42
        }
    }
}

# 태스크별 평가 메트릭은 ml_metrics.py로 분리됨