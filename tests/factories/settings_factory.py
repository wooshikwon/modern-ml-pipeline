"""설정 객체 동적 생성 팩토리 - 파일 의존성 제거

현재 테스트에서 load_settings_by_file() 하드코딩 패턴을 제거하고
동적으로 Settings 딕셔너리를 생성하는 팩토리
"""
from typing import Dict, Any, Optional, List


class SettingsFactory:
    """설정 객체 동적 생성 팩토리"""
    
    @staticmethod
    def create_base_settings(**overrides) -> Dict[str, Any]:
        """기본 설정 딕셔너리 생성 - 모든 테스트의 공통 기반 (완전한 Settings 스키마)"""
        base = {
            "environment": {
                "env_name": "test",
                "gcp_project_id": "test-project"
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5002",
                "experiment_name": "test_experiment"
            },
            "serving": {
                "enabled": False,
                "model_stage": "None",
                "realtime_feature_store": {
                    "store_type": "redis",
                    "connection": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0
                    }
                }
            },
            "artifact_stores": {
                "local": {
                    "enabled": True,
                    "base_uri": "./test_artifacts"
                }
            },
            "data_adapters": {
                "default_loader": "test_loader",
                "default_storage": "storage",
                "adapters": {
                    "storage": {
                        "class_name": "StorageAdapter",
                        "config": {
                            "storage_options": {}
                        }
                    },
                    "sql": {
                        "class_name": "SqlAdapter",
                        "config": {
                            "connection_uri": "sqlite:///:memory:",
                            "query_timeout": 30
                        }
                    }
                }
            },
            "hyperparameter_tuning": {
                "enabled": False
            }
        }
        
        # 깊은 딕셔너리 병합
        return SettingsFactory._deep_merge(base, overrides)
    
    @staticmethod
    def create_classification_settings(env: str = "test", **overrides) -> Dict[str, Any]:
        """분류 작업용 설정 생성 - local_classification_test.yaml 패턴 기반"""
        classification_config = {
            "recipe": {
                "name": f"test_classification_{env}",
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "hyperparameters": {
                        "n_estimators": 50,
                        "max_depth": 10,
                        "random_state": 42,
                        "n_jobs": 2
                    },
                    "loader": {
                        "name": f"test_data_loader_{env}",
                        "source_uri": f"tests/fixtures/data/test_classification_data.csv",
                        "adapter": "storage",
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "event_timestamp"
                        }
                    },
                    "augmenter": {
                        "type": "pass_through"
                    },
                    "preprocessor": {
                        "name": "simple_scaler",
                        "params": {
                            "criterion_col": None,
                            "exclude_cols": ["user_id", "event_timestamp"]
                        }
                    },
                    "data_interface": {
                        "task_type": "classification", 
                        "target_column": "target"  # approved -> target로 표준화
                    },
                    "hyperparameter_tuning": {
                        "enabled": False
                    }
                },
                "evaluation": {
                    "metrics": [
                        "accuracy", "precision_weighted", "recall_weighted", 
                        "f1_weighted", "roc_auc"
                    ],
                    "validation": {
                        "method": "train_test_split",
                        "test_size": 0.2,
                        "stratify": True,
                        "random_state": 42
                    }
                }
            }
        }
        
        settings = SettingsFactory.create_base_settings(**classification_config)
        return SettingsFactory._deep_merge(settings, overrides)
    
    @staticmethod  
    def create_regression_settings(env: str = "test", **overrides) -> Dict[str, Any]:
        """회귀 작업용 설정 생성"""
        regression_config = {
            "recipe": {
                "name": f"test_regression_{env}",
                "model": {
                    "class_path": "sklearn.linear_model.LinearRegression",
                    "hyperparameters": {
                        "fit_intercept": True
                    },
                    "loader": {
                        "name": f"test_regression_loader_{env}",
                        "source_uri": f"tests/fixtures/data/test_regression_data.csv",
                        "adapter": "storage",
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "event_timestamp"
                        }
                    },
                    "augmenter": {
                        "type": "pass_through"
                    },
                    "preprocessor": {
                        "name": "standard_scaler",
                        "params": {
                            "exclude_cols": ["user_id", "event_timestamp", "target"]
                        }
                    },
                    "data_interface": {
                        "task_type": "regression",
                        "target_column": "target"
                    },
                    "hyperparameter_tuning": {
                        "enabled": False
                    }
                },
                "evaluation": {
                    "metrics": [
                        "mse", "rmse", "mae", "r2"
                    ],
                    "validation": {
                        "method": "train_test_split",
                        "test_size": 0.2,
                        "stratify": False,
                        "random_state": 42
                    }
                }
            }
        }
        
        settings = SettingsFactory.create_base_settings(**regression_config)
        return SettingsFactory._deep_merge(settings, overrides)
    
    @staticmethod
    def create_local_settings(**overrides) -> Dict[str, Any]:
        """로컬 환경 설정 생성"""
        local_config = {
            "environment": {
                "env_name": "local"
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5002",
                "experiment_name": "local-test-Campaign-Uplift-Modeling"
            }
        }
        
        return SettingsFactory.create_base_settings(**SettingsFactory._deep_merge(local_config, overrides))
    
    @staticmethod
    def create_dev_settings(**overrides) -> Dict[str, Any]:
        """개발 환경 설정 생성"""
        dev_config = {
            "environment": {
                "env_name": "dev"
            },
            "feature_store": {
                "provider": "feast",
                "config": {
                    "feature_store_path": "/tmp/test_feature_store"
                }
            }
        }
        
        return SettingsFactory.create_base_settings(**SettingsFactory._deep_merge(dev_config, overrides))
    
    @staticmethod
    def create_minimal_settings(task_type: str = "classification", **overrides) -> Dict[str, Any]:
        """최소한의 설정 - 빠른 테스트용"""
        minimal_config = {
            "environment": {"env_name": "test"},
            "mlflow": {"tracking_uri": "memory://", "experiment_name": "test"},
            "serving": {"host": "localhost", "port": 8000},
            "artifact_stores": {"local": {"type": "local", "path": "/tmp"}},
            "recipe": {
                "name": f"minimal_{task_type}_test",
                "model": {
                    "class_path": "sklearn.ensemble.RandomForestClassifier" if task_type == "classification" else "sklearn.linear_model.LinearRegression",
                    "hyperparameters": {"random_state": 42} if task_type == "classification" else {},
                    "data_interface": {
                        "task_type": task_type,
                        "target_column": "target"
                    },
                    "loader": {
                        "adapter": "storage",
                        "entity_schema": {
                            "entity_columns": ["user_id"],
                            "timestamp_column": "event_timestamp"
                        }
                    },
                    "augmenter": {"type": "pass_through"}
                }
            }
        }
        
        return SettingsFactory._deep_merge(minimal_config, overrides)
    
    @staticmethod
    def create_feature_store_settings(provider: str = "feast", **overrides) -> Dict[str, Any]:
        """Feature Store 설정 생성"""
        fs_config = {
            "environment": {"env_name": "dev"},
            "feature_store": {
                "provider": provider,
                "config": {
                    "feature_store_path": "/tmp/test_feature_store" if provider == "feast" else None,
                    "redis_connection_string": "redis://localhost:6379" if provider == "feast" else None
                }
            },
            "recipe": {
                "name": "test_feature_store_recipe",
                "model": {
                    "augmenter": {
                        "type": "feature_store",
                        "provider": provider
                    }
                }
            }
        }
        
        return SettingsFactory.create_classification_settings(**SettingsFactory._deep_merge(fs_config, overrides))
    
    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """딕셔너리 깊은 병합"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SettingsFactory._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def get_standard_entity_schema() -> Dict[str, Any]:
        """표준 Entity 스키마"""
        return {
            "entity_columns": ["user_id"],
            "timestamp_column": "event_timestamp"
        }
    
    @staticmethod
    def get_standard_preprocessor_config(task_type: str = "classification") -> Dict[str, Any]:
        """표준 전처리기 설정"""
        return {
            "name": "simple_scaler" if task_type == "classification" else "standard_scaler",
            "params": {
                "exclude_cols": ["user_id", "event_timestamp", "target", "approved", "label"]
            }
        }