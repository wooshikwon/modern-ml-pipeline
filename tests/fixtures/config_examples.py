"""
Config Examples - Pydantic Model Example Data

Config 스키마에서 분리한 예제 데이터들을 제공합니다.
이전에 model_config에 하드코딩되어 있던 예제들을 체계적으로 관리합니다.
"""

from typing import Dict, Any

# Config 모델의 기본 예제 데이터 (원래 model_config에 있던 것)
CONFIG_LOCAL_EXAMPLE: Dict[str, Any] = {
    "environment": {
        "name": "local"
    },
    "mlflow": {
        "tracking_uri": "./mlruns",
        "experiment_name": "mmp-local"
    },
    "data_source": {
        "name": "PostgreSQL",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "postgresql://user:pass@localhost:5432/db",
            "query_timeout": 30
        }
    },
    "feature_store": {
        "provider": "feast",
        "feast_config": {
            "project": "feast_local",
            "registry": "./feast_repo/registry.db",
            "online_store": {
                "type": "sqlite",
                "path": "./feast_repo/online_store.db"
            },
            "offline_store": {
                "type": "file",
                "path": "./feast_repo/data"
            }
        }
    },
    "serving": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1
    }
}

# 추가 환경별 예제 데이터들
CONFIG_DEVELOPMENT_EXAMPLE: Dict[str, Any] = {
    "environment": {
        "name": "development"
    },
    "mlflow": {
        "tracking_uri": "http://mlflow-dev.company.com:5000",
        "experiment_name": "mmp-development",
        "tracking_username": "dev_user",
        "tracking_password": "dev_password"
    },
    "data_source": {
        "name": "PostgreSQL Development",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "postgresql://dev_user:dev_pass@dev-db.company.com:5432/mmp_dev",
            "query_timeout": 60,
            "pool_size": 10
        }
    },
    "feature_store": {
        "provider": "feast",
        "feast_config": {
            "project": "feast_dev",
            "registry": "gs://feast-dev-bucket/registry.db",
            "online_store": {
                "type": "redis",
                "connection_string": "redis://dev-redis.company.com:6379",
                "password": "redis_dev_password"
            },
            "offline_store": {
                "type": "bigquery",
                "project_id": "company-ml-dev",
                "dataset_id": "feast_offline"
            }
        }
    },
    "serving": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "workers": 2,
        "model_stage": "Staging",
        "auth": {
            "enabled": True,
            "type": "jwt",
            "secret_key": "dev_jwt_secret_key"
        }
    },
    "artifact_store": {
        "type": "s3",
        "config": {
            "bucket": "mmp-artifacts-dev",
            "region": "us-west-2",
            "access_key_id": "dev_access_key",
            "secret_access_key": "dev_secret_key"
        }
    },
    "output": {
        "inference": {
            "name": "batch_inference_results",
            "enabled": True,
            "adapter_type": "storage",
            "config": {
                "base_path": "s3://mmp-results-dev/inference/",
                "format": "parquet"
            }
        },
        "preprocessed": {
            "name": "preprocessed_data",
            "enabled": True,
            "adapter_type": "storage",
            "config": {
                "base_path": "s3://mmp-results-dev/preprocessed/",
                "format": "parquet"
            }
        }
    }
}

CONFIG_PRODUCTION_EXAMPLE: Dict[str, Any] = {
    "environment": {
        "name": "production"
    },
    "mlflow": {
        "tracking_uri": "https://mlflow.company.com",
        "experiment_name": "mmp-production",
        "tracking_username": "prod_service",
        "tracking_password": "secure_prod_password",
        "s3_endpoint_url": "https://s3.amazonaws.com"
    },
    "data_source": {
        "name": "PostgreSQL Production",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "postgresql://prod_user:prod_pass@prod-db-cluster.company.com:5432/mmp_prod",
            "query_timeout": 120,
            "pool_size": 20,
            "ssl_mode": "require"
        }
    },
    "feature_store": {
        "provider": "feast",
        "feast_config": {
            "project": "feast_prod",
            "registry": "s3://feast-prod-bucket/registry.db",
            "online_store": {
                "type": "dynamodb",
                "region": "us-east-1",
                "table_name": "feast_online_features"
            },
            "offline_store": {
                "type": "bigquery",
                "project_id": "company-ml-prod",
                "dataset_id": "feast_offline_prod"
            },
            "entity_key_serialization_version": 2
        }
    },
    "serving": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "model_stage": "Production",
        "auth": {
            "enabled": True,
            "type": "oauth",
            "secret_key": "highly_secure_prod_jwt_secret"
        }
    },
    "artifact_store": {
        "type": "s3",
        "config": {
            "bucket": "mmp-artifacts-prod",
            "region": "us-east-1",
            "kms_key_id": "arn:aws:kms:us-east-1:123456789:key/12345678-1234-1234-1234-123456789012"
        }
    },
    "output": {
        "inference": {
            "name": "production_inference",
            "enabled": True,
            "adapter_type": "sql",
            "config": {
                "table": "ml_predictions.batch_inference_results",
                "connection_uri": "postgresql://prod_writer:writer_pass@prod-db-cluster.company.com:5432/mmp_prod"
            }
        },
        "preprocessed": {
            "name": "production_preprocessed",
            "enabled": False,  # 프로덕션에서는 전처리 결과 저장 비활성화
            "adapter_type": "storage",
            "config": {
                "base_path": "s3://mmp-results-prod/preprocessed/",
                "format": "parquet"
            }
        }
    }
}

# Feature Store 없는 간단한 구성 예제
CONFIG_SIMPLE_EXAMPLE: Dict[str, Any] = {
    "environment": {
        "name": "simple"
    },
    "mlflow": {
        "tracking_uri": "./mlruns",
        "experiment_name": "simple-experiment"
    },
    "data_source": {
        "name": "Local CSV",
        "adapter_type": "storage",
        "config": {
            "base_path": "./data/",
            "supported_formats": ["csv", "parquet", "json"]
        }
    },
    "feature_store": {
        "provider": "none"
    }
}

# 모든 예제를 모은 딕셔너리
CONFIG_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "local": CONFIG_LOCAL_EXAMPLE,
    "development": CONFIG_DEVELOPMENT_EXAMPLE,
    "production": CONFIG_PRODUCTION_EXAMPLE,
    "simple": CONFIG_SIMPLE_EXAMPLE,
}

# 컴포넌트별 예제들
MLFLOW_EXAMPLES = {
    "local": {
        "tracking_uri": "./mlruns",
        "experiment_name": "local-experiment"
    },
    "remote": {
        "tracking_uri": "http://mlflow-server:5000",
        "experiment_name": "remote-experiment",
        "tracking_username": "mlflow_user",
        "tracking_password": "mlflow_password"
    },
    "s3_backend": {
        "tracking_uri": "http://mlflow-server:5000",
        "experiment_name": "s3-backed-experiment",
        "s3_endpoint_url": "https://s3.amazonaws.com"
    }
}

DATA_SOURCE_EXAMPLES = {
    "postgresql": {
        "name": "PostgreSQL",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "postgresql://user:pass@localhost:5432/database",
            "query_timeout": 30,
            "pool_size": 5
        }
    },
    "mysql": {
        "name": "MySQL",
        "adapter_type": "sql",
        "config": {
            "connection_uri": "mysql://user:pass@localhost:3306/database",
            "query_timeout": 60
        }
    },
    "local_storage": {
        "name": "Local Storage",
        "adapter_type": "storage",
        "config": {
            "base_path": "./data/",
            "supported_formats": ["csv", "parquet", "json", "feather"]
        }
    },
    "s3_storage": {
        "name": "S3 Storage",
        "adapter_type": "storage",
        "config": {
            "base_path": "s3://my-bucket/data/",
            "aws_access_key_id": "your_access_key",
            "aws_secret_access_key": "your_secret_key",
            "region": "us-west-2"
        }
    }
}

FEAST_EXAMPLES = {
    "local_sqlite": {
        "provider": "feast",
        "feast_config": {
            "project": "local_feast_project",
            "registry": "./feast_repo/registry.db",
            "online_store": {
                "type": "sqlite",
                "path": "./feast_repo/online_store.db"
            },
            "offline_store": {
                "type": "file",
                "path": "./feast_repo/data"
            }
        }
    },
    "cloud_setup": {
        "provider": "feast",
        "feast_config": {
            "project": "production_feast",
            "registry": "s3://feast-bucket/registry.db",
            "online_store": {
                "type": "redis",
                "connection_string": "redis://redis-cluster:6379",
                "password": "redis_password"
            },
            "offline_store": {
                "type": "bigquery",
                "project_id": "my-gcp-project",
                "dataset_id": "feast_dataset"
            }
        }
    },
    "disabled": {
        "provider": "none"
    }
}