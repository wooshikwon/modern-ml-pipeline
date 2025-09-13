"""
Recipe Examples - Pydantic Model Example Data

Recipe 스키마에서 분리한 예제 데이터들을 제공합니다.
이전에 model_config에 하드코딩되어 있던 예제들을 체계적으로 관리합니다.
"""

from typing import Dict, Any

# Recipe 모델의 기본 예제 데이터 (원래 model_config에 있던 것)
RECIPE_CLASSIFICATION_EXAMPLE: Dict[str, Any] = {
    "name": "classification_rf",
    "task_choice": "classification",
    "model": {
        "class_path": "sklearn.ensemble.RandomForestClassifier",
        "library": "sklearn",
        "hyperparameters": {
            "tuning_enabled": True,
            "fixed": {
                "random_state": 42,
                "n_jobs": -1
            },
            "tunable": {
                "n_estimators": {
                    "type": "int",
                    "range": [50, 200]
                },
                "max_depth": {
                    "type": "int",
                    "range": [5, 20]
                }
            }
        }
    },
    "data": {
        "loader": {
            "source_uri": "sql/train_data.sql",
            "entity_schema": {
                "entity_columns": ["user_id"],
                "timestamp_column": "event_timestamp"
            }
        },
        "fetcher": {
            "type": "feature_store",
            "feature_views": {
                "user_features": {
                    "join_key": "user_id",
                    "features": ["age", "gender", "location"]
                }
            },
            "timestamp_column": "event_timestamp"
        },
        "data_interface": {
            "target_column": "label",
            "entity_columns": ["user_id"],
            "feature_columns": None
        },
        "split": {
            "train": 0.7,
            "validation": 0.2,
            "test": 0.1,
            "calibration": 0.0
        }
    },
    "preprocessor": {
        "steps": [
            {
                "type": "standard_scaler",
                "columns": ["age", "income"]
            },
            {
                "type": "one_hot_encoder",
                "columns": ["gender", "location"]
            }
        ]
    },
    "evaluation": {
        "metrics": ["accuracy", "f1", "roc_auc"],
        "validation": {
            "method": "train_test_split",
            "test_size": 0.2,
            "random_state": 42
        }
    },
    "metadata": {
        "author": "Data Scientist",
        "created_at": "2024-01-01 12:00:00",
        "description": "Random Forest classifier with Optuna tuning",
        "tuning_note": "Optuna will optimize n_estimators and max_depth"
    }
}

# 추가 예제 데이터들
RECIPE_REGRESSION_EXAMPLE: Dict[str, Any] = {
    "name": "regression_xgb",
    "task_choice": "regression",
    "model": {
        "class_path": "xgboost.XGBRegressor",
        "library": "xgboost",
        "hyperparameters": {
            "tuning_enabled": False,
            "values": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
        }
    },
    "data": {
        "loader": {
            "source_uri": "data/housing.csv"
        },
        "fetcher": {
            "type": "pass_through"
        },
        "data_interface": {
            "target_column": "price",
            "entity_columns": ["house_id"],
            "feature_columns": ["bedrooms", "bathrooms", "sqft_living", "sqft_lot"]
        },
        "split": {
            "train": 0.7,
            "validation": 0.2,
            "test": 0.1,
            "calibration": 0.0
        }
    },
    "evaluation": {
        "metrics": ["mae", "mse", "r2"],
        "validation": {
            "method": "cross_validation",
            "n_folds": 5,
            "random_state": 42
        }
    },
    "metadata": {
        "author": "ML Engineer",
        "created_at": "2024-01-02 10:30:00",
        "description": "XGBoost regression for house price prediction"
    }
}

RECIPE_CLUSTERING_EXAMPLE: Dict[str, Any] = {
    "name": "clustering_kmeans",
    "task_choice": "clustering",
    "model": {
        "class_path": "sklearn.cluster.KMeans",
        "library": "sklearn",
        "hyperparameters": {
            "tuning_enabled": True,
            "optimization_metric": "silhouette_score",
            "direction": "maximize",
            "n_trials": 50,
            "fixed": {
                "random_state": 42,
                "init": "k-means++",
                "n_init": 10
            },
            "tunable": {
                "n_clusters": {
                    "type": "int",
                    "range": [2, 10]
                }
            }
        }
    },
    "data": {
        "loader": {
            "source_uri": "data/customers.parquet"
        },
        "fetcher": {
            "type": "pass_through"
        },
        "data_interface": {
            "target_column": None,  # clustering has no target
            "entity_columns": ["customer_id"],
            "feature_columns": ["age", "income", "spending_score"]
        },
        "split": {
            "train": 1.0,
            "validation": 0.0,
            "test": 0.0,
            "calibration": 0.0
        }
    },
    "preprocessor": {
        "steps": [
            {
                "type": "standard_scaler"  # Global scaler, no columns specified
            }
        ]
    },
    "evaluation": {
        "metrics": ["silhouette_score", "calinski_harabasz_score"],
        "validation": {
            "method": "train_test_split",
            "test_size": 0.2,
            "random_state": 42
        }
    },
    "metadata": {
        "author": "Data Analyst",
        "created_at": "2024-01-03 14:15:00",
        "description": "Customer segmentation using K-Means clustering"
    }
}

# 모든 예제를 모은 딕셔너리
RECIPE_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "classification": RECIPE_CLASSIFICATION_EXAMPLE,
    "regression": RECIPE_REGRESSION_EXAMPLE,
    "clustering": RECIPE_CLUSTERING_EXAMPLE,
}

# 특정 컴포넌트별 예제들
HYPERPARAMETER_EXAMPLES = {
    "tuning_enabled": {
        "tuning_enabled": True,
        "optimization_metric": "f1_macro",
        "direction": "maximize",
        "n_trials": 100,
        "timeout": 3600,
        "fixed": {
            "random_state": 42,
            "n_jobs": -1
        },
        "tunable": {
            "n_estimators": {
                "type": "int",
                "range": [50, 200]
            },
            "max_depth": {
                "type": "int",
                "range": [5, 20]
            },
            "learning_rate": {
                "type": "float",
                "range": [0.01, 0.3]
            }
        }
    },
    "tuning_disabled": {
        "tuning_enabled": False,
        "values": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
}

PREPROCESSOR_EXAMPLES = {
    "full_pipeline": [
        {
            "type": "simple_imputer",
            "columns": ["age", "income"],
            "strategy": "mean",
            "create_missing_indicators": False
        },
        {
            "type": "standard_scaler",
            "columns": ["age", "income", "score"]
        },
        {
            "type": "one_hot_encoder",
            "columns": ["gender", "education"]
        },
        {
            "type": "polynomial_features",
            "columns": ["age", "income"],
            "degree": 2
        }
    ],
    "minimal_pipeline": [
        {
            "type": "standard_scaler"
        }
    ]
}

FEATURE_STORE_EXAMPLES = {
    "feast_configuration": {
        "type": "feature_store",
        "feature_views": {
            "user_features": {
                "join_key": "user_id",
                "features": ["age", "gender", "location", "registration_date"]
            },
            "transaction_features": {
                "join_key": "user_id",
                "features": ["total_spend", "transaction_count", "avg_order_value"]
            }
        },
        "timestamp_column": "event_timestamp"
    },
    "pass_through": {
        "type": "pass_through"
    }
}