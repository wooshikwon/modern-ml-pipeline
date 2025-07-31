# src/settings/compatibility_maps.py
"""
시스템 내의 다양한 컴포넌트 간 호환성 정보를 정의하는 중앙 저장소.
"""

TASK_METRIC_COMPATIBILITY = {
    "classification": [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
    ],
    "regression": ["mse", "rmse", "mae", "r2"],
    "clustering": ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"],
    "causal": ["uplift_at_k", "qini_auc_score", "auuc_score"],
} 