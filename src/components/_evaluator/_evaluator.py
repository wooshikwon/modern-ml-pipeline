import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Clustering metrics
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)

from src.interface import BaseEvaluator
from src.engine import EvaluatorRegistry
from src.utils.system.logger import logger

from ._classification import ClassificationEvaluator
from ._regression import RegressionEvaluator
from ._clustering import ClusteringEvaluator
from ._causal import CausalEvaluator

# 각 Evaluator를 task_type에 따라 레지스트리에 자동 등록
EvaluatorRegistry.register("classification", ClassificationEvaluator)
EvaluatorRegistry.register("regression", RegressionEvaluator)
EvaluatorRegistry.register("clustering", ClusteringEvaluator)
EvaluatorRegistry.register("causal", CausalEvaluator) 