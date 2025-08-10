# 🎯 Model Catalog

이 문서는 프로젝트에 포함된 대표 레시피들의 카탈로그입니다. 모든 모델은 공통적으로 다음 원칙을 따릅니다.

- 선언적 레시피: `class_path` 직접 임포트, `hyperparameters` 사전/탐색 공간 지원
- 데이터 인터페이스: `data_interface.task_type`, `data_interface.target_column` 등 표준화
- 보안/정책: SQL SELECT * 금지, DDL/DML 금칙어 차단, Jinja 허용 키 화이트리스트
- 재현성/튜닝: `set_global_seeds`, Optuna 기반 HPO(선택)
- 서빙 정책: API 서빙은 Feature Store 기반 Augmenter가 필수이며 `MinimalPredictionResponse` 사용

## 📊 분류 (Classification)

### Random Forest Classifier
- 파일: `recipes/models/classification/random_forest_classifier.yaml`
- class_path: `sklearn.ensemble.RandomForestClassifier`
- 주요 탐색: `n_estimators`, `max_depth`, `min_samples_split/leaf`, `max_features`
- 권장 메트릭: F1/ROC AUC

### Logistic Regression
- 파일: `recipes/models/classification/logistic_regression.yaml`
- class_path: `sklearn.linear_model.LogisticRegression`
- 주요 탐색: `C`, `penalty`, `solver`, `l1_ratio`
- 권장 메트릭: ROC AUC

### XGBoost Classifier
- 파일: `recipes/models/classification/xgboost_classifier.yaml`
- class_path: `xgboost.XGBClassifier`
- 주요 탐색: `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`
- 권장 메트릭: ROC AUC

### LightGBM Classifier
- 파일: `recipes/models/classification/lightgbm_classifier.yaml`
- class_path: `lightgbm.LGBMClassifier`
- 권장 메트릭: F1/ROC AUC

### CatBoost Classifier
- 파일: `recipes/models/classification/catboost_classifier.yaml`
- class_path: `catboost.CatBoostClassifier`
- 권장 메트릭: ROC AUC

### SVM Classifier
- 파일: `recipes/models/classification/svm_classifier.yaml`
- class_path: `sklearn.svm.SVC`
- 권장 메트릭: Accuracy/F1

### Gaussian Naive Bayes
- 파일: `recipes/models/classification/naive_bayes.yaml`
- class_path: `sklearn.naive_bayes.GaussianNB`
- 권장 메트릭: F1

### KNN Classifier
- 파일: `recipes/models/classification/knn_classifier.yaml`
- class_path: `sklearn.neighbors.KNeighborsClassifier`
- 권장 메트릭: Accuracy

---

## 📈 회귀 (Regression)

### Linear Regression
- 파일: `recipes/models/regression/linear_regression.yaml`
- class_path: `sklearn.linear_model.LinearRegression`
- 권장 메트릭: R²/MAE

### Ridge Regression
- 파일: `recipes/models/regression/ridge_regression.yaml`
- class_path: `sklearn.linear_model.Ridge`
- 권장 메트릭: R²/MAE

### Lasso Regression
- 파일: `recipes/models/regression/lasso_regression.yaml`
- class_path: `sklearn.linear_model.Lasso`
- 권장 메트릭: R²/MAE

### Random Forest Regressor
- 파일: `recipes/models/regression/random_forest_regressor.yaml`
- class_path: `sklearn.ensemble.RandomForestRegressor`
- 권장 메트릭: RMSE/MAE

### XGBoost Regressor
- 파일: `recipes/models/regression/xgboost_regressor.yaml`
- class_path: `xgboost.XGBRegressor`
- 권장 메트릭: RMSE/MAE

### LightGBM Regressor
- 파일: `recipes/models/regression/lightgbm_regressor.yaml`
- class_path: `lightgbm.LGBMRegressor`
- 권장 메트릭: RMSE/MAE

### SVR
- 파일: `recipes/models/regression/svr.yaml`
- class_path: `sklearn.svm.SVR`
- 권장 메트릭: RMSE/MAE

### Elastic Net
- 파일: `recipes/models/regression/elastic_net.yaml`
- class_path: `sklearn.linear_model.ElasticNet`
- 권장 메트릭: R²/MAE

---

## 🎯 클러스터링 (Clustering)

### KMeans
- 파일: `recipes/models/clustering/kmeans.yaml`
- class_path: `sklearn.cluster.KMeans`
- 권장 메트릭: Silhouette Score

### DBSCAN
- 파일: `recipes/models/clustering/dbscan.yaml`
- class_path: `sklearn.cluster.DBSCAN`
- 권장 메트릭: Calinski-Harabasz

### Hierarchical Clustering
- 파일: `recipes/models/clustering/hierarchical_clustering.yaml`
- class_path: `sklearn.cluster.AgglomerativeClustering`
- 권장 메트릭: Silhouette Score

---

## 🎭 인과추론 (Causal Inference)

### Causal Random Forest
- 파일: `recipes/models/causal/causal_random_forest.yaml`
- class_path: `causalml.inference.tree.CausalRandomForestRegressor`
- 권장 메트릭: Uplift AUC

### XGBoost T-Learner
- 파일: `recipes/models/causal/xgb_t_learner.yaml`
- class_path: `causalml.inference.meta.XGBTRegressor`
- 권장 메트릭: Uplift AUC

### S-Learner
- 파일: `recipes/models/causal/s_learner.yaml`
- class_path: `causalml.inference.meta.SRegressor`
- 권장 메트릭: Uplift AUC

### T-Learner
- 파일: `recipes/models/causal/t_learner.yaml`
- class_path: `causalml.inference.meta.TRegressor`
- 권장 메트릭: Uplift AUC

---

## 🚀 사용법 요약

```bash
# 학습
uv run python main.py train --recipe-file recipes/models/classification/xgboost_classifier.yaml

# 배치 추론 (모델 run_id 사용)
uv run python main.py batch-inference --run-id <RUN_ID>

# API 서빙 (Feature Store + serving.enabled: true 필요)
uv run python main.py serve-api --run-id <RUN_ID>
```

API `/predict` 응답은 다음 최소 스키마를 따릅니다.
```json
{"prediction": <value>, "model_uri": "runs:/<RUN_ID>/model"}
``` 