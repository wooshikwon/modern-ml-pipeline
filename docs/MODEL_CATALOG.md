# Blueprint v17.0 Model Catalog 🎯

이 문서는 **Blueprint v17.0 "Automated Excellence Vision"**에서 제공하는 **23개 모델 패키지**의 완전한 카탈로그입니다. 모든 모델은 **자동화된 하이퍼파라미터 최적화**, **Data Leakage 방지**, **환경별 Feature Store 연결**을 지원합니다.

## 📊 분류 모델 (Classification) - 8개

### 1. Random Forest Classifier
- **파일**: `recipes/models/classification/random_forest_classifier.yaml`
- **class_path**: `sklearn.ensemble.RandomForestClassifier`
- **특징**: 앙상블 기반, 강력한 일반화 성능, 해석 가능성
- **최적화**: n_estimators, max_depth, min_samples_split/leaf, max_features
- **메트릭**: F1 Score (maximize)
- **trials**: 100회

### 2. Logistic Regression
- **파일**: `recipes/models/classification/logistic_regression.yaml`
- **class_path**: `sklearn.linear_model.LogisticRegression`
- **특징**: 간단하고 해석 가능한 선형 분류, 확률 출력
- **최적화**: C (정규화 강도), penalty (L1/L2/ElasticNet), solver
- **메트릭**: ROC AUC (maximize)
- **trials**: 50회

### 3. XGBoost Classifier
- **파일**: `recipes/models/classification/xgboost_classifier.yaml`
- **class_path**: `xgboost.XGBClassifier`
- **특징**: Gradient Boosting의 혁신, 뛰어난 성능과 효율성
- **최적화**: learning_rate, n_estimators, max_depth, subsample, colsample_bytree, regularization
- **메트릭**: ROC AUC (maximize)
- **trials**: 100회

### 4. LightGBM Classifier
- **파일**: `recipes/models/classification/lightgbm_classifier.yaml`
- **class_path**: `lightgbm.LGBMClassifier`
- **특징**: 메모리 효율적, 빠른 Gradient Boosting
- **최적화**: learning_rate, n_estimators, num_leaves, feature_fraction, bagging
- **메트릭**: F1 Score (maximize)
- **trials**: 80회

### 5. CatBoost Classifier
- **파일**: `recipes/models/classification/catboost_classifier.yaml`
- **class_path**: `catboost.CatBoostClassifier`
- **특징**: 범주형 피처 자동 처리, 높은 성능
- **최적화**: learning_rate, iterations, depth, l2_leaf_reg, border_count
- **메트릭**: ROC AUC (maximize)
- **trials**: 60회

### 6. Support Vector Machine
- **파일**: `recipes/models/classification/svm_classifier.yaml`
- **class_path**: `sklearn.svm.SVC`
- **특징**: 고차원 데이터에서 최적 결정 경계 탐색
- **최적화**: C, kernel (linear/poly/rbf/sigmoid), gamma, degree
- **메트릭**: Accuracy (maximize)
- **trials**: 30회

### 7. Gaussian Naive Bayes
- **파일**: `recipes/models/classification/naive_bayes.yaml`
- **class_path**: `sklearn.naive_bayes.GaussianNB`
- **특징**: 확률 기반, 간단하고 효과적
- **최적화**: var_smoothing (스무딩 파라미터)
- **메트릭**: F1 Score (maximize)
- **trials**: 20회

### 8. K-Nearest Neighbors
- **파일**: `recipes/models/classification/knn_classifier.yaml`
- **class_path**: `sklearn.neighbors.KNeighborsClassifier`
- **특징**: 거리 기반, 직관적이고 유연
- **최적화**: n_neighbors, weights, metric, algorithm
- **메트릭**: Accuracy (maximize)
- **trials**: 40회

---

## 📈 회귀 모델 (Regression) - 8개

### 1. Linear Regression
- **파일**: `recipes/models/regression/linear_regression.yaml`
- **class_path**: `sklearn.linear_model.LinearRegression`
- **특징**: 단순하고 해석 가능한 선형 관계 모델링
- **최적화**: fit_intercept, positive (제약)
- **메트릭**: R² Score (maximize)
- **trials**: 10회

### 2. Ridge Regression
- **파일**: `recipes/models/regression/ridge_regression.yaml`
- **class_path**: `sklearn.linear_model.Ridge`
- **특징**: L2 정규화로 과적합 방지
- **최적화**: alpha (정규화 강도), solver
- **메트릭**: R² Score (maximize)
- **trials**: 30회

### 3. Lasso Regression
- **파일**: `recipes/models/regression/lasso_regression.yaml`
- **class_path**: `sklearn.linear_model.Lasso`
- **특징**: L1 정규화로 자동 피처 선택
- **최적화**: alpha (정규화 강도), selection
- **메트릭**: R² Score (maximize)
- **trials**: 25회

### 4. Random Forest Regressor
- **파일**: `recipes/models/regression/random_forest_regressor.yaml`
- **class_path**: `sklearn.ensemble.RandomForestRegressor`
- **특징**: 앙상블 기반, 강력하고 안정적인 비선형 회귀
- **최적화**: n_estimators, max_depth, min_samples_split/leaf, max_features, bootstrap
- **메트릭**: RMSE (minimize)
- **trials**: 80회

### 5. XGBoost Regressor
- **파일**: `recipes/models/regression/xgboost_regressor.yaml`
- **class_path**: `xgboost.XGBRegressor`
- **특징**: Gradient Boosting의 회귀 특화, 뛰어난 성능
- **최적화**: learning_rate, n_estimators, max_depth, subsample, regularization
- **메트릭**: RMSE (minimize)
- **trials**: 100회

### 6. LightGBM Regressor
- **파일**: `recipes/models/regression/lightgbm_regressor.yaml`
- **class_path**: `lightgbm.LGBMRegressor`
- **특징**: 메모리 효율적, 빠른 Gradient Boosting 회귀
- **최적화**: learning_rate, n_estimators, num_leaves, feature_fraction, bagging
- **메트릭**: RMSE (minimize)
- **trials**: 80회

### 7. Support Vector Regressor
- **파일**: `recipes/models/regression/svr.yaml`
- **class_path**: `sklearn.svm.SVR`
- **특징**: 고차원 데이터에서 강건한 회귀 예측
- **최적화**: C, epsilon, kernel, gamma, degree
- **메트릭**: RMSE (minimize)
- **trials**: 30회

### 8. Elastic Net Regression
- **파일**: `recipes/models/regression/elastic_net.yaml`
- **class_path**: `sklearn.linear_model.ElasticNet`
- **특징**: L1과 L2 정규화 결합, 균형잡힌 선형 회귀
- **최적화**: alpha (정규화 강도), l1_ratio (L1/L2 비율), selection
- **메트릭**: R² Score (maximize)
- **trials**: 40회

---

## 🎯 클러스터링 모델 (Clustering) - 3개

### 1. K-Means Clustering
- **파일**: `recipes/models/clustering/kmeans.yaml`
- **class_path**: `sklearn.cluster.KMeans`
- **특징**: 중심점 기반, 직관적이고 효율적
- **최적화**: n_clusters, init, n_init, algorithm
- **메트릭**: Silhouette Score (maximize)
- **trials**: 30회

### 2. DBSCAN Clustering
- **파일**: `recipes/models/clustering/dbscan.yaml`
- **class_path**: `sklearn.cluster.DBSCAN`
- **특징**: 밀도 기반, 노이즈 제거 및 임의 형태 클러스터 발견
- **최적화**: eps (이웃 거리), min_samples, metric, algorithm
- **메트릭**: Calinski-Harabasz Score (maximize)
- **trials**: 50회

### 3. Hierarchical Clustering
- **파일**: `recipes/models/clustering/hierarchical_clustering.yaml`
- **class_path**: `sklearn.cluster.AgglomerativeClustering`
- **특징**: 계층적 구조 발견, 트리 기반
- **최적화**: n_clusters, linkage, metric
- **메트릭**: Silhouette Score (maximize)
- **trials**: 25회

---

## 🎭 인과추론 모델 (Causal Inference) - 4개

### 1. Causal Random Forest
- **파일**: `recipes/models/causal/causal_random_forest.yaml`
- **class_path**: `causalml.inference.tree.CausalRandomForestRegressor`
- **특징**: 트리 기반 강력한 업리프트 모델링
- **최적화**: n_estimators, max_depth, min_samples_split/leaf, max_features
- **메트릭**: Uplift AUC (maximize)
- **trials**: 80회

### 2. XGBoost T-Learner
- **파일**: `recipes/models/causal/xgb_t_learner.yaml`
- **class_path**: `causalml.inference.meta.XGBTRegressor`
- **특징**: Meta-Learning 기반 XGBoost 고성능 인과효과 추정
- **최적화**: learning_rate, n_estimators, max_depth, subsample, regularization
- **메트릭**: Uplift AUC (maximize)
- **trials**: 100회

### 3. S-Learner (Single Model)
- **파일**: `recipes/models/causal/s_learner.yaml`
- **class_path**: `causalml.inference.meta.SRegressor`
- **특징**: 단일 모델로 처리군과 대조군 함께 학습
- **최적화**: n_estimators, max_depth, min_samples_split/leaf, max_features
- **메트릭**: Uplift AUC (maximize)
- **trials**: 60회

### 4. T-Learner (Two Model)
- **파일**: `recipes/models/causal/t_learner.yaml`
- **class_path**: `causalml.inference.meta.TRegressor`
- **특징**: 처리군과 대조군을 별도 모델로 학습하는 클래식 인과추론
- **최적화**: n_estimators, max_depth, min_samples_split/leaf, max_features, bootstrap
- **메트릭**: Uplift AUC (maximize)
- **trials**: 70회

---

## 🚀 모든 모델의 공통 특징

### Blueprint v17.0 핵심 기능
- ✅ **자동화된 하이퍼파라미터 최적화**: Optuna 기반 범위 탐색
- ✅ **Data Leakage 완전 방지**: Train-only preprocessing fit
- ✅ **환경별 Feature Store 연결**: 동적 피처 증강
- ✅ **완전한 재현성**: 모든 최적화 과정 추적 및 저장
- ✅ **100% 하위 호환성**: 기존 코드 변경 없이 점진적 활성화

### 사용법 예시
```bash
# 임의의 모델로 자동 최적화 학습
python main.py train --recipe-file "models/classification/xgboost_classifier"

# 자동 최적화 비활성화 (기존 방식)
# recipe 파일에서 hyperparameter_tuning.enabled: false로 설정

# 배치 추론 (최적 하이퍼파라미터 자동 적용)
python main.py batch-inference --run-id "abc123"

# API 서빙 (최적화된 모델로 실시간 서빙)
python main.py serve-api --run-id "abc123"
```

---

**🎉 이제 23개의 다양한 모델 패키지로 무제한적인 실험을 즐기세요!** 