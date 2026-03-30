# AGENT.md -- LLM/AI Agent Guide for mmp

## 1. Quick Reference

mmp (Modern ML Pipeline) is a YAML-driven CLI tool for config-driven ML pipelines. It separates infrastructure configuration (Config YAML) from experiment definition (Recipe YAML). Available commands: `init`, `get-config`, `get-recipe`, `train`, `batch-inference`, `serve-api`, `validate`, `list`. To run a pipeline, you need exactly one Config file and one Recipe file.

## 2. Recipe YAML Schema

The Recipe defines the ML experiment: what model to train, what data to use, how to preprocess, and how to evaluate. The top-level Pydantic model is `Recipe`.

### Top-Level Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | **(required)** | -- | Unique recipe name |
| `task_choice` | str | **(required)** | -- | Task type. Must match a catalog subdirectory (case-insensitive): `classification`, `regression`, `timeseries`, `clustering`, `causal` |
| `model` | Model | **(required)** | -- | Model specification (see below) |
| `data` | Data | **(required)** | -- | Data loading and splitting (see below) |
| `preprocessor` | Preprocessor | (optional, default: null) | null | Preprocessing pipeline |
| `evaluation` | Evaluation | (optional) | null | Evaluation metrics and random seed |
| `metadata` | Metadata | **(required)** | -- | Description and authorship |
| `monitoring` | MonitoringConfig | (optional, default: null) | null | Data drift and prediction drift monitoring |

### model (Model)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `class_path` | str | **(required)** | -- | Full Python class path (e.g., `lightgbm.LGBMRegressor`). Must exist in the catalog for the chosen task. |
| `library` | str | **(required)** | -- | Library name (e.g., `lightgbm`, `sklearn`, `xgboost`, `catboost`, `causalml`, `pytorch`, `statsmodels`, `rtdl_revisiting_models`, `pytorch-tabnet`, `mmp-custom`). This field is required and causes validation errors if omitted. |
| `hyperparameters` | HyperparametersTuning | **(required)** | -- | Hyperparameter configuration (see below) |
| `calibration` | Calibration | (optional, default: null) | null | Calibration settings (classification only) |
| `computed` | dict | (optional, default: {}) | {} | Runtime-computed fields. Do not set manually. |

### model.hyperparameters (HyperparametersTuning)

**CRITICAL: The key structure changes depending on `tuning_enabled`.**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `tuning_enabled` | bool | (optional, default: false) | false | Enable Optuna hyperparameter tuning |
| `optimization_metric` | str | (optional, default: null) | null | Metric to optimize during tuning |
| `n_trials` | int | (optional, default: null) | null | Number of Optuna trials |
| `timeout` | int | (optional, default: null) | null | Tuning timeout in seconds |
| `fixed` | dict | (optional, default: null) | null | Fixed parameters (used ONLY when `tuning_enabled: true`, alongside `tunable`) |
| `tunable` | dict | (optional, default: null) | null | Tunable parameter search space (used ONLY when `tuning_enabled: true`) |
| `values` | dict | (optional, default: null) | null | Static hyperparameter values (used ONLY when `tuning_enabled: false`) |

**When `tuning_enabled: false` (default):**
- Put ALL hyperparameters under `values`.
- Do NOT use `fixed`. The validator requires `values` to be non-null.

```yaml
hyperparameters:
  tuning_enabled: false
  values:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
```

**When `tuning_enabled: true`:**
- Put parameters you do NOT want tuned under `fixed`.
- Put parameters you want Optuna to search under `tunable`, each with `type` and `range`.
- `values` is ignored.
- `optimization_metric`, `n_trials` should also be set.
- Validation split must be > 0.

```yaml
hyperparameters:
  tuning_enabled: true
  optimization_metric: rmse
  n_trials: 50
  timeout: 600
  fixed:
    n_estimators: 500
    verbose: -1
  tunable:
    learning_rate:
      type: float
      range: [0.01, 0.3]
    max_depth:
      type: int
      range: [3, 10]
```

### model.calibration (Calibration)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `enabled` | bool | (optional, default: false) | false | Enable calibration |
| `method` | str | (optional, default: null) | null | Calibration method. Only for classification tasks. |

### data (Data)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `loader` | Loader | **(required)** | -- | Data source URI |
| `fetcher` | Fetcher | **(required)** | -- | Data fetching strategy |
| `data_interface` | DataInterface | **(required)** | -- | Column role definitions |
| `split` | DataSplit | **(required)** | -- | Train/test/validation ratios |

### data.loader (Loader)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source_uri` | str | (optional, default: null) | null | Path to data file or SQL query file. Examples: `./data/train.csv`, `./queries/fetch_data.sql` |

### data.fetcher (Fetcher)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `type` | str | **(required)** | -- | Fetcher type: `pass_through` (direct data load) or `feature_store` (Feast-based). **Always include this field.** When not using Feature Store, set `type: pass_through`. |
| `feature_views` | dict | (optional, default: null) | null | Feature View configuration (only for `feature_store` type) |
| `timestamp_column` | str | (optional, default: null) | null | Timestamp column for point-in-time joins (only for `feature_store` type) |

### data.data_interface (DataInterface)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `target_column` | str | (optional, default: null) | null | Target column name. **Required for all tasks except clustering.** |
| `treatment_column` | str | (optional, default: null) | null | Treatment column. **Required for causal task.** |
| `timestamp_column` | str | (optional, default: null) | null | Timestamp column. **Required for timeseries task.** |
| `entity_columns` | list[str] | **(required)** | -- | Entity/ID columns (excluded from features). Must be a list. |
| `feature_columns` | list[str] | (optional, default: null) | null | Explicit feature column list. If null, all columns not in entity/target/treatment/timestamp are used. |
| `sequence_length` | int | (optional, default: null) | null | Sequence length for LSTM timeseries models. |

### data.split (DataSplit)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `strategy` | str | (optional) | "random" | Split strategy: `random` (sklearn shuffle) or `temporal` (time-ordered, no future leakage) |
| `temporal_column` | str | required if strategy=temporal | null | Column name to sort by for temporal split (e.g., `created_at`) |
| `train` | float | **(required)** | -- | Training data ratio (e.g., 0.7) |
| `test` | float | **(required)** | -- | Test data ratio (e.g., 0.15) |
| `validation` | float | **(required)** | -- | Validation data ratio (e.g., 0.15). **Must be > 0 when `tuning_enabled: true`.** |
| `calibration` | float | (optional, default: null) | null | Calibration data ratio (only when calibration is enabled) |

### preprocessor (Preprocessor)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `steps` | list[PreprocessorStep] | **(required)** | -- | Ordered list of preprocessing steps |

### preprocessor.steps[] (PreprocessorStep)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `type` | str | **(required)** | -- | Preprocessor type name (must be a registered name, see Section 4) |
| `columns` | list[str] | (optional, default: null) | null | Target columns. Null means auto-detect applicable columns. |
| `strategy` | str | (optional, default: null) | null | Strategy for imputer/discretizer |
| `degree` | int | (optional, default: null) | null | Polynomial degree for polynomial_features |
| `n_bins` | int | (optional, default: null) | null | Number of bins for kbins_discretizer |
| `create_missing_indicators` | bool | (optional, default: null) | null | Create missing indicator columns (for simple_imputer) |
| `handle_unknown` | str | (optional, default: null) | null | How to handle unknown categories (for encoders) |
| `unknown_value` | any | (optional, default: null) | null | Value for unknown categories (for ordinal_encoder with `handle_unknown: use_encoded_value`) |

### evaluation (Evaluation)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `metrics` | list[str] | (optional, default: null) | null | Metric names. If null, task-specific defaults are used. |
| `random_state` | int | (optional, default: 42) | 42 | Random seed for reproducibility |

**Quantile regression metrics**: When the model outputs `pred_pN` columns (e.g., `pred_p50`, `pred_p99`), the evaluator automatically computes per-quantile metrics logged to MLflow:

| Metric | Description |
|---|---|
| `coverage_rate_p<N>` | Fraction of actuals ≤ prediction at quantile N (calibration check: p90 should be ~0.90) |
| `pinball_loss_p<N>` | Pinball (quantile) loss at quantile N |
| `mae_p<N>` | Mean absolute error at quantile N |
| `mean_pred_p<N>` | Mean prediction value at quantile N |
| `mean_pinball_loss` | Average pinball loss across all quantiles |
| `interval_coverage` | Fraction of actuals within [lowest, highest] quantile interval |
| `r2_score`, `mse`, `rmse`, `mae` | Standard regression metrics on p50 (if available) |

### metadata (Metadata)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `author` | str | (optional, default: "CLI Recipe Builder") | "CLI Recipe Builder" | Author name |
| `description` | str | **(required)** | -- | Recipe description. Causes validation error if omitted. |
| `tuning_note` | str | (optional, default: null) | null | Notes about tuning configuration |

### monitoring (MonitoringConfig)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `enabled` | bool | (optional, default: false) | false | Enable monitoring |
| `data_drift` | DataDriftConfig | (optional) | defaults | PSI thresholds for data drift detection |
| `prediction_drift` | PredictionDriftConfig | (optional) | defaults | PSI thresholds for prediction drift detection |

## 3. Config YAML Schema

The Config defines infrastructure: where data comes from, where results go, MLflow tracking, and serving settings. The top-level Pydantic model is `Config`.

### Top-Level Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `environment` | Environment | **(required)** | -- | Environment metadata |
| `mlflow` | MLflow | (optional, default: null) | null | MLflow tracking settings |
| `data_source` | DataSource | **(required)** | -- | Data source adapter |
| `feature_store` | FeatureStore | **(required)** | -- | Feature Store settings |
| `serving` | Serving | (optional, default: null) | null | API serving settings |
| `output` | Output | **(required)** | -- | Inference output destination |
| `logging` | Logging | (optional) | defaults | Logging configuration |

### environment (Environment)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | **(required)** | -- | Environment name (e.g., `local`, `dev`, `prod`) |
| `description` | str | (optional, default: null) | null | Environment description |

### mlflow (MLflow)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `tracking_uri` | str | **(required)** | -- | MLflow tracking URI. Use `./mlruns` for local directory-based tracking. Do NOT use `sqlite:///mlruns.db`. |
| `experiment_name` | str | **(required)** | -- | Experiment name |
| `tracking_username` | str | (optional, default: null) | null | MLflow auth username |
| `tracking_password` | str | (optional, default: null) | null | MLflow auth password |

### data_source (DataSource)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | **(required)** | -- | Data source name |
| `adapter_type` | str | **(required)** | -- | Adapter type: `local_files`, `bigquery`, `postgresql`, `s3`, `gcs` |
| `config` | varies | **(required)** | -- | Adapter-specific config (see sub-types below) |

**data_source.config variants by adapter_type:**

**LocalFilesConfig** (for `adapter_type: local_files`):

| Field | Type | Required | Default |
|---|---|---|---|
| `base_path` | str | **(required)** | -- |
| `storage_options` | dict | (optional, default: {}) | {} |

**BigQueryConfig** (for `adapter_type: bigquery`):

| Field | Type | Required | Default |
|---|---|---|---|
| `connection_uri` | str | **(required)** | -- |
| `project_id` | str | **(required)** | -- |
| `dataset_id` | str | **(required)** | -- |
| `location` | str | (optional, default: "US") | "US" |
| `use_pandas_gbq` | bool | (optional, default: true) | true |
| `query_timeout` | int | (optional, default: 300) | 300 |

**PostgreSQLConfig** (for `adapter_type: postgresql`):

| Field | Type | Required | Default |
|---|---|---|---|
| `connection_uri` | str | **(required)** | -- |
| `query_timeout` | int | (optional, default: 300) | 300 |

**S3Config** (for `adapter_type: s3`):

| Field | Type | Required | Default |
|---|---|---|---|
| `base_path` | str | **(required)** | -- |
| `storage_options` | S3StorageOptions | **(required)** | -- |

**GCSConfig** (for `adapter_type: gcs`):

| Field | Type | Required | Default |
|---|---|---|---|
| `base_path` | str | **(required)** | -- |
| `storage_options` | GCSStorageOptions | **(required)** | -- |

### feature_store (FeatureStore)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `provider` | str | (optional, default: "none") | "none" | Feature Store provider: `none` or `feast` |
| `enabled` | bool | (optional, default: false) | false | Enable Feature Store |
| `feast_config` | FeastConfig | (optional, default: null) | null | Feast configuration (required when provider is `feast`) |

### output (Output)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `inference` | OutputTarget | **(required)** | -- | Inference result output target |

### output.inference (OutputTarget)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | str | **(required)** | -- | Output target name |
| `enabled` | bool | (optional, default: true) | true | Enable this output |
| `adapter_type` | str | **(required)** | -- | Output adapter type: `storage`, `sql`, `bigquery` |
| `config` | varies | **(required)** | -- | Adapter-specific output config |

**output.inference.config variants by adapter_type:**

**StorageOutputConfig** (for `adapter_type: storage`):

| Field | Type | Required | Default |
|---|---|---|---|
| `base_path` | str | **(required)** | -- |
| `file_name` | str | (optional, default: null) | null (defaults to `predictions_{run_id}`) |
| `file_format` | str | (optional, default: "parquet") | "parquet". One of: `parquet`, `csv`, `json` |
| `storage_options` | dict | (optional, default: {}) | {} |

**BigQueryOutputConfig** (for `adapter_type: bigquery`):

| Field | Type | Required | Default |
|---|---|---|---|
| `connection_uri` | str | **(required)** | -- |
| `project_id` | str | **(required)** | -- |
| `dataset_id` | str | **(required)** | -- |
| `table` | str | **(required)** | -- |
| `location` | str | (optional, default: "US") | "US" |
| `use_pandas_gbq` | bool | (optional, default: true) | true |

### serving (Serving)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `enabled` | bool | (optional, default: false) | false | Enable serving |
| `host` | str | (optional, default: "0.0.0.0") | "0.0.0.0" | Serving host |
| `port` | int | (optional, default: 8000) | 8000 | Serving port |
| `workers` | int | (optional, default: 1) | 1 | Number of workers |
| `model_stage` | str | (optional, default: null) | null | Model stage |
| `auth` | AuthConfig | (optional, default: null) | null | Authentication settings |
| `cors` | CORSConfig | (optional, default: null) | null | CORS settings |
| `request_timeout_seconds` | int | (optional, default: 30) | 30 | Request timeout |
| `metrics_enabled` | bool | (optional, default: true) | true | Prometheus metrics |

### logging (Logging)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_path` | str | (optional, default: "./logs") | "./logs" | Log storage path |
| `level` | str | (optional, default: "INFO") | "INFO" | Log level: DEBUG, INFO, WARNING, ERROR |
| `retention_days` | int | (optional, default: 30) | 30 | Log file retention days |
| `upload_to_mlflow` | bool | (optional, default: true) | true | Upload logs to MLflow artifacts |

## 4. Available Preprocessors

All preprocessor types are registered in `PreprocessorStepRegistry`. Use the exact `type` string in Recipe YAML.

Recommended step ordering: missing handling -> encoding -> feature generation -> scaling.

### Missing Value Handlers

| type | Parameters | Description | Use When |
|---|---|---|---|
| `simple_imputer` | `strategy` (str, default: "mean": "mean", "median", "most_frequent", "constant"), `columns` (list[str]), `create_missing_indicators` (bool, default: false) | Replaces missing values using the chosen strategy. Optionally creates binary indicator columns for missingness. | You have missing numeric values and want statistical imputation. |
| `drop_missing` | `axis` (str, default: "rows": "rows" or "columns"), `threshold` (float, default: 0.0: fraction of missingness above which to drop), `columns` (list[str]) | Drops rows or columns exceeding the missing threshold. | You want to remove rows/columns with too many missing values. |
| `forward_fill` | `limit` (int, optional), `columns` (list[str]) | Propagates the last valid observation forward. | Time series data where previous values are a reasonable fill. |
| `backward_fill` | `limit` (int, optional), `columns` (list[str]) | Propagates the next valid observation backward. | Time series data where future values are a reasonable fill. |
| `constant_fill` | `fill_value` (any or dict, default: 0), `columns` (list[str]) | Fills missing values with a constant. `fill_value` can be a single value or a dict mapping column names to values. | You know the correct replacement value (e.g., 0 for counts). |
| `interpolation` | `method` (str, default: "linear": "linear", "polynomial", "spline", etc.), `order` (int, optional: for polynomial/spline), `limit` (int, optional), `columns` (list[str]) | Interpolates missing values using pandas interpolation methods. | Numeric time series with gaps that should follow a smooth curve. |

### Encoders

| type | Parameters | Description | Use When |
|---|---|---|---|
| `one_hot_encoder` | `handle_unknown` (str, default: "ignore": "error", "ignore", "infrequent_if_exist"), `sparse_output` (bool, default: false), `columns` (list[str]) | Creates binary columns for each category. Column names are NOT preserved (new columns generated). | Low-cardinality categorical features. |
| `ordinal_encoder` | `handle_unknown` (str, default: "error": "error" or "use_encoded_value"), `unknown_value` (any, default: null: required when `handle_unknown: use_encoded_value`), `columns` (list[str]) | Maps categories to integers. Column names are preserved. **Set `handle_unknown: use_encoded_value` and `unknown_value: -1` for production use**, otherwise unseen categories at inference time will raise an error. | Ordinal categorical features, or tree-based models that handle integer-encoded categories well. |
| `catboost_encoder` | `sigma` (float, default: 0.05), `columns` (list[str]) | Target-based encoding (CatBoost-style). Requires target variable y during fit (supervised). Column names are preserved. Prevents target leakage via ordered encoding. | High-cardinality categorical features (e.g., zone IDs, user IDs) where one-hot would create too many columns. Preferred over passing raw numeric IDs as features. |

### Feature Generators

| type | Parameters | Description | Use When |
|---|---|---|---|
| `polynomial_features` | `degree` (int, default: 2), `include_bias` (bool, default: false), `interaction_only` (bool, default: false), `columns` (list[str]) | Creates polynomial and interaction terms. Column names are NOT preserved. | You want to capture non-linear relationships in linear models. |
| `tree_based_feature_generator` | `n_estimators` (int, default: 10), `max_depth` (int, default: 3), `random_state` (int, default: 42), `columns` (list[str]) | Uses a Random Forest to generate leaf-node one-hot features. Requires target y (supervised). Column names are NOT preserved. | You want to automatically capture feature interactions. |

### Discretizers

| type | Parameters | Description | Use When |
|---|---|---|---|
| `kbins_discretizer` | `n_bins` (int, default: 5), `encode` (str, default: "ordinal": "ordinal", "onehot", "onehot-dense"), `strategy` (str, default: "quantile": "uniform", "quantile", "kmeans"), `columns` (list[str]) | Bins continuous features into discrete intervals. Column names are NOT preserved. | You want to discretize continuous features for models that handle categorical data better. |

### Scalers

Scalers are "global" type -- they auto-apply to all numeric columns regardless of the `columns` parameter.

| type | Parameters | Description | Use When |
|---|---|---|---|
| `standard_scaler` | `columns` (list[str], ignored) | Zero mean, unit variance. Skips constant and all-NaN columns. | Default choice for most numeric features. |
| `min_max_scaler` | `columns` (list[str], ignored) | Scales to [0, 1] range. Skips constant and all-NaN columns. | Features need bounded range (e.g., neural networks). |
| `robust_scaler` | `columns` (list[str], ignored) | Uses median and IQR. Less sensitive to outliers. | Data contains significant outliers. |

## 5. Available Models by Task

Each model is defined in the catalog at `mmp/models/catalog/{Task}/{ModelName}.yaml`. The `class_path` and `library` values below are the exact strings to use in Recipe YAML.

### Classification

| class_path | library |
|---|---|
| `lightgbm.LGBMClassifier` | `lightgbm` |
| `xgboost.XGBClassifier` | `xgboost` |
| `catboost.CatBoostClassifier` | `catboost` |
| `sklearn.ensemble.RandomForestClassifier` | `scikit-learn` |
| `sklearn.linear_model.LogisticRegression` | `scikit-learn` |
| `sklearn.svm.SVC` | `scikit-learn` |
| `sklearn.neighbors.KNeighborsClassifier` | `scikit-learn` |
| `sklearn.naive_bayes.GaussianNB` | `scikit-learn` |
| `mmp.models.custom.ft_transformer.FTTransformerClassifier` | `rtdl_revisiting_models` |
| `mmp.models.custom.tabnet_wrapper.TabNetClassifierWrapper` | `pytorch-tabnet` |

### Regression

| class_path | library |
|---|---|
| `lightgbm.LGBMRegressor` | `lightgbm` |
| `xgboost.XGBRegressor` | `xgboost` |
| `sklearn.ensemble.RandomForestRegressor` | `scikit-learn` |
| `sklearn.linear_model.LinearRegression` | `scikit-learn` |
| `sklearn.linear_model.Ridge` | `scikit-learn` |
| `sklearn.linear_model.Lasso` | `scikit-learn` |
| `sklearn.linear_model.ElasticNet` | `scikit-learn` |
| `sklearn.svm.SVR` | `scikit-learn` |
| `sklearn.neighbors.KNeighborsRegressor` | `scikit-learn` |
| `mmp.models.custom.ft_transformer.FTTransformerRegressor` | `rtdl_revisiting_models` |
| `mmp.models.custom.tabnet_wrapper.TabNetRegressorWrapper` | `pytorch-tabnet` |
| `rtdl_revisiting_models.ResNetRegressor` | `rtdl_revisiting_models` |
| `mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble` | `mmp-custom` |

### Timeseries

| class_path | library |
|---|---|
| `mmp.models.custom.timeseries_wrappers.ARIMA` | `statsmodels` |
| `mmp.models.custom.timeseries_wrappers.ExponentialSmoothing` | `statsmodels` |
| `mmp.models.custom.lstm_timeseries.LSTMTimeSeries` | `pytorch` |
| `sklearn.linear_model.LinearRegression` | `scikit-learn` |

### Clustering

| class_path | library |
|---|---|
| `sklearn.cluster.KMeans` | `scikit-learn` |
| `sklearn.cluster.Birch` | `scikit-learn` |
| `sklearn.mixture.GaussianMixture` | `scikit-learn` |

### Causal

| class_path | library |
|---|---|
| `causalml.inference.tree.CausalRandomForestRegressor` | `causalml` |
| `causalml.inference.meta.LRSRegressor` | `causalml` |
| `causalml.inference.meta.XGBTRegressor` | `causalml` |

### QuantileRegressorEnsemble (Custom Model)

This model trains separate quantile regression models for each specified quantile. It wraps any gradient boosting library (lightgbm, xgboost, catboost).

Special `values` parameters (when `tuning_enabled: false`):
- `base_class_path` **(required)**: The base model class path (e.g., `lightgbm.LGBMRegressor`)
- `quantiles` **(required)**: List of quantile values (e.g., `[0.1, 0.5, 0.9]`)
- Any additional keys are passed as hyperparameters to each base model instance

```yaml
model:
  class_path: mmp.models.custom.quantile_ensemble.QuantileRegressorEnsemble
  library: mmp-custom
  hyperparameters:
    tuning_enabled: false
    values:
      base_class_path: lightgbm.LGBMRegressor
      quantiles: [0.1, 0.5, 0.9]
      n_estimators: 300
      learning_rate: 0.05
```

## 6. Validation Rules

Three validators run during `mmp validate` and before `mmp train`.

### CatalogValidator

- Checks that `task_choice` matches a known catalog subdirectory.
- Checks that `model.class_path` exists in the catalog for the given task.
- When `tuning_enabled: true`: requires `tunable` to be non-empty.
- When `tuning_enabled: false`: requires `values` to be non-empty.
- Validates that tunable parameter ranges fall within catalog-defined limits.

### BusinessValidator

- Checks that every `preprocessor.steps[].type` is a registered preprocessor name.
- Warns if preprocessor step ordering deviates from the recommended order: missing -> encoder -> feature_gen -> scaler.
- Validates calibration is only used with classification tasks.
- Validates evaluation metrics exist for the chosen task type.
- When `tuning_enabled: true`: requires `data.split.validation > 0`.
- Task-specific data_interface validation:
  - All tasks except clustering: `target_column` is required.
  - Causal: `treatment_column` is required.
  - Timeseries: `timestamp_column` is required.

### CompatibilityValidator

- If Recipe fetcher is `feature_store`, Config must have a non-"none" feature_store provider with valid feast_config.
- Validates data source adapter type matches the source_uri pattern (SQL vs storage vs BigQuery).
- Warns if MLflow is not configured.

## 7. Common Pitfalls

1. **Missing `model.library` field.** This is a required string on the Model schema. Omitting it causes a Pydantic validation error. Always include it with the exact value from the catalog (e.g., `lightgbm`, `scikit-learn`, `xgboost`).

2. **Missing `data.fetcher` field.** Even when not using Feature Store, the fetcher is required. Use `{type: pass_through}` for direct data loading.

3. **Using `fixed` instead of `values` when tuning is disabled.** When `tuning_enabled: false`, put hyperparameters under `values`, not `fixed`. The CatalogValidator explicitly checks for this.

4. **Using non-existent preprocessor names.** The only valid registered type names are: `simple_imputer`, `drop_missing`, `forward_fill`, `backward_fill`, `constant_fill`, `interpolation`, `ordinal_encoder`, `one_hot_encoder`, `catboost_encoder`, `polynomial_features`, `tree_based_feature_generator`, `kbins_discretizer`, `standard_scaler`, `min_max_scaler`, `robust_scaler`. Any other string will fail validation.

5. **Not handling unknown categories in ordinal_encoder.** The default `handle_unknown` is `"error"`, which will crash at inference time if new categories appear. For production, always set `handle_unknown: use_encoded_value` and `unknown_value: -1`.

6. **Passing meaningless numeric IDs as numeric features.** Columns like `zone_id`, `car_id`, or `user_id` are numeric but semantically categorical. Passing them as-is to a model treats them as continuous values, which is wrong. Use `catboost_encoder` to convert them into meaningful target-encoded features.

7. **MLflow tracking_uri format.** Use `./mlruns` (a directory path) for local tracking. Do NOT use `sqlite:///mlruns.db` -- this format causes issues with mmp's MLflow integration.

8. **Forgetting `metadata.description`.** This is a required field. Omitting it causes a Pydantic validation error.

9. **Preprocessor step ordering.** Place steps in this order: missing value handling first, then encoding, then feature generation, then scaling. Wrong ordering (e.g., scaling before encoding) triggers a validation warning and may produce incorrect results.

10. **Scalers ignore the `columns` parameter.** Standard, MinMax, and Robust scalers are "global" type -- they automatically apply to all numeric (int64/float64) columns. Setting `columns` has no effect.

## 8. Complete Working Examples

### Example A: Regression with CSV (LocalFiles)

**config_local.yaml**

```yaml
environment:
  name: local
  description: Local development environment

mlflow:
  tracking_uri: ./mlruns
  experiment_name: house-price-prediction

data_source:
  name: local-csv
  adapter_type: local_files
  config:
    base_path: ./data

feature_store:
  provider: none
  enabled: false

output:
  inference:
    name: local-output
    enabled: true
    adapter_type: storage
    config:
      base_path: ./output
      file_format: parquet

logging:
  base_path: ./logs
  level: INFO
```

**recipe_regression.yaml**

```yaml
name: house-price-regression
task_choice: regression

model:
  class_path: lightgbm.LGBMRegressor
  library: lightgbm
  hyperparameters:
    tuning_enabled: false
    values:
      n_estimators: 200
      learning_rate: 0.05
      max_depth: 6
      num_leaves: 31
      verbose: -1

data:
  loader:
    source_uri: ./data/houses.csv
  fetcher:
    type: pass_through
  data_interface:
    target_column: sale_price
    entity_columns:
      - house_id
    feature_columns:
      - lot_area
      - year_built
      - total_rooms
      - garage_area
      - neighborhood
      - condition
  split:
    # strategy: temporal          # uncomment for time-ordered split
    # temporal_column: sale_date  # required when strategy=temporal
    train: 0.7
    test: 0.15
    validation: 0.15

preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
      columns:
        - lot_area
        - garage_area
    - type: ordinal_encoder
      columns:
        - condition
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: catboost_encoder
      columns:
        - neighborhood
    - type: standard_scaler

evaluation:
  metrics:
    - rmse
    - mae
    - r2
  random_state: 42

metadata:
  author: ml-team
  description: House price prediction using LightGBM with median imputation and target encoding for neighborhood.
```

### Example B: Classification with BigQuery

**config_bigquery.yaml**

```yaml
environment:
  name: dev
  description: Development environment with BigQuery

mlflow:
  tracking_uri: ./mlruns
  experiment_name: churn-prediction

data_source:
  name: bigquery-source
  adapter_type: bigquery
  config:
    connection_uri: bigquery://my-gcp-project
    project_id: my-gcp-project
    dataset_id: ml_features
    location: US

feature_store:
  provider: none
  enabled: false

output:
  inference:
    name: bigquery-output
    enabled: true
    adapter_type: bigquery
    config:
      connection_uri: bigquery://my-gcp-project
      project_id: my-gcp-project
      dataset_id: ml_predictions
      table: churn_predictions
      location: US

logging:
  base_path: ./logs
  level: INFO
```

**recipe_classification.yaml**

```yaml
name: customer-churn-classification
task_choice: classification

model:
  class_path: xgboost.XGBClassifier
  library: xgboost
  hyperparameters:
    tuning_enabled: true
    optimization_metric: f1
    n_trials: 30
    timeout: 600
    fixed:
      n_estimators: 500
      use_label_encoder: false
      eval_metric: logloss
    tunable:
      learning_rate:
        type: float
        range: [0.01, 0.3]
      max_depth:
        type: int
        range: [3, 10]
      subsample:
        type: float
        range: [0.6, 1.0]
      colsample_bytree:
        type: float
        range: [0.6, 1.0]

data:
  loader:
    source_uri: ./queries/fetch_churn_data.sql
  fetcher:
    type: pass_through
  data_interface:
    target_column: is_churned
    entity_columns:
      - customer_id
    feature_columns:
      - tenure_months
      - monthly_charges
      - total_charges
      - contract_type
      - payment_method
      - num_support_tickets
      - plan_tier
  split:
    train: 0.7
    test: 0.15
    validation: 0.15

preprocessor:
  steps:
    - type: simple_imputer
      strategy: median
      columns:
        - total_charges
      create_missing_indicators: true
    - type: one_hot_encoder
      columns:
        - contract_type
        - payment_method
    - type: ordinal_encoder
      columns:
        - plan_tier
      handle_unknown: use_encoded_value
      unknown_value: -1
    - type: standard_scaler

evaluation:
  metrics:
    - accuracy
    - f1
    - precision
    - recall
    - roc_auc
  random_state: 42

metadata:
  author: ml-team
  description: Customer churn prediction using XGBoost with Optuna tuning. One-hot encoding for nominal categoricals, ordinal for plan tier.
```
