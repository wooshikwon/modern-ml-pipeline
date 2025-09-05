"""
Unit tests for the Recipe module.
Tests the Recipe schema and validation logic.
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from src.settings.recipe import (
    Recipe, Model, HyperparametersTuning, Data, Loader,
    Fetcher, FeatureView, DataInterface, Preprocessor, PreprocessorStep,
    Evaluation, ValidationConfig, Metadata
)
from tests.helpers.builders import RecipeBuilder
from tests.helpers.assertions import assert_recipe_valid


class TestHyperparametersTuning:
    """Test the HyperparametersTuning configuration class."""
    
    def test_tuning_disabled_with_values(self):
        """Test hyperparameter tuning disabled with fixed values."""
        hp = HyperparametersTuning(
            tuning_enabled=False,
            values={'n_estimators': 100, 'random_state': 42}
        )
        assert hp.tuning_enabled is False
        assert hp.values == {'n_estimators': 100, 'random_state': 42}
        assert hp.fixed is None
        assert hp.tunable is None
    
    def test_tuning_enabled_with_params(self):
        """Test hyperparameter tuning enabled with fixed and tunable params."""
        hp = HyperparametersTuning(
            tuning_enabled=True,
            fixed={'random_state': 42, 'n_jobs': -1},
            tunable={
                'n_estimators': {'type': 'int', 'range': [50, 200]},
                'max_depth': {'type': 'int', 'range': [5, 20]}
            }
        )
        assert hp.tuning_enabled is True
        assert hp.fixed == {'random_state': 42, 'n_jobs': -1}
        assert 'n_estimators' in hp.tunable
        assert hp.tunable['n_estimators']['type'] == 'int'
    
    def test_tunable_param_validation(self):
        """Test validation of tunable parameter structure."""
        with pytest.raises(ValidationError) as exc_info:
            HyperparametersTuning(
                tuning_enabled=True,
                tunable={
                    'n_estimators': {'range': [50, 200]}  # Missing 'type'
                }
            )
        assert "type" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            HyperparametersTuning(
                tuning_enabled=True,
                tunable={
                    'n_estimators': {'type': 'int'}  # Missing 'range'
                }
            )
        assert "range" in str(exc_info.value).lower()
    
    def test_invalid_tunable_type(self):
        """Test validation of tunable parameter type."""
        with pytest.raises(ValidationError) as exc_info:
            HyperparametersTuning(
                tuning_enabled=True,
                tunable={
                    'n_estimators': {
                        'type': 'invalid',  # Invalid type
                        'range': [50, 200]
                    }
                }
            )
        assert "int/float/categorical" in str(exc_info.value)


class TestModel:
    """Test the Model configuration class."""
    
    def test_model_creation(self):
        """Test creating a Model object."""
        model = Model(
            class_path="sklearn.ensemble.RandomForestClassifier",
            library="sklearn",
            hyperparameters=HyperparametersTuning(
                tuning_enabled=False,
                values={'n_estimators': 100}
            )
        )
        assert model.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert model.library == "sklearn"
        assert model.hyperparameters.values == {'n_estimators': 100}
    
    def test_model_with_computed(self):
        """Test Model with computed fields."""
        model = Model(
            class_path="xgboost.XGBClassifier",
            library="xgboost",
            hyperparameters=HyperparametersTuning(tuning_enabled=False),
            computed={'run_name': 'test_run_123', 'seed': 42}
        )
        assert model.computed['run_name'] == 'test_run_123'
        assert model.computed['seed'] == 42
    
    def test_model_validation(self):
        """Test Model validation."""
        with pytest.raises(ValidationError):
            # class_path is required
            Model(library="sklearn")


class TestFeatureView:
    """Test the FeatureView configuration class."""
    
    def test_feature_view_creation(self):
        """Test creating a FeatureView object."""
        view = FeatureView(
            join_key="user_id",
            features=["age", "gender", "location"]
        )
        assert view.join_key == "user_id"
        assert view.features == ["age", "gender", "location"]
    
    def test_feature_view_validation(self):
        """Test FeatureView validation."""
        with pytest.raises(ValidationError):
            # join_key is required
            FeatureView(features=["age"])
        
        with pytest.raises(ValidationError):
            # features is required
            FeatureView(join_key="user_id")


class TestLoader:
    """Test the Loader configuration class."""
    
    def test_loader_with_sql_file(self):
        """Test Loader with SQL file source."""
        loader = Loader(
            source_uri="queries/train_data.sql"
        )
        assert loader.source_uri == "queries/train_data.sql"
        assert loader.get_adapter_type() == "sql"
    
    def test_loader_with_csv_file(self):
        """Test Loader with CSV file source."""
        loader = Loader(
            source_uri="data/train_data.csv"
        )
        assert loader.get_adapter_type() == "storage"
    
    @pytest.mark.parametrize("file_ext,expected_type", [
        (".csv", "storage"),
        (".parquet", "storage"),
        (".json", "storage"),
        (".feather", "storage"),
        (".sql", "sql"),
        ("", "sql")  # No extension defaults to SQL
    ])
    def test_loader_adapter_type_detection(self, file_ext, expected_type):
        """Test automatic adapter type detection from source URI."""
        loader = Loader(
            source_uri=f"path/to/data{file_ext}"
        )
        assert loader.get_adapter_type() == expected_type


class TestFetcher:
    """Test the Fetcher configuration class."""
    
    def test_fetcher_pass_through(self):
        """Test Fetcher with pass_through type."""
        fetcher = Fetcher(type="pass_through")
        assert fetcher.type == "pass_through"
        assert fetcher.feature_views is None
    
    def test_fetcher_feature_store(self):
        """Test Fetcher with feature_store type."""
        fetcher = Fetcher(
            type="feature_store",
            timestamp_column="event_timestamp",
            feature_views={
                "user_features": FeatureView(
                    join_key="user_id",
                    features=["age", "gender", "location"]
                )
            }
        )
        assert fetcher.type == "feature_store"
        assert fetcher.timestamp_column == "event_timestamp"
        assert "user_features" in fetcher.feature_views
        assert fetcher.feature_views["user_features"].join_key == "user_id"
    
    def test_fetcher_feature_store_empty_features(self):
        """Test Fetcher with feature_store type and empty feature_views."""
        # Empty feature_views dict should be allowed
        fetcher = Fetcher(type="feature_store", feature_views={})
        assert fetcher.type == "feature_store"
        assert fetcher.feature_views == {}
    
    def test_fetcher_validation(self):
        """Test Fetcher validation."""
        with pytest.raises(ValidationError):
            # Invalid type
            Fetcher(type="invalid")


class TestDataInterface:
    """Test the DataInterface configuration class."""
    
    @pytest.mark.parametrize("task_type", ["classification", "regression", "clustering", "causal"])
    def test_data_interface_task_types(self, task_type):
        """Test DataInterface with different task types."""
        di = DataInterface(
            task_type=task_type,
            target_column="target",
            feature_columns=["f1", "f2", "f3"],
            entity_columns=["id"]
        )
        assert di.task_type == task_type
        assert di.target_column == "target"
        assert di.feature_columns == ["f1", "f2", "f3"]
        assert di.entity_columns == ["id"]
    
    def test_data_interface_no_feature_columns(self):
        """Test DataInterface without explicit feature columns."""
        di = DataInterface(
            task_type="classification",
            target_column="label",
            entity_columns=["id"]
            # feature_columns is optional (None means all except target)
        )
        assert di.feature_columns is None
        assert di.entity_columns == ["id"]
    
    
    def test_data_interface_validation(self):
        """Test DataInterface validation."""
        with pytest.raises(ValidationError):
            # Invalid task_type
            DataInterface(
                task_type="invalid",
                target_column="target",
                entity_columns=["id"]
            )


class TestPreprocessorStep:
    """Test the PreprocessorStep configuration class."""
    
    @pytest.mark.parametrize("step_type", [
        "standard_scaler", "min_max_scaler", "robust_scaler",
        "one_hot_encoder", "ordinal_encoder", "catboost_encoder",
        "polynomial_features", "tree_based_feature_generator",
        "missing_indicator", "kbins_discretizer"
    ])
    def test_preprocessor_step_types(self, step_type):
        """Test PreprocessorStep with all valid types (except simple_imputer which needs strategy)."""
        step = PreprocessorStep(
            type=step_type,
            columns=["col1", "col2"]
        )
        assert step.type == step_type
        assert step.columns == ["col1", "col2"]
    
    def test_simple_imputer_step_type(self):
        """Test PreprocessorStep with simple_imputer type (requires strategy)."""
        step = PreprocessorStep(
            type="simple_imputer",
            columns=["col1", "col2"],
            strategy="mean"
        )
        assert step.type == "simple_imputer"
        assert step.columns == ["col1", "col2"]
        assert step.strategy == "mean"
    
    def test_simple_imputer_with_strategy(self):
        """Test SimpleImputer step with strategy."""
        step = PreprocessorStep(
            type="simple_imputer",
            columns=["age", "income"],
            strategy="mean"
        )
        assert step.strategy == "mean"
    
    def test_simple_imputer_validation(self):
        """Test SimpleImputer validation requires strategy."""
        with pytest.raises(ValidationError) as exc_info:
            PreprocessorStep(
                type="simple_imputer",
                columns=["col1"]
                # Missing required strategy
            )
        assert "strategy" in str(exc_info.value).lower()
    
    def test_polynomial_features_with_degree(self):
        """Test PolynomialFeatures step with degree."""
        step = PreprocessorStep(
            type="polynomial_features",
            columns=["f1", "f2"],
            degree=3
        )
        assert step.degree == 3
    
    def test_polynomial_features_default_degree(self):
        """Test PolynomialFeatures step gets default degree."""
        step = PreprocessorStep(
            type="polynomial_features",
            columns=["f1", "f2"]
            # degree not specified, should get default
        )
        assert step.degree == 2  # Default value
    
    def test_kbins_discretizer_with_n_bins(self):
        """Test KBinsDiscretizer step with n_bins."""
        step = PreprocessorStep(
            type="kbins_discretizer",
            columns=["age"],
            n_bins=10
        )
        assert step.n_bins == 10
    
    def test_catboost_encoder_with_sigma(self):
        """Test CatBoostEncoder step with sigma."""
        step = PreprocessorStep(
            type="catboost_encoder",
            columns=["category"],
            sigma=0.05
        )
        assert step.sigma == 0.05
    
    def test_preprocessor_step_validation(self):
        """Test PreprocessorStep validation."""
        with pytest.raises(ValidationError):
            # Invalid type
            PreprocessorStep(
                type="invalid_scaler",
                columns=["col1"]
            )
        
        # Test degree bounds
        with pytest.raises(ValidationError):
            PreprocessorStep(
                type="polynomial_features",
                columns=["f1"],
                degree=6  # Too high (max is 5)
            )
        
        # Test n_bins bounds
        with pytest.raises(ValidationError):
            PreprocessorStep(
                type="kbins_discretizer",
                columns=["age"],
                n_bins=25  # Too high (max is 20)
            )


class TestPreprocessor:
    """Test the Preprocessor configuration class."""
    
    def test_preprocessor_with_steps(self):
        """Test Preprocessor with multiple steps."""
        preprocessor = Preprocessor(
            steps=[
                PreprocessorStep(
                    type="standard_scaler",
                    columns=["age", "income"]
                ),
                PreprocessorStep(
                    type="one_hot_encoder",
                    columns=["category", "region"]
                )
            ]
        )
        assert len(preprocessor.steps) == 2
        assert preprocessor.steps[0].type == "standard_scaler"
        assert preprocessor.steps[1].type == "one_hot_encoder"
    
    def test_preprocessor_empty_steps(self):
        """Test Preprocessor with empty steps list."""
        preprocessor = Preprocessor()
        assert preprocessor.steps == []


class TestValidationConfig:
    """Test the ValidationConfig class."""
    
    def test_validation_train_test_split(self):
        """Test ValidationConfig with train_test_split."""
        val = ValidationConfig(
            method="train_test_split",
            test_size=0.3,
            random_state=123
        )
        assert val.method == "train_test_split"
        assert val.test_size == 0.3
        assert val.random_state == 123
        assert val.n_folds is None
    
    def test_validation_cross_validation(self):
        """Test ValidationConfig with cross_validation."""
        val = ValidationConfig(
            method="cross_validation",
            n_folds=5
        )
        assert val.method == "cross_validation"
        assert val.n_folds == 5
    
    def test_validation_defaults(self):
        """Test ValidationConfig default values."""
        val = ValidationConfig()
        assert val.method == "train_test_split"
        assert val.test_size == 0.2
        assert val.random_state == 42
    
    def test_validation_cross_val_default_folds(self):
        """Test cross_validation gets default n_folds."""
        val = ValidationConfig(method="cross_validation")
        assert val.n_folds == 5  # Default value
    
    def test_validation_test_size_bounds(self):
        """Test ValidationConfig test_size bounds."""
        # Valid range
        val1 = ValidationConfig(test_size=0.1)  # Minimum
        assert val1.test_size == 0.1
        
        val2 = ValidationConfig(test_size=0.5)  # Maximum
        assert val2.test_size == 0.5
        
        # Invalid range
        with pytest.raises(ValidationError):
            ValidationConfig(test_size=0.05)  # Too small
        
        with pytest.raises(ValidationError):
            ValidationConfig(test_size=0.6)  # Too large


class TestEvaluation:
    """Test the Evaluation configuration class."""
    
    def test_evaluation_creation(self):
        """Test creating an Evaluation object."""
        eval_config = Evaluation(
            metrics=["accuracy", "f1", "precision", "recall"]
        )
        assert len(eval_config.metrics) == 4
        assert "accuracy" in eval_config.metrics
    
    def test_evaluation_metric_normalization(self):
        """Test that metrics are normalized to lowercase."""
        eval_config = Evaluation(
            metrics=["Accuracy", "F1", "ROC_AUC"]
        )
        assert eval_config.metrics == ["accuracy", "f1", "roc_auc"]
    
    def test_evaluation_with_validation_config(self):
        """Test Evaluation with custom validation config."""
        eval_config = Evaluation(
            metrics=["rmse", "mae"],
            validation=ValidationConfig(
                method="cross_validation",
                n_folds=10
            )
        )
        assert eval_config.validation.method == "cross_validation"
        assert eval_config.validation.n_folds == 10
    
    def test_evaluation_validation(self):
        """Test Evaluation validation."""
        with pytest.raises(ValidationError):
            # Empty metrics not allowed
            Evaluation(metrics=[])


class TestMetadata:
    """Test the Metadata configuration class."""
    
    def test_metadata_creation(self):
        """Test creating a Metadata object."""
        metadata = Metadata(
            author="Data Scientist",
            created_at="2024-01-01 12:00:00",
            description="Test recipe",
            tuning_note="Optuna tuning enabled"
        )
        assert metadata.author == "Data Scientist"
        assert metadata.created_at == "2024-01-01 12:00:00"
        assert metadata.description == "Test recipe"
        assert metadata.tuning_note == "Optuna tuning enabled"
    
    def test_metadata_minimal(self):
        """Test Metadata with minimal required fields."""
        metadata = Metadata(created_at="2024-01-01")
        assert metadata.created_at == "2024-01-01"
        assert metadata.author == "CLI Recipe Builder"  # Default
        assert metadata.description is None
        assert metadata.tuning_note is None
    
    def test_metadata_validation(self):
        """Test Metadata validation."""
        with pytest.raises(ValidationError):
            # created_at is required
            Metadata(author="Test")


class TestData:
    """Test the Data configuration class."""
    
    def test_data_creation(self):
        """Test creating a Data object."""
        data = Data(
            loader=Loader(
                source_uri="data.csv"
            ),
            fetcher=Fetcher(type="pass_through", timestamp_column="ts"),
            data_interface=DataInterface(
                task_type="classification",
                target_column="label",
                entity_columns=["id"]
            )
        )
        assert data.loader.source_uri == "data.csv"
        assert data.fetcher.type == "pass_through"
        assert data.data_interface.task_type == "classification"


class TestRecipe:
    """Test the main Recipe class."""
    
    def test_recipe_creation_minimal(self):
        """Test creating Recipe with minimal required fields."""
        recipe = Recipe(
            name="test_recipe",
            model=Model(
                class_path="sklearn.tree.DecisionTreeClassifier",
                library="sklearn"
            ),
            data=Data(
                loader=Loader(
                    source_uri="train.csv"
                ),
                fetcher=Fetcher(type="pass_through", timestamp_column="ts"),
                data_interface=DataInterface(
                    task_type="classification",
                    target_column="target",
                    entity_columns=["id"]
                )
            ),
            training={
                "validation": {
                    "type": "train_test_split",
                    "test_size": 0.2
                }
            },
            evaluation=Evaluation(metrics=["accuracy"])
        )
        assert_recipe_valid(recipe)
        assert recipe.name == "test_recipe"
        assert recipe.preprocessor is None
        assert recipe.metadata is None
    
    def test_recipe_creation_full(self):
        """Test creating Recipe with all fields."""
        recipe = Recipe(
            name="full_recipe",
            model=Model(
                class_path="xgboost.XGBClassifier",
                library="xgboost",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=True,
                    fixed={'random_state': 42},
                    tunable={
                        'max_depth': {'type': 'int', 'range': [3, 10]}
                    }
                )
            ),
            data=Data(
                loader=Loader(
                    source_uri="query.sql"
                ),
                fetcher=Fetcher(
                    type="feature_store",
                    timestamp_column="timestamp",
                    feature_views={
                        "user_features": FeatureView(
                            join_key="user_id",
                            features=["age", "gender"]
                        )
                    }
                ),
                data_interface=DataInterface(
                    task_type="classification",
                    target_column="clicked",
                    feature_columns=["f1", "f2"],
                    entity_columns=["user_id", "item_id"]
                )
            ),
            preprocessor=Preprocessor(
                steps=[
                    PreprocessorStep(
                        type="standard_scaler",
                        columns=["age"]
                    )
                ]
            ),
            training={
                "validation": {
                    "type": "train_test_split",
                    "test_size": 0.2
                }
            },
            evaluation=Evaluation(
                metrics=["accuracy", "f1", "roc_auc"],
                validation=ValidationConfig(
                    method="cross_validation",
                    n_folds=5
                )
            ),
            metadata=Metadata(
                author="Test Author",
                created_at="2024-01-01",
                description="Full test recipe"
            )
        )
        assert_recipe_valid(recipe)
        assert recipe.preprocessor is not None
        assert recipe.metadata is not None
    
    def test_recipe_from_dict(self):
        """Test creating Recipe from dictionary."""
        recipe_dict = {
            "name": "dict_recipe",
            "model": {
                "class_path": "sklearn.svm.SVC",
                "library": "sklearn",
                "hyperparameters": {
                    "tuning_enabled": False,
                    "values": {"kernel": "rbf"}
                }
            },
            "data": {
                "loader": {
                    "source_uri": "data.parquet"
                },
                "fetcher": {
                    "type": "pass_through",
                    "timestamp_column": "ts"
                },
                "data_interface": {
                    "task_type": "classification",
                    "target_column": "y",
                    "entity_columns": ["id"]
                }
            },
            "training": {
                "validation": {
                    "type": "train_test_split",
                    "test_size": 0.2
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "precision"]
            }
        }
        recipe = Recipe(**recipe_dict)
        assert_recipe_valid(recipe)
        assert recipe.name == "dict_recipe"
    
    def test_recipe_helper_methods(self):
        """Test Recipe helper methods."""
        recipe = RecipeBuilder.build(
            name="helper_test",
            task_type="regression",
            model_class_path="sklearn.linear_model.LinearRegression"
        )
        
        # Test get_task_type
        assert recipe.get_task_type() == "regression"
        
        # Test get_metrics
        metrics = recipe.get_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        # Test is_tuning_enabled
        assert recipe.is_tuning_enabled() is False
        
        # Test get_hyperparameters
        hp = recipe.get_hyperparameters()
        assert isinstance(hp, dict)
        
        # Test get_tunable_params
        tunable = recipe.get_tunable_params()
        assert tunable is None  # Since tuning is disabled
    
    def test_recipe_with_tuning_enabled(self):
        """Test Recipe methods with tuning enabled."""
        recipe = Recipe(
            name="tuning_recipe",
            model=Model(
                class_path="sklearn.ensemble.RandomForestRegressor",
                library="sklearn",
                hyperparameters=HyperparametersTuning(
                    tuning_enabled=True,
                    fixed={'random_state': 42},
                    tunable={
                        'n_estimators': {'type': 'int', 'range': [50, 200]},
                        'max_depth': {'type': 'int', 'range': [5, 15]}
                    }
                )
            ),
            data=Data(
                loader=Loader(
                    source_uri="train.csv"
                ),
                fetcher=Fetcher(type="pass_through", timestamp_column="ts"),
                data_interface=DataInterface(
                    task_type="regression",
                    target_column="value",
                    entity_columns=["id"]
                )
            ),
            training={
                "validation": {
                    "type": "train_test_split",
                    "test_size": 0.2
                }
            },
            evaluation=Evaluation(metrics=["rmse", "mae"])
        )
        
        assert recipe.is_tuning_enabled() is True
        
        # With tuning enabled, get_hyperparameters returns fixed params
        hp = recipe.get_hyperparameters()
        assert hp == {'random_state': 42}
        
        # get_tunable_params returns the tunable parameters
        tunable = recipe.get_tunable_params()
        assert 'n_estimators' in tunable
        assert 'max_depth' in tunable
        assert tunable['n_estimators']['range'] == [50, 200]
    
    def test_recipe_builder_helper(self):
        """Test using the RecipeBuilder helper."""
        recipe = RecipeBuilder.build(
            name="builder_test",
            model_class_path="catboost.CatBoostClassifier",
            task_type="classification",
            source_uri="s3://bucket/data.parquet",
            fetcher_type="feature_store"
        )
        assert recipe.name == "builder_test"
        assert "catboost" in recipe.model.class_path.lower()
        assert recipe.data.fetcher.type == "feature_store"
    
    def test_recipe_validation(self):
        """Test Recipe validation."""
        with pytest.raises(ValidationError):
            # Missing required fields
            Recipe(name="invalid")
        
        with pytest.raises(ValidationError):
            # Missing data
            Recipe(
                name="invalid",
                model=Model(
                    class_path="sklearn.svm.SVC",
                    library="sklearn"
                ),
                evaluation=Evaluation(metrics=["accuracy"])
            )