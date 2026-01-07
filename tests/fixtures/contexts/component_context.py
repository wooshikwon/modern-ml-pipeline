from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

from src.factory import Factory


class ComponentTestContext:
    """Component stack context for focused data-flow tests.
    Provides: settings, factory, adapter, model, and helpers for data prep/validation.
    """

    def __init__(
        self, isolated_temp_directory, settings_builder, test_data_generator, seed: int = 42
    ):
        self.temp_dir = isolated_temp_directory
        # make temp_dir available to nested contexts
        setattr(settings_builder, "temp_dir", isolated_temp_directory)
        self.settings_builder = settings_builder
        self.data_generator = test_data_generator
        self.seed = seed

    @contextmanager
    def classification_stack(self, model: str = "RandomForestClassifier"):
        # 1) Deterministic data
        X, y = self.data_generator.classification_data(
            n_samples=50, n_features=4, random_state=self.seed
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
        df["target"] = y
        data_path = self.temp_dir / "component_cls.csv"
        df.to_csv(data_path, index=False)

        # 2) Minimal settings
        settings = (
            self.settings_builder.with_task("classification")
            .with_model(f"sklearn.ensemble.{model}")
            .with_data_path(str(data_path))
            .build()
        )

        # 3) Factory + components
        factory = Factory(settings)
        adapter = factory.create_data_adapter()
        model_obj = factory.create_model()
        evaluator = factory.create_evaluator()
        preprocessor = factory.create_preprocessor()

        ctx = _ComponentStackContext(
            settings=settings,
            factory=factory,
            adapter=adapter,
            model=model_obj,
            evaluator=evaluator,
            preprocessor=preprocessor,
            data_path=str(data_path),
            settings_builder=self.settings_builder,
        )
        try:
            yield ctx
        finally:
            pass

    @contextmanager
    def regression_stack(self, model: str = "RandomForestRegressor"):
        # 1) Deterministic data
        X, y = self.data_generator.regression_data(
            n_samples=50, n_features=4, random_state=self.seed
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
        df["target"] = y
        data_path = self.temp_dir / "component_reg.csv"
        df.to_csv(data_path, index=False)

        # 2) Minimal settings
        settings = (
            self.settings_builder.with_task("regression")
            .with_model(f"sklearn.ensemble.{model}")
            .with_data_path(str(data_path))
            .build()
        )

        # 3) Factory + components
        factory = Factory(settings)
        adapter = factory.create_data_adapter()
        model_obj = factory.create_model()
        evaluator = factory.create_evaluator()
        preprocessor = factory.create_preprocessor()

        ctx = _ComponentStackContext(
            settings=settings,
            factory=factory,
            adapter=adapter,
            model=model_obj,
            evaluator=evaluator,
            preprocessor=preprocessor,
            data_path=str(data_path),
            settings_builder=self.settings_builder,
        )
        try:
            yield ctx
        finally:
            pass


class _ComponentStackContext:
    def __init__(
        self,
        settings,
        factory,
        adapter,
        model,
        evaluator,
        preprocessor,
        data_path: str,
        settings_builder=None,
    ):
        self.settings = settings
        self.factory = factory
        self.adapter = adapter
        self.model = model
        self.evaluator = evaluator
        self.preprocessor = preprocessor
        self.data_path = data_path
        self.settings_builder = settings_builder  # Add settings_builder access
        # expose temp_dir from outer ComponentTestContext
        if hasattr(settings_builder, "temp_dir"):
            self.temp_dir = settings_builder.temp_dir
        else:
            self.temp_dir = None

    def prepare_model_input(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Simple preparation: exclude entity/target if present
        di = self.settings.recipe.data.data_interface
        drop_cols = set((di.target_column or [],)) | set(di.entity_columns or [])
        return raw_df[[c for c in raw_df.columns if c not in drop_cols]]

    def validate_data_flow(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> bool:
        return len(raw_df) == len(processed_df) and processed_df.shape[1] > 0

    def adapter_is_compatible_with_model(self) -> bool:
        # Minimal placeholder compatibility check for pilot
        return hasattr(self.adapter, "read") and hasattr(self.model, "__class__")

    def create_feature_data(self, n_samples: int = 1) -> pd.DataFrame:
        """Create minimal feature dataframe aligned to current recipe's data_interface.
        Excludes target column; fills simple constant/string placeholders for features.
        """
        di = self.settings.recipe.data.data_interface
        feature_columns = di.feature_columns or []
        entity_columns = di.entity_columns or []

        columns = []
        # Order: entities then features
        columns.extend(entity_columns)
        for col in feature_columns:
            if col and col not in (di.target_column or []):
                columns.append(col)

        import pandas as pd  # local import to avoid global shadowing

        data = {}
        for c in columns:
            # simple filler: numeric-like names get 0.0 else 'x'
            try:
                data[c] = [0.0] * n_samples
            except Exception:
                data[c] = ["x"] * n_samples

        return pd.DataFrame(data)
