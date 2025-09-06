from .dataframe_builder import DataFrameBuilder

class ModelBuilder:
    @staticmethod
    def build_sklearn_classifier():
        from sklearn.ensemble import RandomForestClassifier
        df = DataFrameBuilder.build_classification_data(n_samples=120, n_features=5, n_classes=2, add_entity_column=False, random_state=42)
        X = df[[c for c in df.columns if c.startswith('feature_')]]
        y = df['target']
        model = RandomForestClassifier(n_estimators=16, random_state=42)
        model.fit(X, y)
        return model

    @staticmethod
    def build_sklearn_regressor():
        from sklearn.linear_model import LinearRegression
        df = DataFrameBuilder.build_regression_data(n_samples=120, n_features=5, add_entity_column=False, random_state=42)
        X = df[[c for c in df.columns if c.startswith('feature_')]]
        y = df['target']
        model = LinearRegression()
        model.fit(X, y)
        return model
