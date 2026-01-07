from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


def render_recipe(context: dict) -> str:
    templates_dir = Path(__file__).parents[3] / "src" / "cli" / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    tmpl = env.get_template("recipes/recipe.yaml.j2")
    return tmpl.render(**context)


class TestRecipeTemplateSnapshot:
    def test_timeseries_feature_store_template_contains_required_sections(self, tmp_path):
        context = {
            "recipe_name": "ts_recipe",
            "timestamp": "2025-01-01 00:00:00",
            "task": "timeseries",
            "model_class": "sklearn.linear_model.LinearRegression",
            "model_library": "sklearn",
            "enable_tuning": False,
            "all_hyperparameters": {"random_state": 42},
            # data
            "fetcher_type": "feature_store",
            "feature_views": {
                "user_features": {
                    "join_key": "user_id",
                    "features": ["f1", "f2"],
                }
            },
            "timestamp_column": "event_ts",
            # data_interface
            "target_column": "target",
            "treatment_column": None,
            "timeseries_timestamp_column": "timestamp",
            "entity_columns": ["entity_id"],
            # evaluation
            "metrics": ["mse"],
            "test_size": 0.2,
            # metadata
            "author": "unit_test",
        }

        rendered = render_recipe(context)

        # 스냅샷적 검증: 주석/필수 키 존재
        assert "# 시계열 작업은 timestamp_column이 필수입니다." in rendered
        assert "fetcher:" in rendered
        assert "feature_views:" in rendered
        assert "timestamp_column:" in rendered
        assert "data_interface:" in rendered
        assert "entity_columns:" in rendered
        assert "evaluation:" in rendered

        # YAML 파싱이 가능한지(구조 일관성) 확인
        doc = yaml.safe_load(rendered)
        assert doc["task_choice"] == "timeseries"
        assert doc["data"]["fetcher"]["type"] == "feature_store"
        assert "data_interface" in doc["data"]
        assert doc["data"]["data_interface"]["timestamp_column"] == "timestamp"
        assert doc["data"]["data_interface"]["target_column"] == "target"
        assert doc["data"]["data_interface"]["entity_columns"] == ["entity_id"]
