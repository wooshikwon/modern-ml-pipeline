from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from .dataframe_builder import DataFrameBuilder


class FileBuilder:
    @staticmethod
    def create_yaml_file(path: Path, content: Dict[str, Any]) -> Path:
        import yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)
        return path

    @staticmethod
    def create_csv_file(path: Path, dataframe: Optional[pd.DataFrame] = None, n_rows: int = 100, n_cols: int = 5) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if dataframe is None:
            dataframe = DataFrameBuilder.build_classification_data(n_samples=n_rows, n_features=n_cols - 1)
        dataframe.to_csv(path, index=False)
        return path

    @staticmethod
    def create_sql_file(path: Path, query: Optional[str] = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if query is None:
            query = (
                "SELECT\n"
                "    user_id,\n"
                "    feature_1,\n"
                "    feature_2,\n"
                "    target\n"
                "FROM training_data\n"
                "WHERE created_at >= '2024-01-01'\n"
            )
        with open(path, 'w') as f:
            f.write(query)
        return path

    @staticmethod
    def build_csv_string(data: Optional[pd.DataFrame] = None, **kwargs):
        from io import StringIO
        if data is None:
            data = DataFrameBuilder.build_classification_data(**kwargs)
        buffer = StringIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @staticmethod
    def build_yaml_string(content: Optional[Dict[str, Any]] = None) -> str:
        import yaml
        if content is None:
            content = {"environment": {"name": "test"}}
        return yaml.dump(content)

    @staticmethod
    def build_csv_file_context(data: Optional[pd.DataFrame] = None, **kwargs):
        from contextlib import contextmanager
        import tempfile, os
        @contextmanager
        def _ctx():
            nonlocal data
            if data is None:
                data = DataFrameBuilder.build_classification_data(**kwargs)
            fd, path = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
            try:
                data.to_csv(path, index=False)
                yield path
            finally:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return _ctx()

    @staticmethod
    def build_yaml_file_context(content: Optional[Dict[str, Any]] = None):
        from contextlib import contextmanager
        import tempfile, os, yaml
        @contextmanager
        def _ctx():
            nonlocal content
            if content is None:
                content = {"environment": {"name": "test"}}
            fd, path = tempfile.mkstemp(suffix='.yaml')
            os.close(fd)
            try:
                with open(path, 'w') as f:
                    yaml.dump(content, f)
                yield path
            finally:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return _ctx()
