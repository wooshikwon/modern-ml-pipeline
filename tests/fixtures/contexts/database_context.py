from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict


class DatabaseTestContext:
    """SQLite-backed database context for integration tests.
    Provides: db_path, connection_uri, table helpers.
    """

    def __init__(self, isolated_temp_directory):
        self.temp_dir = isolated_temp_directory
        self.db_path = self.temp_dir / "test.db"
        self.connection_uri = f"sqlite:///{self.db_path}"

    @contextmanager
    def sqlite_db(self, tables: Dict[str, Any]):
        # tables: {table_name: pandas.DataFrame}
        import pandas as pd  # deferred import for tests-only

        conn = sqlite3.connect(self.db_path)
        try:
            for name, df in tables.items():
                assert isinstance(df, pd.DataFrame)
                df.to_sql(name, conn, index=False, if_exists="replace")
            yield self
        finally:
            conn.close()
