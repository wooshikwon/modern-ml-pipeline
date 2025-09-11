import pytest
from pathlib import Path

from src.cli.utils.config_loader import load_config_with_env


def test_config_loader_invalid_yaml_raises(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    bad = tmp_path / "configs" / "local.yaml"
    # 고의적 문법 오류
    bad.write_text("invalid: [yaml")

    # .env.local 파일도 요구되므로 더미 파일 생성
    (tmp_path / ".env.local").write_text("EXAMPLE=1\n")

    with pytest.raises(Exception):
        load_config_with_env("local", base_path=tmp_path)

