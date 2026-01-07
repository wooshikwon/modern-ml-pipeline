from typer.testing import CliRunner

from src.cli.main_commands import app


def test_cli_help_shows_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    # 일부 주요 커맨드가 도움말에 노출되는지 확인
    assert "train" in result.stdout
    assert "batch-inference" in result.stdout
    assert "serve-api" in result.stdout
