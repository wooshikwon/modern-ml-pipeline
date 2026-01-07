from pathlib import Path

from typer.testing import CliRunner

from src.cli.main_commands import app


class TestCLITrainHappy:
    def test_mmp_train_happy_path_logs_run_and_model(self, cli_test_environment):
        runner = CliRunner()

        # 1) 준비: CLI 테스트 격리 환경(레시피/컨피그/데이터) 사용
        env = cli_test_environment
        data_path = Path(env["data_path"])  # 픽스처가 생성한 테스트 데이터 사용

        # 2) Settings: README 정책 준수 - 명시적으로 recipe/config 경로를 전달

        # 3) 실행: 필요한 모든 인자 전달
        result = runner.invoke(
            app,
            [
                "train",
                "-r",
                str(env["recipe_path"]),
                "-c",
                str(env["config_path"]),
                "-d",
                str(data_path),
            ],
        )

        # 4) 검증: 정상 종료 및 콘솔 출력에 run id/MLflow 모델 로깅 힌트 존재
        if result.exit_code != 0:
            # 디버깅을 위해 Typer 출력 원문을 덤프
            print(result.stdout)
            print(result.stderr)
        assert result.exit_code == 0
        # 최소한의 성공 텍스트 단서 확인 - Run ID와 Model URI가 출력되면 성공
        assert "Run ID:" in result.stdout or "run_id" in result.stdout.lower()
        assert "Model URI:" in result.stdout or "model" in result.stdout.lower()
