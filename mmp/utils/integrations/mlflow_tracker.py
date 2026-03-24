# mmp/utils/integrations/mlflow_tracker.py
"""MLflow 실험 추적, 로깅, 아티팩트 관리를 담당하는 모듈.

``MLflowTracker`` 클래스가 실험 lifecycle 전체를 관리한다:
setup, run 시작/종료, 메트릭 로깅, 아티팩트 다운로드.
"""

from __future__ import annotations

import datetime
import os
import re
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.pyfunc import PyFuncModel

    from mmp.settings import Settings

from mmp.utils.core.logger import log_mlflow, logger


class MLflowTracker:
    """MLflow 실험 추적, 로깅, 아티팩트 관리를 담당하는 클래스.

    ``settings`` 를 한 번 주입하면 이후 메서드 호출 시 반복 전달할 필요가 없다.
    ``settings`` 없이 인스턴스화한 뒤 개별 메서드의 ``settings`` 인자로 전달하는 것도
    가능하다 (하위 호환성).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings

    # -- internal helpers ---------------------------------------------------

    def _get_settings(self, settings: Settings | None) -> Settings:
        """명시적으로 넘어온 settings 우선, 없으면 인스턴스 설정 사용."""
        s = settings or self._settings
        if s is None:
            raise ValueError(
                "settings가 필요합니다. MLflowTracker 생성 시 또는 메서드 호출 시 전달하세요."
            )
        return s

    # -- utility ------------------------------------------------------------

    @staticmethod
    def generate_unique_run_name(base_run_name: str) -> str:
        """기본 run name에 timestamp + random suffix를 추가하여 유니크한 이름을 생성한다.

        병렬 테스트 실행 시 MLflow run name 충돌을 방지한다.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        unique_run_name = f"{base_run_name}_{timestamp}_{random_suffix}"
        logger.debug(f"[MLFLOW] 유니크 run name 생성: {base_run_name} -> {unique_run_name}")
        return unique_run_name

    # -- setup & run --------------------------------------------------------

    def setup_mlflow(self, settings: Settings | None = None) -> None:
        """주입된 settings 객체를 기반으로 MLflow 클라이언트를 설정한다."""
        s = self._get_settings(settings)
        mlflow.set_tracking_uri(s.config.mlflow.tracking_uri)
        mlflow.set_experiment(s.config.mlflow.experiment_name)
        log_mlflow("설정 완료")
        logger.debug(f"[MLFLOW] Tracking URI: {s.config.mlflow.tracking_uri}")
        logger.debug(f"[MLFLOW] Experiment: {s.config.mlflow.experiment_name}")

    @contextmanager
    def start_run(
        self, settings: Settings | None = None, run_name: str = "run"
    ) -> Generator[Run, None, None]:
        """MLflow 실행을 시작하고 관리하는 컨텍스트 매니저.

        자동으로 유니크한 run name을 생성하여 병렬 실행 시 충돌을 방지한다.
        """
        s = self._get_settings(settings)
        unique_run_name = self.generate_unique_run_name(run_name)

        tracking_uri = s.config.mlflow.tracking_uri
        if tracking_uri:
            parsed = urlparse(tracking_uri)
            if parsed.scheme == "file" and parsed.path:
                try:
                    os.makedirs(parsed.path, exist_ok=True)
                except Exception:
                    pass
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(s.config.mlflow.experiment_name)

        try:
            with mlflow.start_run(run_name=unique_run_name) as run:
                log_mlflow(f"Run 시작 - ID: {run.info.run_id[:8]}...")
                mlflow.set_tag("original_run_name", run_name)
                mlflow.set_tag("unique_run_name", unique_run_name)
                try:
                    yield run
                    mlflow.set_tag("status", "success")
                    log_mlflow("Run 완료")
                except Exception as e:
                    mlflow.set_tag("status", "failed")
                    logger.error(f"[MLFLOW] Run 실패: {e}", exc_info=True)
                    raise
        except Exception as mlflow_error:
            if (
                "already exists" in str(mlflow_error).lower()
                or "duplicate" in str(mlflow_error).lower()
            ):
                logger.warning(f"[MLFLOW] Run name 충돌 감지: {unique_run_name}")
                retry_run_name = f"{unique_run_name}_{uuid.uuid4().hex[:4]}"
                logger.debug(f"[MLFLOW] 재시도: {retry_run_name}")

                with mlflow.start_run(run_name=retry_run_name) as run:
                    log_mlflow(f"Run 시작 (재시도) - ID: {run.info.run_id[:8]}...")
                    mlflow.set_tag("original_run_name", run_name)
                    mlflow.set_tag("unique_run_name", retry_run_name)
                    mlflow.set_tag("retry_count", "1")
                    try:
                        yield run
                        mlflow.set_tag("status", "success")
                        log_mlflow("Run 완료 (재시도)")
                    except Exception as e:
                        mlflow.set_tag("status", "failed")
                        logger.error(f"[MLFLOW] Run 실패 (재시도): {e}", exc_info=True)
                        raise
            else:
                raise

    # -- query --------------------------------------------------------------

    def get_latest_run_id(
        self, settings: Settings | None = None, experiment_name: str | None = None
    ) -> str:
        """지정된 experiment에서 가장 최근에 성공한 run의 ID를 반환한다."""
        s = self._get_settings(settings)
        self.setup_mlflow(s)
        exp_name = experiment_name or s.config.mlflow.experiment_name
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if not experiment:
                raise ValueError(f"Experiment '{exp_name}'을 찾을 수 없습니다.")

            runs_df = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.status = 'success'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs_df.empty:
                raise ValueError(f"Experiment '{exp_name}'에서 성공한 run을 찾을 수 없습니다.")

            latest_run_id = runs_df.iloc[0]["run_id"]
            logger.debug(f"[MLFLOW] 최근 Run ID 조회: {latest_run_id[:8]}...")
            return latest_run_id
        except Exception as e:
            logger.error(f"[MLFLOW] 최근 Run ID 조회 실패: {e}")
            raise

    @staticmethod
    def get_model_uri(run_id: str, artifact_path: str = "model") -> str:
        """Run ID와 아티팩트 경로를 사용하여 모델 URI를 생성한다."""
        uri = f"runs:/{run_id}/{artifact_path}"
        logger.debug(f"생성된 모델 URI: {uri}")
        return uri

    # -- model loading ------------------------------------------------------

    def load_pyfunc_model(
        self, settings: Settings | None = None, model_uri: str = ""
    ) -> PyFuncModel:
        """지정된 URI에서 모델을 로드하여 Pyfunc 모델 객체를 반환한다."""
        s = self._get_settings(settings)
        log_mlflow(f"모델 로딩 시작: {model_uri}")
        try:
            if model_uri.startswith("runs:/"):
                def _parse_runs_uri(uri: str) -> tuple[str, str]:
                    match = re.match(r"runs:/([^/]+)/(.+)", uri)
                    if not match:
                        raise ValueError(f"'{uri}'는 올바른 'runs:/' URI가 아닙니다.")
                    return match.group(1), match.group(2)

                client = MlflowClient(tracking_uri=s.config.mlflow.tracking_uri)
                run_id, artifact_path = _parse_runs_uri(model_uri)
                local_path = client.download_artifacts(run_id=run_id, path=artifact_path)
                logger.debug(f"[MLFLOW] 아티팩트 다운로드 완료: {local_path}")
                return mlflow.pyfunc.load_model(model_uri=local_path)
            else:
                mlflow.set_tracking_uri(s.config.mlflow.tracking_uri)
                return mlflow.pyfunc.load_model(model_uri=model_uri)
        except Exception as e:
            logger.error(f"[MLFLOW] 모델 로딩 실패: {model_uri}, 오류: {e}", exc_info=True)
            raise

    # -- artifact download --------------------------------------------------

    def download_artifacts(
        self,
        settings: Settings | None = None,
        run_id: str = "",
        artifact_path: str = "",
        dst_path: str | None = None,
    ) -> str:
        """지정된 Run ID에서 특정 아티팩트를 다운로드하고, 로컬 경로를 반환한다."""
        s = self._get_settings(settings)
        mlflow.set_tracking_uri(s.config.mlflow.tracking_uri)
        logger.debug(f"[MLFLOW] 아티팩트 다운로드 시작: {artifact_path}")
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
            )
            logger.debug(f"[MLFLOW] 아티팩트 다운로드 완료: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"[MLFLOW] 아티팩트 다운로드 실패: {e}", exc_info=True)
            raise

    # -- logging helpers ----------------------------------------------------

    def log_training_results(
        self,
        settings: Settings | None = None,
        metrics: dict | None = None,
        training_results: dict | None = None,
    ) -> None:
        """파이프라인에서 간결하게 호출하기 위한 결과 로깅 헬퍼.

        메트릭 로깅 및 HPO(on/off) 분기, 하이퍼파라미터/최적 점수를 로깅한다.
        """
        s = self._get_settings(settings)

        # 1) Metrics
        if metrics:
            mlflow.log_metrics(metrics)
            items = [
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ]
            log_mlflow(f"메트릭 기록 ({len(items)}개)")
            for i in range(0, len(items), 4):
                chunk = items[i : i + 4]
                log_mlflow(f"{', '.join(chunk)}")

        # 2) Hyperparameters / HPO
        hpo = (training_results or {}).get("trainer", {}).get("hyperparameter_optimization")
        if hpo and hpo.get("enabled"):
            best_params = hpo.get("best_params") or {}
            if best_params:
                mlflow.log_params(best_params)
            if "best_score" in hpo:
                mlflow.log_metric("best_score", hpo["best_score"])
            if "total_trials" in hpo:
                mlflow.log_metric("total_trials", hpo["total_trials"])
        else:
            try:
                hp = s.recipe.model.hyperparameters
                if hasattr(hp, "values") and hp.values:
                    mlflow.log_params(hp.values)
            except Exception:
                pass
