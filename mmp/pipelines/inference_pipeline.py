"""배치 추론 파이프라인 — 학습된 모델로 새 데이터에 대해 예측을 실행한다.

학습 파이프라인(train_pipeline)과의 핵심 차이:
  - 모델을 새로 학습하지 않는다. MLflow에 저장된 기존 모델을 Run ID로 복원한다.
  - 전처리·피처 증강·후처리가 PyfuncWrapper 안에 내장되어 있으므로,
    predict() 한 번 호출이 전체 추론 파이프라인을 수행한다.

흐름도:

    CLI (inference_command.py)
     └→ run_inference_pipeline()
         ├── [1] Settings 복원 (MLflow artifact + override)
         ├── [2] 모델 로드 (MLflow pyfunc — 전처리+모델+후처리 일체형)
         ├── [3] 데이터 로딩
         ├── [4] 예측 실행 + 결과 포맷팅
         ├── [5] Monitoring: 드리프트 감지 (학습 시 baseline 대비)
         └── [6] 결과 저장
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

import mlflow

from mmp.factory import Factory
from mmp.settings.factory import SettingsFactory
from mmp.utils.core.console import Console
from mmp.utils.core.logger import log_data, log_infer, log_mlflow, log_pipe
from mmp.utils.core.reproducibility import set_global_seeds
from mmp.utils.data.data_io import format_predictions, load_inference_data, save_output
from mmp.utils.integrations import mlflow_integration as mlflow_utils
from mmp.utils.integrations.version_checker import log_version_warnings


def run_inference_pipeline(
    run_id: str,
    recipe_path: Optional[str] = None,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    context_params: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
):
    """
    배치 추론 파이프라인을 실행합니다.

    Args:
        run_id: 학습된 모델의 MLflow Run ID
        recipe_path: Override할 Recipe 파일 경로 (None이면 artifact에서 복원)
        config_path: Override할 Config 파일 경로 (None이면 artifact에서 복원)
        data_path: 추론할 데이터 경로 (None이면 artifact의 SQL 사용)
        context_params: SQL 렌더링에 사용할 파라미터
        output_path: 결과 저장 전체 경로 (Config override, 파일명+확장자 포함)
    """
    console = Console()
    context_params = context_params or {}

    # [1] Settings 복원: MLflow artifact에서 학습 시 사용한 Recipe/Config를 복원한다.
    #     recipe_path/config_path가 주어지면 해당 항목만 사용자 값으로 override한다.
    #     이를 통해 학습 때와 동일한 전처리 설정을 보장하면서도 유연한 재설정이 가능하다.
    log_pipe("설정 로드 중")
    settings = SettingsFactory.for_inference(
        run_id=run_id,
        config_path=config_path,
        recipe_path=recipe_path,
        data_path=data_path,
        context_params=context_params,
    )

    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, "computed", {}).get("seed", 42)
    set_global_seeds(seed)

    # 파이프라인 시작 로깅
    log_pipe("========== 배치 추론 파이프라인 시작 ==========")
    log_pipe(f"Model Run ID: {run_id} | 환경: {settings.config.environment.name}")

    # MLflow 실행 컨텍스트 시작
    run_name = f"batch_inference_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow_utils.start_run(settings, run_name=run_name) as run:
        inference_run_id = run.info.run_id

        # Factory 생성
        factory = Factory(settings)

        # [2] 모델 로드: mlflow.pyfunc.load_model()은 PyfuncWrapper를 복원한다.
        #     PyfuncWrapper 내부에 전처리기·피처 증강기·모델·후처리가 모두 포함되어 있어,
        #     predict() 한 번 호출로 전체 추론 파이프라인이 실행된다.
        log_infer("모델 로드 시작")
        model_uri = f"runs:/{run_id}/model"
        with console.progress_tracker(
            "model_loading", 100, f"Loading model from {model_uri}"
        ) as update:
            model = mlflow.pyfunc.load_model(model_uri)
            update(100)

        log_infer(f"모델 로드 완료 - {model_uri}")

        # 학습 시와 추론 시의 패키지 버전이 다르면 경고를 출력한다.
        # 버전 불일치(예: scikit-learn, pandas)는 예측 결과 오류의 흔한 원인이다.
        log_version_warnings(run_id)

        # [3] 데이터 로딩: data_path가 있으면 해당 파일을 직접 로드하고,
        #     없으면 학습 시 MLflow artifact로 저장한 SQL을 사용해 DB에서 조회한다.
        log_data("데이터 로드 시작")
        data_adapter = factory.create_data_adapter()
        df = load_inference_data(
            data_adapter=data_adapter,
            data_path=data_path,
            model=model,
            run_id=run_id,
            context_params=context_params,
            console=console,
        )

        log_data(f"로드 완료 - {len(df):,}행, {len(df.columns)}열")
        mlflow.log_metric("inference_input_rows", len(df))
        mlflow.log_metric("inference_input_columns", len(df.columns))

        # [4] 예측 실행 + 결과 포맷팅
        log_infer("모델 추론 시작")
        with console.progress_tracker("inference", 100, "Running model prediction") as update:
            # run_mode="batch": Feature Store를 Offline Store 모드로 사용 (실시간 API 대신 배치 조회)
            # return_dataframe=True: PyfuncWrapper가 ndarray 대신 DataFrame으로 결과를 반환
            predictions_result = model.predict(
                df, params={"run_mode": "batch", "return_dataframe": True}
            )

            # PyfuncWrapper에서 data_interface_schema 정보 가져오기
            wrapped_model = model.unwrap_python_model()
            data_interface_schema = getattr(wrapped_model, "data_interface_schema", {}) or {}

            # 예측 결과에 원본 데이터의 식별 컬럼(entity_columns)을 합쳐 최종 출력 형태로 만든다.
            predictions_df = format_predictions(predictions_result, df, data_interface_schema)
            update(100)

        log_infer(f"추론 완료 - {len(predictions_df):,}개 예측 생성")
        mlflow.log_metric("inference_output_rows", len(predictions_df))

        # [5] 결과 저장: Config의 output 설정에 따라 로컬/GCS/S3 등에 저장한다.
        #     factory가 output 설정을 읽어 적절한 storage adapter를 생성한다.
        #     Monitoring보다 먼저 실행하여, 모니터링 실패가 결과 저장을 막지 않도록 한다.
        log_mlflow("결과 저장 시작")
        save_output(
            df=predictions_df,
            settings=settings,
            output_type="inference",
            factory=factory,
            run_id=inference_run_id,
            console=console,
            additional_metadata={"model_run_id": run_id},
            output_path_override=output_path,
        )

        # [6] Monitoring: 데이터 드리프트(data drift) 감지
        #     학습 시 train_pipeline에서 저장한 baseline.json(학습 데이터의 분포 통계)을 로드하고,
        #     현재 추론 데이터의 분포를 baseline과 비교하여 드리프트를 감지한다.
        #     alert/warning 수준에 따라 로그를 출력하지만, 추론 자체를 중단하지는 않는다.
        #
        #     중요: baseline은 학습 파이프라인에서 전처리(preprocessor.transform) 후의 데이터로
        #     계산된다. 따라서 evaluate()에도 전처리 후 데이터를 전달해야 한다.
        #     PyfuncWrapper.predict()가 내부적으로 전처리를 수행하면서
        #     _last_preprocessed_input에 저장한 데이터를 사용한다.
        monitors = factory.create_monitors()
        if monitors:
            try:
                import json

                from mmp.components.monitor.base import MonitorReport

                baseline = None
                try:
                    client = mlflow.tracking.MlflowClient()
                    baseline_path = client.download_artifacts(run_id, "monitoring/baseline.json")
                    with open(baseline_path) as f:
                        baseline = json.load(f)
                except Exception as e:
                    log_pipe(f"Monitoring baseline 없음 - 건너뜀: {e}")

                if baseline is not None:
                    # PyfuncWrapper가 predict() 중 전처리한 데이터를 꺼낸다.
                    # baseline이 전처리 후 데이터로 계산되었으므로, 동일한 상태의 데이터로 비교해야 한다.
                    X_for_monitor = getattr(wrapped_model, "_last_preprocessed_input", df)

                    combined = MonitorReport()
                    for monitor in monitors:
                        key = type(monitor).__name__
                        monitor_baseline = baseline.get(key, {})
                        report = monitor.evaluate(X_for_monitor, predictions_result, monitor_baseline)
                        combined.merge(report)

                    for k, v in combined.metrics.items():
                        mlflow.log_metric(f"monitor__{k}", v)
                    mlflow.log_dict(combined.to_dict(), "monitoring/report.json")

                    if combined.status == "alert":
                        log_pipe(f"[MONITOR] ALERT: {len(combined.alerts)}건 감지")
                        for alert in combined.alerts:
                            log_pipe(f"  [{alert.severity}] {alert.message}")
                    elif combined.status == "warning":
                        log_pipe(f"[MONITOR] warnings: {len(combined.alerts)}건")
            except Exception as e:
                log_pipe(f"[MONITOR] Monitoring 실패 (비치명적): {e}")

        log_pipe("========== 배치 추론 파이프라인 완료 ==========")
        return SimpleNamespace(
            run_id=inference_run_id, model_uri=model_uri, prediction_count=len(predictions_df)
        )
