from typing import Optional, Dict, Any
from types import SimpleNamespace

import mlflow

from src.settings import Settings
from src.factory import Factory
from src.utils.core.logger import logger
from src.utils.core.console import Console
from src.utils.integrations import mlflow_integration as mlflow_utils
from src.utils.core.environment_check import get_pip_requirements
from src.utils.core.reproducibility import set_global_seeds
from src.utils.integrations.mlflow_integration import log_training_results
from src.utils.integrations.mlflow_integration import log_enhanced_model_with_schema

def _display_mlflow_ui_info(
    run_id: str,
    run: Any,
    metrics: Dict[str, Any],
    console: Console
):
    """Display MLflow UI access information after training"""
    try:
        from src.utils.integrations.ui_helper import MLflowUIHelper, MLflowRunSummary
        
        # Get MLflow tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        
        # Get experiment info
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        experiment_name = experiment.name if experiment else "Default"
        
        # Display run summary
        summary = MLflowRunSummary(console)
        summary.display_run_summary(
            run_id=run_id,
            metrics=metrics,
            params=run.data.params if hasattr(run, 'data') else {}
        )
        
        # Display UI access info
        ui_helper = MLflowUIHelper(tracking_uri, console)
        
        ui_helper.display_access_info(
            run_id=run_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name
        )
        
    except Exception as e:
        logger.debug(f"Could not display MLflow UI info: {e}")
        # Just log the basic info
        console.log_milestone(
            f"MLflow Run ID: {run_id}",
            "success"
        )

def run_train_pipeline(
    settings: Settings,
    context_params: Optional[Dict[str, Any]] = None,
    record_requirements: bool = False,
):
    """
    모델 학습 파이프라인을 실행합니다.
    Factory를 통해 데이터 어댑터와 모든 컴포넌트를 생성하고,
    PyfuncWrapper를 생성하여 MLflow에 저장합니다.
    """
    console = Console()
    
    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)

    # Pipeline context start
    task_type = settings.recipe.task_choice
    model_name = getattr(settings.recipe.model, 'class_path', 'Unknown')
    pipeline_description = f"Environment: {settings.config.environment.name} | Task: {task_type} | Model: {model_name.split('.')[-1]}"
    
    with console.pipeline_context("Training Pipeline", pipeline_description):
        context_params = context_params or {}

        # MLflow 실행 컨텍스트 시작
        with mlflow_utils.start_run(settings, run_name=settings.recipe.model.computed["run_name"]) as run:
            run_id = run.info.run_id
            
            # Factory 생성
            factory = Factory(settings)

            # 1. 컴포넌트 생성
            console.log_phase("Component Initialization", "🔧")
            data_adapter = factory.create_data_adapter()
            fetcher = factory.create_fetcher()
            datahandler = factory.create_datahandler()
            preprocessor = factory.create_preprocessor()
            trainer = factory.create_trainer()
            model = factory.create_model()
            calibrator = factory.create_calibrator()
            evaluator = factory.create_evaluator()

            # 2. 데이터 어댑터를 사용하여 데이터 로딩
            console.log_phase("Data Loading", "📥")
            with console.progress_tracker("data_loading", 100, "Loading and preparing data") as update:
                df = data_adapter.read(settings.recipe.data.loader.source_uri)
                update(100)
            
            console.log_milestone(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns", "success")

            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))

            # 3. 피처 증강
            console.log_phase("Feature Augmentation", "✨")
            augmented_df = fetcher.fetch(df, run_mode="train") if fetcher else df

            # 4. 데이터 준비
            console.log_phase("Data Preparation", "✂️")
            X_train, y_train, add_train, X_val, y_val, add_val, X_test, y_test, add_test, calibration_data = datahandler.split_and_prepare(augmented_df)
            
            # 5. 전처리
            console.log_phase("Preprocessing", "🔍")
            if preprocessor:
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)
                X_val = preprocessor.transform(X_val) if not X_val.empty else X_val
                X_test = preprocessor.transform(X_test) if not X_test.empty else X_test
                
                # Calibration data 전처리 (있는 경우)
                if calibration_data is not None:
                    X_calib, y_calib, add_calib = calibration_data
                    X_calib = preprocessor.transform(X_calib)
                    calibration_data = (X_calib, y_calib, add_calib)
            else:
                # preprocessor가 없는 경우 원본 데이터를 그대로 사용
                pass  # 모든 데이터 변수는 이미 설정됨
            
            # 6. 학습
            console.log_phase("Training", "🧠")
            trained_model, trainer_info = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model=model, # factory로 생성된 모델 인스턴스
                additional_data={'train': add_train, 'val': add_val},
            )

            # 7. Calibration
            console.log_phase("Probability Calibration", "🎯")
            trained_calibrator = None
            
            if calibrator and calibration_data is not None:
                X_calib, y_calib, add_calib = calibration_data
                y_prob_calib = trained_model.predict_proba(X_calib)
                trained_calibrator = calibrator.fit(y_prob_calib, y_calib)
                console.log_milestone("Calibration training completed", "success")
            elif calibrator:
                console.log_milestone("Warning: Calibrator created but no calibration data available", "warning")

            # 8. 평가 및 평가 결과 MLflow에 저장 (Calibration 평가 Factory로 단순화)
            console.log_phase("Evaluation & Logging", "🎯")
            metrics = evaluator.evaluate(trained_model, X_test, y_test, add_test)
            
            # Calibration 평가 (Factory로 모든 복잡한 로직 위임)
            calibration_evaluator = factory.create_calibration_evaluator(trained_model, trained_calibrator)
            if calibration_evaluator:
                calibration_metrics = calibration_evaluator.evaluate(X_test, y_test)
                metrics.update(calibration_metrics)
            
            training_results = {
                'evaluation_metrics': metrics,
                'trainer': trainer_info,
            }
            log_training_results(settings, metrics, training_results)

            # 9. PyfuncWrapper 생성 및 MLflow에 저장
            console.log_phase("Model Packaging", "📦")
            pyfunc_wrapper = factory.create_pyfunc_wrapper(
                trained_model=trained_model,
                trained_datahandler=datahandler,  # 추론 시 재현성을 위한 DataHandler
                trained_preprocessor=preprocessor,
                trained_fetcher=fetcher, # 학습에 사용된 fetcher를 직접 전달
                trained_calibrator=trained_calibrator,  # 학습된 calibrator 추가
                training_df=augmented_df,
                training_results=training_results,
            )
            
            pip_reqs = get_pip_requirements() if record_requirements else []
            
            if not (pyfunc_wrapper.signature and pyfunc_wrapper.data_schema):
                raise ValueError("Failed to generate signature and data_schema. This should not happen.")
        
            log_enhanced_model_with_schema(
                python_model=pyfunc_wrapper,
                signature=pyfunc_wrapper.signature,
                data_schema=pyfunc_wrapper.data_schema,
                input_example=df.head(5),  # 입력 예제
                pip_requirements=pip_reqs
            )
            
            _display_mlflow_ui_info(
                run_id=run_id,
                run=run,
                metrics=metrics,
                console=console
            )
            
            return SimpleNamespace(run_id=run_id, model_uri=f"runs:/{run_id}/model")