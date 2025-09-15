from datetime import datetime
from typing import Optional, Dict, Any
from types import SimpleNamespace

import mlflow

from src.settings import Settings
from src.factory import Factory
from src.utils.core.console import Console
from src.utils.integrations import mlflow_integration as mlflow_utils
from src.utils.core.reproducibility import set_global_seeds
from src.utils.data.data_io import save_output, load_inference_data, format_predictions


def run_inference_pipeline(
    settings: Settings, 
    run_id: str, 
    data_path: Optional[str] = None, 
    context_params: Optional[Dict[str, Any]] = None
):
    """
    배치 추론 파이프라인을 실행합니다.
    지정된 Run ID의 모델을 로드하여 예측을 수행하고,
    결과를 설정된 출력 어댑터에 저장합니다.
    """
    console = Console()
    
    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)
    
    # Pipeline context start
    pipeline_description = f"Model Run ID: {run_id} | Environment: {settings.config.environment.name}"
    
    with console.pipeline_context("Batch Inference Pipeline", pipeline_description):
        context_params = context_params or {}
        
        # MLflow 실행 컨텍스트 시작
        run_name = f"batch_inference_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow_utils.start_run(settings, run_name=run_name) as run:
            inference_run_id = run.info.run_id
            
            # Factory 생성
            factory = Factory(settings)
            
            # 1. 모델 로드
            console.log_phase("Model Loading", "🤖")
            model_uri = f"runs:/{run_id}/model"
            with console.progress_tracker("model_loading", 100, f"Loading model from {model_uri}") as update:
                model = mlflow.pyfunc.load_model(model_uri)
                update(100)
            
            console.log_milestone(f"Model loaded successfully from {model_uri}", "success")
            
            # 2. 데이터 준비
            console.log_phase("Data Preparation", "📊")
            data_adapter = factory.create_data_adapter()
            df = load_inference_data(
                data_adapter=data_adapter,
                data_path=data_path,
                model=model,
                run_id=run_id,
                context_params=context_params,
                console=console
            )
            
            console.log_milestone(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns", "success")
            mlflow.log_metric("inference_input_rows", len(df))
            mlflow.log_metric("inference_input_columns", len(df.columns))
            
            # 3. 예측 실행
            console.log_phase("Model Inference", "🔮")
            with console.progress_tracker("inference", 100, "Running model prediction") as update:
                predictions_result = model.predict(df)
                
                # PyfuncWrapper에서 data_interface 정보 가져오기
                wrapped_model = model.unwrap_python_model()
                data_interface = wrapped_model.data_interface_schema.get('data_interface_config', {}) if hasattr(wrapped_model, 'data_interface_schema') else {}
                
                # data_interface 정의를 사용하여 format
                predictions_df = format_predictions(predictions_result, df, data_interface)
                update(100)
            
            console.log_milestone(f"Predictions generated: {len(predictions_df)} samples", "success")
            mlflow.log_metric("inference_output_rows", len(predictions_df))
            
            # 4. 결과 저장
            console.log_phase("Output Saving", "💾")
            save_output(
                df=predictions_df,
                settings=settings,
                output_type="inference",
                factory=factory,
                run_id=inference_run_id,
                console=console,
                additional_metadata={
                    'model_run_id': run_id
                }
            )
            
            return SimpleNamespace(
                run_id=inference_run_id,
                model_uri=model_uri,
                prediction_count=len(predictions_df)
            )
