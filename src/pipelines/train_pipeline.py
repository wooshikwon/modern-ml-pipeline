import json
from pathlib import Path
from typing import Optional, Dict, Any
from types import SimpleNamespace

import mlflow
import pandas as pd

from src.settings import Settings
from src.factory import Factory
from src.utils.system.logger import logger
from src.utils.system.console_manager import RichConsoleManager
from src.utils.integrations import mlflow_integration as mlflow_utils
from src.utils.system.environment_check import get_pip_requirements
from src.utils.system.reproducibility import set_global_seeds


def run_train_pipeline(settings: Settings, context_params: Optional[Dict[str, Any]] = None):
    """
    모델 학습 파이프라인을 실행합니다.
    Factory를 통해 데이터 어댑터와 모든 컴포넌트를 생성하고, 최종적으로
    순수 로직 PyfuncWrapper를 생성하여 MLflow에 저장합니다.
    """
    console = RichConsoleManager()
    
    # 재현성을 위한 전역 시드 설정
    seed = getattr(settings.recipe.model, 'computed', {}).get('seed', 42)
    set_global_seeds(seed)

    # Pipeline context start
    task_type = settings.recipe.data.data_interface.task_type
    model_name = getattr(settings.recipe.model, 'class_path', 'Unknown')
    pipeline_description = f"Environment: {settings.config.environment.name} | Task: {task_type} | Model: {model_name.split('.')[-1]}"
    
    with console.pipeline_context("Training Pipeline", pipeline_description):
        context_params = context_params or {}

        # MLflow 실행 컨텍스트 시작
        with mlflow_utils.start_run(settings, run_name=settings.recipe.model.computed["run_name"]) as run:
            run_id = run.info.run_id
            
            # Factory 생성
            factory = Factory(settings)

            # 1. 데이터 어댑터를 사용하여 데이터 로딩
            console.log_phase("Data Loading", "📊")
            with console.progress_tracker("data_loading", 100, "Loading and preparing data") as update:
                data_adapter = factory.create_data_adapter()
                df = data_adapter.read(settings.recipe.data.loader.source_uri)
                update(100)
            
            console.log_milestone(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns", "success")

            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))

            # 2. 학습에 사용할 컴포넌트 생성
            console.log_phase("Component Initialization", "🔧")
            fetcher = factory.create_fetcher()
            datahandler = factory.create_datahandler()  # 일관된 Factory 패턴
            preprocessor = factory.create_preprocessor()
            model = factory.create_model()
            evaluator = factory.create_evaluator()
            trainer = factory.create_trainer()  # 일관된 Factory 패턴

            # 3. 모델 학습
            console.log_phase("Model Training", "🤖")
            trained_model, trained_preprocessor, metrics, training_results = trainer.train(
                df=df,
                model=model,
                fetcher=fetcher,
                datahandler=datahandler,  # 일관된 Factory 패턴
                preprocessor=preprocessor,
                evaluator=evaluator,
                context_params=context_params,
            )
            
            # 4. 결과 로깅 및 평가
            console.log_phase("Evaluation & Logging", "📊")
            
            if metrics:
                mlflow.log_metrics(metrics)
                console.display_metrics_table(metrics, "Model Performance Metrics")
            
            # 🆕 하이퍼파라미터 최적화 결과 로깅
            if 'hyperparameter_optimization' in training_results:
                hpo_result = training_results['hyperparameter_optimization']
                if hpo_result['enabled']:
                    mlflow.log_params(hpo_result['best_params'])
                    mlflow.log_metric('best_score', hpo_result['best_score'])
                    mlflow.log_metric('total_trials', hpo_result['total_trials'])

            # 5. Enhanced PyfuncWrapper 생성
            console.log_phase("Model Packaging", "📦")
            pyfunc_wrapper = factory.create_pyfunc_wrapper(
                trained_model=trained_model,
                trained_datahandler=datahandler,  # 추론 시 재현성을 위한 DataHandler
                trained_preprocessor=trained_preprocessor,
                trained_fetcher=fetcher, # 학습에 사용된 fetcher를 직접 전달
                training_df=df,
                training_results=training_results,
            )
            
            # 6. Enhanced Model + 메타데이터 저장 (RichConsoleManager는 mlflow_integration에서 처리)
            pip_reqs = get_pip_requirements()
            
            # Signature와 data_schema 검증
            if not (pyfunc_wrapper.signature and pyfunc_wrapper.data_schema):
                raise ValueError("Failed to generate signature and data_schema. This should not happen.")
            
            # Phase 5 Enhanced 저장 로직 사용 (내부에서 RichConsoleManager 사용)
            from src.utils.integrations.mlflow_integration import log_enhanced_model_with_schema
            
            log_enhanced_model_with_schema(
                python_model=pyfunc_wrapper,
                signature=pyfunc_wrapper.signature,
                data_schema=pyfunc_wrapper.data_schema,
                input_example=df.head(5),  # 입력 예제
                pip_requirements=pip_reqs
            )
            
            # 7. 메타데이터 저장
            model_name = getattr(settings.recipe.model, 'name', None) or settings.recipe.model.computed['run_name']
            metadata = {"run_id": run_id, "model_name": model_name}
            local_dir = Path("./local/artifacts")
            local_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = local_dir / f"metadata-{run_id}.json"
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, default=str)
            mlflow.log_artifact(str(metadata_path), "metadata")

            # 8. 결과 객체 반환(run_id 및 model_uri 포함)
            return SimpleNamespace(run_id=run_id, model_uri=f"runs:/{run_id}/model")