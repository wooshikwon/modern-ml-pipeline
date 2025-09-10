import json
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
from src.utils.integrations.mlflow_integration import log_training_results
from src.utils.integrations.mlflow_integration import log_enhanced_model_with_schema

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
    console = RichConsoleManager()
    
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

            # 1. 데이터 어댑터를 사용하여 데이터 로딩
            console.log_phase("Data Loading", "📊")
            with console.progress_tracker("data_loading", 100, "Loading and preparing data") as update:
                data_adapter = factory.create_data_adapter()
                df = data_adapter.read(settings.recipe.data.loader.source_uri)
                update(100)
            
            console.log_milestone(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns", "success")

            mlflow.log_metric("row_count", len(df))
            mlflow.log_metric("column_count", len(df.columns))

            # 2. 컴포넌트 생성
            console.log_phase("Component Initialization", "🔧")
            fetcher = factory.create_fetcher()
            datahandler = factory.create_datahandler()  # 일관된 Factory 패턴
            preprocessor = factory.create_preprocessor()
            model = factory.create_model()
            evaluator = factory.create_evaluator()
            trainer = factory.create_trainer()  # 일관된 Factory 패턴

            # 3. 피처 증강
            console.log_phase("Feature Augmentation", "✨")
            augmented_df = fetcher.fetch(df, run_mode="train") if fetcher else df

            # 4. 데이터 준비
            console.log_phase("Model Training", "🤖")
            X_train, y_train, add_train, X_test, y_test, add_test = datahandler.split_and_prepare(augmented_df)
            
            # 5. 전처리
            if preprocessor:
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)
                X_test = preprocessor.transform(X_test)
            
            # 6. 학습
            trained_model, trainer_info = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                model=model,
                additional_data={'train': add_train, 'val': add_test},
            )

            # 7. 평가 및 평가결과 저장
            metrics = evaluator.evaluate(trained_model, X_test, y_test, add_test)
            training_results = {
                'evaluation_metrics': metrics,
                'trainer': trainer_info,
            }
            console.log_phase("Evaluation & Logging", "📊")
            log_training_results(settings, metrics, training_results)

            # 8. PyfuncWrapper 생성 및 저장
            console.log_phase("Model Packaging", "📦")
            pyfunc_wrapper = factory.create_pyfunc_wrapper(
                trained_model=trained_model,
                trained_datahandler=datahandler,  # 추론 시 재현성을 위한 DataHandler
                trained_preprocessor=preprocessor,
                trained_fetcher=fetcher, # 학습에 사용된 fetcher를 직접 전달
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
            
            return SimpleNamespace(run_id=run_id, model_uri=f"runs:/{run_id}/model")