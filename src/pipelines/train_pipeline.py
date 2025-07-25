import json
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import pandas as pd

from src.settings import Settings
from src.engine.factory import Factory
from src.components.trainer import Trainer
from src.utils.system.logger import logger
from src.utils.integrations import mlflow_integration as mlflow_utils


def run_training(settings: Settings, context_params: Optional[Dict[str, Any]] = None):
    """
    모델 학습 파이프라인을 실행합니다.
    Factory를 통해 데이터 어댑터와 모든 컴포넌트를 생성하고, 최종적으로
    순수 로직 PyfuncWrapper를 생성하여 MLflow에 저장합니다.
    """
    logger.info(f"'{settings.model.computed['run_name']}' 모델 학습 파이프라인을 시작합니다.")
    context_params = context_params or {}

    # MLflow 실행 컨텍스트 시작
    with mlflow_utils.start_run(settings, run_name=settings.model.computed["run_name"]) as run:
        run_id = run.info.run_id
        
        # Factory 생성
        factory = Factory(settings)

        # 1. 데이터 어댑터를 사용하여 데이터 로딩
        # E2E 테스트가 아닌 경우에만 실제 어댑터를 사용합니다.
        data_adapter = factory.create_data_adapter(settings.data_adapters.default_loader)
        
        # --- E2E 테스트를 위한 임시 Mocking 로직 ---
        is_e2e_test_run = "LIMIT 100" in settings.model.loader.source_uri
        if is_e2e_test_run:
            logger.warning("E2E 테스트 모드: 실제 데이터 로딩 대신 Mock DataFrame을 생성합니다.")
            df = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(100)],
                'item_id': [f'item_{i % 10}' for i in range(100)],
                'timestamp': pd.to_datetime('2024-01-01'),
                'target_date': context_params.get('target_date', '2024-01-01'),
                'target': [i % 2 for i in range(100)]
            })
        else:
            df = data_adapter.read(settings.model.loader.source_uri)

        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 2. 학습에 사용할 컴포넌트 생성
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()

        # 3. 모델 학습
        trainer = Trainer(settings=settings)
        trained_preprocessor, trained_model, training_results = trainer.train(  # ← training_results 활용
            df=df,
            model=model,
            augmenter=augmenter,
            preprocessor=preprocessor,
            context_params=context_params,
        )
        
        # 4. 결과 로깅 (확장)
        if 'metrics' in training_results:
            mlflow.log_metrics(training_results['metrics'])
        
        # 🆕 하이퍼파라미터 최적화 결과 로깅
        if 'hyperparameter_optimization' in training_results:
            hpo_result = training_results['hyperparameter_optimization']
            if hpo_result['enabled']:
                mlflow.log_params(hpo_result['best_params'])
                mlflow.log_metric('best_score', hpo_result['best_score'])
                mlflow.log_metric('total_trials', hpo_result['total_trials'])

        # 5. 확장된 PyfuncWrapper 생성 및 저장
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            training_results=training_results,  # 🆕 결과 전달
        )
        
        # 🆕 Dynamic Signature 생성 (Day 3)
        # 학습 데이터로 샘플 예측을 수행하여 signature 생성
        logger.info("ModelSignature 생성을 위한 샘플 예측 수행 중...")
        
        # 작은 샘플 데이터로 예측 수행 (처음 5개 행 사용)
        sample_input = df.head(5)
        sample_output = pyfunc_wrapper.predict(
            context=None,
            model_input=sample_input,
            params={"run_mode": "batch", "return_intermediate": False}
        )
        
        # DataFrame이 아닌 경우 DataFrame으로 변환
        if not hasattr(sample_output, 'columns'):
            import pandas as pd
            sample_output = pd.DataFrame(sample_output)
        
        # ModelSignature 생성
        signature = mlflow_utils.create_model_signature(
            input_df=sample_input,
            output_df=sample_output
        )
        
        # model.name이 정의되지 않은 경우 run_name을 사용
        model_name = getattr(settings.model, 'name', None) or settings.model.computed['run_name']
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_wrapper,
            signature=signature,  # 🆕 signature 추가
        )
        logger.info(f"순수 로직 모델 '{model_name}'을 MLflow에 성공적으로 저장했습니다.")

        # 6. (선택적) 메타데이터 저장
        metadata = {"run_id": run_id, "model_name": model_name}
        local_dir = Path("./local/artifacts")
        local_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = local_dir / f"metadata-{run_id}.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=str)
        mlflow.log_artifact(str(metadata_path), "metadata")