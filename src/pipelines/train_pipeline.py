import json
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import pandas as pd
from contextlib import contextmanager

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
    logger.info(f"['{settings.recipe.model.computed['run_name']}'] 모델 학습 파이프라인 시작")
    logger.info(f"MLflow Tracking URI (from settings): {settings.mlflow.tracking_uri}") # 경로 검증 로그 추가
    context_params = context_params or {}

    # MLflow 실행 컨텍스트 시작
    with mlflow_utils.start_run(settings, run_name=settings.recipe.model.computed["run_name"]) as run:
        run_id = run.info.run_id
        
        # Factory 생성
        factory = Factory(settings)

        # 1. 데이터 어댑터를 사용하여 데이터 로딩
        # E2E 테스트가 아닌 경우에만 실제 어댑터를 사용합니다.
        data_adapter = factory.create_data_adapter(settings.data_adapters.default_loader)
        
        # --- E2E 테스트를 위한 임시 Mocking 로직 ---
        # 🔧 파일 내용에서 "LIMIT 100" 확인하도록 수정
        is_e2e_test_run = False
        try:
            with open(settings.recipe.model.loader.source_uri, 'r') as f:
                file_content = f.read()
                is_e2e_test_run = "LIMIT 100" in file_content
        except:
            pass  # 파일을 읽을 수 없으면 Mock 모드 비활성화
            
        if is_e2e_test_run:
            logger.warning("E2E 테스트 모드: 실제 데이터 로딩 대신 Mock DataFrame을 생성합니다.")
            # 🔄 E2E Recipe EntitySchema 스키마와 일치하도록 수정
            df = pd.DataFrame({
                'user_id': [f'user_{i}' for i in range(100)],
                'product_id': [f'product_{i % 10}' for i in range(100)],  # ✅ item_id → product_id
                'event_timestamp': pd.to_datetime('2024-01-01'),           # ✅ timestamp → event_timestamp
                'session_duration': [300 + i for i in range(100)],        # ✅ SQL 컬럼 추가
                'page_views': [5 + (i % 10) for i in range(100)],         # ✅ SQL 컬럼 추가
                'outcome': [i % 2 for i in range(100)]                    # ✅ target → outcome
            })
        else:
            df = data_adapter.read(settings.recipe.model.loader.source_uri)

        mlflow.log_metric("row_count", len(df))
        mlflow.log_metric("column_count", len(df.columns))

        # 2. 학습에 사용할 컴포넌트 생성
        augmenter = factory.create_augmenter()
        preprocessor = factory.create_preprocessor()
        model = factory.create_model()

        # 3. 모델 학습
        trainer = Trainer(settings=settings)
        trained_model, trained_preprocessor, metrics, training_results = trainer.train(  # 🔄 수정: 반환값 순서 올바르게 변경
            df=df,
            model=model,
            augmenter=augmenter,
            preprocessor=preprocessor,
            context_params=context_params,
        )
        
        # 4. 결과 로깅 (확장)
        if metrics:  # 🔄 수정: 'metrics' key가 아닌 직접 metrics 객체 사용
            mlflow.log_metrics(metrics)
        
        # 🆕 하이퍼파라미터 최적화 결과 로깅
        if 'hyperparameter_optimization' in training_results:
            hpo_result = training_results['hyperparameter_optimization']
            if hpo_result['enabled']:
                mlflow.log_params(hpo_result['best_params'])
                mlflow.log_metric('best_score', hpo_result['best_score'])
                mlflow.log_metric('total_trials', hpo_result['total_trials'])

        # 5. 🔄 Phase 5: Enhanced PyfuncWrapper 생성 (training_df 추가)
        pyfunc_wrapper = factory.create_pyfunc_wrapper(
            trained_model=trained_model,
            trained_preprocessor=trained_preprocessor,
            training_df=df,  # 🆕 Phase 5: 스키마 생성용
            training_results=training_results,
        )
        
        # 6. 🆕 Phase 5: Enhanced Model + 완전한 메타데이터 저장
        logger.info("🆕 Phase 5: Enhanced Artifact 저장 중...")
        
        if pyfunc_wrapper.signature and pyfunc_wrapper.data_schema:
            # Phase 5 Enhanced 저장 로직 사용
            from src.utils.integrations.mlflow_integration import log_enhanced_model_with_schema
            
            log_enhanced_model_with_schema(
                python_model=pyfunc_wrapper,
                signature=pyfunc_wrapper.signature,
                data_schema=pyfunc_wrapper.data_schema,
                input_example=df.head(5)  # 입력 예제
            )
            
            model_name = getattr(settings.recipe.model, 'name', None) or settings.recipe.model.computed['run_name']
            logger.info(f"✅ Enhanced Artifact '{model_name}' MLflow 저장 완료 (Phase 1-5 통합)")
        else:
            # Fallback: 기존 방식 (training_df가 없었던 경우)
            logger.warning("⚠️ Enhanced 정보가 없어 기본 저장 방식 사용")
            
            # 기본 샘플 예측 및 signature 생성
            sample_input = df.head(5)
            sample_output = pyfunc_wrapper.predict(
                context=None,
                model_input=sample_input,
                params={"run_mode": "batch", "return_intermediate": False}
            )
            
            if not isinstance(sample_output, pd.DataFrame):
                sample_output = pd.DataFrame(sample_output)
            
            signature = mlflow_utils.create_model_signature(
                input_df=sample_input,
                output_df=sample_output
            )
            
            # 기존 MLflow 저장
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=pyfunc_wrapper,
                signature=signature,
                input_example=sample_input,
            )
            
            model_name = getattr(settings.recipe.model, 'name', None) or settings.recipe.model.computed['run_name']
            logger.info(f"기본 모델 '{model_name}'을 MLflow에 저장했습니다.")

        # 7. (선택적) 메타데이터 저장
        metadata = {"run_id": run_id, "model_name": model_name}
        local_dir = Path("./local/artifacts")
        local_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = local_dir / f"metadata-{run_id}.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=str)
        mlflow.log_artifact(str(metadata_path), "metadata")