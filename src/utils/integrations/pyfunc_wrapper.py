from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

from src.utils.core.logger import logger


class PyfuncWrapper(mlflow.pyfunc.PythonModel):
    """
    학습된 컴포넌트와 설정 스냅샷을 캡슐화하는 MLflow PythonModel 구현체.
    기존 구현을 이동하되, 직렬화 안전성을 유지합니다.
    """

    def __init__(
        self,
        settings: Any,
        trained_model: Any,
        trained_datahandler: Optional[Any] = None,
        trained_preprocessor: Optional[Any] = None,
        trained_fetcher: Optional[Any] = None,
        trained_calibrator: Optional[Any] = None,
        training_results: Optional[Dict[str, Any]] = None,
        signature: Optional[Any] = None,  # mlflow.models.ModelSignature
        data_schema: Optional[Any] = None,  # mlflow.types.Schema
        data_interface_schema: Optional[Dict[str, Any]] = None,
    ):
        # 직렬화 가능한 최소한의 설정 정보만 추출
        self._task_type, self.settings_dict = self._extract_serializable_settings(settings)

        self.trained_model = trained_model
        # 재현성 확보: 가능한 한 학습된 객체를 보존하되, 직렬화 시 불필요 필드는 제거
        self.trained_datahandler = trained_datahandler if trained_datahandler is not None else None
        self.trained_preprocessor = (
            trained_preprocessor if trained_preprocessor is not None else None
        )
        self.trained_fetcher = trained_fetcher if trained_fetcher is not None else None
        self.trained_calibrator = trained_calibrator
        self.training_results = training_results or {}
        self.signature = signature
        self.data_schema = data_schema
        self.data_interface_schema = data_interface_schema

        # Task type별 추론 파이프라인 필요성
        self._requires_datahandler = self._task_type in ["timeseries"]

    def _extract_serializable_settings(self, settings):
        """설정에서 직렬화 가능한 최소한의 정보만 추출"""
        try:
            if hasattr(settings, "model_dump"):
                task_type = settings.recipe.task_choice
                # loader 섹션에서 source_uri 추출 (추론 시 동적 파라미터 지원용)
                loader_source_uri = ""
                if hasattr(settings.recipe.data, "loader") and settings.recipe.data.loader:
                    loader_source_uri = getattr(settings.recipe.data.loader, "source_uri", "")
                    # source_uri가 파일 경로인 경우 파일 내용을 읽어서 저장
                    if loader_source_uri and (
                        loader_source_uri.endswith(".sql.j2") or loader_source_uri.endswith(".sql")
                    ):
                        try:
                            from pathlib import Path

                            sql_path = Path(loader_source_uri)
                            if sql_path.exists():
                                loader_source_uri = sql_path.read_text(encoding="utf-8")
                                logger.debug(f"[PYFUNC] SQL 템플릿 파일 로드: {sql_path}")
                        except Exception as e:
                            logger.warning(f"[PYFUNC] SQL 파일 로드 실패, 경로 유지: {e}")

                settings_dict = {
                    "recipe": {
                        "task_choice": task_type,
                        "model": {
                            "class_path": getattr(settings.recipe.model, "class_path", "unknown")
                        },
                        "data": {
                            "loader": {"source_uri": loader_source_uri},
                            "data_interface": {
                                "target_column": getattr(
                                    settings.recipe.data.data_interface, "target_column", None
                                ),
                                "feature_columns": getattr(
                                    settings.recipe.data.data_interface, "feature_columns", None
                                ),
                                "entity_columns": getattr(
                                    settings.recipe.data.data_interface, "entity_columns", []
                                ),
                            },
                        },
                    }
                }
                return task_type, settings_dict
            elif isinstance(settings, dict):
                task_type = settings.get("recipe", {}).get("task_choice", "unknown")
                return task_type, settings
            else:
                task_type = (
                    getattr(settings.recipe, "task_choice", "unknown")
                    if hasattr(settings, "recipe")
                    else "unknown"
                )
                settings_dict = {"recipe": {"task_choice": task_type}}
                return task_type, settings_dict
        except Exception:
            return "unknown", {"recipe": {"task_choice": "unknown"}}

    def __getstate__(self):
        state = self.__dict__.copy()
        # settings를 직렬화 가능 최소 dict로 제한
        try:
            if "settings_dict" in state and not isinstance(state["settings_dict"], dict):
                state["settings_dict"] = {"recipe": {"task_choice": "unknown"}}
        except Exception:
            pass
        # 직렬화 안전화를 위해 중첩 객체의 비직렬화 필드 제거 시도
        try:
            pre = state.get("trained_preprocessor")
            if pre is not None:
                if hasattr(pre, "console"):
                    pre.console = None
                if hasattr(pre, "settings"):
                    pre.settings = None
        except Exception:
            pass
        try:
            dh = state.get("trained_datahandler")
            if dh is not None and hasattr(dh, "console"):
                dh.console = None
        except Exception:
            pass
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _validate_input_schema(self, df: pd.DataFrame):
        if self.data_schema:
            try:
                ts_col = (
                    self.data_schema.get("timestamp_column")
                    if isinstance(self.data_schema, dict)
                    else None
                )
                if (
                    ts_col
                    and ts_col in df.columns
                    and not pd.api.types.is_datetime64_any_dtype(df[ts_col])
                ):
                    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                from src.utils.schema.schema_utils import SchemaConsistencyValidator

                validator = SchemaConsistencyValidator(self.data_schema)
                validator.validate_inference_consistency(df)
                logger.debug("[INFER] 입력 스키마 검증 완료")
            except ValueError as e:
                logger.error(f"[INFER] 스키마 검증 실패 (Schema Drift 감지): {e}")
                raise

    @property
    def model_class_path(self) -> str:
        return self.settings_dict.get("recipe", {}).get("model", {}).get("class_path", "unknown")

    @property
    def loader_sql_snapshot(self) -> str:
        return (
            self.settings_dict.get("recipe", {})
            .get("data", {})
            .get("loader", {})
            .get("source_uri", "")
        )

    @property
    def fetcher_config_snapshot(self) -> Dict[str, Any]:
        fetcher = self.settings_dict.get("recipe", {}).get("data", {}).get("fetcher", {})
        return fetcher if fetcher else {}

    @property
    def recipe_yaml_snapshot(self) -> str:
        import yaml

        recipe = self.settings_dict.get("recipe", {})
        return yaml.dump(recipe)

    @property
    def hyperparameter_optimization(self) -> Dict[str, Any]:
        return self.training_results.get("hyperparameter_optimization", {})

    @property
    def training_methodology(self) -> Dict[str, Any]:
        return self.training_results.get("training_methodology", {})

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray, List]:
        run_mode = params.get("run_mode", "batch") if params else "batch"
        logger.debug(f"[INFER] 예측 호출 - params: {params}")

        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Fetcher를 통한 피처 증강 (Online/Offline Store)
        if self.trained_fetcher is not None:
            try:
                model_input = self.trained_fetcher.fetch(model_input, run_mode=run_mode)
                logger.debug(f"[INFER] 피처 증강 완료 (run_mode={run_mode}): {model_input.shape}")
            except Exception as e:
                logger.warning(f"[INFER] 피처 증강 실패, 원본 데이터로 진행: {e}")

        try:
            if self.data_interface_schema:
                logger.debug("[INFER] 기본 입력 검증 통과")
        except Exception:
            logger.warning("[INFER] 입력 검증 건너뜀")

        try:
            # 피처 컬럼 정보 추출 (전처리 전/후 분리)
            model_feature_columns = None
            target_col = None

            if self.data_interface_schema:
                model_feature_columns = self.data_interface_schema.get("model_feature_columns")
                target_col = self.data_interface_schema.get("target_column")

            # 1단계: DataHandler 변환 적용 (시퀀스 변환, 시간 피처 생성 등)
            if self.trained_datahandler is not None and hasattr(self.trained_datahandler, "transform"):
                try:
                    X = self.trained_datahandler.transform(model_input)
                    logger.debug(f"[INFER] DataHandler 변환 완료: {X.shape}")
                except Exception as e:
                    logger.warning(f"[INFER] DataHandler 변환 실패, 원본으로 진행: {e}")
                    # 폴백: target 컬럼 제외하고 진행
                    exclude_cols = {target_col} if target_col else set()
                    feature_columns = [col for col in model_input.columns if col not in exclude_cols]
                    X = model_input[feature_columns] if feature_columns else model_input
            else:
                # DataHandler 없을 시 target 제외 전체 컬럼 사용
                exclude_cols = {target_col} if target_col else set()
                feature_columns = [col for col in model_input.columns if col not in exclude_cols]
                X = model_input[feature_columns] if feature_columns else model_input
                logger.debug("[INFER] DataHandler 없음, target 제외 전체 컬럼 사용")

            # 2단계: Preprocessor 변환 적용 (스케일링, 결측치 처리 등)
            if self.trained_preprocessor is not None:
                try:
                    X = self.trained_preprocessor.transform(X, dataset_name="infer")
                    logger.debug(f"[INFER] Preprocessor 변환 완료: {X.shape}")
                except Exception as e:
                    logger.warning(f"[INFER] Preprocessor 변환 실패, 이전 단계 결과로 진행: {e}")

            # 3단계: 모델 피처 순서 정렬
            if model_feature_columns:
                available_model_features = [col for col in model_feature_columns if col in X.columns]
                if len(available_model_features) == len(model_feature_columns):
                    X = X[model_feature_columns]
                    logger.debug(f"[INFER] 모델 피처 정렬: {len(model_feature_columns)}개 컬럼")
                else:
                    missing = set(model_feature_columns) - set(X.columns)
                    if missing:
                        logger.debug(f"[INFER] 일부 모델 피처 누락 (정상일 수 있음): {len(missing)}개")

            # Check if we should return probabilities or classes
            return_probabilities = params and params.get("return_probabilities", False)

            if return_probabilities and hasattr(self.trained_model, "predict_proba"):
                # Get probability predictions
                predictions = self.trained_model.predict_proba(X)

                # Apply calibration if available
                if self.trained_calibrator is not None and self._task_type == "classification":
                    logger.debug("[INFER] 확률 캘리브레이션 적용")
                    predictions = self.trained_calibrator.transform(predictions)

            elif (
                self._task_type == "classification"
                and hasattr(self.trained_model, "predict_proba")
                and self.trained_calibrator is not None
            ):
                # For classification with calibrator, always use calibrated probabilities for consistency
                predictions = self.trained_model.predict_proba(X)
                predictions = self.trained_calibrator.transform(predictions)

                # Convert probabilities to class predictions if not explicitly requesting probabilities
                if not return_probabilities:
                    if predictions.ndim == 2:
                        predictions = predictions.argmax(axis=1)
                    else:
                        # Binary classification case with calibrated probabilities
                        predictions = (predictions > 0.5).astype(int)
            else:
                # Standard prediction without calibration
                predictions = self.trained_model.predict(X)

            should_return_dataframe = params and params.get("return_dataframe", False)
            if should_return_dataframe:
                if not isinstance(predictions, pd.DataFrame):
                    # Causal 모델: CATE (Conditional Average Treatment Effect) 컬럼명 사용
                    if self._task_type == "causal":
                        predictions_array = np.atleast_1d(predictions)
                        if predictions_array.ndim == 2:
                            # 다중 treatment 효과
                            cols = [
                                f"cate_treatment_{i}" for i in range(predictions_array.shape[1])
                            ]
                            predictions_df = pd.DataFrame(
                                predictions_array, columns=cols, index=model_input.index
                            )
                        else:
                            predictions_df = pd.DataFrame(
                                {"cate": predictions_array}, index=model_input.index
                            )
                    # Classification 모델: 확률 컬럼명
                    elif return_probabilities and predictions.ndim == 2:
                        prob_cols = [f"prob_class_{i}" for i in range(predictions.shape[1])]
                        predictions_df = pd.DataFrame(
                            predictions, columns=prob_cols, index=model_input.index
                        )
                    elif return_probabilities and predictions.ndim == 1:
                        # Binary classification probabilities
                        predictions_df = pd.DataFrame(
                            {"prob_positive": predictions}, index=model_input.index
                        )
                    else:
                        # Regression 등 일반 예측
                        predictions_df = pd.DataFrame(
                            {"prediction": predictions}, index=model_input.index
                        )
                    logger.info(f"[INFER] 예측 완료: {len(predictions_df)}샘플 (DataFrame)")
                    return predictions_df
                else:
                    logger.info(f"[INFER] 예측 완료: {len(predictions)}샘플 (DataFrame)")
                    return predictions
            else:
                if isinstance(predictions, pd.DataFrame):
                    predictions = predictions.values.flatten()
                elif hasattr(predictions, "tolist"):
                    predictions = predictions.tolist()
                logger.info(f"[INFER] 예측 완료: {len(predictions)}샘플")
                return predictions

        except Exception as e:
            logger.debug(f"[INFER] 예측 실패: {e}")
            # 예외를 전파하여 호출자가 실패를 인지할 수 있도록 함 (Silent Failure 방지)
            raise
