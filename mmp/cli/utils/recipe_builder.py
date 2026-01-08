"""
Recipe Builder for Modern ML Pipeline CLI

Registry 기반 동적 Recipe 생성기입니다.
대화형 인터페이스를 통해 Task, Model, Preprocessor 등을 선택하여 Recipe 파일을 생성합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from mmp.cli.utils.interactive_ui import InteractiveUI
from mmp.cli.utils.template_engine import TemplateEngine
from mmp.settings.validation.business_validator import BusinessValidator
from mmp.settings.validation.catalog_validator import CatalogValidator
from mmp.utils.core.logger import logger


class RecipeBuilder:
    """Registry 기반 동적 Recipe 빌더 - 하드코딩 완전 제거"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.ui = InteractiveUI()

        # 템플릿 엔진 초기화
        templates_dir = Path(__file__).parent.parent / "templates"
        self.template_engine = TemplateEngine(templates_dir)

    def get_available_tasks(self) -> Set[str]:
        """models/catalog 기반 동적 Task 목록"""
        return self.catalog_validator.get_available_tasks()

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들"""
        return self.catalog_validator.get_available_models_for_task(task_type)

    def get_available_preprocessors(self) -> Dict[str, List[str]]:
        """Registry 기반 전처리기 분류 및 제공 (상호 배타성 고려)"""
        available_types = self.business_validator.get_available_preprocessor_types()

        categorized = {
            "Scalers": [],
            "Encoders": [],
            "Missing_Statistical": [],
            "Missing_TimeSeries": [],
            "Missing_Drop": [],
            "Feature Engineering": [],
        }

        for step_type in available_types:
            category = self._categorize_preprocessor(step_type)
            if category in categorized:
                categorized[category].append(step_type)

        return {k: sorted(v) for k, v in categorized.items() if v}

    def get_available_metrics_for_task(self, task_type: str) -> List[str]:
        """Registry 기반 Task별 메트릭 제공"""
        return self.business_validator.get_available_evaluators_for_task(task_type)

    def build_recipe_interactively(self) -> Dict[str, Any]:
        """사용자 대화형 Recipe 생성 - 모든 옵션이 동적"""
        total_steps = 6

        # Step 1: Task 선택
        self.ui.show_step(1, total_steps, "Task 선택")
        self.ui.show_info(
            "해결하려는 ML 문제 유형을 선택합니다."
        )
        available_tasks = self.get_available_tasks()
        if not available_tasks:
            raise ValueError(
                "사용 가능한 Task가 없습니다. models/catalog 디렉토리를 확인해주세요."
            )

        task_descriptions = {
            "classification": "classification - 범주 예측 (이진/다중 분류)",
            "regression": "regression - 연속값 예측",
            "timeseries": "timeseries - 시계열 예측",
            "clustering": "clustering - 비지도 군집화",
            "causal": "causal - 인과 추론 (처리 효과 추정)",
        }
        task_options = [
            task_descriptions.get(t, t) for t in available_tasks
        ]

        task_choice_display = self.ui.select_from_list(
            "Task 유형:", task_options, allow_cancel=False
        )
        task_choice = task_choice_display.split(" - ")[0]

        # Step 2: 모델 선택
        self.ui.show_step(2, total_steps, "Model 선택")
        self.ui.show_info(
            f"'{task_choice}' Task에 사용할 알고리즘을 선택합니다."
        )
        available_models = self.get_available_models_for_task(task_choice)
        if not available_models:
            raise ValueError(f"{task_choice}에 사용 가능한 모델이 없습니다.")

        model_name = self.ui.select_from_list(
            "Model:", list(available_models.keys()), allow_cancel=False
        )
        selected_model = available_models[model_name]

        # Step 3: Recipe 생성 방식 선택
        self.ui.show_step(3, total_steps, "Recipe 생성 방식")
        self.ui.show_info(
            "Cheat sheet(권장 설정 자동) 또는 Manual(직접 설정)을 선택합니다."
        )
        recipe_mode = self.ui.select_from_list(
            "생성 방식:",
            [
                "Cheat sheet - 권장 설정 자동 적용 (Optuna, Calibration, 전처리)",
                "Manual - 세부사항 직접 설정",
            ],
            allow_cancel=False,
        )

        if recipe_mode.startswith("Cheat sheet"):
            return self._build_cheat_sheet_recipe(task_choice, selected_model, model_name)

        # Step 4: 전처리기 선택 (Manual 모드)
        self.ui.show_step(4, total_steps, "Preprocessor 설정")
        self.ui.show_info(
            "Scaler, Encoder, 결측값 처리 등 전처리 파이프라인을 구성합니다."
        )
        preprocessor_steps = self._collect_preprocessor_steps()

        # Step 5: 모델 설정 (Manual 모드)
        self.ui.show_step(5, total_steps, "Model 설정")
        self.ui.show_info(
            "Hyperparameter, Calibration, 데이터 분할 비율을 설정합니다."
        )
        model_config = self._build_model_config(selected_model, task_choice)
        data_interface_config = self._build_data_interface_config(task_choice, selected_model)
        data_split_config = self._build_data_split_config(
            task_choice, model_config.get("calibration", {})
        )

        # Recipe 조립
        recipe_data = {
            "name": f"{task_choice}_{model_name}_{datetime.now().strftime('%Y%m%d')}",
            "task_choice": task_choice,
            "model": model_config,
            "data": {
                "loader": {"source_uri": None},
                "fetcher": {"type": "pass_through"},
                "data_interface": data_interface_config,
                "split": data_split_config,
            },
            "preprocessor": {"steps": preprocessor_steps} if preprocessor_steps else None,
            "metadata": {
                "author": "CLI Recipe Builder",
                "description": f"{task_choice} task using {selected_model['library']}",
            },
        }

        # Step 6: 설정 확인
        self.ui.show_step(6, total_steps, "설정 확인")
        if not self._show_manual_summary_and_confirm(recipe_data, task_choice, preprocessor_steps):
            raise ValueError("사용자가 설정을 취소했습니다.")

        return recipe_data

    def _build_cheat_sheet_recipe(
        self, task_choice: str, selected_model: Dict, model_name: str
    ) -> Dict[str, Any]:
        """권장 설정으로 자동 Recipe 생성 (Cheat Sheet 모드)"""
        self.ui.show_step(4, 6, "Cheat Sheet 적용")

        # 1. 카탈로그에서 권장 전처리기 로드
        preprocessor_steps = self._get_recommended_preprocessor(selected_model)
        if preprocessor_steps:
            self.ui.show_info(f"권장 전처리기 {len(preprocessor_steps)}개 자동 적용")

        # 2. 모델 설정 (Optuna 활성화, 캘리브레이션 자동)
        model_config = self._build_cheat_sheet_model_config(selected_model, task_choice)

        # 3. 데이터 인터페이스 설정 (sequence 모델은 사용자 입력 필요)
        data_interface_config = self._build_data_interface_config(task_choice, selected_model)

        # 4. 데이터 분할 설정 (캘리브레이션 고려)
        has_calibration = (
            task_choice.lower() == "classification"
            and model_config.get("calibration", {}).get("enabled", False)
        )
        if has_calibration:
            data_split_config = {
                "train": 0.7,
                "validation": 0.15,
                "test": 0.1,
                "calibration": 0.05,
            }
        else:
            data_split_config = {"train": 0.7, "validation": 0.15, "test": 0.15}

        # 5. Recipe 조립
        recipe_data = {
            "name": f"{task_choice}_{model_name}_{datetime.now().strftime('%Y%m%d')}",
            "task_choice": task_choice,
            "model": model_config,
            "data": {
                "loader": {"source_uri": None},
                "fetcher": {"type": "pass_through"},
                "data_interface": data_interface_config,
                "split": data_split_config,
            },
            "preprocessor": {"steps": preprocessor_steps} if preprocessor_steps else None,
            "metadata": {
                "author": "CLI Recipe Builder (Cheat Sheet)",
                "description": f"{task_choice} task using {selected_model['library']}",
            },
        }

        # Step 5: 설정 요약 및 확인
        self.ui.show_step(5, 6, "설정 확인")
        if not self._show_cheat_sheet_summary_and_confirm(recipe_data, task_choice, preprocessor_steps):
            raise ValueError("사용자가 설정을 취소했습니다.")

        # Step 6: Recipe 생성
        self.ui.show_step(6, 6, "Recipe 생성")

        return recipe_data

    def _get_recommended_preprocessor(self, model_spec: Dict) -> List[Dict]:
        """카탈로그에서 권장 전처리기 로드"""
        recommended = model_spec.get("recommended_preprocessor")
        if not recommended:
            return []

        steps = []
        for step_config in recommended:
            step = {"type": step_config["type"]}
            for key, value in step_config.items():
                if key != "type":
                    step[key] = value
            steps.append(step)

        return steps

    def _build_cheat_sheet_model_config(
        self, selected_model: Dict, task_choice: str
    ) -> Dict:
        """Cheat Sheet용 모델 설정 (Optuna 활성화, 캘리브레이션 자동)"""
        hyperparams = selected_model.get("hyperparameters", {}) or {}

        default_metric = self.business_validator.get_default_optimization_metric(task_choice)

        model_config = {
            "class_path": selected_model["class_path"],
            "library": selected_model["library"],
            "hyperparameters": {
                "tuning_enabled": True,
                "optimization_metric": default_metric,
                "n_trials": 20,
                "timeout": 600,
                "fixed": hyperparams.get("fixed", {}),
                "tunable": hyperparams.get("tunable", {}),
            },
        }

        # Classification: 권장 캘리브레이션 적용
        if task_choice.lower() == "classification":
            recommended_cal = selected_model.get("recommended_calibration")
            if recommended_cal:
                model_config["calibration"] = {"enabled": True, "method": recommended_cal}
            else:
                model_config["calibration"] = {"enabled": True, "method": "isotonic"}

        return model_config

    def _show_cheat_sheet_summary_and_confirm(
        self, recipe_data: Dict, task_choice: str, preprocessor_steps: List[Dict]
    ) -> bool:
        """Cheat Sheet 적용 결과 요약 및 확인"""
        hp = recipe_data["model"]["hyperparameters"]

        # 요약 데이터 구성
        summary_data = {
            "Optuna": f"n_trials={hp['n_trials']}, timeout={hp['timeout']}s, metric={hp['optimization_metric']}",
        }

        # Calibration
        if task_choice.lower() == "classification":
            cal = recipe_data["model"].get("calibration", {})
            if cal.get("enabled"):
                summary_data["Calibration"] = cal["method"]

        # Preprocessor
        if preprocessor_steps:
            step_types = [s["type"] for s in preprocessor_steps]
            summary_data["Preprocessor"] = " -> ".join(step_types)

        # Data Split
        split = recipe_data["data"]["split"]
        split_str = ", ".join(f"{k}={int(v*100)}%" for k, v in split.items())
        summary_data["Data Split"] = split_str

        return self.ui.show_summary_and_confirm(summary_data, title="Cheat Sheet 설정 요약")

    def _show_manual_summary_and_confirm(
        self, recipe_data: Dict, task_choice: str, preprocessor_steps: List[Dict]
    ) -> bool:
        """Manual 모드 설정 요약 및 확인"""
        hp = recipe_data["model"]["hyperparameters"]

        # 요약 데이터 구성
        summary_data = {}

        # HPO 설정
        if hp.get("tuning_enabled"):
            summary_data["HPO"] = f"n_trials={hp.get('n_trials', 10)}, timeout={hp.get('timeout', 300)}s, metric={hp.get('optimization_metric', 'N/A')}"
        else:
            summary_data["HPO"] = "비활성화"

        # Calibration
        if task_choice.lower() == "classification":
            cal = recipe_data["model"].get("calibration", {})
            if cal.get("enabled"):
                summary_data["Calibration"] = cal["method"]
            else:
                summary_data["Calibration"] = "비활성화"

        # Preprocessor
        if preprocessor_steps:
            step_types = [s["type"] for s in preprocessor_steps]
            summary_data["Preprocessor"] = " -> ".join(step_types)
        else:
            summary_data["Preprocessor"] = "없음"

        # Data Split
        split = recipe_data["data"]["split"]
        split_str = ", ".join(f"{k}={int(v*100)}%" for k, v in split.items())
        summary_data["Data Split"] = split_str

        return self.ui.show_summary_and_confirm(summary_data, title="Manual 설정 요약")

    def _build_model_config(self, selected_model: Dict, task_choice: str) -> Dict:
        """모델 설정 구성"""
        model_config = {
            "class_path": selected_model["class_path"],
            "library": selected_model["library"],
            "hyperparameters": self._configure_hyperparameters(selected_model, task_choice),
        }

        if task_choice.lower() == "classification":
            self.ui.show_info("Calibration: 예측 확률을 실제 확률에 가깝게 보정합니다.")
            calibration_enabled = self.ui.confirm(
                "Calibration을 사용하시겠습니까?", default=False
            )
            if calibration_enabled:
                available_methods = self.business_validator.get_available_calibrators()
                calibration_method = self.ui.select_from_list(
                    "Calibration method:", list(available_methods), allow_cancel=False
                )
                model_config["calibration"] = {"enabled": True, "method": calibration_method}
            else:
                model_config["calibration"] = {"enabled": False}

        return model_config

    def _build_data_interface_config(self, task_choice: str, model_spec: Dict) -> Dict:
        """데이터 인터페이스 설정 구성"""
        data_interface = {
            "entity_columns": [],
            "feature_columns": None,
        }

        if task_choice.lower() != "clustering":
            data_interface["target_column"] = None

        if task_choice.lower() == "causal":
            data_interface["treatment_column"] = None

        if task_choice.lower() == "timeseries":
            data_interface["timestamp_column"] = None

        data_handler = model_spec.get("data_handler", "tabular")
        if data_handler == "sequence":
            self.ui.show_info("Sequence length: 시계열 예측에 사용할 과거 데이터 기간")
            seq_len = self.ui.number_input(
                "Sequence length (lookback window):", default=7, min_value=1, max_value=365
            )
            data_interface["sequence_length"] = int(seq_len)

        return data_interface

    def _build_data_split_config(self, task_choice: str, calibration_config: Dict) -> Dict:
        """데이터 분할 설정 구성"""
        has_calibration = (
            task_choice.lower() == "classification" and calibration_config.get("enabled", False)
        )

        if has_calibration:
            default_train, default_val, default_test, default_cal = 70, 15, 10, 5
        else:
            default_train, default_val, default_test = 70, 15, 15

        if self.ui.confirm("데이터 분할 비율을 직접 설정하시겠습니까?", default=False):
            train_pct = self.ui.number_input(
                "Train (%):", default=default_train, min_value=50, max_value=90
            )
            val_pct = self.ui.number_input(
                "Validation (%):", default=default_val, min_value=0, max_value=30
            )
            test_pct = self.ui.number_input(
                "Test (%):", default=default_test, min_value=5, max_value=30
            )

            if has_calibration:
                cal_pct = self.ui.number_input(
                    "Calibration (%):", default=default_cal, min_value=5, max_value=20
                )
                total = train_pct + val_pct + test_pct + cal_pct
            else:
                cal_pct = 0
                total = train_pct + val_pct + test_pct

            if total != 100:
                self.ui.show_warning(f"비율 합계가 {total}%입니다. 100%로 정규화합니다.")
                train_pct = train_pct * 100 / total
                val_pct = val_pct * 100 / total
                test_pct = test_pct * 100 / total
                if has_calibration:
                    cal_pct = cal_pct * 100 / total

            data_split = {
                "train": round(train_pct / 100, 2),
                "validation": round(val_pct / 100, 2),
                "test": round(test_pct / 100, 2),
            }
            if has_calibration:
                data_split["calibration"] = round(cal_pct / 100, 2)
        else:
            if has_calibration:
                data_split = {"train": 0.7, "validation": 0.15, "test": 0.1, "calibration": 0.05}
            else:
                data_split = {"train": 0.7, "validation": 0.15, "test": 0.15}

        return data_split

    def _configure_hyperparameters(self, selected_model: Dict, task_choice: str) -> Dict:
        """하이퍼파라미터 설정 구성"""
        self.ui.show_info("HPO: Optuna 기반 하이퍼파라미터 자동 최적화")
        enable_tuning = self.ui.confirm(
            "Hyperparameter tuning을 활성화하시겠습니까?", default=False
        )

        hyperparams = selected_model.get("hyperparameters", {})
        if hyperparams is None:
            hyperparams = {}

        if enable_tuning:
            default_metric = self.business_validator.get_default_optimization_metric(task_choice)
            return {
                "tuning_enabled": True,
                "optimization_metric": default_metric,
                "n_trials": 10,
                "timeout": 300,
                "fixed": hyperparams.get("fixed", {}),
                "tunable": hyperparams.get("tunable", {}),
            }
        else:
            fixed_params = hyperparams.get("fixed") or {}
            values = fixed_params.copy()

            tunable_params = hyperparams.get("tunable") or {}
            for param, config in tunable_params.items():
                values[param] = config.get("default", config.get("range", [1])[0])

            return {"tuning_enabled": False, "values": values}

    def _collect_preprocessor_steps(self) -> List[Dict]:
        """Registry 기반 전처리기 수집 (상호 배타성 적용)"""
        available = self.get_available_preprocessors()
        selected_steps = []

        # 1. Scalers (상호 배타)
        if "Scalers" in available and available["Scalers"]:
            self.ui.show_info("Scaler: 숫자형 피처의 스케일을 조정합니다.")
            scalers = available["Scalers"]
            scaler_options = {
                "standard_scaler": "StandardScaler - 평균=0, 표준편차=1로 정규화",
                "min_max_scaler": "MinMaxScaler - 0~1 범위로 스케일링",
                "robust_scaler": "RobustScaler - 중앙값/IQR 기반 (이상치에 강건)",
            }
            display_options = [scaler_options.get(s, s) for s in scalers]
            display_options.append("(사용 안 함)")

            choice = self.ui.select_from_list(
                "Scaler:", display_options, allow_cancel=False
            )
            if choice != "(사용 안 함)":
                actual_choice = choice.split(" - ")[0].lower().replace("scaler", "_scaler")
                # StandardScaler -> standard_scaler 변환
                type_map = {
                    "standard_scaler": "standard_scaler",
                    "minmax_scaler": "min_max_scaler",
                    "robust_scaler": "robust_scaler",
                }
                for key in scalers:
                    if key in choice.lower() or choice.lower().startswith(key.split("_")[0]):
                        actual_choice = key
                        break
                selected_steps.append(
                    {"type": actual_choice, **self._configure_preprocessor_params(actual_choice)}
                )

        # 2. Encoders (상호 배타)
        if "Encoders" in available and available["Encoders"]:
            self.ui.show_info("Encoder: 범주형 피처를 수치로 변환합니다.")
            encoders = available["Encoders"]
            encoder_options = {
                "ordinal_encoder": "OrdinalEncoder - 범주를 정수로 매핑",
                "one_hot_encoder": "OneHotEncoder - 범주별 이진 컬럼 생성",
                "catboost_encoder": "CatBoostEncoder - Target 기반 인코딩",
            }
            display_options = [encoder_options.get(e, e) for e in encoders]
            display_options.append("(사용 안 함)")

            choice = self.ui.select_from_list(
                "Encoder:", display_options, allow_cancel=False
            )
            if choice != "(사용 안 함)":
                actual_choice = None
                for enc in encoders:
                    if enc in choice.lower() or choice.lower().startswith(enc.split("_")[0]):
                        actual_choice = enc
                        break
                if actual_choice:
                    selected_steps.append(
                        {"type": actual_choice, **self._configure_preprocessor_params(actual_choice)}
                    )

        # 3. Missing Value Handling
        has_missing_options = any(
            k in available for k in ("Missing_Drop", "Missing_Statistical", "Missing_TimeSeries")
        )
        if has_missing_options:
            self.ui.show_info("Missing values: 결측값 처리 방법을 선택합니다.")
            missing_approach = self.ui.select_from_list(
                "Missing value 처리:",
                ["Drop - 결측값이 있는 행/열 삭제", "Impute - 값으로 대체", "(사용 안 함)"],
                allow_cancel=False,
            )

            if missing_approach.startswith("Drop"):
                if "Missing_Drop" in available and "drop_missing" in available["Missing_Drop"]:
                    selected_steps.append(
                        {
                            "type": "drop_missing",
                            **self._configure_preprocessor_params("drop_missing"),
                        }
                    )

            elif missing_approach.startswith("Impute"):
                impute_options = []
                impute_labels = {}
                if "Missing_Statistical" in available:
                    for opt in available["Missing_Statistical"]:
                        impute_options.append(opt)
                        if opt == "simple_imputer":
                            impute_labels[opt] = "SimpleImputer - mean/median/mode로 대체"
                        elif opt == "constant_fill":
                            impute_labels[opt] = "ConstantFill - 고정값으로 대체"
                if "Missing_TimeSeries" in available:
                    for opt in available["Missing_TimeSeries"]:
                        impute_options.append(opt)
                        if opt == "forward_fill":
                            impute_labels[opt] = "ForwardFill - 이전 값으로 채움"
                        elif opt == "backward_fill":
                            impute_labels[opt] = "BackwardFill - 다음 값으로 채움"
                        elif opt == "interpolation":
                            impute_labels[opt] = "Interpolation - 보간법"

                if impute_options:
                    display_opts = [impute_labels.get(o, o) for o in impute_options]
                    selected = self.ui.multi_select(
                        "Imputation method (복수 선택 가능):", display_opts
                    )
                    for method_display in selected:
                        for opt in impute_options:
                            if opt in method_display.lower() or method_display.startswith(
                                opt.replace("_", " ").title().split()[0]
                            ):
                                selected_steps.append(
                                    {"type": opt, **self._configure_preprocessor_params(opt)}
                                )
                                break

        # 4. Feature Engineering (다중 선택)
        if "Feature Engineering" in available and available["Feature Engineering"]:
            self.ui.show_info(
                "Feature Engineering: 기존 피처를 조합하여 새로운 컬럼을 자동 생성합니다."
            )
            fe_options = available["Feature Engineering"]
            fe_labels = {
                "polynomial_features": "PolynomialFeatures - 다항식/상호작용 피처",
                "kbins_discretizer": "KBinsDiscretizer - 연속값을 구간으로 이산화",
                "tree_based_feature_generator": "TreeBasedFeatures - 트리 기반 피처 생성",
            }
            display_options = [fe_labels.get(fe, fe) for fe in fe_options]

            if self.ui.confirm(
                "Feature Engineering (자동 추가 컬럼 생성)을 사용하시겠습니까?", default=False
            ):
                selected = self.ui.multi_select("Feature Engineering:", display_options)
                for method_display in selected:
                    for fe in fe_options:
                        if fe in method_display.lower() or method_display.startswith(
                            fe.replace("_", " ").title().split()[0]
                        ):
                            selected_steps.append(
                                {
                                    "type": fe,
                                    **self._configure_preprocessor_params(fe),
                                }
                            )
                            break

        return selected_steps

    def _configure_preprocessor_params(self, step_type: str) -> Dict:
        """전처리기별 파라미터 설정"""
        params = {}

        # === Missing Value Handlers ===
        if step_type == "simple_imputer":
            strategy = self.ui.select_from_list(
                "Imputation strategy:",
                ["mean - 평균값", "median - 중앙값", "most_frequent - 최빈값", "constant - 고정값"],
                allow_cancel=False,
            )
            params["strategy"] = strategy.split(" - ")[0]
            if self.ui.confirm("결측 여부 indicator 컬럼을 생성하시겠습니까?", default=False):
                params["create_missing_indicators"] = True
            self.ui.show_info("columns 미지정 시 결측값이 있는 숫자형 컬럼에 자동 적용됩니다.")

        elif step_type == "drop_missing":
            axis = self.ui.select_from_list(
                "삭제 대상:", ["rows - 행 삭제", "columns - 열 삭제"], allow_cancel=False
            )
            params["axis"] = axis.split(" - ")[0]
            threshold = (
                self.ui.number_input(
                    "Threshold (0.0=결측 있으면 삭제, 1.0=전부 결측일 때만):",
                    default=0,
                    min_value=0,
                    max_value=100,
                )
                / 100.0
            )
            params["threshold"] = threshold

        elif step_type == "forward_fill":
            if self.ui.confirm("최대 연속 채움 횟수를 제한하시겠습니까?", default=False):
                limit = self.ui.number_input("Limit:", default=1, min_value=1, max_value=100)
                params["limit"] = int(limit)

        elif step_type == "backward_fill":
            if self.ui.confirm("최대 연속 채움 횟수를 제한하시겠습니까?", default=False):
                limit = self.ui.number_input("Limit:", default=1, min_value=1, max_value=100)
                params["limit"] = int(limit)

        elif step_type == "constant_fill":
            fill_value = self.ui.number_input(
                "Fill value:", default=0, min_value=-9999, max_value=9999
            )
            params["fill_value"] = fill_value

        elif step_type == "interpolation":
            method = self.ui.select_from_list(
                "Interpolation method:",
                ["linear - 선형 보간", "polynomial - 다항식", "spline - 스플라인", "nearest - 최근접"],
                allow_cancel=False,
            )
            params["method"] = method.split(" - ")[0]
            if params["method"] in ("polynomial", "spline"):
                order = self.ui.number_input("Order (차수):", default=2, min_value=1, max_value=5)
                params["order"] = int(order)

        # === Encoders ===
        elif step_type == "ordinal_encoder":
            self.ui.show_info(
                "handle_unknown: 학습 시 없던 범주가 추론 시 나타날 때 처리 방법"
            )
            handle_choice = self.ui.select_from_list(
                "handle_unknown:",
                ["use_encoded_value - 지정 값으로 대체", "error - 오류 발생"],
                allow_cancel=False,
            )
            params["handle_unknown"] = (
                "use_encoded_value" if "use_encoded" in handle_choice else "error"
            )
            if params["handle_unknown"] == "use_encoded_value":
                unknown_val = self.ui.number_input(
                    "Unknown value (-1 권장):", default=-1, min_value=-999, max_value=999
                )
                params["unknown_value"] = int(unknown_val)
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        elif step_type == "one_hot_encoder":
            self.ui.show_info(
                "handle_unknown: 학습 시 없던 범주가 추론 시 나타날 때 처리 방법"
            )
            handle_choice = self.ui.select_from_list(
                "handle_unknown:",
                ["ignore - 무시 (all zeros)", "error - 오류 발생"],
                allow_cancel=False,
            )
            params["handle_unknown"] = "ignore" if "ignore" in handle_choice else "error"
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        elif step_type == "catboost_encoder":
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        # === Feature Engineering ===
        elif step_type == "polynomial_features":
            degree = self.ui.number_input("Degree (차수):", default=2, min_value=2, max_value=5)
            params["degree"] = int(degree)
            params["interaction_only"] = self.ui.confirm(
                "Interaction only (상호작용 항만)?", default=False
            )
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        elif step_type == "kbins_discretizer":
            n_bins = self.ui.number_input("n_bins (구간 수):", default=5, min_value=2, max_value=20)
            params["n_bins"] = int(n_bins)
            strategy = self.ui.select_from_list(
                "Strategy:",
                ["uniform - 균등 구간", "quantile - 분위수 기반", "kmeans - K-means 클러스터링"],
                allow_cancel=False,
            )
            params["strategy"] = strategy.split(" - ")[0]
            encode = self.ui.select_from_list(
                "Encode:", ["ordinal - 정수", "onehot - One-hot"], allow_cancel=False
            )
            params["encode"] = encode.split(" - ")[0]
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        elif step_type == "tree_based_feature_generator":
            n_estimators = self.ui.number_input(
                "n_estimators (트리 수):", default=10, min_value=1, max_value=100
            )
            params["n_estimators"] = int(n_estimators)
            max_depth = self.ui.number_input(
                "max_depth:", default=3, min_value=1, max_value=10
            )
            params["max_depth"] = int(max_depth)
            self.ui.show_info("적용할 columns는 Recipe 파일에서 직접 지정하세요.")

        return params

    def _categorize_preprocessor(self, step_type: str) -> str:
        """전처리기 타입명 기반 카테고리 분류"""
        step_lower = step_type.lower()

        if "scaler" in step_lower:
            return "Scalers"

        if "encoder" in step_lower:
            return "Encoders"

        if step_lower == "drop_missing":
            return "Missing_Drop"
        if step_lower in ("forward_fill", "backward_fill", "interpolation"):
            return "Missing_TimeSeries"
        if step_lower in ("simple_imputer", "constant_fill"):
            return "Missing_Statistical"

        if "feature" in step_lower or "discretizer" in step_lower:
            return "Feature Engineering"

        return "Other"

    def generate_template_variables(
        self, recipe_data: Dict, config_template_vars: Dict = None
    ) -> Dict:
        """Recipe 데이터로부터 Jinja 템플릿 변수들 생성"""
        template_vars = {
            "recipe_name": recipe_data["name"],
            "task": recipe_data["task_choice"].title(),
            "model_class": recipe_data["model"]["class_path"],
            "model_library": recipe_data["model"]["library"],
            "author": recipe_data["metadata"]["author"],
            "target_column": recipe_data["data"]["data_interface"].get("target_column"),
            "entity_columns": recipe_data["data"]["data_interface"]["entity_columns"],
            "train_ratio": recipe_data["data"]["split"]["train"],
            "test_ratio": recipe_data["data"]["split"]["test"],
            "validation_ratio": recipe_data["data"]["split"]["validation"],
            "enable_tuning": recipe_data["model"]["hyperparameters"]["tuning_enabled"],
        }

        hyperparams = recipe_data["model"]["hyperparameters"]
        if hyperparams["tuning_enabled"]:
            template_vars.update(
                {
                    "optimization_metric": hyperparams.get("optimization_metric"),
                    "n_trials": hyperparams.get("n_trials"),
                    "tuning_timeout": hyperparams.get("timeout"),
                    "fixed_params": hyperparams.get("fixed", {}),
                    "tunable_specs": hyperparams.get("tunable", {}),
                }
            )
        else:
            template_vars["all_hyperparameters"] = hyperparams.get("values", {})

        task_lower = recipe_data["task_choice"].lower()

        if task_lower == "classification":
            calibration = recipe_data["model"].get("calibration", {})
            template_vars.update(
                {
                    "calibration_enabled": calibration.get("enabled", False),
                    "calibration_method": calibration.get("method"),
                    "calibration_ratio": recipe_data["data"]["split"].get("calibration"),
                }
            )

        elif task_lower == "causal":
            template_vars["treatment_column"] = recipe_data["data"]["data_interface"].get(
                "treatment_column"
            )

        elif task_lower == "timeseries":
            template_vars["timeseries_timestamp_column"] = recipe_data["data"][
                "data_interface"
            ].get("timestamp_column")

        fetcher = recipe_data["data"]["fetcher"]
        template_vars.update(
            {
                "fetcher_type": fetcher["type"],
                "feature_views": fetcher.get("feature_views"),
                "timestamp_column": fetcher.get("timestamp_column"),
            }
        )

        preprocessor = recipe_data.get("preprocessor")
        if preprocessor:
            template_vars["preprocessor_steps"] = preprocessor["steps"]

        if config_template_vars:
            template_vars.update(config_template_vars)

        return template_vars

    def create_recipe_file(self, recipe_data: Dict, output_path: Optional[str] = None) -> Path:
        """Recipe 데이터를 Jinja2 템플릿으로 렌더링하여 YAML 파일로 저장.

        템플릿을 사용하여 사용자 친화적인 주석이 포함된 Recipe 파일을 생성합니다.

        Args:
            recipe_data: build_recipe_interactively()에서 생성된 recipe 딕셔너리
            output_path: 출력 파일 경로 (기본값: recipes/{name}.yaml)

        Returns:
            생성된 Recipe 파일 경로
        """
        if output_path is None:
            recipes_dir = Path("recipes")
            recipes_dir.mkdir(exist_ok=True)
            output_path = recipes_dir / f"{recipe_data['name']}.yaml"
        else:
            output_path = Path(output_path)

        # 템플릿 변수 생성 및 렌더링
        template_vars = self.generate_template_variables(recipe_data)
        self.template_engine.write_rendered_file(
            "recipes/recipe.yaml.j2", output_path, template_vars
        )

        logger.debug(f"Recipe 파일 저장됨: {output_path}")
        return output_path


def build_recipe_interactive() -> Dict[str, Any]:
    """편의 함수: 대화형 Recipe 생성"""
    builder = RecipeBuilder()
    return builder.build_recipe_interactively()


def create_recipe_file(recipe_data: Dict, output_path: Optional[str] = None) -> Path:
    """편의 함수: Recipe 파일 생성"""
    builder = RecipeBuilder()
    return builder.create_recipe_file(recipe_data, output_path)
