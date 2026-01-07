"""
Recipe Builder for Modern ML Pipeline CLI

Registry 기반 동적 Recipe 생성기입니다.
대화형 인터페이스를 통해 Task, Model, Preprocessor 등을 선택하여 Recipe 파일을 생성합니다.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.cli.utils.interactive_ui import InteractiveUI
from src.settings.validation.business_validator import BusinessValidator
from src.settings.validation.catalog_validator import CatalogValidator
from src.utils.core.logger import logger


class RecipeBuilder:
    """Registry 기반 동적 Recipe 빌더 - 하드코딩 완전 제거"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.ui = InteractiveUI()

    def get_available_tasks(self) -> Set[str]:
        """models/catalog 기반 동적 Task 목록"""
        return self.catalog_validator.get_available_tasks()

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들"""
        return self.catalog_validator.get_available_models_for_task(task_type)

    def get_available_preprocessors(self) -> Dict[str, List[str]]:
        """Registry 기반 전처리기 분류 및 제공 (상호 배타성 고려)"""
        available_types = self.business_validator.get_available_preprocessor_types()

        # 상호 배타성을 고려한 세분화된 카테고리
        categorized = {
            "Scalers": [],  # 상호 배타 - 1개만 선택
            "Encoders": [],  # 상호 배타 - 1개만 선택
            "Missing_Statistical": [],  # 통계적 결측값 처리
            "Missing_TimeSeries": [],  # 시계열 결측값 처리 (다중 선택 가능)
            "Missing_Drop": [],  # 결측값 삭제
            "Feature Engineering": [],  # 다중 선택 가능
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
        total_steps = 5

        # Step 1: Task 선택
        self.ui.show_step(1, total_steps, "Task 선택")
        available_tasks = self.get_available_tasks()
        if not available_tasks:
            raise ValueError(
                "사용 가능한 Task가 없습니다. models/catalog 디렉토리를 확인해주세요."
            )

        task_choice = self.ui.select_from_list(
            "Task를 선택하세요:", list(available_tasks), allow_cancel=False
        )

        # Step 2: 모델 선택
        self.ui.show_step(2, total_steps, "모델 선택")
        available_models = self.get_available_models_for_task(task_choice)
        if not available_models:
            raise ValueError(f"{task_choice}에 사용 가능한 모델이 없습니다.")

        model_name = self.ui.select_from_list(
            "모델을 선택하세요:", list(available_models.keys()), allow_cancel=False
        )
        selected_model = available_models[model_name]

        # Step 3: 전처리기 선택
        self.ui.show_step(3, total_steps, "전처리기 선택")
        preprocessor_steps = self._collect_preprocessor_steps()

        # Step 4: 모델 설정
        self.ui.show_step(4, total_steps, "모델 설정")
        available_metrics = self.get_available_metrics_for_task(task_choice)
        if not available_metrics:
            raise ValueError(f"{task_choice}에 사용 가능한 메트릭이 없습니다.")

        selected_metrics = available_metrics
        model_config = self._build_model_config(selected_model, task_choice)
        data_interface_config = self._build_data_interface_config(task_choice, selected_model)
        data_split_config = self._build_data_split_config(
            task_choice, model_config.get("calibration", {})
        )

        # Step 5: Recipe 생성
        self.ui.show_step(5, total_steps, "Recipe 생성")
        recipe_data = {
            "name": f"{task_choice}_{model_name}_{datetime.now().strftime('%Y%m%d')}",
            "task_choice": task_choice,
            "model": model_config,
            "data": {
                "loader": {"source_uri": None},  # 나중에 주입
                "fetcher": {"type": "pass_through"},  # 기본값
                "data_interface": data_interface_config,
                "split": data_split_config,
            },
            "preprocessor": {"steps": preprocessor_steps} if preprocessor_steps else None,
            "evaluation": {"metrics": selected_metrics, "random_state": 42},
            "metadata": {
                "author": "CLI Recipe Builder",
                "created_at": datetime.now().isoformat(),
                "description": f"{task_choice} task using {selected_model['library']}",
            },
        }

        # 사용자 필수 입력 항목 안내
        self._show_required_user_inputs(recipe_data, task_choice, preprocessor_steps)

        return recipe_data

    def _build_model_config(self, selected_model: Dict, task_choice: str) -> Dict:
        """모델 설정 구성 (Task별 조건부 로직 포함)"""
        model_config = {
            "class_path": selected_model["class_path"],
            "library": selected_model["library"],
            "hyperparameters": self._configure_hyperparameters(selected_model, task_choice),
        }

        # Classification task에서만 Calibration 지원
        if task_choice.lower() == "classification":
            calibration_enabled = self.ui.confirm("캘리브레이션을 사용하시겠습니까?")
            if calibration_enabled:
                available_methods = self.business_validator.get_available_calibrators()
                calibration_method = self.ui.select_from_list(
                    "캘리브레이션 방법을 선택하세요:", list(available_methods), allow_cancel=False
                )
                model_config["calibration"] = {"enabled": True, "method": calibration_method}
            else:
                model_config["calibration"] = {"enabled": False}

        return model_config

    def _build_data_interface_config(self, task_choice: str, model_spec: Dict) -> Dict:
        """데이터 인터페이스 설정 구성 (Task별, 모델별 특수 필드 포함)"""
        data_interface = {
            "entity_columns": [],  # 나중에 사용자 입력
            "feature_columns": None,  # 자동 추론
        }

        # Clustering에서만 target_column 없음
        if task_choice.lower() != "clustering":
            data_interface["target_column"] = None  # 나중에 주입

        # Causal task에서만 treatment_column 사용
        if task_choice.lower() == "causal":
            data_interface["treatment_column"] = None  # 나중에 주입

        # TimeSeries task에서만 timestamp_column 필수
        if task_choice.lower() == "timeseries":
            data_interface["timestamp_column"] = None  # 나중에 주입

        # 모델 카탈로그의 data_handler 기반 추가 필드
        data_handler = model_spec.get("data_handler", "tabular")
        if data_handler == "sequence":
            seq_len = self.ui.number_input(
                "시퀀스 길이 (lookback window):", default=7, min_value=1, max_value=365
            )
            data_interface["sequence_length"] = int(seq_len)

        return data_interface

    def _build_data_split_config(self, task_choice: str, calibration_config: Dict) -> Dict:
        """데이터 분할 설정 구성 (사용자 커스텀 비율 지원)"""
        has_calibration = (
            task_choice.lower() == "classification" and calibration_config.get("enabled", False)
        )

        # 기본 비율 설정
        if has_calibration:
            default_train, default_val, default_test, default_cal = 70, 15, 10, 5
        else:
            default_train, default_val, default_test = 70, 15, 15

        if self.ui.confirm("데이터 분할 비율을 직접 설정하시겠습니까?", default=False):
            train_pct = self.ui.number_input(
                "Train 비율 (%):", default=default_train, min_value=50, max_value=90
            )
            val_pct = self.ui.number_input(
                "Validation 비율 (%):", default=default_val, min_value=0, max_value=30
            )
            test_pct = self.ui.number_input(
                "Test 비율 (%):", default=default_test, min_value=5, max_value=30
            )

            if has_calibration:
                cal_pct = self.ui.number_input(
                    "Calibration 비율 (%):", default=default_cal, min_value=5, max_value=20
                )
                total = train_pct + val_pct + test_pct + cal_pct
            else:
                cal_pct = 0
                total = train_pct + val_pct + test_pct

            # 합계 검증
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
            # 기본값 사용
            if has_calibration:
                data_split = {"train": 0.7, "validation": 0.15, "test": 0.1, "calibration": 0.05}
            else:
                data_split = {"train": 0.7, "validation": 0.15, "test": 0.15}

        return data_split

    def _configure_hyperparameters(self, selected_model: Dict, task_choice: str) -> Dict:
        """하이퍼파라미터 설정 구성"""
        enable_tuning = self.ui.confirm("하이퍼파라미터 튜닝을 활성화하시겠습니까?")

        hyperparams = selected_model.get("hyperparameters", {})

        # None 처리 추가 - 견고성 개선
        if hyperparams is None:
            hyperparams = {}

        if enable_tuning:
            # Task별 기본 최적화 메트릭 동적 로드
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
            # 기본값 사용
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

        # 1. Scalers - 상호 배타
        if "Scalers" in available and available["Scalers"]:
            scalers = available["Scalers"]
            # 스케일러별 설명 추가
            scaler_labels = {
                "standard_scaler": "standard_scaler (평균=0, 표준편차=1)",
                "min_max_scaler": "min_max_scaler (0~1 범위)",
                "robust_scaler": "robust_scaler (이상치에 강건)",
            }
            display_options = [scaler_labels.get(s, s) for s in scalers]
            display_options.append("(사용 안 함)")

            choice = self.ui.select_from_list(
                "스케일러를 선택하세요 (숫자형 컬럼에 적용):", display_options, allow_cancel=False
            )
            if choice != "(사용 안 함)":
                actual_choice = choice.split(" (")[0]
                selected_steps.append(
                    {"type": actual_choice, **self._configure_preprocessor_params(actual_choice)}
                )

        # 2. Encoders - 상호 배타
        if "Encoders" in available and available["Encoders"]:
            encoders = available["Encoders"]
            # 인코더별 설명 추가
            display_options = []
            encoder_labels = {
                "catboost_encoder": "catboost_encoder (y 기반 인코딩)",
                "one_hot_encoder": "one_hot_encoder (범주별 이진 컬럼)",
                "ordinal_encoder": "ordinal_encoder (범주를 정수로)",
            }
            for enc in encoders:
                display_options.append(encoder_labels.get(enc, enc))
            display_options.append("(사용 안 함)")

            choice = self.ui.select_from_list(
                "인코더를 선택하세요 (범주형 컬럼에 적용):", display_options, allow_cancel=False
            )
            # 선택값에서 설명 제거
            if choice != "(사용 안 함)":
                actual_choice = choice.split(" (")[0]
                selected_steps.append(
                    {"type": actual_choice, **self._configure_preprocessor_params(actual_choice)}
                )

        # 3. Missing Value Handling - 계층적 선택
        has_missing_options = any(
            k in available for k in ("Missing_Drop", "Missing_Statistical", "Missing_TimeSeries")
        )
        if has_missing_options:
            missing_approach = self.ui.select_from_list(
                "결측값 처리 방법을 선택하세요:",
                ["삭제 (행/열 제거)", "대체 (값 채우기)", "(사용 안 함)"],
                allow_cancel=False,
            )

            if missing_approach == "삭제 (행/열 제거)":
                # drop_missing 설정
                if "Missing_Drop" in available and "drop_missing" in available["Missing_Drop"]:
                    selected_steps.append(
                        {
                            "type": "drop_missing",
                            **self._configure_preprocessor_params("drop_missing"),
                        }
                    )

            elif missing_approach == "대체 (값 채우기)":
                # 대체 방식 세분화
                impute_options = []
                if "Missing_Statistical" in available:
                    impute_options.extend(available["Missing_Statistical"])
                if "Missing_TimeSeries" in available:
                    impute_options.extend(available["Missing_TimeSeries"])

                if impute_options:
                    selected = self.ui.multi_select(
                        "결측값 대체 방식 (복수 선택 가능):", impute_options
                    )
                    for method in selected:
                        selected_steps.append(
                            {"type": method, **self._configure_preprocessor_params(method)}
                        )

        # 4. Feature Engineering - 다중 선택 가능
        if "Feature Engineering" in available and available["Feature Engineering"]:
            fe_options = available["Feature Engineering"]
            # Feature Engineering별 설명 추가
            fe_labels = {
                "tree_based_feature_generator": "tree_based_feature_generator (y 기반, 트리 피처)",
                "polynomial_features": "polynomial_features (다항식 피처)",
                "kbins_discretizer": "kbins_discretizer (구간 이산화)",
            }
            display_options = [fe_labels.get(fe, fe) for fe in fe_options]

            if self.ui.confirm("Feature Engineering을 사용하시겠습니까?"):
                selected = self.ui.multi_select("Feature Engineering 방식:", display_options)
                for method in selected:
                    actual_method = method.split(" (")[0]
                    selected_steps.append(
                        {
                            "type": actual_method,
                            **self._configure_preprocessor_params(actual_method),
                        }
                    )

        return selected_steps

    def _configure_preprocessor_params(self, step_type: str) -> Dict:
        """전처리기별 파라미터 설정"""
        params = {}

        # === Missing Value Handlers ===
        if step_type == "simple_imputer":
            strategy = self.ui.select_from_list(
                "결측값 대체 전략:",
                ["mean", "median", "most_frequent", "constant"],
                allow_cancel=False,
            )
            params["strategy"] = strategy
            create_indicators = self.ui.confirm("결측값 지시자를 생성하시겠습니까?")
            params["create_missing_indicators"] = create_indicators
            self.ui.show_info("columns 미지정 시 결측값이 있는 숫자형 컬럼에 자동 적용됩니다.")

        elif step_type == "drop_missing":
            axis = self.ui.select_from_list(
                "삭제 대상:", ["rows (행 삭제)", "columns (열 삭제)"], allow_cancel=False
            )
            params["axis"] = "rows" if "rows" in axis else "columns"
            threshold = (
                self.ui.number_input(
                    "결측 비율 임계값 (0.0=결측 있으면 삭제, 1.0=전부 결측일 때만):",
                    default=0,
                    min_value=0,
                    max_value=100,
                )
                / 100.0
            )
            params["threshold"] = threshold

        elif step_type == "forward_fill":
            if self.ui.confirm("최대 연속 채움 횟수를 제한하시겠습니까?"):
                limit = self.ui.number_input(
                    "최대 연속 채움 횟수:", default=1, min_value=1, max_value=100
                )
                params["limit"] = int(limit)

        elif step_type == "backward_fill":
            if self.ui.confirm("최대 연속 채움 횟수를 제한하시겠습니까?"):
                limit = self.ui.number_input(
                    "최대 연속 채움 횟수:", default=1, min_value=1, max_value=100
                )
                params["limit"] = int(limit)

        elif step_type == "constant_fill":
            fill_value = self.ui.number_input(
                "채울 상수 값:", default=0, min_value=-9999, max_value=9999
            )
            params["fill_value"] = fill_value

        elif step_type == "interpolation":
            method = self.ui.select_from_list(
                "보간 방법:", ["linear", "polynomial", "spline", "nearest"], allow_cancel=False
            )
            params["method"] = method
            if method in ("polynomial", "spline"):
                order = self.ui.number_input("차수:", default=2, min_value=1, max_value=5)
                params["order"] = int(order)

        # === Encoders ===
        elif step_type == "ordinal_encoder":
            handle_choice = self.ui.select_from_list(
                "미지의 범주 처리 방법:",
                ["use_encoded_value (지정값으로 대체)", "error (오류 발생)"],
                allow_cancel=False,
            )
            params["handle_unknown"] = (
                "use_encoded_value" if "use_encoded" in handle_choice else "error"
            )
            if params["handle_unknown"] == "use_encoded_value":
                unknown_val = self.ui.number_input(
                    "미지의 범주에 할당할 값:", default=-1, min_value=-999, max_value=999
                )
                params["unknown_value"] = int(unknown_val)
            self.ui.show_info("적용 컬럼은 Recipe의 columns 필드에서 직접 지정하세요.")

        elif step_type == "one_hot_encoder":
            handle_choice = self.ui.select_from_list(
                "미지의 범주 처리 방법:",
                ["ignore (무시)", "error (오류 발생)"],
                allow_cancel=False,
            )
            params["handle_unknown"] = "ignore" if "ignore" in handle_choice else "error"
            self.ui.show_info("적용 컬럼은 Recipe의 columns 필드에서 직접 지정하세요.")

        # === Feature Engineering ===
        elif step_type == "polynomial_features":
            degree = self.ui.number_input("다항식 차수:", default=2, min_value=2, max_value=5)
            params["degree"] = int(degree)
            params["interaction_only"] = self.ui.confirm("상호작용 항만 생성하시겠습니까?")
            self.ui.show_info("특정 컬럼에만 적용하려면 Recipe의 columns 필드를 지정하세요.")

        elif step_type == "kbins_discretizer":
            n_bins = self.ui.number_input("구간 개수:", default=5, min_value=2, max_value=20)
            params["n_bins"] = int(n_bins)
            strategy = self.ui.select_from_list(
                "구간 분할 전략:", ["uniform", "quantile", "kmeans"], allow_cancel=False
            )
            params["strategy"] = strategy
            encode = self.ui.select_from_list(
                "출력 인코딩:", ["ordinal (정수)", "onehot (원-핫)"], allow_cancel=False
            )
            params["encode"] = "ordinal" if "ordinal" in encode else "onehot"
            self.ui.show_info("특정 컬럼에만 적용하려면 Recipe의 columns 필드를 지정하세요.")

        elif step_type == "tree_based_feature_generator":
            n_estimators = self.ui.number_input(
                "트리 개수:", default=10, min_value=1, max_value=100
            )
            params["n_estimators"] = int(n_estimators)
            max_depth = self.ui.number_input("최대 깊이:", default=3, min_value=1, max_value=10)
            params["max_depth"] = int(max_depth)
            self.ui.show_info("특정 컬럼에만 적용하려면 Recipe의 columns 필드를 지정하세요.")

        return params

    def _categorize_preprocessor(self, step_type: str) -> str:
        """전처리기 타입명 기반 카테고리 분류 (상호 배타성 고려)"""
        step_lower = step_type.lower()

        # Scalers (상호 배타)
        if "scaler" in step_lower:
            return "Scalers"

        # Encoders (상호 배타)
        if "encoder" in step_lower:
            return "Encoders"

        # Missing Value - 세분화
        if step_lower == "drop_missing":
            return "Missing_Drop"
        if step_lower in ("forward_fill", "backward_fill", "interpolation"):
            return "Missing_TimeSeries"
        if step_lower in ("simple_imputer", "constant_fill"):
            return "Missing_Statistical"

        # Feature Engineering (다중 선택 가능)
        if "feature" in step_lower or "discretizer" in step_lower:
            return "Feature Engineering"

        return "Other"

    def _show_required_user_inputs(
        self, recipe_data: Dict, task_choice: str, preprocessor_steps: List[Dict]
    ) -> None:
        """Recipe 생성 후 사용자가 반드시 입력해야 할 항목 안내"""
        self.ui.print_divider()
        self.ui.show_warning("[필수 수정 항목] Recipe 파일에서 직접 지정해야 합니다:")

        items = [
            "  - data.loader.source_uri: 데이터 파일 경로 또는 SQL 파일",
            "  - data.data_interface.entity_columns: 엔티티 컬럼 리스트",
        ]

        # Task별 필수 항목
        task_lower = task_choice.lower()
        if task_lower != "clustering":
            items.append("  - data.data_interface.target_column: 타겟 컬럼명")
        if task_lower == "causal":
            items.append("  - data.data_interface.treatment_column: 처리 컬럼명")
        if task_lower == "timeseries":
            items.append("  - data.data_interface.timestamp_column: 타임스탬프 컬럼명")

        # 인코더가 있으면 columns 안내
        has_encoder = any(s.get("type", "").endswith("_encoder") for s in preprocessor_steps)
        if has_encoder:
            items.append("  - preprocessor.steps[encoder].columns: 인코딩할 범주형 컬럼 리스트")

        for item in items:
            self.ui.show_info(item)

        self.ui.print_divider()

    def generate_template_variables(
        self, recipe_data: Dict, config_template_vars: Dict = None
    ) -> Dict:
        """
        Recipe 데이터로부터 Jinja 템플릿 변수들 생성

        모든 config.yaml.j2, recipe.yaml.j2에서 사용되는 변수들을
        Recipe 데이터에서 추출하여 생성
        """
        template_vars = {
            # Recipe 기본 정보
            "recipe_name": recipe_data["name"],
            "task": recipe_data["task_choice"].title(),
            "model_class": recipe_data["model"]["class_path"],
            "model_library": recipe_data["model"]["library"],
            "timestamp": recipe_data["metadata"]["created_at"],
            "author": recipe_data["metadata"]["author"],
            # 평가 메트릭
            "metrics": recipe_data["evaluation"]["metrics"],
            # 데이터 인터페이스
            "target_column": recipe_data["data"]["data_interface"].get("target_column"),
            "entity_columns": recipe_data["data"]["data_interface"]["entity_columns"],
            # 데이터 분할
            "train_ratio": recipe_data["data"]["split"]["train"],
            "test_ratio": recipe_data["data"]["split"]["test"],
            "validation_ratio": recipe_data["data"]["split"]["validation"],
            # 하이퍼파라미터 관련
            "enable_tuning": recipe_data["model"]["hyperparameters"]["tuning_enabled"],
        }

        # 하이퍼파라미터 체계 처리
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

        # Task별 조건부 변수들
        task_lower = recipe_data["task_choice"].lower()

        # Classification 전용
        if task_lower == "classification":
            calibration = recipe_data["model"].get("calibration", {})
            template_vars.update(
                {
                    "calibration_enabled": calibration.get("enabled", False),
                    "calibration_method": calibration.get("method"),
                    "calibration_ratio": recipe_data["data"]["split"].get("calibration"),
                }
            )

        # Causal 전용
        elif task_lower == "causal":
            template_vars["treatment_column"] = recipe_data["data"]["data_interface"].get(
                "treatment_column"
            )

        # TimeSeries 전용
        elif task_lower == "timeseries":
            template_vars["timeseries_timestamp_column"] = recipe_data["data"][
                "data_interface"
            ].get("timestamp_column")

        # Feature Store 관련
        fetcher = recipe_data["data"]["fetcher"]
        template_vars.update(
            {
                "fetcher_type": fetcher["type"],
                "feature_views": fetcher.get("feature_views"),
                "timestamp_column": fetcher.get("timestamp_column"),
            }
        )

        # 전처리기 관련
        preprocessor = recipe_data.get("preprocessor")
        if preprocessor:
            template_vars["preprocessor_steps"] = preprocessor["steps"]

        # Config 템플릿 변수들 병합 (사용자 제공)
        if config_template_vars:
            template_vars.update(config_template_vars)

        return template_vars

    def create_recipe_file(self, recipe_data: Dict, output_path: Optional[str] = None) -> Path:
        """Recipe 데이터를 YAML 파일로 저장"""
        if output_path is None:
            recipes_dir = Path("recipes")
            recipes_dir.mkdir(exist_ok=True)
            output_path = recipes_dir / f"{recipe_data['name']}.yaml"
        else:
            output_path = Path(output_path)

        # YAML 파일로 저장
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(recipe_data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Recipe 파일 생성 완료: {output_path}")
        return output_path


# 편의 함수들
def build_recipe_interactive() -> Dict[str, Any]:
    """편의 함수: 대화형 Recipe 생성"""
    builder = RecipeBuilder()
    return builder.build_recipe_interactively()


def create_recipe_file(recipe_data: Dict, output_path: Optional[str] = None) -> Path:
    """편의 함수: Recipe 파일 생성"""
    builder = RecipeBuilder()
    return builder.create_recipe_file(recipe_data, output_path)
