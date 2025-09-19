"""
Recipe Builder for Modern ML Pipeline CLI
Phase 4: Registry 기반 동적 Recipe 생성기

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- Registry 기반 동적 컴포넌트 발견
"""

from pathlib import Path
import os
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import yaml

from src.settings.validation.catalog_validator import CatalogValidator
from src.settings.validation.business_validator import BusinessValidator
from src.utils.core.logger import logger


class InteractiveUI:
    """간단한 대화형 UI (CLI용 임시 구현)"""
    
    def text_input(self, prompt: str, default: str = "", validator=None) -> str:
        while True:
            result = input(f"{prompt} [{default}]: ").strip()
            if not result:
                result = default
            if validator is None or validator(result):
                return result
            print("❌ 유효하지 않은 입력입니다. 다시 시도해주세요.")
    
    def confirm(self, prompt: str, default: bool = True) -> bool:
        default_str = "Y/n" if default else "y/N"
        while True:
            result = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not result:
                return default
            if result in ['y', 'yes', '1', 'true']:
                return True
            elif result in ['n', 'no', '0', 'false']:
                return False
            print("❌ y/n으로 답해주세요.")
    
    def select_from_list(self, prompt: str, options: List[str]) -> str:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            try:
                choice = int(input("선택 (번호): "))
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                print(f"❌ 1-{len(options)} 범위의 번호를 입력해주세요.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
    
    def number_input(self, prompt: str, default: float, min_value: float = None, max_value: float = None) -> float:
        while True:
            try:
                result = input(f"{prompt} [{default}]: ").strip()
                if not result:
                    result = default
                else:
                    result = float(result)
                
                if min_value is not None and result < min_value:
                    print(f"❌ {min_value} 이상의 값을 입력해주세요.")
                    continue
                if max_value is not None and result > max_value:
                    print(f"❌ {max_value} 이하의 값을 입력해주세요.")
                    continue
                
                return result
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
    
    def show_info(self, message: str):
        print(f"💡 {message}")

    def show_warning(self, message: str):
        print(f"⚠️  {message}")

    def show_error(self, message: str):
        print(f"❌ {message}")

    def show_panel(self, content: str, title: str = "", style: str = ""):
        """Panel display for summary (matches deprecated format)"""
        print(f"\n{'='*60}")
        if title:
            print(f"{title}")
            print('='*60)
        print(content)
        print('='*60)

    def show_table(self, title: str, headers: List[str], rows: List[List[str]], show_index: bool = False):
        """Table display (matches deprecated format)"""
        print(f"\n{title}")
        print("-" * 50)

        # Headers
        if show_index:
            print(f"{'#':<3} {' | '.join(f'{h:<15}' for h in headers)}")
        else:
            print(f"{' | '.join(f'{h:<15}' for h in headers)}")
        print("-" * 50)

        # Rows
        for i, row in enumerate(rows, 1):
            if show_index:
                print(f"{i:<3} {' | '.join(f'{str(cell):<15}' for cell in row)}")
            else:
                print(f"{' | '.join(f'{str(cell):<15}' for cell in row)}")

    def single_choice(self, prompt: str, choices: List[tuple], default: str = None) -> str:
        """Single choice selection (matches deprecated format)"""
        print(f"\n{prompt}")
        for key, desc in choices:
            marker = " (기본값)" if key == default else ""
            print(f"  {key}: {desc}{marker}")

        while True:
            result = input("선택: ").strip()
            if not result and default:
                return default

            for key, _ in choices:
                if result == key:
                    return key

            # Quiet repeated prompt errors when running non-interactively
            if os.getenv("MMP_QUIET_PROMPTS", "0") != "1":
                print(f"❌ 유효한 선택지를 입력해주세요: {[k for k, _ in choices]}")

    def print_divider(self):
        print("\n" + "="*50 + "\n")


class RecipeBuilder:
    """Registry 기반 동적 Recipe 빌더 - 하드코딩 완전 제거"""

    def __init__(self):
        self.catalog_validator = CatalogValidator()
        self.business_validator = BusinessValidator()
        self.ui = InteractiveUI()

    def get_available_tasks(self) -> Set[str]:
        """src/models/catalog 기반 동적 Task 목록"""
        return self.catalog_validator.get_available_tasks()

    def get_available_models_for_task(self, task_type: str) -> Dict[str, Dict]:
        """특정 Task의 사용 가능한 모델들"""
        return self.catalog_validator.get_available_models_for_task(task_type)

    def get_available_preprocessors(self) -> Dict[str, List[str]]:
        """Registry 기반 전처리기 분류 및 제공"""
        available_types = self.business_validator.get_available_preprocessor_types()

        # 타입명 기반 자동 분류 (하드코딩 대신 규칙 기반)
        categorized = {
            "Missing Value Handling": [],
            "Encoders": [],
            "Scalers": [],
            "Feature Engineering": []
        }

        for step_type in available_types:
            category = self._categorize_preprocessor(step_type)
            if category in categorized:
                categorized[category].append(step_type)

        return {k: v for k, v in categorized.items() if v}  # 빈 카테고리 제거

    def get_available_metrics_for_task(self, task_type: str) -> List[str]:
        """Registry 기반 Task별 메트릭 제공"""
        return self.business_validator.get_available_evaluators_for_task(task_type)

    def build_recipe_interactively(self) -> Dict[str, Any]:
        """사용자 대화형 Recipe 생성 - 모든 옵션이 동적"""

        # 1. Task 선택 (동적)
        available_tasks = self.get_available_tasks()
        if not available_tasks:
            raise ValueError("사용 가능한 Task가 없습니다. src/models/catalog 디렉토리를 확인해주세요.")
        
        task_choice = self.ui.select_from_list("Task를 선택하세요:", list(available_tasks))

        # 2. 모델 선택 (Task별 동적)
        available_models = self.get_available_models_for_task(task_choice)
        if not available_models:
            raise ValueError(f"{task_choice}에 사용 가능한 모델이 없습니다.")
        
        model_name = self.ui.select_from_list("모델을 선택하세요:", list(available_models.keys()))
        selected_model = available_models[model_name]

        # 3. 전처리기 선택 (Registry 기반 동적)
        preprocessor_steps = self._collect_preprocessor_steps()

        # 4. 평가 메트릭 선택 (Task별 동적)
        available_metrics = self.get_available_metrics_for_task(task_choice)
        if not available_metrics:
            raise ValueError(f"{task_choice}에 사용 가능한 메트릭이 없습니다.")
        
        # 기본적으로 모든 메트릭 사용, 사용자가 원하면 선택 가능
        selected_metrics = available_metrics  # 또는 사용자 선택 로직 추가

        # 5. Task별 특수 설정 처리
        model_config = self._build_model_config(selected_model, task_choice)
        data_interface_config = self._build_data_interface_config(task_choice)
        data_split_config = self._build_data_split_config(task_choice, model_config.get('calibration', {}))

        # 6. Recipe 데이터 구성
        recipe_data = {
            "name": f"{task_choice}_{model_name}_{datetime.now().strftime('%Y%m%d')}",
            "task_choice": task_choice,
            "model": model_config,
            "data": {
                "loader": {"source_uri": None},  # 나중에 주입
                "fetcher": {"type": "pass_through"},  # 기본값
                "data_interface": data_interface_config,
                "split": data_split_config
            },
            "preprocessor": {"steps": preprocessor_steps} if preprocessor_steps else None,
            "evaluation": {
                "metrics": selected_metrics,
                "random_state": 42
            },
            "metadata": {
                "author": "CLI Recipe Builder",
                "created_at": datetime.now().isoformat(),
                "description": f"{task_choice} task using {selected_model['library']}"
            }
        }

        return recipe_data

    def _build_model_config(self, selected_model: Dict, task_choice: str) -> Dict:
        """모델 설정 구성 (Task별 조건부 로직 포함)"""
        model_config = {
            "class_path": selected_model["class_path"],
            "library": selected_model["library"],
            "hyperparameters": self._configure_hyperparameters(selected_model)
        }

        # Classification task에서만 Calibration 지원
        if task_choice.lower() == 'classification':
            calibration_enabled = self.ui.confirm("캘리브레이션을 사용하시겠습니까?")
            if calibration_enabled:
                available_methods = self.business_validator.get_available_calibrators()
                calibration_method = self.ui.select_from_list(
                    "캘리브레이션 방법을 선택하세요:",
                    list(available_methods)
                )
                model_config["calibration"] = {
                    "enabled": True,
                    "method": calibration_method
                }
            else:
                model_config["calibration"] = {"enabled": False}

        return model_config

    def _build_data_interface_config(self, task_choice: str) -> Dict:
        """데이터 인터페이스 설정 구성 (Task별 특수 필드 포함)"""
        data_interface = {
            "entity_columns": [],  # 나중에 사용자 입력
            "feature_columns": None  # 자동 추론
        }

        # Clustering에서만 target_column 없음
        if task_choice.lower() != 'clustering':
            data_interface["target_column"] = None  # 나중에 주입

        # Causal task에서만 treatment_column 사용
        if task_choice.lower() == 'causal':
            data_interface["treatment_column"] = None  # 나중에 주입

        # TimeSeries task에서만 timestamp_column 필수
        if task_choice.lower() == 'timeseries':
            data_interface["timestamp_column"] = None  # 나중에 주입

        return data_interface

    def _build_data_split_config(self, task_choice: str, calibration_config: Dict) -> Dict:
        """데이터 분할 설정 구성 (Calibration 조건부 포함)"""
        # 기본 데이터 분할 비율
        data_split = {
            "train": 0.8,
            "test": 0.1,
            "validation": 0.1
        }

        # Classification + Calibration 활성화시에만 calibration split 추가
        if (task_choice.lower() == 'classification' and
            calibration_config.get('enabled', False)):
            # Calibration을 위해 train 비율 조정
            data_split = {
                "train": 0.7,
                "test": 0.1,
                "validation": 0.1,
                "calibration": 0.1
            }

        return data_split

    def _configure_hyperparameters(self, selected_model: Dict) -> Dict:
        """하이퍼파라미터 설정 구성"""
        enable_tuning = self.ui.confirm("하이퍼파라미터 튜닝을 활성화하시겠습니까?")

        hyperparams = selected_model.get("hyperparameters", {})

        # None 처리 추가 - 견고성 개선
        if hyperparams is None:
            hyperparams = {}
        
        if enable_tuning:
            return {
                "tuning_enabled": True,
                "optimization_metric": "accuracy",  # 기본값, 나중에 task별로 조정
                "n_trials": 10,
                "timeout": 300,
                "fixed": hyperparams.get("fixed", {}),
                "tunable": hyperparams.get("tunable", {})
            }
        else:
            # 기본값 사용
            fixed_params = hyperparams.get("fixed") or {}
            values = fixed_params.copy()

            tunable_params = hyperparams.get("tunable") or {}
            for param, config in tunable_params.items():
                values[param] = config.get("default", config.get("range", [1])[0])
            
            return {
                "tuning_enabled": False,
                "values": values
            }

    def _collect_preprocessor_steps(self) -> List[Dict]:
        """Registry 기반 전처리기 수집"""
        available_preprocessors = self.get_available_preprocessors()
        selected_steps = []

        for category, preprocessors in available_preprocessors.items():
            if self.ui.confirm(f"\n{category} 전처리를 사용하시겠습니까?"):
                for preprocessor_type in preprocessors:
                    if self.ui.confirm(f"  {preprocessor_type}를 사용하시겠습니까?"):
                        step_config = {
                            "type": preprocessor_type,
                            # 동적 파라미터 설정 (Registry에서 클래스 분석)
                            **self._configure_preprocessor_params(preprocessor_type)
                        }
                        selected_steps.append(step_config)

        return selected_steps

    def _configure_preprocessor_params(self, step_type: str) -> Dict:
        """전처리기별 파라미터 설정"""
        params = {}
        
        # 타입별 특수 파라미터 설정
        if step_type == "simple_imputer":
            strategy = self.ui.select_from_list(
                "결측값 대체 전략:",
                ["mean", "median", "most_frequent", "constant"]
            )
            params["strategy"] = strategy
            
            create_indicators = self.ui.confirm("결측값 지시자를 생성하시겠습니까?")
            params["create_missing_indicators"] = create_indicators
            
        elif step_type == "polynomial_features":
            degree = self.ui.number_input("다항식 차수:", default=2, min_value=2, max_value=5)
            params["degree"] = int(degree)
            
        elif step_type == "kbins_discretizer":
            n_bins = self.ui.number_input("구간 개수:", default=5, min_value=2, max_value=20)
            params["n_bins"] = int(n_bins)
            
            strategy = self.ui.select_from_list(
                "구간 분할 전략:",
                ["uniform", "quantile", "kmeans"]
            )
            params["strategy"] = strategy

        return params

    def _categorize_preprocessor(self, step_type: str) -> str:
        """전처리기 타입명 기반 카테고리 분류 (하드코딩 제거)"""
        step_lower = step_type.lower()

        if 'imputer' in step_lower or 'missing' in step_lower:
            return "Missing Value Handling"
        elif 'encoder' in step_lower:
            return "Encoders"
        elif 'scaler' in step_lower:
            return "Scalers"
        elif 'feature' in step_lower or 'discretizer' in step_lower:
            return "Feature Engineering"
        else:
            return "Other"

    def generate_template_variables(self, recipe_data: Dict,
                                   config_template_vars: Dict = None) -> Dict:
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
            template_vars.update({
                "optimization_metric": hyperparams.get("optimization_metric"),
                "n_trials": hyperparams.get("n_trials"),
                "tuning_timeout": hyperparams.get("timeout"),
                "fixed_params": hyperparams.get("fixed", {}),
                "tunable_specs": hyperparams.get("tunable", {})
            })
        else:
            template_vars["all_hyperparameters"] = hyperparams.get("values", {})

        # Task별 조건부 변수들
        task_lower = recipe_data["task_choice"].lower()

        # Classification 전용
        if task_lower == 'classification':
            calibration = recipe_data["model"].get("calibration", {})
            template_vars.update({
                "calibration_enabled": calibration.get("enabled", False),
                "calibration_method": calibration.get("method"),
                "calibration_ratio": recipe_data["data"]["split"].get("calibration")
            })

        # Causal 전용
        elif task_lower == 'causal':
            template_vars["treatment_column"] = recipe_data["data"]["data_interface"].get("treatment_column")

        # TimeSeries 전용
        elif task_lower == 'timeseries':
            template_vars["timeseries_timestamp_column"] = recipe_data["data"]["data_interface"].get("timestamp_column")

        # Feature Store 관련
        fetcher = recipe_data["data"]["fetcher"]
        template_vars.update({
            "fetcher_type": fetcher["type"],
            "feature_views": fetcher.get("feature_views"),
            "timestamp_column": fetcher.get("timestamp_column")
        })

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
        with open(output_path, 'w', encoding='utf-8') as f:
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
