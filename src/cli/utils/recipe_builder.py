"""
Recipe Builder for Modern ML Pipeline CLI
Phase 4: Environment-independent recipe generation

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 환경 독립적 설계
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

from src.cli.utils.interactive_ui import InteractiveUI
from src.cli.utils.template_engine import TemplateEngine


class RecipeBuilder:
    """환경 독립적인 Recipe 빌더.
    
    사용자와의 대화형 인터페이스를 통해 모델 선택 및 Recipe 파일을 생성합니다.
    환경 설정과 분리되어 Recipe만을 다룹니다.
    """
    
    # Task별 기본 메트릭 매핑
    TASK_METRICS = {
        "Classification": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "Regression": ["mae", "mse", "rmse", "r2", "mape"],
        "Clustering": ["silhouette_score", "davies_bouldin", "calinski_harabasz"],
        "Causal": ["ate", "att", "confidence_intervals"]
    }
    
    def __init__(self):
        """RecipeBuilder 초기화."""
        self.ui = InteractiveUI()
        templates_dir = Path(__file__).parent.parent / "templates"
        self.template_engine = TemplateEngine(templates_dir)
        self.catalog_dir = Path(__file__).parent.parent.parent / "models" / "catalog"
    
    def run_interactive_flow(self) -> Dict[str, Any]:
        """
        대화형 Recipe 생성 플로우 실행.
        
        Returns:
            Recipe 생성을 위한 선택 사항 딕셔너리
        """
        selections = {}
        
        # 1. Recipe 이름 입력
        recipe_name = self.ui.text_input(
            "Recipe 이름을 입력하세요",
            validator=lambda x: len(x) > 0 and x.replace("-", "").replace("_", "").isalnum()
        )
        selections["recipe_name"] = recipe_name
        
        self.ui.print_divider()
        
        # 2. Task 선택 (DeepLearning 제외)
        self.ui.show_info("Task 선택")
        tasks = self._get_available_tasks()
        
        if not tasks:
            self.ui.show_error("사용 가능한 Task가 없습니다. 모델 카탈로그를 확인해주세요.")
            raise ValueError("No tasks available in catalog")
        
        task = self.ui.select_from_list(
            "Task 종류를 선택하세요",
            tasks
        )
        selections["task"] = task
        
        self.ui.print_divider()
        
        # 3. 모델 선택
        self.ui.show_info(f"{task} 모델 선택")
        models = self._get_models_for_task(task)
        
        if not models:
            self.ui.show_error(f"{task}에 사용 가능한 모델이 없습니다.")
            raise ValueError(f"No models available for task: {task}")
        
        # 모델 정보를 테이블로 표시
        model_info = []
        for model_file in models:
            model_data = self._load_model_spec(model_file)
            model_info.append({
                "name": model_file.stem,
                "class_path": model_data.get("class_path", "Unknown"),
                "library": model_data.get("library", "Unknown"),
                "description": model_data.get("description", "")
            })
        
        # 테이블 표시
        headers = ["모델", "라이브러리", "설명"]
        rows = [
            [info["class_path"], info["library"], info["description"][:50]]
            for info in model_info
        ]
        self.ui.show_table(f"{task} 모델 목록", headers, rows, show_index=True)
        
        # 모델 선택
        model_names = [info["name"] for info in model_info]
        selected_model = self.ui.select_from_list(
            "모델을 선택하세요",
            model_names
        )
        
        # 선택된 모델 정보 저장
        selected_model_info = next(
            info for info in model_info if info["name"] == selected_model
        )
        selections["model_name"] = selected_model
        selections["model_class"] = selected_model_info["class_path"]
        selections["model_library"] = selected_model_info["library"]
        
        # 모델 스펙 로드
        model_spec = self._load_model_spec(
            self.catalog_dir / task / f"{selected_model}.yaml"
        )
        selections["model_spec"] = model_spec
        
        self.ui.print_divider()
        
        # 4. 데이터 설정
        self.ui.show_info("데이터 설정")
        
        # 데이터 소스 타입 선택
        data_source_type = self.ui.select_from_list(
            "데이터 소스 타입을 선택하세요",
            ["SQL 파일", "데이터 파일 (CSV/Parquet)"]
        )
        
        if data_source_type == "SQL 파일":
            source_uri = self.ui.text_input(
                "SQL 파일 경로 (예: sql/train_data.sql)",
                default="sql/train_data.sql"
            )
        else:
            source_uri = self.ui.text_input(
                "데이터 파일 경로 (예: data/train.csv)",
                default="data/train.csv"
            )
        
        selections["source_uri"] = source_uri
        
        # Target column
        target_column = self.ui.text_input(
            "Target column 이름",
            default="target"
        )
        selections["target_column"] = target_column
        
        # Entity schema
        self.ui.show_info("Entity Schema 설정 (엔터로 구분하여 입력, 빈 줄로 종료)")
        entity_schema = []
        while True:
            entity = self.ui.text_input(
                f"Entity {len(entity_schema) + 1} (빈 줄로 종료)",
                default=""
            )
            if not entity:
                break
            entity_schema.append(entity)
        
        if not entity_schema:
            entity_schema = ["user_id", "timestamp"]  # 기본값
        
        selections["entity_schema"] = entity_schema
        
        self.ui.print_divider()
        
        # 5. 전처리 설정
        self.ui.show_info("전처리 설정")
        
        # 전처리 단계 선택
        use_scaler = self.ui.confirm("StandardScaler를 사용하시겠습니까?", default=True)
        use_encoder = self.ui.confirm("OneHotEncoder를 사용하시겠습니까?", default=True)
        
        preprocessor_steps = []
        if use_scaler:
            preprocessor_steps.append({
                "type": "StandardScaler",
                "columns": ["numerical_features"]  # 사용자가 수정 필요
            })
        if use_encoder:
            preprocessor_steps.append({
                "type": "OneHotEncoder",
                "columns": ["categorical_features"]  # 사용자가 수정 필요
            })
        
        selections["preprocessor_steps"] = preprocessor_steps
        
        self.ui.print_divider()
        
        # 6. 평가 설정
        self.ui.show_info("평가 설정")
        
        # Task별 메트릭 자동 설정
        metrics = self.TASK_METRICS.get(task, ["accuracy"])
        selections["metrics"] = metrics
        
        # Validation 설정
        test_size = self.ui.number_input(
            "Test set 비율 (0.1 ~ 0.5)",
            default=0.2,
            min_value=0.1,
            max_value=0.5
        )
        selections["test_size"] = test_size
        
        # Hyperparameter tuning 설정
        enable_tuning = self.ui.confirm(
            "Hyperparameter Tuning (Optuna)을 활성화하시겠습니까?",
            default=False
        )
        selections["enable_tuning"] = enable_tuning
        
        if enable_tuning:
            self.ui.show_info("Hyperparameter Tuning이 활성화되었습니다.")
            self.ui.show_info("학습 시 Optuna가 자동으로 최적의 하이퍼파라미터를 찾습니다.")
            
            n_trials = self.ui.number_input(
                "Tuning trials 수",
                default=10,
                min_value=5,
                max_value=100
            )
            tuning_timeout = self.ui.number_input(
                "Tuning timeout (seconds)",
                default=300,
                min_value=60,
                max_value=3600
            )
            selections["n_trials"] = n_trials
            selections["tuning_timeout"] = tuning_timeout
        else:
            self.ui.show_info("Hyperparameter Tuning이 비활성화되었습니다.")
            self.ui.show_info("모델 카탈로그의 기본값을 사용합니다.")
        
        self.ui.print_divider()
        
        # 선택 사항 확인
        self._show_selections_summary(selections)
        
        if not self.ui.confirm("\n이 설정으로 Recipe를 생성하시겠습니까?", default=True):
            self.ui.show_warning("Recipe 생성이 취소되었습니다. 다시 시작해주세요.")
            return self.run_interactive_flow()
        
        return selections
    
    def generate_recipe_file(self, selections: Dict[str, Any]) -> Path:
        """
        Recipe 파일 생성.
        
        Args:
            selections: 사용자 선택 사항
            
        Returns:
            생성된 Recipe 파일 경로
        """
        # 템플릿 컨텍스트 준비
        context = self._prepare_template_context(selections)
        
        # Recipe 파일 경로
        recipes_dir = Path("recipes")
        recipes_dir.mkdir(exist_ok=True)
        recipe_path = recipes_dir / f"{selections['recipe_name']}.yaml"
        
        # 템플릿 렌더링 및 파일 생성
        self.template_engine.write_rendered_file(
            "recipes/recipe.yaml.j2",
            recipe_path,
            context
        )
        
        return recipe_path
    
    def _get_available_tasks(self) -> List[str]:
        """
        사용 가능한 Task 목록 반환.
        
        Returns:
            Task 이름 리스트 (DeepLearning 제외)
        """
        if not self.catalog_dir.exists():
            return []
        
        tasks = []
        for task_dir in self.catalog_dir.iterdir():
            if task_dir.is_dir() and task_dir.name != "DeepLearning":
                tasks.append(task_dir.name)
        
        return sorted(tasks)
    
    def _get_models_for_task(self, task: str) -> List[Path]:
        """
        특정 Task의 모델 파일 목록 반환.
        
        Args:
            task: Task 이름
            
        Returns:
            모델 YAML 파일 경로 리스트
        """
        task_dir = self.catalog_dir / task
        if not task_dir.exists():
            return []
        
        return sorted(task_dir.glob("*.yaml"))
    
    def _load_model_spec(self, model_file: Path) -> Dict[str, Any]:
        """
        모델 스펙 파일 로드.
        
        Args:
            model_file: 모델 YAML 파일 경로
            
        Returns:
            모델 스펙 딕셔너리
        """
        with open(model_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _prepare_template_context(self, selections: Dict[str, Any]) -> Dict[str, Any]:
        """
        템플릿 렌더링을 위한 컨텍스트 준비.
        
        Args:
            selections: 사용자 선택 사항
            
        Returns:
            템플릿 컨텍스트
        """
        context = {
            "recipe_name": selections["recipe_name"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": selections["task"],
            "model_class": selections["model_class"],
            "model_library": selections["model_library"],
            "hyperparameters": {},
            "metrics": selections["metrics"],
            "source_uri": selections["source_uri"],
            "target_column": selections["target_column"],
            "entity_schema": selections["entity_schema"],
            "preprocessor_steps": selections["preprocessor_steps"],
            "test_size": selections["test_size"],
            "enable_tuning": selections["enable_tuning"]
        }
        
        # 하이퍼파라미터 설정
        model_spec = selections.get("model_spec", {})
        hyperparams = model_spec.get("hyperparameters", {})
        
        # Tuning 활성화 여부에 따라 다르게 처리
        if selections.get("enable_tuning"):
            # 튜닝 활성화시: fixed와 tunable 분리
            context["fixed_params"] = {}
            context["tunable_specs"] = {}
            
            # Fixed 파라미터 (튜닝하지 않음)
            if "fixed" in hyperparams:
                context["fixed_params"].update(hyperparams["fixed"])
            
            # Tunable 파라미터 (Optuna가 탐색)
            if "tunable" in hyperparams:
                for param, config in hyperparams["tunable"].items():
                    context["tunable_specs"][param] = {
                        "type": config.get("type", "float"),
                        "range": config.get("range", [0.1, 1.0])
                    }
        else:
            # 튜닝 비활성화시: 모든 파라미터를 고정값으로
            context["all_hyperparameters"] = {}
            
            # Fixed parameters
            if "fixed" in hyperparams:
                context["all_hyperparameters"].update(hyperparams["fixed"])
            
            # Tunable parameters (기본값 또는 범위의 첫 번째 값 사용)
            if "tunable" in hyperparams:
                for param, config in hyperparams["tunable"].items():
                    if "default" in config:
                        default_value = config["default"]
                    elif "range" in config and isinstance(config["range"], list) and len(config["range"]) > 0:
                        default_value = config["range"][0]
                    else:
                        default_value = 1 if config.get("type") == "int" else 0.1
                    
                    context["all_hyperparameters"][param] = default_value
        
        return context
    
    def _show_selections_summary(self, selections: Dict[str, Any]) -> None:
        """
        선택 사항 요약 표시.
        
        Args:
            selections: 사용자 선택 사항
        """
        summary = f"""
Recipe 이름: {selections['recipe_name']}
Task: {selections['task']}
모델: {selections['model_class']}
라이브러리: {selections['model_library']}
데이터 소스: {selections['source_uri']}
Target Column: {selections['target_column']}
Entity Schema: {', '.join(selections['entity_schema'])}
평가 메트릭: {', '.join(selections['metrics'])}
Test Size: {selections['test_size']}
Hyperparameter Tuning: {'활성화' if selections.get('enable_tuning') else '비활성화'}
"""
        self.ui.show_panel(summary, title="📋 Recipe 설정 요약", style="cyan")