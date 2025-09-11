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
from src.components.evaluator.registry import EvaluatorRegistry


class RecipeBuilder:
    """환경 독립적인 Recipe 빌더.
    
    사용자와의 대화형 인터페이스를 통해 모델 선택 및 Recipe 파일을 생성합니다.
    환경 설정과 분리되어 Recipe만을 다룹니다.
    """
    
    def __init__(self):
        """RecipeBuilder 초기화."""
        self.ui = InteractiveUI()
        templates_dir = Path(__file__).parent.parent / "templates"
        self.template_engine = TemplateEngine(templates_dir)
        self.catalog_dir = Path(__file__).parent.parent.parent / "models" / "catalog"
    
    def _is_global_preprocessor(self, step_type: str) -> bool:
        """전처리기가 Global 타입인지 확인합니다.
        
        Global 전처리기는 모든 적합한 컬럼에 자동 적용되므로 컬럼 지정이 불필요합니다.
        """
        global_preprocessors = {
            "standard_scaler", "min_max_scaler", "robust_scaler"  # Scalers only
        }
        return step_type in global_preprocessors
    
    def _get_columns_hint(self, step_type: str) -> str:
        """전처리 단계에 따른 기본 컬럼 힌트를 반환합니다."""
        hints = {
            # Targeted Imputers
            "simple_imputer": "missing_col1,missing_col2",
            # Targeted Encoders
            "one_hot_encoder": "categorical_col1,categorical_col2",
            "ordinal_encoder": "ordinal_col1,ordinal_col2", 
            "catboost_encoder": "high_cardinality_col1,high_cardinality_col2",
            # Targeted Feature Engineering
            "polynomial_features": "numerical_col1,numerical_col2",
            "tree_based_feature_generator": "all_features",
            "kbins_discretizer": "continuous_col1,continuous_col2"
        }
        return hints.get(step_type, "col1,col2")
    
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
        
        # 5. Fetcher 설정 (Feature Store Integration)
        self.ui.print_divider()
        self.ui.show_info("Feature Store 설정")
        
        use_feature_store = self.ui.confirm(
            "Feature Store를 사용하시겠습니까? (Point-in-time join이 가능한 feature store 보유 시에만 활성화 하세요)",
            default=False
        )
        
        if use_feature_store:
            selections["fetcher_type"] = "feature_store"
            
            # Timestamp column (PIT join 기준)
            def _non_empty(value: str) -> bool:
                return isinstance(value, str) and len(value.strip()) > 0
            timestamp_column = self.ui.text_input(
                "Timestamp column 이름 (Point-in-Time join 기준) — 공백 불가", 
                default="event_timestamp",
                validator=_non_empty
            )
            selections["timestamp_column"] = timestamp_column
            
            # FeatureView 설정 (Feast 표준)
            self.ui.show_info("Feast FeatureView 설정")
            feature_views = {}
            
            while True:
                feature_view_name = self.ui.text_input(
                    f"FeatureView 이름 {len(feature_views) + 1} (예: user_features, 빈 줄로 종료)",
                    default=""
                )
                if not feature_view_name:
                    break
                
                # Join key 설정
                join_key = self.ui.text_input(
                    f"{feature_view_name}의 join key column (예: user_id)",
                    default=f"{feature_view_name.replace('_features', '')}_id"
                )
                
                # Features 설정
                features_str = self.ui.text_input(
                    f"{feature_view_name}에서 가져올 features (쉼표로 구분)",
                    default="feature1,feature2,feature3"
                )
                features = [f.strip() for f in features_str.split(",")]
                
                feature_views[feature_view_name] = {
                    "join_key": join_key,
                    "features": features
                }
                
                self.ui.show_info(f"✓ {feature_view_name} 설정 완료")
            
            if not feature_views:
                # 기본값 제공
                feature_views = {
                    "user_features": {
                        "join_key": "user_id",
                        "features": ["feature1", "feature2"]
                    }
                }
            
            selections["feature_views"] = feature_views
        else:
            selections["fetcher_type"] = "pass_through"
            selections["timestamp_column"] = "event_timestamp"  # 기본값
            selections["feature_views"] = None
        
        # Target column
        target_column = self.ui.text_input(
            "Target column 이름",
            default="target"
        )
        selections["target_column"] = target_column
        
        # Treatment column (causal task에서만)
        if task.lower() == "causal":
            self.ui.show_info("🧪 Causal Inference 설정")
            treatment_column = self.ui.text_input(
                "Treatment column 이름 (처치 변수, 예: campaign_exposure)",
                default="treatment"
            )
            selections["treatment_column"] = treatment_column
        else:
            selections["treatment_column"] = None
        
        # Timestamp column (timeseries task에서만)
        if task.lower() == "timeseries":
            self.ui.show_info("📈 Timeseries 설정")
            def _non_empty(value: str) -> bool:
                return isinstance(value, str) and len(value.strip()) > 0
            timestamp_column = self.ui.text_input(
                "Timestamp column 이름 (시계열 시간 컬럼, 예: timestamp, date) — 공백 불가",
                default="timestamp",
                validator=_non_empty
            )
            selections["timeseries_timestamp_column"] = timestamp_column
        else:
            selections["timeseries_timestamp_column"] = None
        
        # Entity columns 설정 (새로 추가)
        self.ui.show_info("🔗 Entity Columns 설정")
        entity_columns_str = self.ui.text_input(
            "Entity column(s) 이름 (쉼표로 구분, 예: user_id,item_id)",
            default="user_id"
        )
        entity_columns = [col.strip() for col in entity_columns_str.split(",")]
        selections["entity_columns"] = entity_columns
        
        # Feature columns 처리 방법 안내
        self.ui.show_info("📊 Feature Columns 자동 처리")
        self.ui.show_info(
            "💡 Feature columns는 자동으로 처리됩니다:\n"
            "   - Target, Treatment, Entity columns를 제외한 모든 컬럼이 자동으로 feature로 사용됩니다\n"
            "   - 별도 설정이 필요하지 않습니다"
        )
        
        self.ui.print_divider()
        
        # 5. 전처리 설정
        self.ui.show_info("전처리 설정")
        
        # 사용 가능한 전처리 모듈들 (논리적 순서로 배치)
        available_preprocessors = {
            "Missing Value Handling": {
                "simple_imputer": "SimpleImputer (결측값 채우기 + 선택적 지시자 생성)"
            },
            "Encoder": {
                "one_hot_encoder": "OneHotEncoder (범주형 → 원핫 인코딩)",
                "ordinal_encoder": "OrdinalEncoder (범주형 → 순서형 인코딩)",
                "catboost_encoder": "CatBoostEncoder (Target 기반 인코딩)"
            },
            "Feature Engineering": {
                "polynomial_features": "PolynomialFeatures (다항 특성 생성)",
                "tree_based_feature_generator": "TreeBasedFeatures (트리 기반 특성)",
                "kbins_discretizer": "KBinsDiscretizer (연속형 → 구간형)"
            },
            "Scaler": {
                "standard_scaler": "StandardScaler (평균=0, 분산=1 정규화)",
                "min_max_scaler": "MinMaxScaler (0-1 범위 정규화)",
                "robust_scaler": "RobustScaler (이상치에 강건한 정규화)"
            }
        }
        
        preprocessor_steps = []
        
        # 각 카테고리별로 전처리 선택
        for category, preprocessors in available_preprocessors.items():
            if not self.ui.confirm(f"\n{category} 전처리를 사용하시겠습니까?", default=category in ["Missing Value Handling", "Encoder", "Scaler"]):
                continue
            
            # 해당 카테고리의 전처리 방법 선택
            if category == "Feature Engineering":
                # Feature Engineering은 여러 개 선택 가능
                self.ui.show_info("Feature Engineering은 여러 개 선택 가능합니다. 선택 완료 시 '완료' 선택")
                selected_features = []
                options = list(preprocessors.values()) + ["완료"]
                while True:
                    selected = self.ui.select_from_list(
                        f"{category} 유형을 선택하세요 (현재 선택: {len(selected_features)}개)",
                        options
                    )
                    if selected == "완료" or selected is None:
                        break
                    if selected in selected_features:
                        self.ui.show_warning(f"{selected}는 이미 선택되었습니다.")
                    else:
                        selected_features.append(selected)
                        # 선택된 항목을 리스트에서 제거
                        options.remove(selected)
                
                # 선택된 Feature Engineering 처리
                for sel in selected_features:
                    step_type = [k for k, v in preprocessors.items() if v == sel][0]
                    
                    step_config = {"type": step_type}
                    
                    # Global 전처리기가 아닌 경우에만 컬럼 요청
                    if not self._is_global_preprocessor(step_type):
                        columns_hint = self._get_columns_hint(step_type)
                        columns_str = self.ui.text_input(
                            f"{sel}에 적용할 컬럼 (쉼표로 구분)",
                            default=columns_hint
                        )
                        columns = [col.strip() for col in columns_str.split(",")]
                        step_config["columns"] = columns
                    else:
                        self.ui.show_info(f"{sel}는 모든 적합한 컬럼에 자동으로 적용됩니다.")
                    
                    # 특별한 파라미터가 필요한 경우
                    if step_type == "polynomial_features":
                        degree = self.ui.number_input("다항 차수", default=2, min_value=2, max_value=4)
                        step_config["degree"] = degree
                    elif step_type == "kbins_discretizer":
                        n_bins = self.ui.number_input("구간 개수", default=5, min_value=2, max_value=10)
                        step_config["n_bins"] = n_bins
                        
                        strategy = self.ui.single_choice(
                            "구간 분할 전략을 선택하세요",
                            [
                                ("uniform", "Uniform (균등한 간격으로 분할)"),
                                ("quantile", "Quantile (분위수 기반으로 분할)"),
                                ("kmeans", "K-means (클러스터링 기반으로 분할)")
                            ],
                            default="quantile"
                        )
                        step_config["strategy"] = strategy
                    
                    preprocessor_steps.append(step_config)
            else:  # 단일 선택 (Scaler, Encoder, Imputer)
                options = list(preprocessors.values())
                selected = self.ui.select_from_list(
                    f"{category} 유형을 선택하세요",
                    options
                )
                
                if selected:  # 사용자가 선택한 경우
                    step_type = [k for k, v in preprocessors.items() if v == selected][0]
                    
                    step_config = {"type": step_type}
                    
                    # Global 전처리기가 아닌 경우에만 컬럼 요청
                    if not self._is_global_preprocessor(step_type):
                        columns_hint = self._get_columns_hint(step_type)
                        columns_str = self.ui.text_input(
                            f"{selected}에 적용할 컬럼 (쉼표로 구분)",
                            default=columns_hint
                        )
                        columns = [col.strip() for col in columns_str.split(",")]
                        step_config["columns"] = columns
                    else:
                        self.ui.show_info(f"{selected}는 모든 적합한 컬럼에 자동으로 적용됩니다.")
                    
                    # 특별한 파라미터가 필요한 경우
                    if step_type == "simple_imputer":
                        strategy = self.ui.select_from_list(
                            "결측값 대체 전략",
                            ["mean", "median", "most_frequent", "constant"]
                        )
                        step_config["strategy"] = strategy
                        
                        # Missing indicators 생성 여부
                        create_indicators = self.ui.confirm(
                            "결측값 지시자 컬럼을 생성하시겠습니까? (imputation 전 결측값 위치 표시)",
                            default=False
                        )
                        step_config["create_missing_indicators"] = create_indicators
                    elif step_type == "catboost_encoder":
                        # CatBoostEncoder는 target이 필요함
                        step_config["sigma"] = self.ui.number_input(
                            "Regularization strength (sigma)",
                            default=0.05,
                            min_value=0.0,
                            max_value=1.0
                        )
                    
                    preprocessor_steps.append(step_config)
        
        selections["preprocessor_steps"] = preprocessor_steps
        
        self.ui.print_divider()
        
        # 6. 평가 설정
        self.ui.show_info("평가 설정")
        
        # Task별 메트릭 자동 설정
        try:
            evaluator_class = EvaluatorRegistry.get_evaluator_class(task.lower())
            metrics = evaluator_class.METRIC_KEYS
        except KeyError:
            self.ui.show_error(f"사용 가능한 평가 메트릭이 없습니다. {task} Task에 대한 평가 메트릭을 직접 설정해주세요.")
            raise ValueError(f"No metrics available for task: {task}")

        selections["metrics"] = metrics
        
        # Validation 설정
        self.ui.show_info(
            "📊 데이터 분할 전략:\n"
            "• 기본: Train(80%) / Test(20%)\n"
            "• Optuna 사용 시: Train을 다시 Train(64%) / Val(16%)로 분할하여 튜닝"
        )
        test_size = self.ui.number_input(
            "Test set 비율 (0.1 ~ 0.5)",
            default=0.2,
            min_value=0.1,
            max_value=0.5
        )
        selections["test_size"] = test_size
        
        # Hyperparameter tuning 설정
        enable_tuning = self.ui.confirm(
            "Hyperparameter Tuning (Optuna)을 활성화하시겠습니까?\n"
            "(활성화 시 Train 데이터에서 Validation set 자동 생성)",
            default=False
        )
        selections["enable_tuning"] = enable_tuning
        
        if enable_tuning:
            self.ui.show_info("Hyperparameter Tuning이 활성화되었습니다.")
            self.ui.show_info("학습 시 Optuna가 자동으로 최적의 하이퍼파라미터를 찾습니다.")
            
            # 최적화할 metric 선택
            self.ui.show_info(f"🎯 {task} Task의 최적화 기준 지표 선택")
            available_metrics = self.TASK_METRICS.get(task, ["accuracy"])
            
            # 각 metric의 최적화 방향을 표시
            metric_descriptions = available_metrics
            
            optimization_metric = self.ui.select_from_list(
                f"{task}에서 최적화할 지표를 선택하세요 (방향키 사용)",
                metric_descriptions
            )
            
            selections["optimization_metric"] = optimization_metric
            # direction은 recipe 생성 시 자동으로 결정되므로 제거
            
            self.ui.show_info(f"✓ 선택된 최적화 기준: {optimization_metric}")
            
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
            message = "Recipe 생성이 취소되었습니다. 현재 선택 내용을 반환합니다."
            # 일부 테스트 스텁은 show_warning을 제공하지 않음 → 안전 대체
            if hasattr(self.ui, "show_warning"):
                self.ui.show_warning(message)
            elif hasattr(self.ui, "show_info"):
                self.ui.show_info(message)
            # 재귀 재시작 대신 현재 selections를 반환하여 무한루프 방지
            return selections
        
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
            "entity_columns": selections["entity_columns"],  # ✅ entity_columns 추가
            "target_column": selections["target_column"],
            "treatment_column": selections.get("treatment_column", None),
            "feature_columns": selections.get("feature_columns", None),  # ✅ feature_columns 추가 (null 허용)
            "timestamp_column": selections["timestamp_column"],
            "preprocessor_steps": selections["preprocessor_steps"],
            "test_size": selections["test_size"],
            "enable_tuning": selections["enable_tuning"],
            "optimization_metric": selections.get("optimization_metric", "accuracy"),
            "fetcher_type": selections.get("fetcher_type", "pass_through"),
            "feature_views": selections.get("feature_views", None)
        }
        
        # 하이퍼파라미터 설정
        model_spec = selections.get("model_spec", {})
        hyperparams = model_spec.get("hyperparameters", {})
        
        # Tuning 활성화 여부에 따라 다르게 처리
        if selections.get("enable_tuning"):
            # 튜닝 활성화시: fixed와 tunable 분리
            context["fixed_params"] = {}
            context["tunable_specs"] = {}
            context["n_trials"] = selections.get("n_trials", 10)
            context["tuning_timeout"] = selections.get("tuning_timeout", 300)
            
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
        # FeatureView 요약 생성
        feature_store_info = ""
        if selections.get("feature_views"):
            feature_views_summary = []
            for view_name, config in selections["feature_views"].items():
                feature_views_summary.append(f"{view_name} (join_key: {config['join_key']})")
            feature_store_info = f"FeatureViews: {', '.join(feature_views_summary)}\n"

        # Treatment column 정보 (causal task에서만)
        treatment_info = ""
        if selections.get("treatment_column"):
            treatment_info = f"Treatment Column: {selections['treatment_column']}\n"

        summary = f"""
        Recipe 이름: {selections['recipe_name']}
        Task: {selections['task']}
        모델: {selections['model_class']}
        라이브러리: {selections['model_library']}
        데이터 소스: {selections['source_uri']}
        Target Column: {selections['target_column']}
        {treatment_info}{feature_store_info}평가 메트릭: {', '.join(selections['metrics'])}
        Test Size: {selections['test_size']}
        Hyperparameter Tuning: {'활성화' if selections.get('enable_tuning') else '비활성화'}
        """
        self.ui.show_panel(summary, title="📋 Recipe 설정 요약", style="cyan")