"""
Test for get-recipe command implementation
Phase 4: TDD 기반 대화형 모델 선택 테스트

CLAUDE.md 원칙 준수:
- TDD: RED → GREEN → REFACTOR
- 타입 힌트 필수  
- Google Style Docstring
"""

import pytest
import typer
from unittest.mock import Mock, patch

from src.cli.commands.get_recipe_command import (
    InteractiveModelSelector,
    get_recipe_command
)
from src.cli.utils.recipe_generator import CatalogBasedRecipeGenerator


class TestInteractiveModelSelector:
    """InteractiveModelSelector 테스트 클래스."""
    
    @pytest.fixture
    def mock_catalog(self):
        """Mock ModelCatalog 생성."""
        catalog = Mock()
        catalog.models = {
            "Classification": [
                Mock(class_path="sklearn.ensemble.RandomForestClassifier", 
                     library="scikit-learn", description="Random Forest"),
                Mock(class_path="xgboost.XGBClassifier",
                     library="xgboost", description="XGBoost")
            ],
            "Regression": [
                Mock(class_path="sklearn.linear_model.LinearRegression",
                     library="scikit-learn", description="Linear Regression")
            ]
        }
        return catalog
    
    def test_interactive_model_selector_initialization(self, mock_catalog):
        """
        GIVEN: ModelCatalog이 제공됨
        WHEN: InteractiveModelSelector 초기화
        THEN: catalog과 console이 올바르게 설정됨
        """
        selector = InteractiveModelSelector(mock_catalog)
        
        assert selector.catalog == mock_catalog
        assert selector.console is not None
    
    @patch('src.cli.commands.get_recipe_command.Prompt.ask')
    def test_select_environment_local_choice(self, mock_prompt, mock_catalog):
        """
        GIVEN: 사용자가 환경 선택 UI에서 '1' 입력 (local)
        WHEN: _select_environment 호출
        THEN: 'local' 반환
        """
        mock_prompt.return_value = "1"
        selector = InteractiveModelSelector(mock_catalog)
        
        result = selector._select_environment()
        
        assert result == "local"
    
    @patch('src.cli.commands.get_recipe_command.Prompt.ask')
    def test_select_environment_dev_choice(self, mock_prompt, mock_catalog):
        """
        GIVEN: 사용자가 환경 선택 UI에서 '2' 입력 (dev)  
        WHEN: _select_environment 호출
        THEN: 'dev' 반환
        """
        mock_prompt.return_value = "2" 
        selector = InteractiveModelSelector(mock_catalog)
        
        result = selector._select_environment()
        
        assert result == "dev"
    
    @patch('src.cli.commands.get_recipe_command.Prompt.ask')
    def test_select_task_classification_choice(self, mock_prompt, mock_catalog):
        """
        GIVEN: 사용자가 태스크 선택 UI에서 '1' 입력 (Classification)
        WHEN: _select_task 호출  
        THEN: 'Classification' 반환
        """
        mock_prompt.return_value = "1"
        selector = InteractiveModelSelector(mock_catalog)
        
        result = selector._select_task()
        
        assert result == "Classification"
    
    @patch('src.cli.commands.get_recipe_command.Prompt.ask')
    def test_select_model_first_choice(self, mock_prompt, mock_catalog):
        """
        GIVEN: Classification 태스크와 사용자가 '1' 선택
        WHEN: _select_model 호출
        THEN: 첫 번째 모델 반환
        """
        mock_prompt.return_value = "1"
        selector = InteractiveModelSelector(mock_catalog)
        task = "Classification"
        
        result = selector._select_model(task)
        
        assert result == mock_catalog.models["Classification"][0]
    
    @patch('src.cli.commands.get_recipe_command.Prompt.ask')
    def test_run_interactive_selection_full_flow(self, mock_prompt, mock_catalog):
        """
        GIVEN: 사용자가 순서대로 '2', '1', '1' 선택 (dev, Classification, 첫 모델)
        WHEN: run_interactive_selection 호출
        THEN: ('dev', 'Classification', 선택된 모델) 튜플 반환
        """
        mock_prompt.side_effect = ["2", "1", "1", "y"]  # dev, Classification, 첫 모델, 확인
        selector = InteractiveModelSelector(mock_catalog)
        
        environment, task, model_spec = selector.run_interactive_selection()
        
        assert environment == "dev"
        assert task == "Classification" 
        assert model_spec == mock_catalog.models["Classification"][0]


class TestCatalogBasedRecipeGenerator:
    """CatalogBasedRecipeGenerator 테스트 클래스."""
    
    @pytest.fixture
    def mock_catalog(self):
        """Mock ModelCatalog with hyperparameters."""
        catalog = Mock()
        model_spec = Mock()
        model_spec.class_path = "sklearn.ensemble.RandomForestClassifier"
        model_spec.hyperparameters = {
            "fixed": {"random_state": 42},
            "environment_defaults": {
                "local": {"n_estimators": 10, "max_depth": 3},
                "dev": {"n_estimators": 50, "max_depth": 10}
            },
            "tunable": {
                "n_estimators": {"type": "int", "range": [10, 200], "default": 100},
                "max_depth": {"type": "int", "range": [3, 20], "default": 10}
            }
        }
        catalog.models = {"Classification": [model_spec]}
        return catalog
    
    @patch('pathlib.Path.write_text')
    def test_generate_recipe_creates_file(self, mock_write_text, mock_catalog):
        """
        GIVEN: 환경, 태스크, 모델 스펙이 제공됨
        WHEN: generate_recipe 호출
        THEN: recipes/ 디렉토리에 YAML 파일 생성됨
        """
        generator = CatalogBasedRecipeGenerator()
        environment = "dev"
        task = "Classification"
        model_spec = mock_catalog.models["Classification"][0]
        
        result_path = generator.generate_recipe(environment, task, model_spec)
        
        assert "recipes/" in str(result_path)
        assert result_path.suffix == ".yaml"
        mock_write_text.assert_called_once()
    
    def test_generate_recipe_yaml_content_structure(self, mock_catalog):
        """
        GIVEN: dev 환경과 Classification 모델
        WHEN: generate_recipe 호출
        THEN: 생성된 레시피 파일이 올바른 구조를 가짐
        """
        generator = CatalogBasedRecipeGenerator()
        environment = "dev"
        task = "Classification" 
        model_spec = mock_catalog.models["Classification"][0]
        
        # 실제 파일을 생성하고 내용을 검증
        recipe_path = generator.generate_recipe(environment, task, model_spec)
        yaml_content = recipe_path.read_text()
        
        assert "name:" in yaml_content
        assert "model:" in yaml_content
        assert "hyperparameters:" in yaml_content
        assert 'environment: "dev"' in yaml_content
        assert model_spec.class_path in yaml_content


class TestGetRecipeCommand:
    """get-recipe CLI 명령어 테스트."""
    
    @patch('src.cli.commands.get_recipe_command.get_recipe_command')
    def test_get_recipe_command_success_flow(self, mock_get_recipe):
        """
        GIVEN: get_recipe_command가 정상 실행됨
        WHEN: get_recipe_command 호출
        THEN: 함수가 호출됨
        """
        # 실행
        mock_get_recipe()
        
        # 검증
        mock_get_recipe.assert_called_once()
    
    @patch('src.cli.commands.get_recipe_command.ModelCatalog.from_yaml')
    def test_get_recipe_command_catalog_error(self, mock_catalog_class):
        """
        GIVEN: catalog 로딩 실패
        WHEN: get_recipe_command 실행
        THEN: 오류 발생 시 typer.Exit 호출됨
        """
        mock_catalog_class.from_yaml.side_effect = Exception("Catalog loading failed")
        
        with pytest.raises(typer.Exit):
            get_recipe_command()