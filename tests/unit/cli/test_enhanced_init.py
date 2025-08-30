"""
Enhanced CLI Init Command Unit Tests
Blueprint v17.0 - TDD RED Phase for M04-1-1

M04-1-1: 템플릿 시스템 재설계
- 환경별 템플릿 (local/dev/prod)
- 통합 --interactive 인터페이스
- 레시피 내용 환경별 자동 생성
- 기존 호환성 100% 유지

CLAUDE.md 원칙 준수:
- RED → GREEN → REFACTOR 사이클
- 테스트 없는 구현 금지
- 커버리지 ≥ 90%
"""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch, call
import click
import typer.testing
from typer.testing import CliRunner
import pytest

from src.cli import app


# Disable autouse fixtures from conftest.py for this test file
@pytest.fixture(scope="session", autouse=True)
def isolated_test_environment():
    """Isolate this test file from conftest.py fixtures"""
    # Change to a temporary directory to avoid config loading issues
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        yield
        os.chdir(original_cwd)


class TestEnhancedInit:
    """M04-1-1 Enhanced Init Command 테스트 클래스"""
    
    def setup_method(self) -> None:
        """각 테스트 메서드 전 실행되는 설정"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def test_enhanced_init__interactive_flag__should_prompt_for_environment(self) -> None:
        """
        --interactive 플래그가 환경 선택 프롬프트를 표시하는지 검증.
        
        Given: --interactive 플래그로 init 실행
        When: 명령어 실행
        Then: local/dev/prod 환경 선택 프롬프트 표시
        """
        # Given: Interactive 모드로 init 실행 준비
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['local', 'classification']
            
            # When: init --interactive 실행
            result = self.runner.invoke(app, ['init', '--interactive', '--dir', self.temp_dir])
            
            # Then: 프롬프트가 두 번 호출되었는지 확인
            assert mock_prompt.call_count == 2, f"Expected 2 prompt calls, got {mock_prompt.call_count}"
            
            # 첫 번째 호출: 환경 선택
            first_call = mock_prompt.call_args_list[0]
            assert first_call[0][0] == "환경을 선택하세요"
            assert isinstance(first_call[1]['type'], click.Choice)
            assert list(first_call[1]['type'].choices) == ['local', 'dev', 'prod']
            
            # 두 번째 호출: 레시피 타입 선택
            second_call = mock_prompt.call_args_list[1]
            assert second_call[0][0] == "레시피 타입을 선택하세요"
            assert isinstance(second_call[1]['type'], click.Choice)
            assert list(second_call[1]['type'].choices) == ['classification', 'regression', 'clustering', 'causal']
            
            assert result.exit_code == 0
    
    def test_enhanced_init__local_environment__should_generate_storage_based_recipes(self) -> None:
        """
        local 환경 선택시 storage adapter 기반 레시피 생성하는지 검증.
        
        Given: interactive 모드에서 local 환경 선택
        When: 템플릿 생성
        Then: storage adapter, CSV 파일 기반 레시피 생성
        """
        # Given: local 환경 선택
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['local', 'classification']
            
            # When: init 실행
            result = self.runner.invoke(app, ['init', '--interactive', '--dir', self.temp_dir])
            
            # Then: local 환경 레시피 파일 생성 확인
            recipe_path = Path(self.temp_dir) / 'recipes' / 'classification_recipe.yaml'
            assert recipe_path.exists(), "classification 레시피 파일이 생성되지 않았습니다"
            
            # 레시피 내용이 storage adapter 사용하는지 확인
            recipe_content = recipe_path.read_text()
            assert 'adapter: storage' in recipe_content, "local 환경 레시피가 storage adapter를 사용하지 않습니다"
            assert '.csv' in recipe_content, "local 환경 레시피가 CSV 파일을 사용하지 않습니다"
    
    def test_enhanced_init__dev_environment__should_generate_sql_based_recipes(self) -> None:
        """
        dev 환경 선택시 SQL adapter + Feature Store 기반 레시피 생성하는지 검증.
        
        Given: interactive 모드에서 dev 환경 선택
        When: 템플릿 생성
        Then: SQL adapter, Feature Store 기반 레시피 생성
        """
        # Given: dev 환경 선택
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['dev', 'classification']
            
            # When: init 실행
            result = self.runner.invoke(app, ['init', '--interactive', '--dir', self.temp_dir])
            
            # Then: dev 환경 레시피 파일 생성 확인
            recipe_path = Path(self.temp_dir) / 'recipes' / 'classification_recipe.yaml'
            assert recipe_path.exists(), "classification 레시피 파일이 생성되지 않았습니다"
            
            # 레시피 내용이 SQL adapter와 Feature Store 사용하는지 확인
            recipe_content = recipe_path.read_text()
            assert 'adapter: sql' in recipe_content, "dev 환경 레시피가 SQL adapter를 사용하지 않습니다"
            assert 'feature_store' in recipe_content, "dev 환경 레시피가 Feature Store를 사용하지 않습니다"
            assert '.sql' in recipe_content, "dev 환경 레시피가 SQL 파일을 사용하지 않습니다"
    
    def test_enhanced_init__prod_environment__should_generate_sql_based_recipes(self) -> None:
        """
        prod 환경 선택시 SQL adapter + Feature Store 기반 레시피 생성하는지 검증.
        
        Given: interactive 모드에서 prod 환경 선택
        When: 템플릿 생성
        Then: SQL adapter, Feature Store 기반 레시피 생성 (dev와 동일)
        """
        # Given: prod 환경 선택
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['prod', 'regression']
            
            # When: init 실행
            result = self.runner.invoke(app, ['init', '--interactive', '--dir', self.temp_dir])
            
            # Then: prod 환경 레시피 파일 생성 확인
            recipe_path = Path(self.temp_dir) / 'recipes' / 'regression_recipe.yaml'
            assert recipe_path.exists(), "regression 레시피 파일이 생성되지 않았습니다"
            
            # 레시피 내용이 SQL adapter와 Feature Store 사용하는지 확인
            recipe_content = recipe_path.read_text()
            assert 'adapter: sql' in recipe_content, "prod 환경 레시피가 SQL adapter를 사용하지 않습니다"
            assert 'feature_store' in recipe_content, "prod 환경 레시피가 Feature Store를 사용하지 않습니다"
    
    def test_enhanced_init__environment_config_generation__should_create_appropriate_config(self) -> None:
        """
        선택된 환경에 따라 적절한 config 파일이 생성되는지 검증.
        
        Given: 각 환경 선택
        When: 템플릿 생성
        Then: 해당 환경에 맞는 config.yaml 생성
        """
        environments = ['local', 'dev', 'prod']
        
        for env in environments:
            with patch('typer.prompt') as mock_prompt:
                mock_prompt.side_effect = [env, 'classification']
                
                # When: 각 환경으로 init 실행
                temp_dir = tempfile.mkdtemp()
                result = self.runner.invoke(app, ['init', '--interactive', '--dir', temp_dir])
                
                # Then: 해당 환경 config 파일 생성 확인
                config_path = Path(temp_dir) / 'config' / f'{env}.yaml'
                assert config_path.exists(), f"{env} 환경 config 파일이 생성되지 않았습니다"
                
                # config 내용에 환경 정보 포함 확인
                config_content = config_path.read_text()
                assert env in config_content, f"{env} config 파일에 환경 정보가 없습니다"
    
    def test_enhanced_init__backward_compatibility__should_work_without_interactive(self) -> None:
        """
        기존 init 명령어 호환성이 유지되는지 검증.
        
        Given: --interactive 플래그 없이 init 실행
        When: 기존 방식으로 실행
        Then: 기존 템플릿 구조 생성 (기본값: local 환경)
        """
        # Given: 기존 방식으로 init 실행 준비
        
        # When: --interactive 없이 init 실행
        result = self.runner.invoke(app, ['init', '--dir', self.temp_dir])
        
        # Then: 명령어 성공 및 기본 구조 생성
        assert result.exit_code == 0, f"기존 init 명령어 실행 실패: {result.output}"
        
        # 기본 디렉토리 구조 생성 확인 (실제 템플릿 구조와 일치)
        expected_dirs = ['config', 'recipes']
        for dir_name in expected_dirs:
            dir_path = Path(self.temp_dir) / dir_name
            assert dir_path.exists(), f"{dir_name} 디렉토리가 생성되지 않았습니다"
            
        # 기본 파일들이 존재하는지 확인
        expected_files = ['.env.template']
        for file_name in expected_files:
            file_path = Path(self.temp_dir) / file_name
            assert file_path.exists(), f"{file_name} 파일이 생성되지 않았습니다"
    
    def test_enhanced_init__recipe_type_selection__should_generate_specific_recipe(self) -> None:
        """
        레시피 타입 선택에 따라 특정 레시피가 생성되는지 검증.
        
        Given: interactive 모드에서 특정 레시피 타입 선택
        When: 각 레시피 타입 선택
        Then: 해당 타입의 레시피 파일 생성
        """
        recipe_types = ['classification', 'regression', 'clustering', 'causal']
        
        for recipe_type in recipe_types:
            with patch('typer.prompt') as mock_prompt:
                mock_prompt.side_effect = ['local', recipe_type]
                
                # When: 특정 레시피 타입으로 init 실행
                temp_dir = tempfile.mkdtemp()
                result = self.runner.invoke(app, ['init', '--interactive', '--dir', temp_dir])
                
                # Then: 해당 레시피 타입 파일 생성 확인
                recipe_path = Path(temp_dir) / 'recipes' / f'{recipe_type}_recipe.yaml'
                assert recipe_path.exists(), f"{recipe_type} 레시피 파일이 생성되지 않았습니다"
                
                # 레시피 내용에 타입 정보 포함 확인
                recipe_content = recipe_path.read_text()
                assert recipe_type in recipe_content.lower(), f"{recipe_type} 레시피 내용이 올바르지 않습니다"


class TestTemplateContentGeneration:
    """템플릿 내용 생성 로직 테스트 클래스"""
    
    def test_template_content__local_vs_dev_recipes__should_have_different_adapters(self) -> None:
        """
        local과 dev 환경 레시피의 adapter 설정이 다른지 검증.
        
        Given: local과 dev 환경용 레시피 생성
        When: 동일한 레시피 타입으로 생성
        Then: adapter 설정이 다르게 생성 (local: storage, dev: sql)
        """
        runner = CliRunner()
        
        # Given: local 환경 레시피 생성
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['local', 'classification']
            local_temp = tempfile.mkdtemp()
            runner.invoke(app, ['init', '--interactive', '--dir', local_temp])
            
        # dev 환경 레시피 생성
        with patch('typer.prompt') as mock_prompt:
            mock_prompt.side_effect = ['dev', 'classification']
            dev_temp = tempfile.mkdtemp()
            runner.invoke(app, ['init', '--interactive', '--dir', dev_temp])
        
        # When: 생성된 레시피 내용 읽기
        local_recipe = (Path(local_temp) / 'recipes' / 'classification_recipe.yaml').read_text()
        dev_recipe = (Path(dev_temp) / 'recipes' / 'classification_recipe.yaml').read_text()
        
        # Then: adapter 설정 차이 확인
        assert 'adapter: storage' in local_recipe, "local 레시피에 storage adapter가 없습니다"
        assert 'adapter: sql' in dev_recipe, "dev 레시피에 sql adapter가 없습니다"
        assert 'feature_store' not in local_recipe, "local 레시피에 feature_store가 포함되어 있습니다"
        assert 'feature_store' in dev_recipe, "dev 레시피에 feature_store가 없습니다"
    
    def test_template_content__hyperparameter_tuning__should_vary_by_environment(self) -> None:
        """
        환경별로 하이퍼파라미터 튜닝 설정이 다른지 검증.
        
        Given: 각 환경별 레시피 생성
        When: hyperparameter_tuning 섹션 확인
        Then: local은 disabled, dev/prod는 enabled
        """
        runner = CliRunner()
        environments = [
            ('local', False),
            ('dev', True),
            ('prod', True)
        ]
        
        for env, tuning_enabled in environments:
            with patch('typer.prompt') as mock_prompt:
                mock_prompt.side_effect = [env, 'classification']
                temp_dir = tempfile.mkdtemp()
                
                # When: 환경별 레시피 생성
                runner.invoke(app, ['init', '--interactive', '--dir', temp_dir])
                
                # Then: 하이퍼파라미터 튜닝 설정 확인
                recipe_content = (Path(temp_dir) / 'recipes' / 'classification_recipe.yaml').read_text()
                
                if tuning_enabled:
                    assert 'enabled: true' in recipe_content, f"{env} 환경에서 하이퍼파라미터 튜닝이 활성화되지 않았습니다"
                    assert 'n_trials:' in recipe_content, f"{env} 환경에서 n_trials 설정이 없습니다"
                else:
                    assert 'enabled: false' in recipe_content, f"{env} 환경에서 하이퍼파라미터 튜닝이 비활성화되지 않았습니다"