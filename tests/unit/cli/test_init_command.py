"""
Init Command Tests - 실제 구현에 맞춘 테스트
Phase 5: 현재 CLI 구현을 기반으로 한 정확한 테스트

CLAUDE.md 원칙 준수:
- TDD 기반 테스트
- 타입 힌트 필수
- Google Style Docstring
- 실제 구현과 100% 일치하는 테스트
"""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from typer.testing import CliRunner
import pytest
import subprocess

from src.cli import app


@pytest.fixture(scope="session", autouse=True)
def isolated_test_environment():
    """Isolate this test file from conftest.py fixtures"""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        yield
        os.chdir(original_cwd)


class TestInitCommand:
    """Init 명령어 테스트 클래스 - 실제 구현 기반"""
    
    def setup_method(self) -> None:
        """각 테스트 메서드 전 실행되는 설정"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self) -> None:
        """각 테스트 메서드 후 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.unit
    def test_init__with_project_name_option__should_create_project_structure(self) -> None:
        """
        프로젝트명 옵션으로 init 실행 시 프로젝트 구조가 생성되는지 검증.
        
        Given: 프로젝트명을 옵션으로 제공
        When: init 명령어 실행
        Then: 기본 프로젝트 구조가 생성되어야 함
        """
        # Given: 프로젝트명을 옵션으로 제공
        project_name = "test_project"
        
        with patch('typer.confirm', return_value=False) as mock_confirm:
            with patch('src.cli.commands.init_command.create_project_structure') as mock_create:
                # When: init 명령어 실행
                result = self.runner.invoke(app, ['init', '--project-name', project_name])
                
                # Then: 명령어 성공
                assert result.exit_code == 0, f"init 명령어 실행 실패: {result.output}"
                
                # mmp-local-dev 설치 여부 확인 호출
                mock_confirm.assert_called_once_with(
                    "🐳 mmp-local-dev를 함께 설치하시겠습니까? (PostgreSQL, Redis, MLflow 개발 환경)"
                )
                
                # 프로젝트 구조 생성 호출 확인
                mock_create.assert_called_once()
                args, kwargs = mock_create.call_args
                assert str(args[0]).endswith(project_name), "프로젝트 경로가 올바르지 않습니다"
                assert kwargs['with_mmp_dev'] is False, "with_mmp_dev 플래그가 올바르지 않습니다"

    @pytest.mark.unit
    def test_init__without_project_name__should_prompt_for_name(self) -> None:
        """
        프로젝트명 없이 init 실행 시 프로젝트명을 입력받는지 검증.
        
        Given: 프로젝트명 옵션 없이 실행
        When: init 명령어 실행
        Then: 프로젝트명 입력 프롬프트가 표시되어야 함
        """
        # Given: 프로젝트명 없이 실행
        expected_project_name = "prompted_project"
        
        with patch('typer.confirm', return_value=False) as mock_confirm:
            with patch('typer.prompt', return_value=expected_project_name) as mock_prompt:
                with patch('src.cli.commands.init_command.create_project_structure') as mock_create:
                    # When: init 명령어 실행
                    result = self.runner.invoke(app, ['init'])
                    
                    # Then: 명령어 성공
                    assert result.exit_code == 0, f"init 명령어 실행 실패: {result.output}"
                    
                    # 프로젝트명 입력 프롬프트 호출 확인
                    mock_prompt.assert_called_once_with("📁 프로젝트 이름을 입력하세요")
                    
                    # 프로젝트 구조 생성 호출 확인
                    mock_create.assert_called_once()
                    args, kwargs = mock_create.call_args
                    assert str(args[0]).endswith(expected_project_name), "프롬프트로 입력받은 프로젝트명이 사용되지 않았습니다"

    @pytest.mark.unit  
    def test_init__with_mmp_dev_flag__should_install_mmp_local_dev(self) -> None:
        """
        --with-mmp-dev 플래그로 실행 시 mmp-local-dev가 설치되는지 검증.
        
        Given: --with-mmp-dev 플래그로 실행
        When: init 명령어 실행
        Then: mmp-local-dev clone이 실행되어야 함
        """
        # Given: --with-mmp-dev 플래그로 실행
        project_name = "test_project"
        
        with patch('src.cli.commands.init_command.clone_mmp_local_dev') as mock_clone:
            with patch('src.cli.commands.init_command.create_project_structure') as mock_create:
                # When: init 명령어 실행
                result = self.runner.invoke(app, ['init', '--project-name', project_name, '--with-mmp-dev'])
                
                # Then: 명령어 성공
                assert result.exit_code == 0, f"init 명령어 실행 실패: {result.output}"
                
                # mmp-local-dev clone 호출 확인
                mock_clone.assert_called_once()
                args = mock_clone.call_args[0]
                # 실제로는 Path.cwd().parent가 전달되므로, 상위 디렉토리 경로인지 확인
                called_path = args[0]
                assert isinstance(called_path, Path), "Path 객체가 전달되지 않았습니다"
                
                # 프로젝트 구조 생성에 with_mmp_dev=True 전달 확인
                args, kwargs = mock_create.call_args
                assert kwargs['with_mmp_dev'] is True, "with_mmp_dev 플래그가 제대로 전달되지 않았습니다"

    @pytest.mark.unit
    def test_init__mmp_dev_confirm_yes__should_install_mmp_local_dev(self) -> None:
        """
        mmp-local-dev 설치 확인에 yes 응답 시 설치가 진행되는지 검증.
        
        Given: mmp-local-dev 설치 확인에 yes 응답
        When: init 명령어 실행
        Then: mmp-local-dev clone이 실행되어야 함
        """
        # Given: mmp-local-dev 설치 확인에 yes 응답
        project_name = "test_project"
        
        with patch('typer.confirm', return_value=True) as mock_confirm:
            with patch('src.cli.commands.init_command.clone_mmp_local_dev') as mock_clone:
                with patch('src.cli.commands.init_command.create_project_structure') as mock_create:
                    # When: init 명령어 실행
                    result = self.runner.invoke(app, ['init', '--project-name', project_name])
                    
                    # Then: 명령어 성공
                    assert result.exit_code == 0, f"init 명령어 실행 실패: {result.output}"
                    
                    # mmp-local-dev 설치 확인 호출
                    mock_confirm.assert_called_once()
                    
                    # mmp-local-dev clone 호출 확인
                    mock_clone.assert_called_once()

    @pytest.mark.unit
    def test_init__mmp_dev_clone_failure__should_continue_with_warning(self) -> None:
        """
        mmp-local-dev clone 실패 시 경고와 함께 계속 진행되는지 검증.
        
        Given: mmp-local-dev clone이 실패하는 상황
        When: init 명령어 실행
        Then: 경고 메시지와 함께 프로젝트 생성은 계속되어야 함
        """
        # Given: mmp-local-dev clone이 실패하는 상황
        project_name = "test_project"
        clone_error = subprocess.CalledProcessError(1, ["git", "clone"], "Clone failed")
        
        with patch('typer.confirm', return_value=True):
            with patch('src.cli.commands.init_command.clone_mmp_local_dev', side_effect=clone_error):
                with patch('src.cli.commands.init_command.create_project_structure') as mock_create:
                    # When: init 명령어 실행
                    result = self.runner.invoke(app, ['init', '--project-name', project_name])
                    
                    # Then: 명령어 성공 (실패해도 계속 진행)
                    assert result.exit_code == 0, f"init 명령어 실행 실패: {result.output}"
                    
                    # 경고 메시지 출력 확인
                    assert "⚠️ mmp-local-dev 설치 중 오류 발생" in result.output
                    
                    # 프로젝트 구조 생성은 계속 실행
                    mock_create.assert_called_once()

    @pytest.mark.unit
    def test_init__keyboard_interrupt__should_exit_gracefully(self) -> None:
        """
        키보드 인터럽트 발생 시 우아하게 종료되는지 검증.
        
        Given: 프롬프트 중 키보드 인터럽트 발생
        When: init 명령어 실행
        Then: 적절한 에러 메시지와 함께 종료되어야 함
        """
        # Given: 프롬프트 중 키보드 인터럽트 발생
        with patch('typer.confirm', side_effect=KeyboardInterrupt()):
            # When: init 명령어 실행
            result = self.runner.invoke(app, ['init'])
            
            # Then: 에러 코드 1로 종료
            assert result.exit_code == 1, "키보드 인터럽트 시 에러 코드가 올바르지 않습니다"
            
            # 적절한 에러 메시지 출력 확인
            assert "❌ 프로젝트 초기화가 취소되었습니다" in result.output


class TestCreateProjectStructure:
    """create_project_structure 함수 테스트 클래스"""
    
    def setup_method(self) -> None:
        """각 테스트 메서드 전 실행되는 설정"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self) -> None:
        """각 테스트 메서드 후 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.unit
    def test_create_project_structure__basic_creation__should_create_all_directories(self) -> None:
        """
        기본 프로젝트 구조 생성이 모든 디렉토리를 생성하는지 검증.
        
        Given: 빈 디렉토리
        When: create_project_structure 실행
        Then: config, recipes, data, docs 디렉토리가 생성되어야 함
        """
        # Given: 빈 디렉토리
        from src.cli.commands.init_command import create_project_structure
        project_path = self.temp_dir / "test_project"
        
        # When: create_project_structure 실행
        with patch('src.cli.commands.init_command._generate_config_files') as mock_config:
            with patch('src.cli.commands.init_command._generate_sample_data') as mock_data:
                with patch('src.cli.commands.init_command._generate_project_docs') as mock_docs:
                    create_project_structure(project_path, with_mmp_dev=False)
        
        # Then: 모든 디렉토리가 생성되어야 함
        assert project_path.exists(), "프로젝트 디렉토리가 생성되지 않았습니다"
        assert (project_path / "config").exists(), "config 디렉토리가 생성되지 않았습니다"
        assert (project_path / "recipes").exists(), "recipes 디렉토리가 생성되지 않았습니다"
        assert (project_path / "data").exists(), "data 디렉토리가 생성되지 않았습니다"
        assert (project_path / "docs").exists(), "docs 디렉토리가 생성되지 않았습니다"
        
        # 헬퍼 함수들이 호출되었는지 확인
        mock_config.assert_called_once_with(project_path, False)
        mock_data.assert_called_once_with(project_path)
        mock_docs.assert_called_once_with(project_path)

    @pytest.mark.unit
    def test_create_project_structure__with_mmp_dev__should_pass_flag_to_config_generation(self) -> None:
        """
        with_mmp_dev=True일 때 config 생성에 플래그가 전달되는지 검증.
        
        Given: with_mmp_dev=True
        When: create_project_structure 실행
        Then: _generate_config_files에 True가 전달되어야 함
        """
        # Given: with_mmp_dev=True
        from src.cli.commands.init_command import create_project_structure
        project_path = self.temp_dir / "test_project"
        
        # When: create_project_structure 실행
        with patch('src.cli.commands.init_command._generate_config_files') as mock_config:
            with patch('src.cli.commands.init_command._generate_sample_data'):
                with patch('src.cli.commands.init_command._generate_project_docs'):
                    create_project_structure(project_path, with_mmp_dev=True)
        
        # Then: _generate_config_files에 True가 전달되어야 함
        mock_config.assert_called_once_with(project_path, True)


class TestCloneMmpLocalDev:
    """clone_mmp_local_dev 함수 테스트 클래스"""

    @pytest.mark.unit
    def test_clone_mmp_local_dev__successful_clone__should_execute_git_command(self) -> None:
        """
        성공적인 clone이 git 명령어를 실행하는지 검증.
        
        Given: git clone이 성공하는 상황
        When: clone_mmp_local_dev 실행
        Then: 적절한 git 명령어가 실행되어야 함
        """
        # Given: git clone이 성공하는 상황
        from src.cli.commands.init_command import clone_mmp_local_dev
        parent_dir = Path(tempfile.mkdtemp())
        
        # subprocess.run을 모킹
        mock_result = Mock()
        mock_result.returncode = 0
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            with patch.object(Path, 'exists', return_value=False):  # mmp-local-dev가 존재하지 않음
                # When: clone_mmp_local_dev 실행
                clone_mmp_local_dev(parent_dir)
        
        # Then: 적절한 git 명령어가 실행되어야 함
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "git", "git 명령어가 실행되지 않았습니다"
        assert cmd[1] == "clone", "git clone이 실행되지 않았습니다"
        assert "mmp-local-dev.git" in cmd[2], "올바른 리포지토리 URL이 사용되지 않았습니다"
        assert str(parent_dir / "mmp-local-dev") == cmd[3], "올바른 대상 경로가 사용되지 않았습니다"

    @pytest.mark.unit
    def test_clone_mmp_local_dev__already_exists__should_skip(self) -> None:
        """
        mmp-local-dev가 이미 존재하는 경우 건너뛰는지 검증.
        
        Given: mmp-local-dev 디렉토리가 이미 존재
        When: clone_mmp_local_dev 실행
        Then: git 명령어가 실행되지 않아야 함
        """
        # Given: mmp-local-dev 디렉토리가 이미 존재
        from src.cli.commands.init_command import clone_mmp_local_dev
        parent_dir = Path(tempfile.mkdtemp())
        
        with patch('subprocess.run') as mock_run:
            with patch.object(Path, 'exists', return_value=True):  # mmp-local-dev가 이미 존재
                # When: clone_mmp_local_dev 실행
                clone_mmp_local_dev(parent_dir)
        
        # Then: git 명령어가 실행되지 않아야 함
        mock_run.assert_not_called()

    @pytest.mark.unit
    def test_clone_mmp_local_dev__clone_failure__should_raise_error(self) -> None:
        """
        git clone 실패 시 에러를 발생시키는지 검증.
        
        Given: git clone이 실패하는 상황
        When: clone_mmp_local_dev 실행
        Then: CalledProcessError가 발생해야 함
        """
        # Given: git clone이 실패하는 상황
        from src.cli.commands.init_command import clone_mmp_local_dev
        parent_dir = Path(tempfile.mkdtemp())
        
        # subprocess.run이 실패를 반환
        mock_result = Mock()
        mock_result.returncode = 128  # git 에러 코드
        mock_result.stdout = "Clone output"
        mock_result.stderr = "Clone error"
        
        with patch('subprocess.run', return_value=mock_result):
            with patch.object(Path, 'exists', return_value=False):
                # When & Then: clone_mmp_local_dev 실행 시 CalledProcessError 발생
                with pytest.raises(subprocess.CalledProcessError) as exc_info:
                    clone_mmp_local_dev(parent_dir)
                
                # 에러 정보 확인
                assert exc_info.value.returncode == 128
                assert "git" in str(exc_info.value.cmd)


class TestRealIntegration:
    """실제 파일 시스템을 사용한 통합 테스트"""

    @pytest.mark.unit
    def test_init_integration__real_file_system__should_create_actual_files(self) -> None:
        """
        실제 파일 시스템에서 init이 파일들을 생성하는지 검증.
        
        Given: 실제 임시 디렉토리
        When: create_project_structure 실행 (실제 함수들 호출)
        Then: 실제 파일들이 생성되어야 함
        """
        # Given: 실제 임시 디렉토리
        import tempfile
        import shutil
        
        temp_base = Path(tempfile.mkdtemp())
        project_path = temp_base / "integration_test_project"
        
        try:
            # When: create_project_structure 실행 (모킹 없이)
            from src.cli.commands.init_command import create_project_structure
            
            # 템플릿 디렉토리 존재 확인 후 실행
            templates_dir = Path(__file__).parent.parent.parent / "src/cli/templates/configs"
            if templates_dir.exists():
                create_project_structure(project_path, with_mmp_dev=False)
                
                # Then: 실제 파일들이 생성되어야 함
                assert project_path.exists(), "프로젝트 디렉토리가 생성되지 않았습니다"
                
                # config 파일들 확인
                config_files = ["base.yaml", "local.yaml", "dev.yaml", "prod.yaml"]
                for config_file in config_files:
                    config_path = project_path / "config" / config_file
                    assert config_path.exists(), f"{config_file}이 생성되지 않았습니다"
                    assert config_path.stat().st_size > 0, f"{config_file}이 비어있습니다"
                
                # 샘플 데이터 확인
                data_path = project_path / "data" / "sample_data.csv"
                assert data_path.exists(), "샘플 데이터가 생성되지 않았습니다"
                assert data_path.stat().st_size > 0, "샘플 데이터가 비어있습니다"
                
                # 문서 확인
                docs_path = project_path / "docs" / "README.md"
                assert docs_path.exists(), "README.md가 생성되지 않았습니다"
                assert docs_path.stat().st_size > 0, "README.md가 비어있습니다"
                
                # recipes 디렉토리는 비어있어야 함
                recipes_path = project_path / "recipes"
                assert recipes_path.exists(), "recipes 디렉토리가 생성되지 않았습니다"
                assert len(list(recipes_path.iterdir())) == 0, "recipes 디렉토리가 비어있지 않습니다"
            else:
                pytest.skip("Templates 디렉토리가 존재하지 않아 통합 테스트를 건너뜁니다")
                
        finally:
            # 정리
            shutil.rmtree(temp_base, ignore_errors=True)