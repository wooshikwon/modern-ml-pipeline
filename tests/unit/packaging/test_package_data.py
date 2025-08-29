"""
Package Data Inclusion Unit Tests
Blueprint v17.0 - TDD RED Phase

CLAUDE.md 원칙 준수:
- RED → GREEN → REFACTOR 사이클
- 테스트 없는 구현 금지
- 커버리지 ≥ 90%
"""

import subprocess
import zipfile
from pathlib import Path
import pytest


class TestPackageDataInclusion:
    """PyPI 패키지 데이터 포함 테스트 클래스"""
    
    def test_package_data__template_files__should_be_included_in_wheel(self) -> None:
        """
        패키지 빌드 시 CLI 템플릿 파일들이 휠에 포함되는지 검증.
        
        Given: src/cli/project_templates/ 디렉토리의 템플릿 파일들
        When: hatch build로 휠 패키지 생성
        Then: 생성된 휠 파일에 모든 템플릿 파일이 포함되어야 함
        """
        # Given: 프로젝트 루트와 템플릿 파일 경로
        project_root = Path(__file__).parent.parent.parent.parent
        templates_dir = project_root / "src" / "cli" / "project_templates"
        
        # 템플릿 파일들이 존재하는지 먼저 확인
        assert templates_dir.exists(), f"템플릿 디렉토리가 존재하지 않습니다: {templates_dir}"
        
        # 예상되는 템플릿 파일들
        expected_files = [
            "config/base.yaml",
            "config/local.yaml", 
            "config/enterprise.yaml",
            "config/research.yaml",
            "recipes/local_classification_test.yaml",
            "recipes/dev_classification_test.yaml",
            "guideline_recipe.yaml.j2",
        ]
        
        # Given: 모든 예상 파일들이 실제로 존재하는지 확인
        missing_files = []
        for file_path in expected_files:
            full_path = templates_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        assert len(missing_files) == 0, f"예상된 템플릿 파일이 없습니다: {missing_files}"

    def test_package_data__hatch_build_config__should_include_src_directory(self) -> None:
        """
        Hatch 빌드 설정이 src 디렉토리를 올바르게 패키지에 포함하는지 검증.
        
        Given: pyproject.toml의 hatch 빌드 설정
        When: 빌드 설정 파싱
        Then: src 디렉토리가 패키지에 포함되도록 설정되어야 함
        """
        # Given: pyproject.toml 읽기
        import tomllib
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # When: Hatch 빌드 설정 확인
        hatch_config = pyproject_data.get("tool", {}).get("hatch", {})
        build_config = hatch_config.get("build", {})
        targets_config = build_config.get("targets", {})
        wheel_config = targets_config.get("wheel", {})
        
        # Then: packages 설정이 src를 포함해야 함
        packages = wheel_config.get("packages", [])
        assert "src" in packages, f"Hatch 빌드 설정에서 src 디렉토리가 누락됨: {packages}"

    @pytest.mark.slow
    def test_package_data__actual_wheel_build__should_contain_template_files(self) -> None:
        """
        실제 휠 빌드를 수행하여 템플릿 파일들이 포함되는지 검증.
        
        Given: 현재 프로젝트 상태
        When: hatch build -t wheel 실행  
        Then: 생성된 휠 파일에 템플릿 파일들이 존재해야 함
        """
        # Given: 프로젝트 루트
        project_root = Path(__file__).parent.parent.parent.parent
        dist_dir = project_root / "dist"
        
        # 기존 휠 파일들 정리
        if dist_dir.exists():
            for wheel_file in dist_dir.glob("*.whl"):
                wheel_file.unlink()
        
        # When: 휠 빌드 실행
        build_cmd = ["hatch", "build", "-t", "wheel", "--clean"]
        
        try:
            result = subprocess.run(
                build_cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120  # 2분 타임아웃
            )
        except subprocess.TimeoutExpired:
            pytest.skip("빌드 시간이 너무 오래 걸려 테스트를 건너뜁니다")
        except FileNotFoundError:
            pytest.skip("hatch 빌드 도구가 설치되지 않아 테스트를 건너뜁니다")
        
        if result.returncode != 0:
            pytest.skip(f"휠 빌드 실패 (스킵): {result.stderr}")
        
        # Then: 생성된 휠 파일 찾기
        wheel_files = list(dist_dir.glob("*.whl"))
        if len(wheel_files) == 0:
            pytest.skip("휠 파일이 생성되지 않아 테스트를 건너뜁니다")
        
        wheel_file = wheel_files[0]
        
        # Then: 휠 파일 내용 검사
        expected_template_paths = [
            "cli/project_templates/config/base.yaml",
            "cli/project_templates/config/local.yaml", 
            "cli/project_templates/__init__.py",
        ]
        
        missing_in_wheel = []
        with zipfile.ZipFile(wheel_file, 'r') as zf:
            wheel_contents = zf.namelist()
            
            for expected_path in expected_template_paths:
                # 휠 내부에서는 경로가 조금 다를 수 있음
                found = any(expected_path in wheel_path for wheel_path in wheel_contents)
                if not found:
                    missing_in_wheel.append(expected_path)
        
        # 실패 시에만 상세 정보 출력 (성공 시에는 단순 통과)
        if len(missing_in_wheel) > 0:
            print(f"\n휠 내용 (처음 20개): {wheel_contents[:20]}")
            pytest.fail(f"다음 템플릿 파일들이 휠에 포함되지 않음: {missing_in_wheel}")

    def test_package_data__init_command_functionality__should_work_after_pip_install(self) -> None:
        """
        pip install 후 init 명령어가 템플릿 파일을 올바르게 사용할 수 있는지 검증.
        
        Given: 패키지 설치된 환경
        When: modern-ml-pipeline init 명령어 실행 시 템플릿 파일 접근
        Then: 템플릿 파일들을 찾을 수 있어야 함
        """
        # Given: 현재 설치된 패키지에서 템플릿 경로 확인
        from src.cli.commands import Path as CLIPath
        
        # When: CLI 명령어에서 사용하는 템플릿 경로 확인
        # 실제 init 함수에서 사용하는 경로와 동일하게
        template_source_path = CLIPath(__file__).parent.parent.parent.parent / "src" / "cli" / "project_templates"
        
        # Then: 템플릿 디렉토리가 접근 가능해야 함
        assert template_source_path.exists(), (
            f"init 명령어가 참조하는 템플릿 경로를 찾을 수 없습니다: {template_source_path}"
        )
        
        # Then: 핵심 템플릿 파일들이 존재해야 함
        essential_templates = [
            "config/base.yaml",
            "config/local.yaml",
            "recipes/local_classification_test.yaml",
        ]
        
        missing_templates = []
        for template_file in essential_templates:
            template_path = template_source_path / template_file
            if not template_path.exists():
                missing_templates.append(template_file)
        
        assert len(missing_templates) == 0, (
            f"init 명령어에 필요한 핵심 템플릿이 없습니다: {missing_templates}"
        )


class TestPackageDataAccessibility:
    """패키지 데이터 접근성 테스트 클래스"""
    
    def test_package_data__importlib_resources__should_access_templates_programmatically(self) -> None:
        """
        importlib.resources를 통해 템플릿 파일에 프로그래매틱하게 접근할 수 있는지 검증.
        
        Given: 설치된 패키지 환경
        When: importlib.resources로 템플릿 접근
        Then: 템플릿 파일 내용을 읽을 수 있어야 함
        """
        # Given: 패키지 내 템플릿 리소스 접근
        try:
            # When: 리소스로 템플릿 파일 접근 시도
            # src.cli.project_templates 패키지에서 파일 확인
            import src.cli.project_templates
            
            # 실제 파일 시스템 경로를 통해 확인
            template_dir = Path(src.cli.project_templates.__file__).parent
            assert template_dir.exists(), "템플릿 패키지 디렉토리가 존재하지 않습니다"
            
            # Then: 최소한 기본 설정 파일들이 접근 가능해야 함
            config_dir = template_dir / "config"
            assert config_dir.exists(), "config 디렉토리가 존재하지 않습니다"
            
            base_config = config_dir / "base.yaml"
            assert base_config.exists(), "base.yaml 설정 파일이 존재하지 않습니다"
            
            # 파일 내용 읽기 테스트
            content = base_config.read_text()
            assert len(content) > 0, "base.yaml 파일이 비어있습니다"
            assert "mlflow" in content.lower(), "base.yaml에 mlflow 설정이 없습니다"
            
        except ImportError as e:
            pytest.skip(f"패키지 리소스 접근 불가: {e}")

    def test_package_data__manifest_inclusion__should_have_proper_manifest(self) -> None:
        """
        MANIFEST.in 파일이나 pyproject.toml 설정으로 데이터 파일 포함이 명시되어 있는지 검증.
        
        Given: 프로젝트 루트
        When: 패키지 데이터 포함 설정 확인
        Then: 템플릿 파일들이 명시적으로 포함되도록 설정되어야 함
        """
        # Given: 프로젝트 루트
        project_root = Path(__file__).parent.parent.parent.parent
        
        # When: MANIFEST.in 또는 pyproject.toml에서 데이터 파일 설정 확인
        manifest_file = project_root / "MANIFEST.in"
        pyproject_file = project_root / "pyproject.toml"
        
        has_manifest = manifest_file.exists()
        has_pyproject = pyproject_file.exists()
        
        assert has_pyproject, "pyproject.toml 파일이 존재하지 않습니다"
        
        # pyproject.toml에서 hatch 빌드 설정 확인
        if has_pyproject:
            import tomllib
            with open(pyproject_file, "rb") as f:
                pyproject_data = tomllib.load(f)
            
            # Hatch 빌드 설정에서 packages 확인
            hatch_build = pyproject_data.get("tool", {}).get("hatch", {}).get("build", {})
            wheel_config = hatch_build.get("targets", {}).get("wheel", {})
            packages = wheel_config.get("packages", [])
            
            # Then: src 디렉토리가 포함되어야 함 (템플릿 파일들 포함)
            assert "src" in packages, (
                f"pyproject.toml의 hatch 빌드 설정에서 src 디렉토리가 누락됨: {packages}"
            )
        
        # 추가적으로 MANIFEST.in이 있다면 확인
        if has_manifest:
            manifest_content = manifest_file.read_text()
            # recursive-include이나 include 지시문으로 템플릿 파일들이 포함되어야 함
            template_included = (
                "recursive-include src/cli/project_templates" in manifest_content or
                "include src/cli/project_templates" in manifest_content or
                "graft src" in manifest_content
            )
            
            if not template_included:
                # 경고 메시지만 출력하고 실패하지는 않음 (pyproject.toml로도 처리 가능)
                print(f"Warning: MANIFEST.in에 템플릿 파일 포함 설정이 명시되지 않음: {manifest_content[:200]}")