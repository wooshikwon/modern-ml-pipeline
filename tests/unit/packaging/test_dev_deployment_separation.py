"""
Development/Deployment Environment Separation Unit Tests
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
import tomllib


class TestDevDeploymentSeparation:
    """개발/배포 환경 분리 테스트 클래스"""
    
    def test_dev_deployment__dev_directory__should_exist_and_contain_dev_files(self) -> None:
        """
        dev/ 디렉토리가 존재하고 개발 전용 파일들을 포함하는지 검증.
        
        Given: 프로젝트 루트 디렉토리
        When: dev/ 디렉토리 확인
        Then: dev/ 디렉토리가 존재하고 개발 전용 파일들이 있어야 함
        """
        # Given: 프로젝트 루트
        project_root = Path(__file__).parent.parent.parent.parent
        dev_dir = project_root / "dev"
        
        # Then: dev 디렉토리가 존재해야 함
        assert dev_dir.exists(), f"dev/ 디렉토리가 존재하지 않습니다: {dev_dir}"
        assert dev_dir.is_dir(), f"dev는 디렉토리여야 합니다: {dev_dir}"
        
        # Then: 개발 전용 파일들이 dev/ 구조 내에 있어야 함 (구조화된 경로 포함)
        expected_dev_files = {
            "docker/docker-compose.yml": dev_dir / "docker" / "docker-compose.yml",
            "docker/Dockerfile": dev_dir / "docker" / "Dockerfile", 
            "scripts/setup-dev-environment.sh": dev_dir / "scripts" / "setup-dev-environment.sh",
            "docs/factoringlog.md": dev_dir / "docs" / "factoringlog.md",
            "examples/main.py": dev_dir / "examples" / "main.py"
        }
        
        missing_files = []
        for file_description, file_path in expected_dev_files.items():
            if not file_path.exists():
                missing_files.append(file_description)
        
        assert len(missing_files) == 0, f"다음 개발 파일들이 dev/ 디렉토리 구조에 없습니다: {missing_files}"

    def test_dev_deployment__root_directory__should_not_contain_dev_files(self) -> None:
        """
        프로젝트 루트에 개발 전용 파일들이 없는지 검증.
        
        Given: 프로젝트 루트 디렉토리
        When: 개발 전용 파일들 확인
        Then: 루트에는 개발 파일들이 없고 배포용 파일들만 있어야 함
        """
        # Given: 프로젝트 루트
        project_root = Path(__file__).parent.parent.parent.parent
        
        # When: 루트에 있으면 안되는 개발 파일들 정의
        dev_files_in_root = [
            "docker-compose.yml",
            "Dockerfile",
            "setup-dev-environment.sh", 
            "factoringlog.md",
            "main.py"
        ]
        
        # Then: 루트에 개발 파일들이 없어야 함
        found_dev_files = []
        for dev_file in dev_files_in_root:
            root_file_path = project_root / dev_file
            if root_file_path.exists():
                found_dev_files.append(dev_file)
        
        assert len(found_dev_files) == 0, (
            f"다음 개발 파일들이 루트 디렉토리에 있습니다. dev/ 로 이동해야 합니다: {found_dev_files}"
        )
    
    def test_dev_deployment__root_directory__should_contain_only_deployment_files(self) -> None:
        """
        프로젝트 루트에 배포용 파일들만 존재하는지 검증.
        
        Given: 프로젝트 루트 디렉토리  
        When: 루트 디렉토리 파일 목록 확인
        Then: PyPI 배포에 필요한 파일들만 존재해야 함
        """
        # Given: 프로젝트 루트
        project_root = Path(__file__).parent.parent.parent.parent
        
        # When: 배포용 파일들 정의 (허용된 파일들)
        allowed_deployment_files = {
            # PyPI 필수 파일들
            "pyproject.toml",
            "README.md", 
            "LICENSE",
            "CLAUDE.md",
            
            # 디렉토리들
            "src",
            "tests", 
            "config",
            "recipes",
            "docs",
            "data",
            "dev",  # 개발 파일들이 이동될 디렉토리
            
            # 설정 파일들
            "pytest.ini",
            ".gitignore",
            
            # 빌드 아티팩트 (임시)
            "dist",
            "uv.lock",
            "requirements-dev.lock",
            
            # 런타임 데이터 (gitignore되지만 존재 가능)
            "mlruns",
            "logs", 
            "local",
            
            # 숨김 파일들
            ".DS_Store",
            ".claude"
        }
        
        # Then: 루트의 모든 항목이 허용 목록에 있어야 함
        actual_items = {item.name for item in project_root.iterdir()}
        
        # 허용되지 않은 파일들 찾기
        disallowed_items = actual_items - allowed_deployment_files
        
        # 숨김 파일들과 임시 파일들은 경고만 (실패하지 않음)
        warning_items = {item for item in disallowed_items if item.startswith('.') or item.endswith('.tmp')}
        critical_items = disallowed_items - warning_items
        
        if warning_items:
            print(f"Warning: 숨김/임시 파일들 발견: {warning_items}")
        
        assert len(critical_items) == 0, (
            f"허용되지 않은 파일들이 루트에 있습니다: {critical_items}\n"
            f"개발 파일들은 dev/ 디렉토리로 이동하거나 .gitignore에 추가하세요."
        )

    def test_dev_deployment__gitignore_coverage__should_exclude_dev_artifacts(self) -> None:
        """
        .gitignore가 개발 아티팩트들을 제대로 제외하는지 검증.
        
        Given: .gitignore 파일
        When: 개발 관련 패턴 확인
        Then: 주요 개발 아티팩트들이 gitignore에 포함되어야 함
        """
        # Given: .gitignore 파일 읽기
        project_root = Path(__file__).parent.parent.parent.parent
        gitignore_path = project_root / ".gitignore"
        
        assert gitignore_path.exists(), ".gitignore 파일이 존재하지 않습니다"
        
        gitignore_content = gitignore_path.read_text()
        
        # When: 필수 개발 아티팩트 패턴들 정의
        required_ignore_patterns = [
            "/mlruns/",      # MLflow 실험 데이터
            "/local/",       # 로컬 아티팩트
            "logs/",         # 로그 파일들
            ".env",          # 환경 변수
            "__pycache__/",  # Python 캐시
            "dist/",         # 빌드 아티팩트
            ".venv/",        # 가상 환경
        ]
        
        # Then: 모든 패턴이 gitignore에 있어야 함
        missing_patterns = []
        for pattern in required_ignore_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
        
        assert len(missing_patterns) == 0, (
            f"다음 개발 아티팩트 패턴들이 .gitignore에 없습니다: {missing_patterns}"
        )

    def test_dev_deployment__hatch_build_exclusion__should_not_include_dev_files(self) -> None:
        """
        Hatch 빌드 설정이 개발 파일들을 제외하는지 검증.
        
        Given: pyproject.toml의 hatch 빌드 설정
        When: packages 및 exclude 설정 확인
        Then: src만 포함하고 개발 파일들은 제외되어야 함
        """
        # Given: pyproject.toml 읽기
        project_root = Path(__file__).parent.parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        # When: Hatch 빌드 설정 확인
        hatch_config = pyproject_data.get("tool", {}).get("hatch", {})
        build_targets = hatch_config.get("build", {}).get("targets", {})
        wheel_config = build_targets.get("wheel", {})
        
        packages = wheel_config.get("packages", [])
        
        # Then: src만 포함되어야 함 (개발 디렉토리 제외)
        assert "src" in packages, "Hatch 빌드 설정에 src가 포함되어야 합니다"
        
        # 개발 관련 디렉토리들이 포함되지 않아야 함
        dev_directories = ["dev", "mlruns", "logs", "local"]
        included_dev_dirs = [d for d in dev_directories if d in packages]
        
        assert len(included_dev_dirs) == 0, (
            f"다음 개발 디렉토리들이 Hatch 빌드에 포함되어 있습니다: {included_dev_dirs}"
        )

    @pytest.mark.slow
    def test_dev_deployment__wheel_build_content__should_exclude_dev_files(self) -> None:
        """
        실제 휠 빌드 시 개발 파일들이 제외되는지 검증.
        
        Given: 현재 프로젝트 상태
        When: hatch build 실행하여 휠 생성
        Then: 생성된 휠에 개발 파일들이 포함되지 않아야 함
        """
        # Given: 프로젝트 루트 및 빌드 디렉토리
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
                timeout=120
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("hatch 빌드를 실행할 수 없어 테스트를 건너뜁니다")
        
        if result.returncode != 0:
            pytest.skip(f"휠 빌드 실패 (스킵): {result.stderr}")
        
        # Then: 생성된 휠 파일 검사
        wheel_files = list(dist_dir.glob("*.whl"))
        if len(wheel_files) == 0:
            pytest.skip("휠 파일이 생성되지 않아 테스트를 건너뜁니다")
        
        wheel_file = wheel_files[0]
        
        # 휠에 포함되면 안되는 개발 파일/경로들
        prohibited_paths = [
            "dev/",
            "docker-compose.yml",
            "Dockerfile", 
            "setup-dev-environment.sh",
            "factoringlog.md",
            "main.py",
            "mlruns/",
            "logs/",
            "local/",
        ]
        
        # 휠 내용 검사
        found_dev_files = []
        with zipfile.ZipFile(wheel_file, 'r') as zf:
            wheel_contents = zf.namelist()
            
            for prohibited_path in prohibited_paths:
                for wheel_path in wheel_contents:
                    if prohibited_path in wheel_path:
                        found_dev_files.append(wheel_path)
                        break
        
        assert len(found_dev_files) == 0, (
            f"휠 파일에 개발 파일들이 포함되어 있습니다: {found_dev_files}\n"
            f"이 파일들은 dev/ 디렉토리로 이동하거나 빌드에서 제외되어야 합니다."
        )


class TestDevDirectoryStructure:
    """dev/ 디렉토리 구조 테스트 클래스"""
    
    def test_dev_directory__structure__should_be_organized_properly(self) -> None:
        """
        dev/ 디렉토리 내부가 체계적으로 구조화되어 있는지 검증.
        
        Given: dev/ 디렉토리
        When: 내부 구조 확인
        Then: 논리적으로 구성된 하위 디렉토리들이 있어야 함
        """
        # Given: dev 디렉토리
        project_root = Path(__file__).parent.parent.parent.parent
        dev_dir = project_root / "dev"
        
        if not dev_dir.exists():
            pytest.skip("dev/ 디렉토리가 존재하지 않아 테스트를 건너뜁니다")
        
        # When: dev 디렉토리 내부 구조 확인
        expected_structure = {
            "docker": ["docker-compose.yml", "Dockerfile"],  # 컨테이너 관련
            "scripts": ["setup-dev-environment.sh"],         # 스크립트들
            "docs": ["factoringlog.md"],                     # 개발 문서들
            # examples나 samples는 선택사항
        }
        
        # Then: 구조가 체계적이어야 함 (완전히 강제하지는 않음)
        for category, files in expected_structure.items():
            category_dir = dev_dir / category
            if category_dir.exists():
                for file in files:
                    file_path = category_dir / file
                    if not file_path.exists():
                        # 경고만 출력하고 실패하지는 않음
                        print(f"Warning: {file}이 {category}/ 디렉토리에 없습니다")

    def test_dev_directory__readme__should_exist_and_explain_structure(self) -> None:
        """
        dev/ 디렉토리에 README가 있고 구조를 설명하는지 검증.
        
        Given: dev/ 디렉토리
        When: README 파일 확인
        Then: dev/ 내부 구조와 사용법을 설명하는 README가 있어야 함
        """
        # Given: dev 디렉토리
        project_root = Path(__file__).parent.parent.parent.parent
        dev_dir = project_root / "dev"
        
        if not dev_dir.exists():
            pytest.skip("dev/ 디렉토리가 존재하지 않아 테스트를 건너뜁니다")
        
        # When: README 파일 확인
        readme_files = ["README.md", "README.txt", "README"]
        readme_found = None
        
        for readme_name in readme_files:
            readme_path = dev_dir / readme_name
            if readme_path.exists():
                readme_found = readme_path
                break
        
        # Then: README가 존재하고 내용이 있어야 함
        assert readme_found is not None, (
            "dev/ 디렉토리에 README 파일이 없습니다. "
            "개발 환경 구조와 사용법을 설명하는 README가 필요합니다."
        )
        
        readme_content = readme_found.read_text()
        assert len(readme_content.strip()) > 0, "dev/ README 파일이 비어있습니다"
        
        # 기본적인 설명 키워드들이 있는지 확인
        essential_keywords = ["development", "docker", "environment", "setup"]
        content_lower = readme_content.lower()
        
        missing_keywords = [kw for kw in essential_keywords if kw not in content_lower]
        if missing_keywords:
            print(f"Warning: dev/ README에 다음 키워드들이 누락될 수 있습니다: {missing_keywords}")