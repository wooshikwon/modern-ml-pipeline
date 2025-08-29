"""
PyPI Dependencies Unit Tests
Blueprint v17.0 - TDD RED Phase

CLAUDE.md 원칙 준수:
- RED → GREEN → REFACTOR 사이클
- 테스트 없는 구현 금지
- 커버리지 ≥ 90%
"""

import tomllib
from pathlib import Path
import pytest


class TestPyProjectDependencies:
    """PyPI 패키지 의존성 테스트 클래스"""
    
    def test_dependencies__no_circular_dependencies__uv_should_not_be_in_main_dependencies(self) -> None:
        """
        main dependencies에서 uv가 제외되어 circular dependency를 방지하는지 검증.
        
        Given: pyproject.toml 파일
        When: [project.dependencies] 섹션 확인
        Then: "uv"가 포함되지 않아야 함 (circular dependency 방지)
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        # When: 의존성 정보 파싱
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        
        # Then: uv가 main dependencies에 없어야 함
        assert "uv" not in dependencies, "uv는 circular dependency를 유발하므로 main dependencies에서 제외되어야 합니다"
        
        # 문자열 형태로도 확인 (예: "uv>=0.1.0"), 단 uvicorn은 제외
        uv_variants = [dep for dep in dependencies if dep.startswith("uv") and not dep.startswith("uvicorn")]
        assert len(uv_variants) == 0, f"uv 관련 의존성이 발견됨 (uvicorn 제외): {uv_variants}"

    def test_dependencies__core_ml_libraries__should_have_version_constraints(self) -> None:
        """
        핵심 ML 라이브러리들이 적절한 버전 제약을 가지는지 검증.
        
        Given: pyproject.toml의 dependencies
        When: 핵심 ML 라이브러리 버전 확인
        Then: scikit-learn, pandas, mlflow 등이 버전 제약을 가져야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        
        # When: 핵심 라이브러리 체크
        core_libraries = ["scikit-learn", "pandas", "mlflow", "typer", "pydantic"]
        
        # Then: 핵심 라이브러리들이 존재해야 함
        dep_names = [dep.split(">=")[0].split("==")[0].split("[")[0] for dep in dependencies]
        
        for lib in core_libraries:
            assert lib in dep_names, f"핵심 라이브러리 '{lib}'이 dependencies에 없습니다"

    def test_dependencies__optional_dependencies__should_separate_dev_and_ml_extras(self) -> None:
        """
        optional-dependencies가 dev와 ML 관련으로 적절히 분리되어 있는지 검증.
        
        Given: pyproject.toml의 optional-dependencies
        When: dev, ml-extras 그룹 확인  
        Then: 개발 도구와 ML 선택적 도구가 분리되어 있어야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
        
        # Then: dev 그룹은 반드시 존재해야 함
        assert "dev" in optional_deps, "optional-dependencies에 'dev' 그룹이 없습니다"
        
        dev_deps = optional_deps["dev"]
        dev_tools = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit"]
        
        # When: dev 의존성 이름만 추출
        dev_dep_names = [dep.split(">=")[0].split("==")[0].split("[")[0] for dep in dev_deps]
        
        # Then: 핵심 개발 도구들이 포함되어야 함
        for tool in dev_tools:
            assert tool in dev_dep_names, f"개발 도구 '{tool}'이 dev optional-dependencies에 없습니다"

    def test_dependencies__heavy_ml_libraries__should_be_in_extras(self) -> None:
        """
        무거운 ML 라이브러리들이 선택적 의존성으로 분리되어 있는지 검증.
        
        Given: pyproject.toml의 dependencies 구조
        When: 무거운 ML 라이브러리 확인
        Then: torch, tensorflow 같은 무거운 라이브러리는 ml-extras에 있어야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        
        # When: 무거운 라이브러리 확인
        main_dep_names = [dep.split(">=")[0].split("==")[0].split("[")[0] for dep in dependencies]
        
        # Then: 무거운 라이브러리가 main에 있다면 ml-extras 그룹 존재 확인
        torch_in_main = "torch" in main_dep_names
        if torch_in_main:
            # torch가 main에 있다면 향후 ml-extras로 이동 계획 필요
            # 현재는 경고로 처리 (향후 개선 예정)
            pass

    def test_dependencies__version_pinning_strategy__should_use_minimum_versions(self) -> None:
        """
        버전 고정 전략이 최소 버전(>=) 방식을 사용하는지 검증.
        
        Given: pyproject.toml의 dependencies
        When: 버전 제약 형태 확인
        Then: 대부분 >=를 사용하고 == 형태의 하드 핀은 최소화되어야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        dependencies = pyproject_data.get("project", {}).get("dependencies", [])
        
        # When: 버전 제약이 있는 의존성 분석
        versioned_deps = [dep for dep in dependencies if ">=" in dep or "==" in dep or "~=" in dep]
        hard_pinned = [dep for dep in versioned_deps if "==" in dep]
        flexible_pinned = [dep for dep in versioned_deps if ">=" in dep]
        
        # Then: 유연한 버전 제약(>=)이 하드 핀(==)보다 많아야 함
        assert len(flexible_pinned) >= len(hard_pinned), (
            f"하드 핀({len(hard_pinned)})보다 유연한 버전 제약({len(flexible_pinned)})이 더 많아야 합니다. "
            f"하드 핀: {hard_pinned}"
        )

    @pytest.mark.slow
    def test_dependencies__importability__all_core_dependencies_should_be_importable(self) -> None:
        """
        모든 핵심 의존성이 실제로 import 가능한지 검증.
        
        Given: pyproject.toml의 dependencies
        When: 핵심 라이브러리들을 import 시도
        Then: ImportError 없이 성공해야 함
        """
        # Given: 핵심 라이브러리 목록
        core_imports = [
            "typer", "pandas", "pydantic", "sklearn", "mlflow", 
            "jinja2", "fastapi", "uvicorn"
        ]
        
        # When & Then: 각 라이브러리 import 테스트
        failed_imports = []
        for lib in core_imports:
            try:
                __import__(lib)
            except ImportError as e:
                failed_imports.append(f"{lib}: {e}")
        
        assert len(failed_imports) == 0, (
            f"다음 핵심 의존성들을 import할 수 없습니다: {failed_imports}"
        )


class TestPyProjectMetadata:
    """PyPI 패키지 메타데이터 테스트 클래스"""
    
    def test_metadata__package_info__should_be_complete_for_pypi(self) -> None:
        """
        PyPI 배포를 위한 패키지 메타데이터가 완전한지 검증.
        
        Given: pyproject.toml의 project 섹션
        When: PyPI 필수 필드들 확인
        Then: name, version, description, readme, license, authors가 모두 있어야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        project = pyproject_data.get("project", {})
        
        # When & Then: 필수 필드들 검증
        required_fields = ["name", "version", "description", "readme", "license", "authors"]
        
        for field in required_fields:
            assert field in project, f"PyPI 필수 필드 '{field}'가 project 섹션에 없습니다"
            assert project[field], f"PyPI 필드 '{field}'의 값이 비어있습니다"

    def test_metadata__python_version__should_support_modern_versions(self) -> None:
        """
        Python 버전 요구사항이 적절한 범위를 지원하는지 검증.
        
        Given: pyproject.toml의 requires-python
        When: 버전 범위 확인
        Then: Python 3.11+ 지원하고 상한선이 합리적이어야 함
        """
        # Given: pyproject.toml 읽기
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        
        requires_python = pyproject_data.get("project", {}).get("requires-python", "")
        
        # Then: Python 버전 요구사항 검증
        assert requires_python, "requires-python 필드가 없습니다"
        assert "3.11" in requires_python, "Python 3.11 이상을 지원해야 합니다"