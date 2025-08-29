"""
Environment Health Check Implementation
Blueprint v17.0 - Environment validation and compatibility checks

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 예외 처리 및 로깅
"""

import sys
import subprocess
import pkg_resources

from src.health.models import CheckResult, CheckCategory, HealthCheckError


class EnvironmentHealthCheck:
    """
    환경 관련 건강 검사를 수행하는 클래스.
    
    Python 버전, 핵심 의존성, 템플릿 접근성 등을 검증합니다.
    """
    
    def __init__(self) -> None:
        """EnvironmentHealthCheck 인스턴스를 초기화합니다."""
        self.category = CheckCategory.ENVIRONMENT
        
    def check_python_version(self) -> CheckResult:
        """
        Python 버전이 지원 범위 내에 있는지 확인합니다.
        
        Returns:
            CheckResult: Python 버전 검사 결과
        """
        try:
            current_version = sys.version_info
            major, minor = current_version.major, current_version.minor
            
            # pyproject.toml에서 정의된 지원 버전: >=3.11,<3.12
            is_supported = major == 3 and minor == 11
            
            version_str = f"{major}.{minor}.{current_version.micro}"
            
            if is_supported:
                return CheckResult(
                    is_healthy=True,
                    message=f"Python {version_str} (지원됨)",
                    details=[
                        f"현재 버전: {version_str}",
                        "요구사항: >=3.11,<3.12",
                        "✅ 버전 호환성 확인됨"
                    ]
                )
            else:
                recommendations = [
                    "Python 3.11.x 버전으로 업그레이드하세요",
                    "uv를 사용한 설치: uv python install 3.11",
                    "pyenv를 사용한 설치: pyenv install 3.11.10"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message=f"Python {version_str} (지원되지 않음)",
                    details=[
                        f"현재 버전: {version_str}",
                        "요구사항: >=3.11,<3.12",
                        "❌ 버전 호환성 문제"
                    ],
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"Python 버전 확인 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_core_dependencies(self) -> CheckResult:
        """
        핵심 의존성 패키지들이 설치되어 있는지 확인합니다.
        
        Returns:
            CheckResult: 핵심 의존성 검사 결과
        """
        core_packages = [
            'typer', 'pydantic', 'pyyaml', 'jinja2', 'python-dotenv',
            'pandas', 'scikit-learn', 'pyarrow', 'mlflow', 
            'fastapi', 'uvicorn', 'httpx'
        ]
        
        try:
            installed_packages = {}
            missing_packages = []
            details = []
            
            for package in core_packages:
                try:
                    # importlib.metadata 사용 (Python 3.8+)
                    import importlib.metadata
                    version = importlib.metadata.version(package)
                    installed_packages[package] = version
                    details.append(f"✅ {package}: {version}")
                except (importlib.metadata.PackageNotFoundError, ImportError):
                    # fallback to pkg_resources
                    try:
                        dist = pkg_resources.get_distribution(package)
                        installed_packages[package] = dist.version
                        details.append(f"✅ {package}: {dist.version}")
                    except pkg_resources.DistributionNotFound:
                        missing_packages.append(package)
                        details.append(f"❌ {package}: 설치되지 않음")
            
            if not missing_packages:
                return CheckResult(
                    is_healthy=True,
                    message=f"모든 핵심 패키지 설치됨 ({len(installed_packages)}/{len(core_packages)})",
                    details=details
                )
            else:
                recommendations = [
                    "누락된 패키지들을 설치하세요:",
                    "uv sync",  # 전체 의존성 동기화
                    f"또는 개별 설치: uv add {' '.join(missing_packages)}"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message=f"누락된 패키지: {len(missing_packages)}개",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"의존성 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_template_accessibility(self) -> CheckResult:
        """
        설정 템플릿 파일들에 접근 가능한지 확인합니다.
        
        Returns:
            CheckResult: 템플릿 접근성 검사 결과
        """
        try:
            # CLI 명령어에서 사용하는 템플릿 디렉토리 경로 확인
            from src.cli.commands import _get_templates_directory
            
            templates_dir = _get_templates_directory()
            
            # 필수 템플릿 파일들 확인
            required_templates = {
                'config/base.yaml': templates_dir / 'config' / 'base.yaml',
                'config/enterprise.yaml': templates_dir / 'config' / 'enterprise.yaml',
                'config/local.yaml': templates_dir / 'config' / 'local.yaml',
                'config/research.yaml': templates_dir / 'config' / 'research.yaml',
                'guideline_recipe.yaml.j2': templates_dir / 'guideline_recipe.yaml.j2'
            }
            
            details = []
            missing_templates = []
            accessible_templates = []
            
            for template_name, template_path in required_templates.items():
                if template_path.exists() and template_path.is_file():
                    # 읽기 권한 확인
                    try:
                        template_path.read_text(encoding='utf-8')
                        accessible_templates.append(template_name)
                        details.append(f"✅ {template_name}")
                    except Exception as read_error:
                        missing_templates.append(template_name)
                        details.append(f"❌ {template_name} (읽기 실패: {read_error})")
                else:
                    missing_templates.append(template_name)
                    details.append(f"❌ {template_name} (파일 없음)")
            
            if not missing_templates:
                return CheckResult(
                    is_healthy=True,
                    message=f"모든 템플릿 접근 가능 ({len(accessible_templates)}/{len(required_templates)})",
                    details=details + [f"템플릿 디렉토리: {templates_dir}"]
                )
            else:
                recommendations = [
                    "패키지 재설치를 시도하세요:",
                    "uv sync --reinstall",
                    "또는 개발 모드 설치: uv pip install -e .",
                    f"템플릿 디렉토리 확인: {templates_dir}"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message=f"접근 불가능한 템플릿: {len(missing_templates)}개",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"템플릿 접근성 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_uv_availability(self) -> CheckResult:
        """
        uv 패키지 매니저의 가용성을 확인합니다.
        
        Returns:
            CheckResult: uv 가용성 검사 결과
        """
        try:
            result = subprocess.run(
                ['uv', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_output = result.stdout.strip()
                return CheckResult(
                    is_healthy=True,
                    message="uv 패키지 매니저 사용 가능",
                    details=[
                        f"버전: {version_output}",
                        "✅ 의존성 관리 도구 준비됨"
                    ]
                )
            else:
                recommendations = [
                    "uv 패키지 매니저를 설치하세요:",
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    "또는 pip install uv"
                ]
                
                return CheckResult(
                    is_healthy=False,
                    message="uv 패키지 매니저를 찾을 수 없음",
                    details=[
                        f"오류: {result.stderr}",
                        "❌ 권장 패키지 매니저 없음"
                    ],
                    recommendations=recommendations
                )
                
        except subprocess.TimeoutExpired:
            return CheckResult(
                is_healthy=False,
                message="uv 명령어 실행 시간 초과",
                details=["❌ uv 응답 없음 (10초 초과)"],
                recommendations=["uv 재설치를 고려하세요"]
            )
        except FileNotFoundError:
            recommendations = [
                "uv 패키지 매니저를 설치하세요:",
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                "또는 pip install uv",
                "설치 후 PATH 환경변수 확인"
            ]
            
            return CheckResult(
                is_healthy=False,
                message="uv 패키지 매니저가 설치되지 않음",
                details=["❌ 명령어를 찾을 수 없음"],
                recommendations=recommendations
            )
        except Exception as e:
            raise HealthCheckError(
                message=f"uv 가용성 검사 실패: {e}",
                category=self.category,
                original_error=e
            )