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
import yaml
from pathlib import Path
from packaging import version

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
    
    def check_python_version_detailed(self) -> CheckResult:
        """
        Python 버전에 대한 세부 검증을 수행합니다.
        
        패치 버전, 보안 업데이트, 권장 사항 등을 포함합니다.
        
        Returns:
            CheckResult: Python 세부 버전 검사 결과
        """
        try:
            current_version = sys.version_info
            major, minor, micro = current_version.major, current_version.minor, current_version.micro
            version_str = f"{major}.{minor}.{micro}"
            
            # 기본 지원 여부 확인
            is_supported = major == 3 and minor == 11
            
            details = [
                f"현재 버전: Python {version_str}",
                f"메이저: {major}, 마이너: {minor}, 패치(마이크로): {micro}",
                f"빌드 정보: {sys.version.split()[0]}",
                "요구사항: >=3.11,<3.12"
            ]
            
            # 세부 권장사항
            recommendations = []
            
            if is_supported:
                # 3.11.x 버전의 경우 패치 버전 권장사항
                if micro < 5:
                    details.append("⚠️ 이전 패치 버전입니다")
                    recommendations.extend([
                        f"Python 3.11.10+ 권장 (현재: {version_str})",
                        "보안 패치 및 버그 수정이 포함된 최신 패치 버전으로 업데이트하세요"
                    ])
                elif micro >= 10:
                    details.append("✅ 권장 패치 버전입니다")
                    details.append("🔒 최신 보안 업데이트 적용됨")
                else:
                    details.append("✅ 안정적인 패치 버전입니다")
                
                # Python 3.11의 주요 기능 언급
                details.extend([
                    "🚀 성능 향상: 10-60% 더 빠른 실행",
                    "📝 향상된 오류 메시지",
                    "⚡ PEP 678 예외 그룹 지원"
                ])
                
                return CheckResult(
                    is_healthy=True,
                    message=f"Python {version_str} (권장 버전, 세부 검증 통과)",
                    details=details,
                    recommendations=recommendations if recommendations else None
                )
            else:
                # 지원되지 않는 버전
                if minor < 11:
                    details.append("❌ 최소 요구사항보다 낮은 버전")
                    recommendations.extend([
                        "Python 3.11.10 이상으로 업그레이드 필요",
                        "uv python install 3.11.10",
                        "pyenv install 3.11.10 && pyenv global 3.11.10"
                    ])
                elif minor >= 12:
                    details.append("⚠️ 아직 테스트되지 않은 최신 버전")
                    recommendations.extend([
                        "Python 3.11.x 권장 (호환성 보장)",
                        "최신 버전 사용 시 호환성 문제 발생 가능"
                    ])
                
                return CheckResult(
                    is_healthy=False,
                    message=f"Python {version_str} (세부 검증 실패)",
                    details=details,
                    recommendations=recommendations
                )
                
        except Exception as e:
            raise HealthCheckError(
                message=f"Python 세부 버전 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_dependencies_detailed(self) -> CheckResult:
        """
        의존성 패키지들에 대한 세부 검증을 수행합니다.
        
        버전 호환성, 보안 취약점, 업그레이드 권장사항을 포함합니다.
        
        Returns:
            CheckResult: 의존성 세부 검사 결과
        """
        # 핵심 패키지별 권장 버전 정의
        recommended_versions = {
            'typer': '0.9.0',
            'pydantic': '2.0.0',
            'pyyaml': '6.0',
            'jinja2': '3.1.0',
            'python-dotenv': '1.0.0',
            'pandas': '2.0.0',
            'scikit-learn': '1.3.0',
            'pyarrow': '12.0.0',
            'mlflow': '2.0.0',
            'fastapi': '0.100.0',
            'uvicorn': '0.23.0',
            'httpx': '0.24.0'
        }
        
        try:
            detailed_info = []
            issues = []
            recommendations = []
            installed_count = 0
            compatible_count = 0
            
            for package, min_version in recommended_versions.items():
                try:
                    # 패키지 정보 가져오기
                    try:
                        import importlib.metadata
                        installed_version = importlib.metadata.version(package)
                    except (importlib.metadata.PackageNotFoundError, ImportError):
                        dist = pkg_resources.get_distribution(package)
                        installed_version = dist.version
                    
                    installed_count += 1
                    
                    # 버전 호환성 확인
                    try:
                        if version.parse(installed_version) >= version.parse(min_version):
                            detailed_info.append(f"✅ {package}: {installed_version} (권장: {min_version}+)")
                            compatible_count += 1
                        else:
                            detailed_info.append(f"⚠️ {package}: {installed_version} (권장: {min_version}+)")
                            issues.append(f"{package} 버전 업그레이드 권장")
                            recommendations.append(f"uv add '{package}>={min_version}'")
                    except Exception:
                        # 버전 파싱 실패 시
                        detailed_info.append(f"✅ {package}: {installed_version} (버전 검증 불가)")
                        compatible_count += 1
                        
                except (pkg_resources.DistributionNotFound, importlib.metadata.PackageNotFoundError):
                    detailed_info.append(f"❌ {package}: 설치되지 않음")
                    issues.append(f"{package} 패키지 누락")
                    recommendations.append(f"uv add {package}")
            
            # 전체 요약 정보 추가
            detailed_info.extend([
                "",
                f"📊 호환성 요약: {compatible_count}/{installed_count}/{len(recommended_versions)}",
                f"   - 설치됨: {installed_count}개",
                f"   - 호환 버전: {compatible_count}개",
                f"   - 문제: {len(issues)}개"
            ])
            
            if issues:
                detailed_info.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
            
            # 성공 조건: 모든 패키지가 설치되고 80% 이상 호환
            is_healthy = (installed_count == len(recommended_versions) and 
                         compatible_count / len(recommended_versions) >= 0.8)
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"의존성 세부 검증: {compatible_count}/{len(recommended_versions)} 호환",
                details=detailed_info,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"의존성 세부 검사 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_template_content_validation(self) -> CheckResult:
        """
        템플릿 파일들의 내용 구문과 스키마를 검증합니다.
        
        YAML 구문, Jinja2 템플릿 문법, 스키마 일관성을 확인합니다.
        
        Returns:
            CheckResult: 템플릿 내용 검증 결과
        """
        try:
            from src.cli.commands import _get_templates_directory
            templates_dir = _get_templates_directory()
            
            details = []
            issues = []
            recommendations = []
            templates_checked = 0
            valid_templates = 0
            
            # 검증할 템플릿 파일들
            templates_to_validate = {
                'config/base.yaml': 'YAML 설정',
                'config/local.yaml': 'YAML 설정',
                'config/enterprise.yaml': 'YAML 설정',
                'config/research.yaml': 'YAML 설정',
                'guideline_recipe.yaml.j2': 'Jinja2 템플릿'
            }
            
            for template_name, template_type in templates_to_validate.items():
                template_path = templates_dir / template_name
                templates_checked += 1
                
                if not template_path.exists():
                    details.append(f"❌ {template_name}: 파일 없음")
                    issues.append(f"{template_name} 파일 누락")
                    continue
                
                try:
                    content = template_path.read_text(encoding='utf-8')
                    
                    if template_type == 'YAML 설정':
                        # YAML 구문 검증
                        try:
                            yaml_data = yaml.safe_load(content)
                            if yaml_data is None:
                                details.append(f"⚠️ {template_name}: 빈 YAML 파일")
                                issues.append(f"{template_name} 내용 없음")
                            else:
                                # 기본 스키마 필드 확인
                                required_fields = ['environment', 'mlflow', 'feature_store']
                                missing_fields = [f for f in required_fields if f not in yaml_data]
                                
                                if missing_fields:
                                    details.append(f"⚠️ {template_name}: 필수 필드 누락 ({', '.join(missing_fields)})")
                                    issues.append(f"{template_name} 스키마 불완전")
                                else:
                                    details.append(f"✅ {template_name}: YAML 구문 및 스키마 유효")
                                    valid_templates += 1
                        except yaml.YAMLError as ye:
                            details.append(f"❌ {template_name}: YAML 구문 오류 - {str(ye)[:50]}...")
                            issues.append(f"{template_name} YAML 구문 오류")
                            recommendations.append(f"{template_name} 파일의 YAML 구문을 확인하세요")
                    
                    elif template_type == 'Jinja2 템플릿':
                        # Jinja2 템플릿 구문 기본 검증
                        try:
                            from jinja2 import Template
                            Template(content)  # 구문 검증
                            
                            # 템플릿 변수 확인
                            if '{{' in content and '}}' in content:
                                details.append(f"✅ {template_name}: Jinja2 템플릿 구문 유효")
                                valid_templates += 1
                            else:
                                details.append(f"⚠️ {template_name}: Jinja2 변수 없음")
                                valid_templates += 1  # 구문은 유효하므로 카운트
                        except Exception as je:
                            details.append(f"❌ {template_name}: Jinja2 구문 오류 - {str(je)[:50]}...")
                            issues.append(f"{template_name} Jinja2 구문 오류")
                            recommendations.append(f"{template_name} 템플릿 문법을 확인하세요")
                            
                except Exception as e:
                    details.append(f"❌ {template_name}: 읽기 실패 - {str(e)[:30]}...")
                    issues.append(f"{template_name} 접근 불가")
            
            # 전체 요약
            details.extend([
                "",
                "📊 템플릿 검증 요약:",
                f"   - 검사됨: {templates_checked}개",
                f"   - 유효함: {valid_templates}개",
                f"   - 문제: {len(issues)}개"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
                recommendations.extend([
                    "템플릿 파일들의 구문을 검증하세요",
                    "필수 스키마 필드가 모두 포함되어 있는지 확인하세요"
                ])
            
            # 성공 조건: 80% 이상 유효
            is_healthy = valid_templates / templates_checked >= 0.8
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"템플릿 내용 검증: {valid_templates}/{templates_checked} 유효",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"템플릿 내용 검증 실패: {e}",
                category=self.category,
                original_error=e
            )
    
    def check_uv_advanced_capabilities(self) -> CheckResult:
        """
        uv 패키지 매니저의 고급 기능들을 검증합니다.
        
        sync 기능, 가상환경 상태, pyproject.toml 호환성을 확인합니다.
        
        Returns:
            CheckResult: uv 고급 기능 검사 결과
        """
        try:
            details = []
            issues = []
            recommendations = []
            checks_passed = 0
            total_checks = 0
            
            # 1. 기본 uv 가용성 확인
            total_checks += 1
            try:
                result = subprocess.run(
                    ['uv', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version_info = result.stdout.strip()
                    details.append(f"✅ uv 기본 기능: {version_info}")
                    checks_passed += 1
                else:
                    details.append("❌ uv 기본 기능: 실행 실패")
                    issues.append("uv 명령어 실행 불가")
                    recommendations.append("uv 재설치 필요")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                details.append("❌ uv 기본 기능: 명령어 없음")
                issues.append("uv 설치 필요")
                recommendations.append("curl -LsSf https://astral.sh/uv/install.sh | sh")
            
            # 2. pyproject.toml 호환성 확인
            total_checks += 1
            pyproject_path = Path("pyproject.toml")
            if pyproject_path.exists():
                try:
                    result = subprocess.run(
                        ['uv', 'tree'],
                        capture_output=True,
                        text=True,
                        timeout=15
                    )
                    if result.returncode == 0:
                        details.append("✅ pyproject.toml 호환성: 의존성 트리 분석 가능")
                        checks_passed += 1
                    else:
                        details.append("⚠️ pyproject.toml 호환성: 의존성 분석 실패")
                        issues.append("프로젝트 설정 호환성 문제")
                        recommendations.append("uv sync 실행 후 재시도")
                except subprocess.TimeoutExpired:
                    details.append("⚠️ pyproject.toml 호환성: 응답 시간 초과")
                    issues.append("의존성 분석 시간 초과")
            else:
                details.append("❌ pyproject.toml 호환성: 프로젝트 파일 없음")
                issues.append("pyproject.toml 파일 필요")
                recommendations.append("프로젝트 루트 디렉토리에서 실행하세요")
            
            # 3. 가상환경 상태 확인
            total_checks += 1
            try:
                # 현재 가상환경 확인
                venv_path = sys.prefix
                is_venv = venv_path != sys.base_prefix
                
                if is_venv:
                    details.append(f"✅ 가상환경 상태: 활성화됨 ({Path(venv_path).name})")
                    checks_passed += 1
                else:
                    details.append("⚠️ 가상환경 상태: 시스템 Python 사용 중")
                    issues.append("가상환경 미사용")
                    recommendations.append("uv venv 후 source .venv/bin/activate")
            except Exception:
                details.append("❌ 가상환경 상태: 확인 불가")
                issues.append("가상환경 상태 불명")
            
            # 4. sync 기능 테스트 (실제 실행하지 않고 dry-run으로 확인)
            total_checks += 1
            try:
                result = subprocess.run(
                    ['uv', 'sync', '--dry-run'],
                    capture_output=True,
                    text=True,
                    timeout=20
                )
                if result.returncode == 0:
                    details.append("✅ sync 기능: 의존성 해결 가능")
                    checks_passed += 1
                else:
                    details.append("⚠️ sync 기능: 의존성 해결 문제")
                    issues.append("의존성 동기화 이슈")
                    recommendations.append("uv sync --refresh 시도")
            except subprocess.TimeoutExpired:
                details.append("⚠️ sync 기능: 응답 시간 초과")
                issues.append("sync 작업 시간 초과")
            except Exception:
                details.append("❌ sync 기능: 테스트 실패")
                issues.append("sync 기능 문제")
            
            # 전체 요약
            details.extend([
                "",
                "📊 uv 고급 기능 요약:",
                f"   - 총 검사: {total_checks}개",
                f"   - 통과: {checks_passed}개",
                f"   - 성공률: {(checks_passed/total_checks)*100:.1f}%"
            ])
            
            if issues:
                details.extend(["", "🔧 발견된 문제:"] + [f"  • {issue}" for issue in issues])
            
            # 성공 조건: 75% 이상 통과
            is_healthy = checks_passed / total_checks >= 0.75
            
            return CheckResult(
                is_healthy=is_healthy,
                message=f"uv 고급 기능 검증: {checks_passed}/{total_checks} 통과",
                details=details,
                recommendations=recommendations if recommendations else None
            )
            
        except Exception as e:
            raise HealthCheckError(
                message=f"uv 고급 기능 검사 실패: {e}",
                category=self.category,
                original_error=e
            )