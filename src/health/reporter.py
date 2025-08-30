"""
Health Check Reporter Implementation
Blueprint v17.0 - Health check results formatting and display

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 사용자 친화적 출력
"""

from typing import Dict, List, Optional, Any
import typer

from src.health.models import (
    CheckResult, HealthCheckSummary, HealthCheckConfig,
    CheckCategory
)


class HealthReporter:
    """
    건강 검사 결과를 포맷팅하고 출력하는 클래스.
    
    컬러 출력, 추천 사항, 요약 정보 등을 사용자 친화적으로 제공합니다.
    """
    
    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """
        HealthReporter 인스턴스를 초기화합니다.
        
        Args:
            config: 건강 검사 설정
        """
        self.config = config or HealthCheckConfig()
    
    def display_summary(self, summary: HealthCheckSummary) -> None:
        """
        건강 검사 요약을 콘솔에 출력합니다.
        
        Args:
            summary: 건강 검사 요약 정보
        """
        # 헤더 출력
        self._print_header(summary)
        
        # 카테고리별 결과 출력
        for category, result in summary.categories.items():
            self._print_category_result(category, result)
            
        # 전체 요약 출력
        self._print_overall_summary(summary)
        
        # 추천 사항 출력
        if not summary.overall_healthy:
            self._print_recommendations(summary)
    
    def format_result(self, category_name: str, result: CheckResult, use_colors: bool = True) -> str:
        """
        개별 검사 결과를 포맷팅합니다.
        
        Args:
            category_name: 카테고리명
            result: 검사 결과
            use_colors: 컬러 출력 사용 여부
            
        Returns:
            str: 포맷팅된 결과 문자열
        """
        # 상태 아이콘 및 색상 결정
        if result.is_healthy:
            status_icon = "✅"
            color = typer.colors.GREEN if use_colors else None
        else:
            status_icon = "❌"
            color = typer.colors.RED if use_colors else None
        
        # 메시지 포맷팅
        formatted_message = f"{status_icon} {category_name}: {result.message}"
        
        if color and use_colors:
            formatted_message = typer.style(formatted_message, fg=color)
        
        return formatted_message
    
    def get_recommendations(self, category_name: str, result: CheckResult) -> List[str]:
        """
        검사 결과에 기반한 추천 사항을 생성합니다.
        
        Args:
            category_name: 카테고리명
            result: 검사 결과
            
        Returns:
            List[str]: 추천 사항 목록
        """
        if result.is_healthy:
            return []
        
        # 기본 추천사항 (결과에서 제공된 것)
        recommendations = result.recommendations or []
        
        # 카테고리별 추가 추천사항
        additional_recommendations = self._get_category_specific_recommendations(
            category_name, result
        )
        
        return recommendations + additional_recommendations
    
    def generate_summary(self, results: Dict[str, CheckResult]) -> Dict[str, any]:
        """
        검사 결과들을 종합하여 요약 정보를 생성합니다.
        
        Args:
            results: 카테고리별 검사 결과
            
        Returns:
            Dict[str, any]: 요약 정보
        """
        total_checks = len(results)
        passed_checks = sum(1 for result in results.values() if result.is_healthy)
        failed_checks = total_checks - passed_checks
        
        return {
            'overall_healthy': failed_checks == 0,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
    
    def _print_header(self, summary: HealthCheckSummary) -> None:
        """건강 검사 결과 헤더를 출력합니다."""
        typer.echo("=" * 60)
        typer.secho("🏥 Modern ML Pipeline - 시스템 건강 상태 보고서", 
                   fg=typer.colors.CYAN, bold=True)
        typer.echo("=" * 60)
        typer.echo(f"검사 시간: {summary.timestamp}")
        typer.echo(f"실행 시간: {summary.execution_time_seconds:.2f}초")
        typer.echo()
    
    def _print_category_result(self, category: CheckCategory, result: CheckResult) -> None:
        """카테고리별 검사 결과를 출력합니다."""
        category_display_names = {
            CheckCategory.ENVIRONMENT: "환경 검사",
            CheckCategory.MLFLOW: "MLflow 검사", 
            CheckCategory.EXTERNAL_SERVICES: "외부 서비스 검사",
            CheckCategory.TEMPLATES: "템플릿 검사",
            CheckCategory.SYSTEM: "시스템 검사"
        }
        
        category_name = category_display_names.get(category, category.value)
        
        # 카테고리 헤더
        typer.secho(f"\n📋 {category_name}", fg=typer.colors.BLUE, bold=True)
        typer.echo("-" * 40)
        
        # 메인 결과
        formatted_result = self.format_result(category_name, result, self.config.use_colors)
        typer.echo(formatted_result)
        
        # 상세 정보 (verbose 모드에서만)
        if self.config.verbose and result.details:
            typer.echo("  상세 정보:")
            for detail in result.details:
                typer.echo(f"    • {detail}")
    
    def _print_overall_summary(self, summary: HealthCheckSummary) -> None:
        """전체 요약 정보를 출력합니다."""
        typer.echo("\n" + "=" * 60)
        typer.secho("📊 전체 요약", fg=typer.colors.BLUE, bold=True)
        typer.echo("=" * 60)
        
        # 전체 상태
        if summary.overall_healthy:
            status_text = "✅ 모든 시스템 정상"
            status_color = typer.colors.GREEN
        else:
            status_text = "❌ 일부 시스템에 문제 발견"
            status_color = typer.colors.RED
        
        typer.secho(f"상태: {status_text}", fg=status_color, bold=True)
        
        # 통계
        typer.echo(f"전체 검사: {summary.total_checks}개")
        typer.echo(f"통과: {summary.passed_checks}개")
        typer.echo(f"실패: {summary.failed_checks}개")
        typer.echo(f"성공률: {summary.success_rate:.1f}%")
        
        if summary.has_warnings:
            typer.echo(f"경고: {summary.warning_checks}개")
    
    def _print_recommendations(self, summary: HealthCheckSummary) -> None:
        """추천 사항들을 출력합니다."""
        typer.echo("\n" + "=" * 60)
        typer.secho("💡 문제 해결 방법", fg=typer.colors.YELLOW, bold=True)
        typer.echo("=" * 60)
        
        recommendation_count = 1
        
        for category, result in summary.categories.items():
            if not result.is_healthy and result.recommendations:
                category_display_names = {
                    CheckCategory.ENVIRONMENT: "환경 문제",
                    CheckCategory.MLFLOW: "MLflow 문제",
                    CheckCategory.EXTERNAL_SERVICES: "외부 서비스 문제",
                    CheckCategory.TEMPLATES: "템플릿 문제",
                    CheckCategory.SYSTEM: "시스템 문제"
                }
                
                category_name = category_display_names.get(category, category.value)
                typer.secho(f"\n🔧 {category_name} 해결책:", fg=typer.colors.YELLOW)
                
                for recommendation in result.recommendations:
                    typer.echo(f"  {recommendation_count}. {recommendation}")
                    recommendation_count += 1
        
        # 일반적인 도움말
        typer.echo("\n💡 추가 도움이 필요하다면:")
        typer.echo("  • GitHub Issues: https://github.com/your-org/modern-ml-pipeline/issues")
        typer.echo("  • Documentation: 프로젝트 README.md 참조")
        typer.echo("  • 상세 로그: --verbose 옵션 사용")
    
    def _get_category_specific_recommendations(self, category_name: str, result: CheckResult) -> List[str]:
        """카테고리별 추가 추천사항을 생성합니다."""
        additional_recommendations = []
        
        if "environment" in category_name.lower():
            additional_recommendations.extend([
                "개발 환경 가이드: dev/README.md 참조",
                "의존성 문제 시: uv sync --reinstall 실행"
            ])
        elif "mlflow" in category_name.lower():
            additional_recommendations.extend([
                "MLflow 서버 설정: config/base.yaml에서 tracking_uri 확인",
                "로컬 모드 전환: MLFLOW_TRACKING_URI 환경변수 제거"
            ])
        elif "external" in category_name.lower():
            additional_recommendations.extend([
                "Docker 컨테이너 상태 확인: docker-compose ps",
                "서비스 재시작: docker-compose restart"
            ])
        
        return additional_recommendations


class ActionableReporter:
    """
    M04-2-5 Enhanced Actionable Reporting System
    
    실행 가능한 해결책과 우선순위 기반 보고서를 제공하는 클래스.
    """
    
    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """
        ActionableReporter 인스턴스를 초기화합니다.
        
        Args:
            config: 건강 검사 설정
        """
        self.config = config or HealthCheckConfig()
    
    def generate_executable_commands(self, summary: HealthCheckSummary) -> Dict[CheckCategory, List[str]]:
        """
        문제 상황별 실행 가능한 명령어를 생성합니다.
        
        Args:
            summary: 건강 검사 요약 정보
            
        Returns:
            Dict[CheckCategory, List[str]]: 카테고리별 실행 가능한 명령어 목록
        """
        executable_commands = {}
        
        for category, result in summary.categories.items():
            if not result.is_healthy:
                commands = []
                
                if category == CheckCategory.ENVIRONMENT:
                    commands.extend([
                        "uv sync --reinstall",
                        "uv run python --version",
                        "uv add python@3.11"
                    ])
                elif category == CheckCategory.MLFLOW:
                    commands.extend([
                        "mlflow server --host 0.0.0.0 --port 5000",
                        "docker-compose -f ../mmp-local-dev/docker-compose.yml restart mlflow",
                        "export MLFLOW_TRACKING_URI=http://localhost:5000"
                    ])
                elif category == CheckCategory.EXTERNAL_SERVICES:
                    commands.extend([
                        "docker-compose -f ../mmp-local-dev/docker-compose.yml ps",
                        "docker-compose -f ../mmp-local-dev/docker-compose.yml restart",
                        "docker-compose -f ../mmp-local-dev/docker-compose.yml up -d"
                    ])
                elif category == CheckCategory.TEMPLATES:
                    commands.extend([
                        "yamllint config/",
                        "find config/ -name '*.yaml' -exec yamllint {} \\;",
                        "uv run python -c \"import yaml; yaml.safe_load(open('config/base.yaml'))\""
                    ])
                
                if commands:
                    executable_commands[category] = commands
        
        return executable_commands
    
    def sort_issues_by_priority(self, summary: HealthCheckSummary) -> List[Dict[str, Any]]:
        """
        문제들을 우선순위순으로 정렬합니다.
        
        Args:
            summary: 건강 검사 요약 정보
            
        Returns:
            List[Dict[str, Any]]: 우선순위순 정렬된 문제 목록
        """
        issues = []
        priority_order = {'critical': 0, 'important': 1, 'warning': 2}
        
        for category, result in summary.categories.items():
            if not result.is_healthy:
                severity = result.severity or 'warning'
                issues.append({
                    'category': category,
                    'result': result,
                    'severity': severity,
                    'priority': priority_order.get(severity, 3)
                })
        
        # 우선순위순 정렬 (숫자가 작을수록 높은 우선순위)
        return sorted(issues, key=lambda x: x['priority'])
    
    def generate_step_by_step_resolution(self, category: CheckCategory, result: CheckResult) -> List[Dict[str, Any]]:
        """
        단계별 해결 가이드를 생성합니다.
        
        Args:
            category: 검사 카테고리
            result: 검사 결과
            
        Returns:
            List[Dict[str, Any]]: 단계별 해결 가이드
        """
        steps = []
        
        if category == CheckCategory.EXTERNAL_SERVICES:
            steps = [
                {
                    'step_number': 1,
                    'action': 'Docker 컨테이너 상태 확인',
                    'command': 'docker-compose -f ../mmp-local-dev/docker-compose.yml ps',
                    'description': 'mmp-local-dev의 모든 컨테이너 상태를 확인합니다.'
                },
                {
                    'step_number': 2,
                    'action': '서비스별 연결 테스트',
                    'command': 'telnet localhost 5432; telnet localhost 6379',
                    'description': 'PostgreSQL과 Redis 포트 연결을 개별 테스트합니다.'
                },
                {
                    'step_number': 3,
                    'action': '컨테이너 재시작',
                    'command': 'docker-compose -f ../mmp-local-dev/docker-compose.yml restart',
                    'description': '문제가 있는 서비스를 재시작합니다.'
                }
            ]
        elif category == CheckCategory.MLFLOW:
            steps = [
                {
                    'step_number': 1,
                    'action': 'MLflow 서버 상태 확인',
                    'command': 'curl -f http://localhost:5000/health',
                    'description': 'MLflow 서버 응답을 확인합니다.'
                },
                {
                    'step_number': 2,
                    'action': '로컬 모드로 전환',
                    'command': 'unset MLFLOW_TRACKING_URI',
                    'description': '서버 연결 실패 시 로컬 모드로 전환합니다.'
                }
            ]
        
        return steps
    
    def format_with_priority_colors(self, category_name: str, result: CheckResult) -> str:
        """
        우선순위 기반 색상으로 포맷팅합니다.
        
        Args:
            category_name: 카테고리명
            result: 검사 결과
            
        Returns:
            str: 색상 코딩된 결과 문자열
        """
        severity = result.severity or 'warning'
        
        # 우선순위별 색상 설정
        if severity == 'critical':
            color = typer.colors.BRIGHT_RED
            icon = "🔥"
        elif severity == 'important':
            color = typer.colors.BRIGHT_YELLOW
            icon = "⚠️"
        else:  # warning
            color = typer.colors.YELLOW
            icon = "💡"
        
        formatted_message = f"{icon} {category_name}: {result.message}"
        
        if self.config.use_colors:
            formatted_message = typer.style(formatted_message, fg=color, bold=True)
        
        return formatted_message
    
    def generate_interactive_resolution_guide(self, category: CheckCategory, result: CheckResult) -> Dict[str, Any]:
        """
        대화형 해결 가이드를 생성합니다.
        
        Args:
            category: 검사 카테고리
            result: 검사 결과
            
        Returns:
            Dict[str, Any]: 대화형 가이드 정보
        """
        guide = {
            'options': [],
            'prompts': [],
            'next_steps': []
        }
        
        if category == CheckCategory.MLFLOW:
            guide['options'] = [
                {
                    'choice': '1',
                    'description': 'MLflow 서버 재시작',
                    'commands': [
                        'docker-compose -f ../mmp-local-dev/docker-compose.yml restart mlflow',
                        'sleep 10',
                        'curl -f http://localhost:5000/health'
                    ]
                },
                {
                    'choice': '2',
                    'description': '로컬 모드로 전환',
                    'commands': [
                        'unset MLFLOW_TRACKING_URI',
                        'mkdir -p ./mlruns',
                        'echo "로컬 모드로 전환됨"'
                    ]
                }
            ]
            
            guide['prompts'] = [
                '해결 방법을 선택하세요:',
                '1) MLflow 서버 재시작',
                '2) 로컬 모드로 전환',
                '선택 (1-2): '
            ]
        
        guide['next_steps'] = [
            'modern-ml-pipeline self-check 재실행하여 문제 해결 확인',
            '문제가 지속되면 로그 파일 확인 또는 GitHub Issues에 보고'
        ]
        
        return guide
    
    def generate_comprehensive_actionable_report(self, summary: HealthCheckSummary) -> Dict[str, Any]:
        """
        종합적인 액션 가능한 보고서를 생성합니다.
        
        Args:
            summary: 건강 검사 요약 정보
            
        Returns:
            Dict[str, Any]: 종합 액션 가능한 보고서
        """
        report = {
            'summary': {
                'overall_healthy': summary.overall_healthy,
                'total_checks': summary.total_checks,
                'failed_checks': summary.failed_checks,
                'success_rate': ((summary.total_checks - summary.failed_checks) / summary.total_checks * 100) if summary.total_checks > 0 else 0
            },
            'prioritized_actions': self.sort_issues_by_priority(summary),
            'executable_commands': self.generate_executable_commands(summary),
            'interactive_guides': {}
        }
        
        # 실패한 각 카테고리에 대해 대화형 가이드 생성
        for category, result in summary.categories.items():
            if not result.is_healthy:
                report['interactive_guides'][category] = self.generate_interactive_resolution_guide(category, result)
        
        return report