"""
Health Check Reporter Implementation
Blueprint v17.0 - Health check results formatting and display

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 사용자 친화적 출력
"""

from typing import Dict, List, Optional
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