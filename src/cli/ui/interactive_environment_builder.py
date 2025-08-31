"""
Interactive Environment Builder for Modern ML Pipeline v2.0

대화형 환경 설정 구축기 - 기존 복잡한 Jinja2 템플릿 방식을 완전 대체.
Rich UI를 사용한 직관적이고 사용자 친화적인 환경 설정 경험 제공.

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- Rich UI로 사용자 경험 최적화
"""

from typing import Dict, Any
from datetime import datetime
import sys

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

# 새로운 모델들과 유틸리티 import
from src.cli.utils.config_management import (
    EnvironmentConfig,
    DataAdapterConfig,
    ServiceType
)
from src.cli.utils.system_integration import SimplifiedServiceCatalog


class InteractiveEnvironmentBuilder:
    """
    완전 대화형 환경 설정 구축기
    
    Recipe-Config 분리 아키텍처의 핵심 UI 컴포넌트.
    사용자가 4개 핵심 서비스를 대화형으로 선택하고
    환경별 완전한 설정을 자동 구성.
    
    지원 서비스:
    - ML_TRACKING: MLflow (Local/Server/Disabled)
    - DATABASE: PostgreSQL, MySQL, BigQuery, Disabled
    - FEATURE_STORE: Redis, Disabled  
    - STORAGE: S3, GCS, Local
    
    Examples:
        builder = InteractiveEnvironmentBuilder()
        env_config = builder.create_environment("local")
        # → EnvironmentConfig 객체 반환 (대화형 선택 완료)
    """
    
    def __init__(self):
        """Interactive Environment Builder 초기화"""
        self.console = Console()
        self.catalog = SimplifiedServiceCatalog()
    
    def create_environment(self, env_name: str) -> EnvironmentConfig:
        """
        대화형 환경 생성 프로세스 실행
        
        Args:
            env_name: 생성할 환경 이름
            
        Returns:
            EnvironmentConfig: 완전한 환경 설정 객체
            
        Raises:
            KeyboardInterrupt: 사용자가 Ctrl+C로 중단한 경우
            
        Process:
        1. Welcome 메시지 출력
        2. 4개 핵심 서비스 순차 선택
        3. 데이터 어댑터 자동 구성
        4. 선택 결과 확인
        5. EnvironmentConfig 객체 생성 및 반환
        """
        
        try:
            # Welcome 메시지
            self._show_welcome_message(env_name)
            
            # 4개 핵심 서비스 대화형 선택
            self.console.print("\n[bold cyan]🔧 서비스 설정[/bold cyan]")
            self.console.print("필요한 서비스들을 순서대로 선택하세요.\n")
            
            ml_tracking_config = self._configure_ml_tracking()
            database_config = self._configure_database()  
            feature_store_config = self._configure_feature_store()
            storage_config = self._configure_storage()
            
            # 데이터 어댑터 자동 구성
            data_adapters = self._build_data_adapters(database_config, storage_config)
            
            # 선택 결과 요약 및 확인
            env_config = EnvironmentConfig(
                name=env_name,
                ml_tracking=ml_tracking_config,
                data_adapters=data_adapters,
                feature_store=feature_store_config,
                storage=storage_config,
                created_at=datetime.now().isoformat(),
                description=f"Interactive configuration for {env_name} environment"
            )
            
            self._show_configuration_summary(env_config)
            
            if not self._confirm_configuration():
                self.console.print("[yellow]설정이 취소되었습니다.[/yellow]")
                sys.exit(0)
            
            return env_config
            
        except KeyboardInterrupt:
            self.console.print("\n[red]사용자에 의해 취소되었습니다.[/red]")
            sys.exit(0)
    
    def _show_welcome_message(self, env_name: str) -> None:
        """Welcome 메시지 및 안내 출력"""
        welcome_panel = Panel(
            f"""[bold green]🚀 '{env_name}' 환경 설정을 시작합니다![/bold green]

[bold yellow]Modern ML Pipeline v2.0 특징:[/bold yellow]
• Recipe-Config 완전 분리 아키텍처
• 환경에 독립적인 Recipe 재사용
• 대화형 UI로 직관적 설정

[bold cyan]설정할 서비스 (4개):[/bold cyan]
1️⃣ ML Tracking (실험 추적)
2️⃣ Database (SQL 데이터 소스) 
3️⃣ Feature Store (Point-in-Time 조인)
4️⃣ Storage (아티팩트 저장)""",
            title="Environment Configuration",
            border_style="green"
        )
        
        self.console.print(welcome_panel)
    
    def _configure_ml_tracking(self) -> Dict[str, Any]:
        """ML Tracking 서비스 선택"""
        return self._select_service_interactive(
            service_type=ServiceType.ML_TRACKING,
            title="1️⃣ ML Tracking (실험 추적)",
            description="MLflow를 사용한 실험 추적 설정"
        )
    
    def _configure_database(self) -> Dict[str, Any]:
        """Database 서비스 선택"""
        return self._select_service_interactive(
            service_type=ServiceType.DATABASE,
            title="2️⃣ Database (SQL 데이터 소스)",
            description="SQL 기반 데이터 소스 설정"
        )
    
    def _configure_feature_store(self) -> Dict[str, Any]:
        """Feature Store 서비스 선택"""
        return self._select_service_interactive(
            service_type=ServiceType.FEATURE_STORE,
            title="3️⃣ Feature Store (Point-in-Time 조인)",
            description="피처 스토어 및 Point-in-Time 조인 설정"
        )
    
    def _configure_storage(self) -> Dict[str, Any]:
        """Storage 서비스 선택"""
        return self._select_service_interactive(
            service_type=ServiceType.STORAGE,
            title="4️⃣ Storage (아티팩트 저장)",
            description="모델 및 아티팩트 저장소 설정"
        )
    
    def _select_service_interactive(
        self, 
        service_type: ServiceType, 
        title: str, 
        description: str
    ) -> Dict[str, Any]:
        """
        특정 서비스 타입에 대한 대화형 선택
        
        Args:
            service_type: 선택할 서비스 타입
            title: 섹션 제목
            description: 서비스 설명
            
        Returns:
            Dict[str, Any]: 선택된 서비스 설정
        """
        
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        self.console.print(f"[dim]{description}[/dim]")
        
        # 사용 가능한 서비스 옵션 가져오기
        options = self.catalog.get_service_options(service_type)
        
        if not options:
            self.console.print("[red]사용 가능한 옵션이 없습니다.[/red]")
            return {"provider": "none", "config": {}}
        
        # 옵션 테이블 생성
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("번호", style="dim", width=6)
        table.add_column("서비스", style="cyan", min_width=15)
        table.add_column("설명", style="yellow")
        
        for i, option in enumerate(options, 1):
            table.add_row(
                str(i), 
                option["name"], 
                option.get("description", "")
            )
        
        self.console.print(table)
        
        # 사용자 선택 받기
        choices = [str(i) for i in range(1, len(options) + 1)]
        choice_idx = int(Prompt.ask("선택하세요", choices=choices)) - 1
        selected_option = options[choice_idx]
        
        self.console.print(f"[green]✓ {selected_option['name']} 선택됨[/green]")
        
        # 선택된 서비스에 따른 기본 설정 반환
        return self._build_service_config(selected_option)
    
    def _build_service_config(self, selected_option: Dict[str, Any]) -> Dict[str, Any]:
        """
        선택된 서비스 옵션을 바탕으로 기본 설정 구성
        
        Args:
            selected_option: 선택된 서비스 옵션
            
        Returns:
            Dict[str, Any]: 서비스별 기본 설정
        """
        
        provider = selected_option["provider"]
        
        # Provider별 기본 설정 구성
        if provider == "mlflow_local":
            return {
                "provider": provider,
                "tracking_uri": "./mlruns",
                "experiment_name": "${EXPERIMENT_NAME:default-experiment}"
            }
        elif provider == "mlflow_server":
            return {
                "provider": provider,
                "tracking_uri": "${MLFLOW_TRACKING_URI:http://localhost:5000}",
                "experiment_name": "${EXPERIMENT_NAME:default-experiment}"
            }
        elif provider in ["postgresql", "mysql"]:
            return {
                "provider": provider,
                "connection": {
                    "host": "${DATABASE_HOST:localhost}",
                    "port": 5432 if provider == "postgresql" else 3306,
                    "database": "${DATABASE_NAME:mlpipeline}",
                    "username": "${DATABASE_USER:user}",
                    "password": "${DATABASE_PASSWORD:password}"
                }
            }
        elif provider == "bigquery":
            return {
                "provider": provider,
                "connection": {
                    "project_id": "${BIGQUERY_PROJECT_ID:your-project}",
                    "dataset": "${BIGQUERY_DATASET:ml_pipeline}",
                    "credentials": "${GOOGLE_APPLICATION_CREDENTIALS:./credentials/bigquery.json}"
                }
            }
        elif provider == "redis":
            return {
                "provider": provider,
                "connection_url": "${REDIS_URL:redis://localhost:6379/0}",
                "namespace": "${FEATURE_STORE_NAMESPACE:ml_features}"
            }
        elif provider == "s3":
            return {
                "provider": provider,
                "bucket": "${S3_BUCKET:ml-pipeline-artifacts}",
                "region": "${AWS_REGION:us-west-2}",
                "access_key": "${AWS_ACCESS_KEY_ID}",
                "secret_key": "${AWS_SECRET_ACCESS_KEY}"
            }
        elif provider == "gcs":
            return {
                "provider": provider,
                "bucket": "${GCS_BUCKET:ml-pipeline-artifacts}",
                "credentials": "${GOOGLE_APPLICATION_CREDENTIALS:./credentials/gcs.json}"
            }
        elif provider == "local":
            return {
                "provider": provider,
                "base_path": "${STORAGE_BASE_PATH:./artifacts}"
            }
        else:  # "none" or unknown
            return {
                "provider": "none",
                "config": {}
            }
    
    def _build_data_adapters(
        self, 
        database_config: Dict[str, Any], 
        storage_config: Dict[str, Any]
    ) -> Dict[str, DataAdapterConfig]:
        """
        Database와 Storage 설정을 기반으로 데이터 어댑터 자동 구성
        
        Args:
            database_config: 선택된 데이터베이스 설정
            storage_config: 선택된 스토리지 설정
            
        Returns:
            Dict[str, DataAdapterConfig]: 구성된 데이터 어댑터들
        """
        
        adapters = {}
        
        # SQL 어댑터 (Database 설정 기반)
        if database_config["provider"] != "none":
            adapters["sql"] = DataAdapterConfig(
                type=database_config["provider"],
                connection_params=database_config.get("connection", {})
            )
        
        # Storage 어댑터 (Storage 설정 기반, 항상 생성)  
        adapters["storage"] = DataAdapterConfig(
            type=storage_config["provider"],
            connection_params=storage_config
        )
        
        return adapters
    
    def _show_configuration_summary(self, env_config: EnvironmentConfig) -> None:
        """구성된 환경 설정 요약 표시"""
        
        self.console.print(f"\n[bold cyan]📋 '{env_config.name}' 환경 설정 요약[/bold cyan]")
        
        # 설정 요약 테이블
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("서비스", style="cyan", min_width=15)
        summary_table.add_column("선택된 옵션", style="green")
        summary_table.add_column("상태", style="yellow")
        
        # ML Tracking
        ml_provider = env_config.ml_tracking.get("provider", "unknown")
        ml_status = "✅ 활성화" if ml_provider != "none" else "⏸️ 비활성화"
        summary_table.add_row("ML Tracking", ml_provider, ml_status)
        
        # Database
        sql_adapter = env_config.data_adapters.get("sql")
        if sql_adapter:
            db_status = f"✅ {sql_adapter.type}"
        else:
            db_status = "⏸️ 비활성화"
        summary_table.add_row("Database", sql_adapter.type if sql_adapter else "none", db_status)
        
        # Feature Store
        fs_provider = env_config.feature_store.get("provider", "none")
        fs_status = "✅ 활성화" if fs_provider != "none" else "⏸️ 비활성화"
        summary_table.add_row("Feature Store", fs_provider, fs_status)
        
        # Storage
        storage_adapter = env_config.data_adapters.get("storage")
        storage_status = f"✅ {storage_adapter.type}" if storage_adapter else "❌ 오류"
        summary_table.add_row("Storage", storage_adapter.type if storage_adapter else "unknown", storage_status)
        
        self.console.print(summary_table)
    
    def _confirm_configuration(self) -> bool:
        """설정 확인"""
        return Confirm.ask(
            "\n[bold yellow]이 설정으로 환경을 생성하시겠습니까?[/bold yellow]",
            default=True
        )