"""
Interactive Config Builder
Phase 1: 대화형 설정 생성기

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- TDD 기반 개발
"""

from typing import Dict, Any, List, Optional, Tuple
import yaml
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.cli.ui.interactive_selector import InteractiveSelector


class InteractiveConfigBuilder:
    """대화형 Config 빌더."""
    
    def __init__(self) -> None:
        """Initialize InteractiveConfigBuilder."""
        self.console = Console()
        self.selector = InteractiveSelector()
        
    def run_interactive_flow(self, env_name: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 대화형 플로우 실행.
        
        Args:
            env_name: 환경 이름 (선택적)
            
        Returns:
            사용자 선택 사항을 담은 딕셔너리
        """
        selections = {}
        
        # 1. 기본 정보
        self.console.print("\n[bold cyan]🏷️ 기본 정보[/bold cyan]")
        selections['env_name'] = env_name or Prompt.ask(
            "환경 이름을 입력하세요", 
            default="local"
        )
        selections['project_name'] = Prompt.ask(
            "프로젝트 이름", 
            default="ml-pipeline"
        )
        
        # 2. 데이터 소스
        self.console.print("\n[bold cyan]🗄️ 데이터 소스 설정[/bold cyan]")
        selections.update(self._select_data_source())
        
        # 3. MLflow
        self.console.print("\n[bold cyan]📊 MLflow 설정[/bold cyan]")
        selections.update(self._select_mlflow())
        
        # 4. Feature Store
        self.console.print("\n[bold cyan]🎯 Feature Store 설정[/bold cyan]")
        selections.update(self._select_feature_store())
        
        # 5. 아티팩트 저장소
        self.console.print("\n[bold cyan]💾 아티팩트 저장소 설정[/bold cyan]")
        selections.update(self._select_storage())
        
        # 6. 고급 설정
        if Confirm.ask("\n⚙️ 고급 설정을 구성하시겠습니까?", default=False):
            selections.update(self._advanced_settings())
        
        return selections
    
    def _select_data_source(self) -> Dict[str, Any]:
        """
        데이터 소스 선택 및 설정.
        
        Returns:
            데이터 소스 관련 설정
        """
        options = [
            ("PostgreSQL (로컬 개발용)", "postgresql"),
            ("MySQL (팀 개발 서버)", "mysql"),
            ("BigQuery (GCP 프로덕션)", "bigquery"),
            ("SQLite (테스트용)", "sqlite"),
        ]
        
        data_source = self.selector.select(
            "데이터 소스를 선택하세요",
            options
        )
        
        config = {'data_source': data_source}
        
        # 데이터 소스별 추가 설정
        if data_source == 'postgresql':
            config['db_host'] = Prompt.ask("  호스트", default="localhost")
            config['db_port'] = Prompt.ask("  포트", default="5432")
            config['db_name'] = Prompt.ask("  데이터베이스명", default="mlflow")
            config['db_user'] = Prompt.ask("  사용자명", default="postgres")
            config['db_connection_uri'] = (
                f"postgresql://${{DB_USER:={config['db_user']}}}:${{DB_PASSWORD}}@"
                f"{config['db_host']}:{config['db_port']}/{config['db_name']}"
            )
        elif data_source == 'mysql':
            config['db_host'] = Prompt.ask("  호스트", default="localhost")
            config['db_port'] = Prompt.ask("  포트", default="3306")
            config['db_name'] = Prompt.ask("  데이터베이스명", default="mlflow")
            config['db_user'] = Prompt.ask("  사용자명", default="root")
            config['db_connection_uri'] = (
                f"mysql+pymysql://${{DB_USER:={config['db_user']}}}:${{DB_PASSWORD}}@"
                f"{config['db_host']}:{config['db_port']}/{config['db_name']}"
            )
        elif data_source == 'bigquery':
            config['bq_project'] = Prompt.ask("  GCP 프로젝트 ID", default="your-project")
            config['bq_dataset'] = Prompt.ask("  BigQuery 데이터셋", default="ml_data")
            config['db_connection_uri'] = f"bigquery://{config['bq_project']}/{config['bq_dataset']}"
        elif data_source == 'sqlite':
            config['db_path'] = Prompt.ask("  데이터베이스 파일 경로", default="./data/mlflow.db")
            config['db_connection_uri'] = f"sqlite:///{config['db_path']}"
        
        return config
    
    def _select_mlflow(self) -> Dict[str, Any]:
        """
        MLflow 설정 선택.
        
        Returns:
            MLflow 관련 설정
        """
        options = [
            ("Local (./mlruns)", "local"),
            ("Remote Server", "remote"),
            ("Cloud Storage (GCS/S3)", "cloud"),
        ]
        
        mlflow_type = self.selector.select(
            "MLflow Tracking 방식을 선택하세요",
            options
        )
        
        config = {'mlflow_type': mlflow_type}
        
        if mlflow_type == 'local':
            config['mlflow_uri'] = Prompt.ask("  저장 경로", default="./mlruns")
            config['mlflow_experiment'] = Prompt.ask("  기본 Experiment 이름", default="${ENV_NAME}-experiment")
        elif mlflow_type == 'remote':
            config['mlflow_uri'] = Prompt.ask("  서버 URL", default="http://localhost:5000")
            config['mlflow_experiment'] = Prompt.ask("  기본 Experiment 이름", default="${ENV_NAME}-experiment")
        elif mlflow_type == 'cloud':
            storage_type = self.selector.select(
                "  클라우드 스토리지 타입",
                [("Google Cloud Storage", "gcs"), ("AWS S3", "s3")]
            )
            if storage_type == 'gcs':
                config['mlflow_uri'] = Prompt.ask("    GCS 버킷 경로", default="gs://your-bucket/mlflow")
            else:
                config['mlflow_uri'] = Prompt.ask("    S3 버킷 경로", default="s3://your-bucket/mlflow")
            config['mlflow_experiment'] = Prompt.ask("  기본 Experiment 이름", default="${ENV_NAME}-experiment")
        
        return config
    
    def _select_feature_store(self) -> Dict[str, Any]:
        """
        Feature Store 설정 선택.
        
        Returns:
            Feature Store 관련 설정
        """
        config = {}
        
        if Confirm.ask("Feature Store를 사용하시겠습니까?", default=True):
            config['feature_store_enabled'] = True
            
            # Offline Store (항상 필요)
            self.console.print("\n  [cyan]Offline Store 설정[/cyan]")
            config['offline_store_type'] = 'file'  # 기본값
            config['offline_store_path'] = Prompt.ask("    저장 경로", default="./feature_repo/data")
            
            # Online Store
            self.console.print("\n  [cyan]Online Store 설정[/cyan]")
            options = [
                ("Redis (실시간 서빙)", "redis"),
                ("SQLite (개발용)", "sqlite"),
                ("None (배치 추론만)", "none"),
            ]
            
            online_store = self.selector.select(
                "    Online Store를 선택하세요",
                options
            )
            
            config['online_store_type'] = online_store
            
            if online_store == 'redis':
                config['redis_host'] = Prompt.ask("    Redis 호스트", default="localhost")
                config['redis_port'] = Prompt.ask("    Redis 포트", default="6379")
                config['redis_db'] = Prompt.ask("    Redis DB 번호", default="0")
            elif online_store == 'sqlite':
                config['sqlite_path'] = Prompt.ask("    SQLite 경로", default="./feature_repo/online_store.db")
        else:
            config['feature_store_enabled'] = False
        
        return config
    
    def _select_storage(self) -> Dict[str, Any]:
        """
        아티팩트 저장소 설정 선택.
        
        Returns:
            스토리지 관련 설정
        """
        options = [
            ("Local (./data)", "local"),
            ("Google Cloud Storage", "gcs"),
            ("AWS S3", "s3"),
        ]
        
        storage_type = self.selector.select(
            "아티팩트 저장소를 선택하세요",
            options
        )
        
        config = {'storage_type': storage_type}
        
        if storage_type == 'local':
            config['storage_path'] = Prompt.ask("  저장 경로", default="./data")
        elif storage_type == 'gcs':
            config['gcs_bucket'] = Prompt.ask("  GCS 버킷 이름", default="your-bucket")
            config['gcs_prefix'] = Prompt.ask("  버킷 내 경로", default="ml-artifacts")
        elif storage_type == 's3':
            config['s3_bucket'] = Prompt.ask("  S3 버킷 이름", default="your-bucket")
            config['s3_prefix'] = Prompt.ask("  버킷 내 경로", default="ml-artifacts")
        
        return config
    
    def _advanced_settings(self) -> Dict[str, Any]:
        """
        고급 설정 구성.
        
        Returns:
            고급 설정 사항
        """
        config = {}
        
        # 로깅 레벨
        log_level = self.selector.select(
            "로깅 레벨을 선택하세요",
            [
                ("DEBUG", "DEBUG"),
                ("INFO", "INFO"),
                ("WARNING", "WARNING"),
                ("ERROR", "ERROR"),
            ]
        )
        config['log_level'] = log_level
        
        # 병렬 처리
        config['n_jobs'] = Prompt.ask("병렬 처리 워커 수", default="4")
        
        # 캐시 설정
        config['enable_cache'] = Confirm.ask("캐시를 활성화하시겠습니까?", default=True)
        
        return config
    
    def generate_config_file(self, env_name: str, selections: Dict[str, Any]) -> Path:
        """
        선택 사항을 기반으로 config YAML 파일 생성.
        
        Args:
            env_name: 환경 이름
            selections: 사용자 선택 사항
            
        Returns:
            생성된 config 파일 경로
        """
        # configs 디렉토리 생성
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        # Config 구조 생성
        config = self._build_config_structure(selections)
        
        # YAML 파일 저장
        config_path = config_dir / f"{env_name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return config_path
    
    def generate_env_template(self, env_name: str, selections: Dict[str, Any]) -> Path:
        """
        선택 사항을 기반으로 .env 템플릿 파일 생성.
        
        Args:
            env_name: 환경 이름
            selections: 사용자 선택 사항
            
        Returns:
            생성된 .env 템플릿 파일 경로
        """
        env_vars = []
        
        # 기본 환경 변수
        env_vars.append(f"# Environment: {env_name}")
        env_vars.append(f"ENV_NAME={env_name}")
        env_vars.append(f"PROJECT_NAME={selections.get('project_name', 'ml-pipeline')}")
        env_vars.append("")
        
        # 데이터베이스 관련
        if selections.get('data_source') in ['postgresql', 'mysql']:
            env_vars.append("# Database")
            env_vars.append(f"DB_USER={selections.get('db_user', 'user')}")
            env_vars.append("DB_PASSWORD=your_password_here")
            env_vars.append("")
        
        # BigQuery 관련
        if selections.get('data_source') == 'bigquery':
            env_vars.append("# Google Cloud")
            env_vars.append(f"GCP_PROJECT={selections.get('bq_project', 'your-project')}")
            env_vars.append("GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
            env_vars.append("")
        
        # MLflow 관련
        if selections.get('mlflow_type') == 'remote':
            env_vars.append("# MLflow")
            env_vars.append(f"MLFLOW_TRACKING_URI={selections.get('mlflow_uri', 'http://localhost:5000')}")
            env_vars.append("")
        
        # Feature Store (Redis) 관련
        if selections.get('online_store_type') == 'redis':
            env_vars.append("# Redis (Feature Store)")
            env_vars.append(f"REDIS_HOST={selections.get('redis_host', 'localhost')}")
            env_vars.append(f"REDIS_PORT={selections.get('redis_port', '6379')}")
            env_vars.append("REDIS_PASSWORD=")  # Optional
            env_vars.append("")
        
        # Cloud Storage 관련
        if selections.get('storage_type') == 'gcs':
            env_vars.append("# Google Cloud Storage")
            env_vars.append(f"GCS_BUCKET={selections.get('gcs_bucket', 'your-bucket')}")
            env_vars.append("")
        elif selections.get('storage_type') == 's3':
            env_vars.append("# AWS S3")
            env_vars.append(f"S3_BUCKET={selections.get('s3_bucket', 'your-bucket')}")
            env_vars.append("AWS_ACCESS_KEY_ID=your_access_key")
            env_vars.append("AWS_SECRET_ACCESS_KEY=your_secret_key")
            env_vars.append("AWS_REGION=us-west-2")
            env_vars.append("")
        
        # 파일 저장
        env_template_path = Path(f".env.{env_name}.template")
        with open(env_template_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_vars))
        
        return env_template_path
    
    def _build_config_structure(self, selections: Dict[str, Any]) -> Dict[str, Any]:
        """
        선택 사항을 기반으로 config 구조 생성.
        
        Args:
            selections: 사용자 선택 사항
            
        Returns:
            Config 딕셔너리 구조
        """
        config = {
            'environment': {
                'app_env': selections['env_name'],
                'gcp_project_id': selections.get('bq_project', '${GCP_PROJECT:}')
            },
            'mlflow': {
                'tracking_uri': selections.get('mlflow_uri', './mlruns'),
                'experiment_name': selections.get('mlflow_experiment', f"{selections['env_name']}-experiment")
            },
            'data_adapters': {
                'default_loader': 'sql',
                'default_storage': 'storage',
                'adapters': {}
            },
            'serving': {
                'enabled': False,
                'model_stage': 'None'
            },
            'artifact_stores': {},
            'hyperparameter_tuning': {
                'enabled': False
            }
        }
        
        # SQL Adapter 설정
        if 'db_connection_uri' in selections:
            config['data_adapters']['adapters']['sql'] = {
                'class_name': 'SqlAdapter',
                'config': {
                    'connection_uri': selections['db_connection_uri'],
                    'query_timeout': 30
                }
            }
        
        # Storage Adapter 설정
        if selections.get('storage_type') == 'local':
            config['data_adapters']['adapters']['storage'] = {
                'class_name': 'StorageAdapter',
                'config': {
                    'storage_options': {}
                }
            }
            config['artifact_stores']['local'] = {
                'enabled': True,
                'base_uri': selections.get('storage_path', './data')
            }
        elif selections.get('storage_type') == 'gcs':
            config['data_adapters']['adapters']['storage'] = {
                'class_name': 'StorageAdapter',
                'config': {
                    'storage_options': {
                        'project': '${GCP_PROJECT}',
                        'token': '${GOOGLE_APPLICATION_CREDENTIALS:}'
                    }
                }
            }
            config['artifact_stores']['gcs'] = {
                'enabled': True,
                'bucket': selections.get('gcs_bucket', '${GCS_BUCKET}'),
                'prefix': selections.get('gcs_prefix', 'ml-artifacts')
            }
        
        # Feature Store 설정
        if selections.get('feature_store_enabled'):
            config['serving']['realtime_feature_store'] = {
                'store_type': selections.get('online_store_type', 'redis')
            }
            
            if selections.get('online_store_type') == 'redis':
                config['serving']['realtime_feature_store']['connection'] = {
                    'host': selections.get('redis_host', '${REDIS_HOST:localhost}'),
                    'port': int(selections.get('redis_port', 6379)),
                    'db': int(selections.get('redis_db', 0))
                }
        
        # 고급 설정
        if 'log_level' in selections:
            config['environment']['log_level'] = selections['log_level']
        
        return config