"""
Common fixtures for E2E tests
"""

import pytest
import tempfile
import shutil
import mlflow
import uuid


@pytest.fixture
def isolated_mlflow():
    """MLflow 테스트 격리를 위한 고유한 tracking URI 설정
    
    각 테스트마다 완전히 분리된 MLflow tracking database를 사용하여
    병렬 실행 시 run name 충돌이나 experiment 충돌을 방지합니다.
    """
    # 기존 active run이 있으면 종료
    if mlflow.active_run():
        mlflow.end_run()
    
    # 각 테스트마다 완전히 분리된 MLflow tracking URI 생성
    unique_id = str(uuid.uuid4())[:8]
    test_mlflow_dir = tempfile.mkdtemp(prefix=f"mlflow_test_{unique_id}_")
    isolated_tracking_uri = f"file://{test_mlflow_dir}"
    
    # 원본 MLflow 설정 백업
    original_tracking_uri = mlflow.get_tracking_uri()
    
    # 테스트용 MLflow 설정 적용
    mlflow.set_tracking_uri(isolated_tracking_uri)
    
    yield isolated_tracking_uri
    
    # 테스트 완료 후 active run 정리
    if mlflow.active_run():
        mlflow.end_run()
    
    # MLflow 설정 복원 및 임시 디렉토리 정리
    mlflow.set_tracking_uri(original_tracking_uri)
    shutil.rmtree(test_mlflow_dir, ignore_errors=True)


@pytest.fixture
def unique_experiment_name():
    """각 테스트마다 고유한 experiment name을 생성합니다."""
    return f"e2e_test_{uuid.uuid4().hex[:8]}"