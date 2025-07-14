"""
Settings Extensions
Blueprint v17.0 확장 기능 모듈

이 모듈은 Blueprint v17.0에서 추가된 확장 기능들을 제공합니다.
필요한 경우에만 선택적으로 import하여 사용할 수 있습니다.
"""

import yaml
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from .models import Settings
from .loaders import get_feast_config, get_app_env


def create_feast_config_file(settings: Settings, output_path: str = None) -> str:
    """
    Blueprint v17.0: config에서 임시 Feast 설정 파일 생성
    
    이 함수는 config/*.yaml에 통합된 feast_config를 
    임시 파일로 추출하여 Feast 라이브러리가 읽을 수 있도록 합니다.
    
    Args:
        settings: Settings 객체
        output_path: 출력 파일 경로 (None이면 임시 파일)
        
    Returns:
        생성된 설정 파일 경로
    """
    feast_config = get_feast_config(settings)
    
    if output_path is None:
        # 임시 파일 생성
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True)
        output_path = temp_dir / f"feast_config_{get_app_env()}.yaml"
    
    # YAML 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(feast_config, f, default_flow_style=False, allow_unicode=True)
    
    return str(output_path)


def validate_environment_settings(settings: Settings) -> Dict[str, Any]:
    """
    환경별 설정 유효성 검증
    
    Returns:
        검증 결과 딕셔너리
    """
    validation_results = {
        "app_env": settings.environment.app_env,
        "errors": [],
        "warnings": [],
        "status": "valid"
    }
    
    app_env = settings.environment.app_env
    
    # LOCAL 환경 검증
    if app_env == "local":
        if settings.model.augmenter and settings.model.augmenter.type == "feature_store":
            validation_results["warnings"].append(
                "LOCAL 환경에서는 PassThroughAugmenter 사용을 권장합니다 (Blueprint 원칙 9)"
            )
    
    # DEV 환경 검증
    elif app_env == "dev":
        if not settings.feature_store or not settings.feature_store.feast_config:
            validation_results["errors"].append(
                "DEV 환경에는 완전한 Feature Store 설정이 필요합니다"
            )
    
    # PROD 환경 검증
    elif app_env == "prod":
        if not settings.feature_store or not settings.feature_store.feast_config:
            validation_results["errors"].append(
                "PROD 환경에는 완전한 Feature Store 설정이 필요합니다"
            )
        
        if settings.environment.gcp_project_id == "local-project":
            validation_results["errors"].append(
                "PROD 환경에서는 실제 GCP 프로젝트 ID가 필요합니다"
            )
    
    # 오류가 있으면 상태 변경
    if validation_results["errors"]:
        validation_results["status"] = "invalid"
    elif validation_results["warnings"]:
        validation_results["status"] = "warning"
    
    return validation_results


def print_settings_summary(settings: Settings) -> None:
    """Settings 객체 요약 출력 (개발용)"""
    print(f"""
🎯 Blueprint v17.0 Settings Summary
=====================================
환경: {settings.environment.app_env}
GCP 프로젝트: {settings.environment.gcp_project_id}
MLflow: {settings.mlflow.tracking_uri}

모델 설정:
- 클래스: {settings.model.class_path}
- 로더: {settings.model.loader.source_uri}
- 증강기: {settings.model.augmenter.type if settings.model.augmenter else 'None'}
- 태스크: {settings.model.data_interface.task_type}

Feature Store: {'✅ 설정됨' if settings.feature_store else '❌ 미설정'}
HPO: {'✅ 활성화' if settings.hyperparameter_tuning and settings.hyperparameter_tuning.enabled else '❌ 비활성화'}
=====================================
    """)


def get_settings_diff(settings1: Settings, settings2: Settings) -> Dict[str, Any]:
    """두 Settings 객체 간의 차이점 분석 (개발용)"""
    dict1 = json.loads(settings1.model_dump_json())
    dict2 = json.loads(settings2.model_dump_json())
    
    def find_diff(d1, d2, path=""):
        diff = {}
        all_keys = set(d1.keys()) | set(d2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                diff[current_path] = {"type": "added", "value": d2[key]}
            elif key not in d2:
                diff[current_path] = {"type": "removed", "value": d1[key]}
            elif d1[key] != d2[key]:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    nested_diff = find_diff(d1[key], d2[key], current_path)
                    diff.update(nested_diff)
                else:
                    diff[current_path] = {
                        "type": "changed", 
                        "old": d1[key], 
                        "new": d2[key]
                    }
        
        return diff
    
    return find_diff(dict1, dict2)


def check_blueprint_compliance(settings: Settings) -> Dict[str, Any]:
    """
    Blueprint v17.0 9대 원칙 준수 여부 검사
    
    Returns:
        원칙별 준수 여부와 개선 사항
    """
    compliance = {
        "overall_score": 0,
        "principles": {},
        "recommendations": []
    }
    
    # 원칙 1: 레시피는 논리, 설정은 인프라
    principle_1 = {
        "name": "레시피는 논리, 설정은 인프라",
        "score": 10,  # 기본 점수
        "issues": []
    }
    
    if not settings.feature_store or not settings.feature_store.feast_config:
        principle_1["score"] -= 3
        principle_1["issues"].append("Feature Store 설정이 config에 통합되지 않음")
    
    compliance["principles"]["principle_1"] = principle_1
    
    # 원칙 9: 환경별 차등적 기능 분리
    principle_9 = {
        "name": "환경별 차등적 기능 분리", 
        "score": 10,
        "issues": []
    }
    
    if settings.environment.app_env == "local":
        if settings.model.augmenter and settings.model.augmenter.type == "feature_store":
            principle_9["score"] -= 5
            principle_9["issues"].append("LOCAL 환경에서 Feature Store 사용")
    
    compliance["principles"]["principle_9"] = principle_9
    
    # 전체 점수 계산
    total_score = sum(p["score"] for p in compliance["principles"].values())
    max_score = len(compliance["principles"]) * 10
    compliance["overall_score"] = round((total_score / max_score) * 100)
    
    # 추천 사항
    if compliance["overall_score"] < 80:
        compliance["recommendations"].append("Blueprint 원칙 준수를 위한 설정 개선이 필요합니다")
    
    return compliance 