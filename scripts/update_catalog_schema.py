#!/usr/bin/env python3
"""
Model Catalog Schema 업데이트 스크립트
- supported_tasks 필드 제거
- data_handler 필드 추가 (디렉토리 기반 매핑)
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any

def update_catalog_files():
    """모든 catalog 파일에서 supported_tasks 제거하고 data_handler 추가"""
    
    # 스크립트 위치 기준으로 catalog 루트 찾기
    script_dir = Path(__file__).parent
    catalog_root = script_dir.parent / "src" / "models" / "catalog"
    
    if not catalog_root.exists():
        print(f"❌ Catalog 디렉토리를 찾을 수 없습니다: {catalog_root}")
        return
    
    print(f"📁 Catalog 루트: {catalog_root}")
    
    # 디렉토리별 data_handler 매핑
    handler_mapping = {
        "Classification": "tabular",
        "Regression": "tabular", 
        "Clustering": "tabular",
        "Causal": "tabular",
        "Timeseries": "timeseries",
        "DeepLearning": "deeplearning"
    }
    
    updated_files = 0
    total_files = 0
    
    for task_dir in catalog_root.iterdir():
        if task_dir.is_dir() and task_dir.name in handler_mapping:
            handler_type = handler_mapping[task_dir.name]
            
            print(f"\n📂 Processing directory: {task_dir.name} → data_handler: {handler_type}")
            
            for yaml_file in task_dir.glob("*.yaml"):
                total_files += 1
                
                try:
                    # YAML 파일 읽기
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if not isinstance(data, dict):
                        print(f"⚠️  Skipped (not a dict): {yaml_file.name}")
                        continue
                    
                    # 변경사항 추적
                    changes_made = False
                    
                    # supported_tasks 제거
                    if 'supported_tasks' in data:
                        removed_tasks = data.pop('supported_tasks')
                        print(f"   🗑️  Removed supported_tasks: {removed_tasks}")
                        changes_made = True
                    
                    # data_handler 추가 (기존 것이 있으면 덮어쓰기)
                    if data.get('data_handler') != handler_type:
                        old_handler = data.get('data_handler', 'None')
                        data['data_handler'] = handler_type
                        print(f"   ✅ Set data_handler: {old_handler} → {handler_type}")
                        changes_made = True
                    
                    # 변경사항이 있으면 파일 업데이트
                    if changes_made:
                        # 백업 생성
                        backup_path = yaml_file.with_suffix('.yaml.backup')
                        if not backup_path.exists():  # 백업이 없을 때만 생성
                            shutil.copy(yaml_file, backup_path)
                        
                        # YAML 파일 업데이트 (순서 유지)
                        with open(yaml_file, 'w', encoding='utf-8') as f:
                            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                                    allow_unicode=True, width=1000)
                        
                        print(f"   💾 Updated: {yaml_file.name}")
                        updated_files += 1
                    else:
                        print(f"   ✨ No changes needed: {yaml_file.name}")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {yaml_file.name}: {e}")
        
        elif task_dir.is_dir():
            print(f"\n⚠️  Unknown directory (skipped): {task_dir.name}")
    
    print(f"\n🎉 업데이트 완료!")
    print(f"   📊 총 파일: {total_files}개")
    print(f"   ✅ 업데이트된 파일: {updated_files}개")
    print(f"   💾 백업 파일: *.yaml.backup")
    
    if updated_files > 0:
        print("\n📋 확인 방법:")
        print("   git diff src/models/catalog/")
        print("   git add src/models/catalog/")
        print("   git commit -m 'Update catalog schema: remove supported_tasks, add data_handler'")

def validate_catalog_structure():
    """업데이트된 catalog 구조 검증"""
    script_dir = Path(__file__).parent
    catalog_root = script_dir.parent / "src" / "models" / "catalog"
    
    print("\n🔍 Catalog 구조 검증 중...")
    
    handler_mapping = {
        "Classification": "tabular",
        "Regression": "tabular", 
        "Clustering": "tabular",
        "Causal": "tabular",
        "Timeseries": "timeseries",
        "DeepLearning": "deeplearning"
    }
    
    validation_errors = []
    
    for task_dir in catalog_root.iterdir():
        if task_dir.is_dir() and task_dir.name in handler_mapping:
            expected_handler = handler_mapping[task_dir.name]
            
            for yaml_file in task_dir.glob("*.yaml"):
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    # supported_tasks 있으면 안됨
                    if 'supported_tasks' in data:
                        validation_errors.append(f"{yaml_file}: supported_tasks 필드가 아직 존재")
                    
                    # data_handler 필수
                    if 'data_handler' not in data:
                        validation_errors.append(f"{yaml_file}: data_handler 필드 누락")
                    elif data['data_handler'] != expected_handler:
                        validation_errors.append(f"{yaml_file}: 잘못된 data_handler - expected: {expected_handler}, got: {data['data_handler']}")
                    
                except Exception as e:
                    validation_errors.append(f"{yaml_file}: 읽기 오류 - {e}")
    
    if validation_errors:
        print("❌ 검증 실패:")
        for error in validation_errors:
            print(f"   {error}")
        return False
    else:
        print("✅ 모든 catalog 파일이 올바르게 업데이트되었습니다!")
        return True

if __name__ == "__main__":
    print("🚀 Model Catalog Schema 업데이트 시작...")
    
    # 1. Catalog 파일들 업데이트
    update_catalog_files()
    
    # 2. 구조 검증
    print("\n" + "="*50)
    is_valid = validate_catalog_structure()
    
    if is_valid:
        print("\n✅ 업데이트 완료! Phase 2 성공!")
    else:
        print("\n❌ 검증 실패. 수동으로 확인이 필요합니다.")