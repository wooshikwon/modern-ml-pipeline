#!/usr/bin/env python3
"""
Recipe 구조 올바른 복원 스크립트
entity_schema (Point-in-Time) + data_interface (ML 설정) 분리
"""

import yaml
import os
from pathlib import Path

def fix_recipe_structure(file_path):
    """Recipe 파일의 구조를 올바르게 수정"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if 'model' not in data or 'loader' not in data['model']:
        return False
    
    # 현재 entity_schema에서 정보 추출
    entity_schema = data['model']['loader'].get('entity_schema', {})
    
    if not entity_schema:
        return False
    
    # 1. entity_schema는 Point-in-Time만 (target_column, task_type 제거)
    new_entity_schema = {
        'entity_columns': entity_schema.get('entity_columns', []),
        'timestamp_column': entity_schema.get('timestamp_column', '')
    }
    
    # 2. data_interface 생성 (ML 설정)
    new_data_interface = {
        'task_type': entity_schema.get('task_type', ''),
        'target_column': entity_schema.get('target_column', '')
    }
    
    # 3. task_type별 추가 설정 복원
    task_type = entity_schema.get('task_type', '')
    
    if task_type == 'causal':
        # Causal 모델용 설정 추가
        new_data_interface.update({
            'treatment_column': 'treatment_group',  # 기본값
            'treatment_value': 'treatment'
        })
    elif task_type == 'classification':
        # Classification 모델용 설정 추가
        new_data_interface.update({
            'class_weight': 'balanced',
            'average': 'weighted'
        })
    elif task_type == 'regression':
        # Regression 모델용 설정 (필요시)
        pass
    elif task_type == 'clustering':
        # Clustering 모델용 설정 (필요시)
        pass
    
    # 4. 구조 업데이트
    data['model']['loader']['entity_schema'] = new_entity_schema
    data['model']['data_interface'] = new_data_interface
    
    # 5. 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    return True

def main():
    """모든 Recipe 파일 처리"""
    
    recipe_dir = Path('recipes')
    fixed_count = 0
    
    print("🚀 Recipe 구조 복원 시작")
    
    # 모든 YAML 파일 찾기
    yaml_files = list(recipe_dir.rglob('*.yaml'))
    
    print(f"📁 처리할 Recipe 파일: {len(yaml_files)}개")
    
    for yaml_file in yaml_files:
        try:
            if fix_recipe_structure(yaml_file):
                print(f"✅ {yaml_file}")
                fixed_count += 1
            else:
                print(f"⏭️  {yaml_file} (entity_schema 없음)")
        except Exception as e:
            print(f"❌ {yaml_file}: {e}")
    
    print(f"\n🎯 Recipe 구조 복원 완료: {fixed_count}개 파일 수정")
    
    # 검증
    print("\n📊 복원 결과 검증:")
    
    entity_schema_count = 0
    data_interface_count = 0
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'entity_schema:' in content:
                    entity_schema_count += 1
                if 'data_interface:' in content:
                    data_interface_count += 1
        except:
            pass
    
    print(f"- entity_schema 있는 Recipe: {entity_schema_count}개")
    print(f"- data_interface 있는 Recipe: {data_interface_count}개")

if __name__ == '__main__':
    main() 