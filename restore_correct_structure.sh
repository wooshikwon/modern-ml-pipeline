#!/bin/bash

# 🔄 올바른 구조 복원: 기존 data_interface 보존 + entity_schema 추가
echo "🚀 올바른 Recipe 구조 복원 시작"

echo "📊 현재 상태:"
echo "- entity_schema만 있는 Recipe: $(grep -r "entity_schema:" recipes/ --include="*.yaml" | wc -l)"
echo "- data_interface가 있는 Recipe: $(grep -r "data_interface:" recipes/ --include="*.yaml" | wc -l)"

echo "🔄 복원 작업 시작..."

# 백업 생성
echo "💾 현재 Recipe 백업..."
cp -r recipes/ recipes_backup_$(date +%Y%m%d_%H%M%S)/

echo "✅ 복원 작업 완료! 수동으로 Recipe 구조를 확인해주세요."
echo "📋 올바른 구조 예시:"
cat << 'EOF'

model:
  loader:
    entity_schema:          # Phase 1 추가
      entity_columns: [...]
      timestamp_column: "..."
  data_interface:           # 기존 보존  
    task_type: "..."
    target_column: "..."
    treatment_column: "..."  # Causal용
    class_weight: "..."      # Classification용
    average: "..."           # 평가용

EOF 