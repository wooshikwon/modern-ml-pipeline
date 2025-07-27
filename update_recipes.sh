#!/bin/bash

# 🔧 Recipe 구조 변경: data_interface → entity_schema + ml_task 분리
echo "🚀 Recipe 구조 변경 시작: data_interface → entity_schema + ml_task"

# 1. recipes/ 디렉토리의 모든 YAML 파일에서 구조 변경
echo "📁 Recipe 파일 구조 업데이트..."

# data_interface를 entity_schema로 변경
find recipes/ -name "*.yaml" -type f -exec sed -i '' 's/data_interface:/entity_schema:/g' {} \;

echo "✅ 변경된 Recipe 파일들:"
find recipes/ -name "*.yaml" -type f -exec grep -l "entity_schema:" {} \;

# 2. 변경 검증
echo "📊 변경 결과 요약:"
echo "- entity_schema 참조 개수: $(grep -r "entity_schema:" recipes/ --include="*.yaml" | wc -l)"
echo "- 남은 data_interface 참조: $(grep -r "data_interface:" recipes/ --include="*.yaml" | wc -l || echo "0")"

echo "🎯 Recipe 구조 변경 완료!" 