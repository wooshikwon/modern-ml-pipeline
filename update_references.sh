#!/bin/bash

# 🔧 Phase 1 구조 변경: 일괄 참조 업데이트 스크립트
echo "🚀 Phase 1 구조 변경 시작: loader.data_interface → loader.entity_schema"

# 1. Python 파일들에서 일괄 변경
echo "📁 Python 파일 참조 업데이트..."
find src/ tests/ -name "*.py" -type f -exec sed -i '' 's/loader\.data_interface/loader.entity_schema/g' {} \;

# 2. 변경된 파일들 확인
echo "✅ 변경된 파일 목록:"
grep -r "loader\.entity_schema" src/ tests/ --include="*.py" | cut -d: -f1 | sort | uniq

# 3. 혹시 남은 참조 확인
echo "⚠️  남은 data_interface 참조 확인:"
grep -r "loader\.data_interface" src/ tests/ --include="*.py" || echo "✅ 모든 참조가 성공적으로 변경됨"

echo "🎯 Step 1 완료: 모든 loader.data_interface → loader.entity_schema 변경 완료" 