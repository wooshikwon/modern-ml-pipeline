#!/bin/bash

# 🔧 Import 참조 수정: 클래스명 변경 반영
echo "🚀 Import 참조 수정 시작"

# 1. DataInterfaceSettings → MLTaskSettings
echo "📁 DataInterfaceSettings → MLTaskSettings..."
find src/ tests/ -name "*.py" -type f -exec sed -i '' 's/DataInterfaceSettings/MLTaskSettings/g' {} \;

# 2. LoaderDataInterface → EntitySchema  
echo "📁 LoaderDataInterface → EntitySchema..."
find src/ tests/ -name "*.py" -type f -exec sed -i '' 's/LoaderDataInterface/EntitySchema/g' {} \;

# 3. 변경 검증
echo "✅ 변경 결과:"
echo "- MLTaskSettings 참조: $(grep -r "MLTaskSettings" src/ tests/ --include="*.py" | wc -l)"
echo "- EntitySchema 참조: $(grep -r "EntitySchema" src/ tests/ --include="*.py" | wc -l)"
echo "- 남은 DataInterfaceSettings: $(grep -r "DataInterfaceSettings" src/ tests/ --include="*.py" | wc -l || echo "0")"
echo "- 남은 LoaderDataInterface: $(grep -r "LoaderDataInterface" src/ tests/ --include="*.py" | wc -l || echo "0")"

echo "🎯 Import 참조 수정 완료!" 