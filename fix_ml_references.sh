#!/bin/bash

# 🔧 ML 설정 참조 수정: loader.entity_schema → model.data_interface (ML 설정만)
echo "🚀 ML 설정 참조 수정 시작"

# 1. Trainer.py 수정 - treatment_column, target_column 등
echo "📁 Trainer ML 설정 참조 수정..."
sed -i '' 's/data_interface = self\.settings\.recipe\.model\.loader\.entity_schema/data_interface = self.settings.recipe.model.data_interface/g' src/components/trainer.py

# 2. Factory.py 수정 - task_type은 양쪽 다 있으니 data_interface에서
echo "📁 Factory ML 설정 참조 수정..."
sed -i '' 's/task_type = self\.model_config\.loader\.entity_schema\.task_type/task_type = self.model_config.data_interface.task_type/g' src/engine/factory.py

# 3. Evaluator 생성 시 data_interface 전달
sed -i '' 's/return evaluator_class(self\.model_config\.loader\.entity_schema)/return evaluator_class(self.model_config.data_interface)/g' src/engine/factory.py

echo "✅ ML 설정 참조 수정 완료"

# 검증
echo "📊 수정 결과:"
echo "- model.data_interface 참조 개수: $(grep -r "model\.data_interface" src/ --include="*.py" | wc -l)"
echo "- loader.entity_schema 참조 개수: $(grep -r "loader\.entity_schema" src/ --include="*.py" | wc -l)" 