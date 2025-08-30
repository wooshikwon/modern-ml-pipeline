# tests/unit/cli/test_enhanced_validate.py
"""
M04-3-1: Enhanced Validate Command TDD Tests
다층 검증 아키텍처 (syntax/schema/connectivity/execution) 테스트

CLAUDE.md 원칙:
- TDD: RED → GREEN → REFACTOR
- 테스트 명명: test_<모듈>__<행동>__<기대>()
- Given/When/Then 주석 권장
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from src.cli.commands import app

# conftest.py fixture 의존성 제거 - 완전히 격리된 테스트


class TestEnhancedValidate:
    """M04-3-1: 다층 검증 아키텍처 테스트"""
    
    def setup_method(self):
        """각 테스트 전 설정"""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """각 테스트 후 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =============================================================================
    # Layer 1: Syntax 검증 테스트
    # =============================================================================
    
    def test_validate__syntax_layer_invalid_yaml__should_return_syntax_error(self):
        """구문 오류가 있는 Recipe 파일은 syntax 검증 실패해야 함"""
        # Given: 잘못된 YAML 구문을 가진 Recipe 파일
        invalid_recipe = self.temp_dir / "invalid_syntax.yaml"
        invalid_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
    invalid_indentation: true  # 잘못된 들여쓰기
  preprocessor:
""")
        
        # When: validate 명령 실행 
        result = self.runner.invoke(app, ['validate', str(invalid_recipe), '--level', 'basic'])
        
        # Then: syntax 오류 감지 및 구체적 메시지 제공
        assert result.exit_code == 1
        assert "❌ [1/4] Syntax 검증: YAML 구문 오류" in result.stdout
        assert "들여쓰기" in result.stdout or "indentation" in result.stdout
    
    def test_validate__syntax_layer_missing_required_fields__should_return_field_error(self):
        """필수 필드가 없는 Recipe 파일은 syntax 검증 실패해야 함"""
        # Given: model 필드가 누락된 Recipe 파일
        incomplete_recipe = self.temp_dir / "missing_model.yaml"
        incomplete_recipe.write_text("""
# model 필드 완전 누락
preprocessor:
  column_transforms: {}
""")
        
        # When: validate 명령 실행
        result = self.runner.invoke(app, ['validate', str(incomplete_recipe), '--level', 'basic'])
        
        # Then: 필수 필드 누락 오류
        assert result.exit_code == 1
        assert "❌ [1/4] Syntax 검증: 필수 필드 누락" in result.stdout
        assert "'model' 섹션이 필요합니다" in result.stdout

    # =============================================================================
    # Layer 2: Schema 검증 테스트 
    # =============================================================================
    
    def test_validate__schema_layer_invalid_class_path__should_return_schema_error(self):
        """잘못된 class_path는 schema 검증 실패해야 함"""
        # Given: 존재하지 않는 클래스 경로를 가진 Recipe 파일
        invalid_class_recipe = self.temp_dir / "invalid_class.yaml"
        invalid_class_recipe.write_text("""
model:
  class_path: non.existent.ModelClass  # 존재하지 않는 클래스
  loader:
    adapter: storage
    source_uri: data/sample.csv
""")
        
        # When: validate 명령 실행 (schema 레벨까지)
        result = self.runner.invoke(app, ['validate', str(invalid_class_recipe), '--level', 'basic'])
        
        # Then: schema 검증 실패
        assert result.exit_code == 1
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "❌ [2/4] Schema 검증: 클래스 경로 오류" in result.stdout
        assert "non.existent.ModelClass" in result.stdout

    def test_validate__schema_layer_missing_target_column__should_suggest_fix(self):
        """target_column 누락 시 구체적 해결책 제공해야 함"""
        # Given: target_column이 누락된 분류 Recipe 파일
        missing_target_recipe = self.temp_dir / "missing_target.yaml" 
        missing_target_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      # target_column 누락!
""")
        
        # When: validate 명령 실행 (fix-suggestions 플래그 포함)
        result = self.runner.invoke(app, ['validate', str(missing_target_recipe), '--fix-suggestions'])
        
        # Then: 구체적 수정 방법 제안
        assert result.exit_code == 1
        assert "❌ [2/4] Schema 검증: target_column 누락" in result.stdout
        assert "🚀 해결 방법:" in result.stdout
        assert "target_column: 'your_target_column_name'" in result.stdout
        assert "entity_schema 섹션에 추가하세요" in result.stdout

    # =============================================================================
    # Layer 3: Connectivity 검증 테스트
    # =============================================================================
    
    def test_validate__connectivity_layer_mlflow_connection__should_test_real_connection(self):
        """MLflow 연결성 검증은 실제 서버 연결을 테스트해야 함"""
        # Given: 유효한 Recipe 파일 (syntax, schema 정상)
        valid_recipe = self.temp_dir / "valid_recipe.yaml"
        valid_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      target_column: target
""")
        
        # When: validate 명령 실행 (full level = connectivity 검증 포함)
        with patch('src.health.mlflow.MLflowChecker') as mock_mlflow_checker:
            # MLflow 서버 연결 실패 시뮬레이션
            mock_checker_instance = MagicMock()
            mock_checker_instance.run_health_check.return_value = MagicMock(
                is_healthy=False,
                details=["MLflow 서버 연결 실패: Connection refused"]
            )
            mock_mlflow_checker.return_value = mock_checker_instance
            
            result = self.runner.invoke(app, ['validate', str(valid_recipe), '--level', 'full'])
        
        # Then: connectivity 검증 실행 및 MLflow 연결 실패 보고
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "✅ [2/4] Schema 검증: 스키마 유효" in result.stdout  
        assert "❌ [3/4] Connectivity 검증: MLflow 연결 실패" in result.stdout
        assert "Connection refused" in result.stdout
        assert result.exit_code == 1

    def test_validate__connectivity_layer_feature_store_available__should_pass(self):
        """Feature Store 연결 가능 시 connectivity 검증 통과해야 함"""
        # Given: Feature Store를 사용하는 Recipe 파일
        fs_recipe = self.temp_dir / "feature_store_recipe.yaml"
        fs_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: sql 
    source_uri: data/query.sql
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      target_column: target
  augmenter:
    type: feature_store
    features:
      - feature_namespace: user_profile
        features: [age, region]
""")
        
        # When: validate 명령 실행 (모든 외부 서비스 연결 정상)
        with patch('src.health.external.ExternalServicesChecker') as mock_fs_checker:
            mock_checker_instance = MagicMock()
            mock_checker_instance.run_health_check.return_value = MagicMock(
                is_healthy=True,
                details=["✅ Feature Store 연결 정상"]
            )
            mock_fs_checker.return_value = mock_checker_instance
            
            with patch('src.health.mlflow.MLflowChecker') as mock_mlflow_checker:
                mock_mlflow_instance = MagicMock()
                mock_mlflow_instance.run_health_check.return_value = MagicMock(
                    is_healthy=True,
                    details=["✅ MLflow 서버 정상 연결"]
                )
                mock_mlflow_checker.return_value = mock_mlflow_instance
                
                result = self.runner.invoke(app, ['validate', str(fs_recipe), '--level', 'full'])
        
        # Then: connectivity 검증 통과
        assert "✅ [3/4] Connectivity 검증: 외부 서비스 정상" in result.stdout

    # =============================================================================
    # Layer 4: Execution 검증 테스트
    # =============================================================================
    
    def test_validate__execution_layer_component_creation__should_test_factory(self):
        """execution 검증은 실제 컴포넌트 생성을 테스트해야 함"""
        # Given: 모든 이전 단계를 통과한 Recipe 파일
        complete_recipe = self.temp_dir / "complete_recipe.yaml"
        complete_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      target_column: target
  preprocessor:
    column_transforms: {}
""")
        
        # When: validate 명령 실행 (full level = execution 검증 포함)
        with patch('src.engine.factory.Factory') as mock_factory:
            # Factory 컴포넌트 생성 성공 시뮬레이션
            mock_factory_instance = MagicMock()
            mock_factory_instance.create_model.return_value = MagicMock()
            mock_factory_instance.create_data_adapter.return_value = MagicMock()
            mock_factory_instance.create_preprocessor.return_value = MagicMock()
            mock_factory.return_value = mock_factory_instance
            
            # 연결성 검증도 통과하도록 설정
            with patch('src.health.mlflow.MLflowChecker') as mock_mlflow:
                mock_mlflow_instance = MagicMock()
                mock_mlflow_instance.run_health_check.return_value = MagicMock(is_healthy=True)
                mock_mlflow.return_value = mock_mlflow_instance
                
                result = self.runner.invoke(app, ['validate', str(complete_recipe), '--level', 'full'])
        
        # Then: execution 검증 통과, 전체 검증 성공
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "✅ [2/4] Schema 검증: 스키마 유효" in result.stdout
        assert "✅ [3/4] Connectivity 검증: 외부 서비스 정상" in result.stdout
        assert "✅ [4/4] Execution 검증: 컴포넌트 생성 가능" in result.stdout
        assert "🎉 전체 검증 성공: Recipe 파일이 모든 요구사항을 충족합니다" in result.stdout
        assert result.exit_code == 0

    def test_validate__execution_layer_factory_failure__should_report_specific_error(self):
        """Factory 컴포넌트 생성 실패 시 구체적 오류 보고해야 함"""
        # Given: syntax/schema/connectivity는 정상이지만 execution 실패하는 Recipe
        factory_fail_recipe = self.temp_dir / "factory_fail.yaml"
        factory_fail_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  params:
    n_estimators: "invalid_string_for_int"  # 타입 오류 유발
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      target_column: target
""")
        
        # When: validate 명령 실행
        with patch('src.engine.factory.Factory') as mock_factory:
            # Factory에서 파라미터 타입 오류 발생 시뮬레이션
            mock_factory_instance = MagicMock()
            mock_factory_instance.create_model.side_effect = ValueError(
                "n_estimators must be an integer, got <class 'str'>"
            )
            mock_factory.return_value = mock_factory_instance
            
            # 이전 단계들은 모두 성공하도록 설정
            with patch('src.health.mlflow.MLflowChecker') as mock_mlflow:
                mock_mlflow_instance = MagicMock()
                mock_mlflow_instance.run_health_check.return_value = MagicMock(is_healthy=True)
                mock_mlflow.return_value = mock_mlflow_instance
                
                result = self.runner.invoke(app, ['validate', str(factory_fail_recipe), '--level', 'full', '--fix-suggestions'])
        
        # Then: execution 검증 실패, 구체적 해결책 제시
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "✅ [2/4] Schema 검증: 스키마 유효" in result.stdout
        assert "✅ [3/4] Connectivity 검증: 외부 서비스 정상" in result.stdout
        assert "❌ [4/4] Execution 검증: 컴포넌트 생성 실패" in result.stdout
        assert "n_estimators must be an integer" in result.stdout
        assert "🚀 해결 방법:" in result.stdout
        assert "n_estimators: 100  # 문자열이 아닌 정수로 설정" in result.stdout
        assert result.exit_code == 1

    # =============================================================================
    # CLI 옵션 테스트
    # =============================================================================
    
    def test_validate__basic_level_option__should_skip_connectivity_and_execution(self):
        """--level basic 옵션은 syntax/schema만 검증해야 함"""
        # Given: 완전한 Recipe 파일
        recipe = self.temp_dir / "complete.yaml"
        recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id] 
      timestamp_column: event_ts
      target_column: target
""")
        
        # When: basic level 검증 실행
        result = self.runner.invoke(app, ['validate', str(recipe), '--level', 'basic'])
        
        # Then: syntax/schema만 검증, connectivity/execution 건너뜀
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "✅ [2/4] Schema 검증: 스키마 유효" in result.stdout
        assert "⏭️  [3/4] Connectivity 검증: 건너뜀 (basic 모드)" in result.stdout
        assert "⏭️  [4/4] Execution 검증: 건너뜀 (basic 모드)" in result.stdout
        assert "✅ Basic 검증 완료" in result.stdout
        assert result.exit_code == 0

    def test_validate__no_fix_suggestions_flag__should_not_show_detailed_solutions(self):
        """--fix-suggestions 없이 실행시 간단한 오류만 표시해야 함"""  
        # Given: 오류가 있는 Recipe 파일
        error_recipe = self.temp_dir / "error.yaml"
        error_recipe.write_text("""
model:
  class_path: non.existent.Class
""")
        
        # When: fix-suggestions 없이 validate 실행
        result = self.runner.invoke(app, ['validate', str(error_recipe)])
        
        # Then: 간단한 오류 메시지만 표시, 상세 해결책 없음
        assert "❌ [2/4] Schema 검증: 클래스 경로 오류" in result.stdout
        assert "🚀 해결 방법:" not in result.stdout  # 상세 해결책 표시 안됨
        assert "더 구체적인 해결 방법이 필요하다면:" in result.stdout
        assert "--fix-suggestions 옵션을 사용하세요" in result.stdout
        assert result.exit_code == 1

    # =============================================================================
    # 통합 시나리오 테스트  
    # =============================================================================
    
    def test_validate__comprehensive_scenario__should_handle_multiple_errors(self):
        """복수 오류가 있는 Recipe에 대해 우선순위별 해결책 제시해야 함"""
        # Given: 여러 레벨에서 오류가 있는 Recipe 파일
        multi_error_recipe = self.temp_dir / "multi_error.yaml"
        multi_error_recipe.write_text("""
model:
  class_path: non.existent.ModelClass
  params:
    n_estimators: "should_be_int"
  loader:
    adapter: storage
    source_uri: data/nonexistent.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      # target_column 누락
""")
        
        # When: full level + fix-suggestions로 검증 실행
        with patch('src.health.mlflow.MLflowChecker') as mock_mlflow:
            mock_mlflow_instance = MagicMock()
            mock_mlflow_instance.run_health_check.return_value = MagicMock(
                is_healthy=False,
                details=["MLflow 서버 연결 실패"]
            )
            mock_mlflow.return_value = mock_mlflow_instance
            
            result = self.runner.invoke(app, ['validate', str(multi_error_recipe), '--level', 'full', '--fix-suggestions'])
        
        # Then: 단계별 오류 진단 및 우선순위별 해결책 제시
        assert "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert "❌ [2/4] Schema 검증: 3개 문제 발견" in result.stdout
        assert "❌ [3/4] Connectivity 검증: MLflow 연결 실패" in result.stdout  
        assert "⚠️  [4/4] Execution 검증: 건너뜀 (이전 단계 실패)" in result.stdout
        
        # 우선순위별 해결책 표시
        assert "🚀 우선순위별 해결 방법:" in result.stdout
        assert "1. [HIGH] Schema 오류 수정:" in result.stdout
        assert "2. [MEDIUM] 외부 서비스 연결 복구:" in result.stdout
        assert result.exit_code == 1

    def test_validate__legacy_compatibility__should_maintain_existing_behavior(self):
        """기존 validate 명령어 호환성 유지해야 함"""
        # Given: 기존 방식으로 작성된 유효한 Recipe 파일
        legacy_recipe = self.temp_dir / "legacy.yaml"  
        legacy_recipe.write_text("""
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  loader:
    adapter: storage
    source_uri: data/sample.csv
    entity_schema:
      entity_columns: [user_id]
      timestamp_column: event_ts
      target_column: target
""")
        
        # When: 플래그 없이 기존 방식으로 validate 실행
        result = self.runner.invoke(app, ['validate', str(legacy_recipe)])
        
        # Then: 기본 동작은 basic level, 간단한 성공/실패 메시지
        assert "✅ 성공: 모든 설정 파일이 유효합니다." in result.stdout or "✅ [1/4] Syntax 검증: YAML 구문 정상" in result.stdout
        assert result.exit_code == 0