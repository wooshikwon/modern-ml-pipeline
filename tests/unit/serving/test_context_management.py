"""서빙 컨텍스트 관리 테스트 - Real + Mock 하이브리드 테스트

Phase 3.3: _context.py의 AppContext 클래스 상태 관리 기능을 검증합니다.

테스트 전략:
- AppContext 클래스는 Real Implementation (단순한 상태 관리)
- MLflow 모델과 Settings는 Mock 처리
- 전역 상태 관리와 동적 스키마 생성 검증
- Factory 패턴과 일관된 구조 유지
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from src.serving._context import AppContext, app_context
from tests.factories.settings_factory import SettingsFactory


class TestAppContext:
    """AppContext 상태 관리 테스트"""
    
    def test_app_context_initialization(self, test_factories):
        """AppContext 초기화 테스트"""
        # Given: 새로운 AppContext 인스턴스 생성
        # When: AppContext 초기화
        context = AppContext()
        
        # Then: 초기 상태 검증
        assert context.model is None
        assert context.model_uri == ""
        assert context.settings is None
        assert context.PredictionRequest is not None
        assert context.BatchPredictionRequest is not None
        
        # 기본 스키마가 BaseModel을 상속받는지 검증
        assert issubclass(context.PredictionRequest, BaseModel)
        assert issubclass(context.BatchPredictionRequest, BaseModel)
        
        # 기본 스키마 이름 검증
        assert context.PredictionRequest.__name__ == "DefaultPredictionRequest"
        assert context.BatchPredictionRequest.__name__ == "DefaultBatchPredictionRequest"
    
    def test_app_context_model_assignment(self, test_factories):
        """AppContext 모델 할당 테스트"""
        # Given: AppContext와 Mock 모델
        context = AppContext()
        mock_model = MagicMock()
        model_uri = "runs:/test_model_123/model"
        
        # When: 모델 할당
        context.model = mock_model
        context.model_uri = model_uri
        
        # Then: 모델 상태 검증
        assert context.model is mock_model
        assert context.model_uri == model_uri
    
    def test_app_context_settings_assignment(self, test_factories):
        """AppContext 설정 할당 테스트"""
        # Given: AppContext와 Mock 설정
        context = AppContext()
        mock_settings = MagicMock()
        
        # When: 설정 할당
        context.settings = mock_settings
        
        # Then: 설정 상태 검증
        assert context.settings is mock_settings
    
    def test_app_context_dynamic_schema_assignment(self, test_factories):
        """AppContext 동적 스키마 할당 테스트"""
        # Given: AppContext와 커스텀 스키마
        context = AppContext()
        
        # 커스텀 Pydantic 모델 생성 (실제 schemas.py 방식 모방)
        from pydantic import create_model, Field
        
        CustomPredictionRequest = create_model(
            "CustomPredictionRequest",
            user_id=(str, Field(..., description="사용자 ID")),
            campaign_id=(str, Field(..., description="캠페인 ID"))
        )
        
        CustomBatchPredictionRequest = create_model(
            "CustomBatchPredictionRequest", 
            samples=(list, Field(..., description="예측 샘플들"))
        )
        
        # When: 동적 스키마 할당
        context.PredictionRequest = CustomPredictionRequest
        context.BatchPredictionRequest = CustomBatchPredictionRequest
        
        # Then: 스키마 상태 검증
        assert context.PredictionRequest.__name__ == "CustomPredictionRequest"
        assert context.BatchPredictionRequest.__name__ == "CustomBatchPredictionRequest"
        
        # 필드 검증
        prediction_fields = context.PredictionRequest.model_fields
        assert "user_id" in prediction_fields
        assert "campaign_id" in prediction_fields
        
        batch_fields = context.BatchPredictionRequest.model_fields
        assert "samples" in batch_fields
    
    def test_app_context_complete_state_setup(self, test_factories):
        """AppContext 완전한 상태 설정 테스트"""
        # Given: 모든 컴포넌트를 가진 완전한 설정
        context = AppContext()
        
        # Mock 컴포넌트들
        mock_model = MagicMock()
        mock_model.model_class_path = "sklearn.ensemble.RandomForestClassifier"
        
        settings_dict = test_factories['settings'].create_classification_settings("test")
        mock_settings = MagicMock()
        mock_settings.recipe.model.name = "test_model"
        
        model_uri = "runs:/complete_test/model"
        
        # 커스텀 스키마
        from pydantic import create_model, Field
        PredictionRequest = create_model(
            "TestPredictionRequest",
            feature1=(float, Field(...)),
            feature2=(float, Field(...))
        )
        
        # When: 완전한 상태 설정
        context.model = mock_model
        context.model_uri = model_uri
        context.settings = mock_settings
        context.PredictionRequest = PredictionRequest
        
        # Then: 모든 상태 검증
        assert context.model is mock_model
        assert context.model_uri == model_uri
        assert context.settings is mock_settings
        assert context.PredictionRequest.__name__ == "TestPredictionRequest"
        
        # 스키마 필드 검증
        fields = context.PredictionRequest.model_fields
        assert "feature1" in fields
        assert "feature2" in fields
    
    def test_global_app_context_instance(self, test_factories):
        """전역 app_context 인스턴스 테스트"""
        # Given: 전역 app_context 인스턴스
        # When: app_context 접근
        global_context = app_context
        
        # Then: 인스턴스 검증
        assert global_context is not None
        assert isinstance(global_context, AppContext)
        
        # 초기 상태 검증
        assert global_context.model is None or hasattr(global_context.model, 'predict')
        assert isinstance(global_context.model_uri, str)
        assert global_context.settings is None or hasattr(global_context.settings, 'recipe')
    
    def test_app_context_state_isolation(self, test_factories):
        """AppContext 인스턴스 격리 테스트"""
        # Given: 두 개의 별도 AppContext 인스턴스
        context1 = AppContext()
        context2 = AppContext()
        
        # Mock 객체들
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_settings1 = MagicMock()
        mock_settings2 = MagicMock()
        
        # When: 각각 다른 상태로 설정
        context1.model = mock_model1
        context1.model_uri = "runs:/model1/model"
        context1.settings = mock_settings1
        
        context2.model = mock_model2  
        context2.model_uri = "runs:/model2/model"
        context2.settings = mock_settings2
        
        # Then: 인스턴스 격리 검증
        assert context1.model is not context2.model
        assert context1.model_uri != context2.model_uri
        assert context1.settings is not context2.settings
        
        assert context1.model is mock_model1
        assert context2.model is mock_model2
        assert context1.model_uri == "runs:/model1/model"
        assert context2.model_uri == "runs:/model2/model"
    
    @patch('src.serving._context.create_model')
    def test_app_context_schema_creation_mock(self, mock_create_model, test_factories):
        """AppContext 스키마 생성 Mock 검증"""
        # Given: create_model Mock 설정
        mock_schema = MagicMock()
        mock_create_model.return_value = mock_schema
        
        # When: AppContext 초기화 (create_model 호출됨)
        context = AppContext()
        
        # Then: create_model 호출 검증
        assert mock_create_model.call_count >= 2  # PredictionRequest, BatchPredictionRequest
        
        # 호출 인수 검증
        calls = mock_create_model.call_args_list
        call_names = [call[0][0] for call in calls]
        assert "DefaultPredictionRequest" in call_names
        assert "DefaultBatchPredictionRequest" in call_names
    
    def test_app_context_state_reset(self, test_factories):
        """AppContext 상태 리셋 테스트"""
        # Given: 상태가 설정된 AppContext
        context = AppContext()
        context.model = MagicMock()
        context.model_uri = "runs:/test/model"
        context.settings = MagicMock()
        
        # When: 상태 리셋
        context.model = None
        context.model_uri = ""
        context.settings = None
        
        # Then: 리셋 검증
        assert context.model is None
        assert context.model_uri == ""
        assert context.settings is None
        
        # 스키마는 유지되는지 검증
        assert context.PredictionRequest is not None
        assert context.BatchPredictionRequest is not None


class TestAppContextIntegration:
    """AppContext 통합 시나리오 테스트"""
    
    def test_app_context_mlflow_model_integration(self, test_factories):
        """AppContext와 MLflow 모델 통합 테스트"""
        # Given: AppContext와 MLflow 스타일 Mock 모델
        context = AppContext()
        
        # MLflow PyFuncModel 스타일 Mock
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock()
        mock_model.unwrap_python_model.return_value = MagicMock()
        
        model_uri = "runs:/integration_test/model"
        
        # When: 모델 통합
        context.model = mock_model
        context.model_uri = model_uri
        
        # Then: 통합 상태 검증
        assert context.model is mock_model
        assert context.model_uri == model_uri
        assert hasattr(context.model, 'predict')
        assert hasattr(context.model, 'unwrap_python_model')
    
    def test_app_context_settings_integration(self, test_factories):
        """AppContext와 Settings 통합 테스트"""
        # Given: AppContext와 실제 Settings 구조의 Mock
        context = AppContext()
        
        # Settings 구조를 모방한 Mock
        mock_settings = MagicMock()
        mock_settings.recipe.model.name = "test_model"
        mock_settings.artifact_stores = {"predictions": MagicMock()}
        
        # When: 설정 통합
        context.settings = mock_settings
        
        # Then: 통합 검증
        assert context.settings is mock_settings
        assert context.settings.recipe.model.name == "test_model"
        assert hasattr(context.settings, 'artifact_stores')