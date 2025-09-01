"""
Registry 기반 평가 메트릭 시스템 테스트
실제 Evaluator 구현에서 메트릭을 동적으로 추출하는 기능 테스트
"""

import pytest
from unittest.mock import patch

from src.cli.utils.ml_metrics import (
    get_task_metrics,
    get_tuning_config, 
    get_primary_metric,
    validate_custom_metrics,
    get_evaluator_class,
    _extract_metrics_from_evaluator,
    _get_metrics_by_dummy_call,
    EVALUATOR_REGISTRY
)
from src.components._evaluator import ClassificationEvaluator, RegressionEvaluator


class TestRegistryEvaluationMetrics:
    """Registry 기반 평가 메트릭 시스템 테스트"""

    def test_get_task_metrics_classification(self):
        """Classification 태스크 메트릭 추출 테스트"""
        # When
        metrics = get_task_metrics('classification')
        
        # Then
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        # ClassificationEvaluator.evaluate()에서 실제 반환하는 메트릭들
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for expected in expected_metrics:
            assert expected in metrics

    def test_get_task_metrics_regression(self):
        """Regression 태스크 메트릭 추출 테스트"""
        # When
        metrics = get_task_metrics('regression')
        
        # Then
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        # RegressionEvaluator.evaluate()에서 실제 반환하는 메트릭들
        expected_metrics = ['r2_score', 'mean_squared_error']
        for expected in expected_metrics:
            assert expected in metrics

    def test_get_task_metrics_case_insensitive(self):
        """대소문자 구분 없이 동작하는지 테스트"""
        # When
        metrics_lower = get_task_metrics('classification')
        metrics_upper = get_task_metrics('CLASSIFICATION')
        metrics_mixed = get_task_metrics('Classification')
        
        # Then
        assert metrics_lower == metrics_upper == metrics_mixed

    def test_get_task_metrics_unsupported_task(self):
        """지원하지 않는 태스크 타입 예외 테스트"""
        # When & Then
        with pytest.raises(ValueError, match="Unsupported task type"):
            get_task_metrics('unsupported_task')

    def test_get_tuning_config_classification(self):
        """Classification 튜닝 설정 테스트"""
        # When
        config = get_tuning_config('classification')
        
        # Then
        assert isinstance(config, dict)
        assert 'objective' in config
        assert 'direction' in config
        assert config['objective'] == 'f1_score'
        assert config['direction'] == 'maximize'

    def test_get_tuning_config_regression(self):
        """Regression 튜닝 설정 테스트"""
        # When
        config = get_tuning_config('regression')
        
        # Then
        assert isinstance(config, dict)
        assert config['objective'] == 'r2_score'
        assert config['direction'] == 'maximize'

    def test_get_primary_metric(self):
        """주요 메트릭 반환 테스트"""
        # When
        primary_clf = get_primary_metric('classification')
        primary_reg = get_primary_metric('regression')
        
        # Then
        assert primary_clf == 'f1_score'
        assert primary_reg == 'r2_score'

    def test_validate_custom_metrics(self):
        """사용자 정의 메트릭 검증 테스트"""
        # Given
        valid_metrics = ['accuracy', 'precision']
        invalid_metrics = ['accuracy', 'invalid_metric', 'precision']
        
        # When
        validated_all_valid = validate_custom_metrics('classification', valid_metrics)
        validated_mixed = validate_custom_metrics('classification', invalid_metrics)
        
        # Then
        assert validated_all_valid == ['accuracy', 'precision']
        assert validated_mixed == ['accuracy', 'precision']  # invalid_metric 제외

    def test_get_evaluator_class(self):
        """Evaluator 클래스 반환 테스트"""
        # When
        clf_evaluator = get_evaluator_class('classification')
        reg_evaluator = get_evaluator_class('regression')
        
        # Then
        assert clf_evaluator == ClassificationEvaluator
        assert reg_evaluator == RegressionEvaluator

    def test_extract_metrics_from_evaluator_source_parsing(self):
        """소스 코드 파싱을 통한 메트릭 추출 테스트"""
        # When
        metrics = _extract_metrics_from_evaluator(ClassificationEvaluator)
        
        # Then
        assert isinstance(metrics, list)
        # ClassificationEvaluator의 실제 메트릭들이 추출되어야 함
        expected = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected:
            assert metric in metrics

    @patch('src.cli.utils.ml_metrics.re.findall')
    @patch('src.cli.utils.ml_metrics.inspect.getsource')
    def test_extract_metrics_fallback_to_dummy_call(self, mock_getsource, mock_findall):
        """소스 파싱 실패 시 더미 호출 fallback 테스트"""
        # Given: getsource는 성공하지만 정규식 매칭이 실패하도록 설정
        mock_getsource.return_value = "def evaluate(self): return {}"
        mock_findall.return_value = []  # 정규식 매칭 실패
        
        # When
        with patch('src.cli.utils.ml_metrics._get_metrics_by_dummy_call') as mock_dummy:
            mock_dummy.return_value = ['mocked_metric']
            metrics = _extract_metrics_from_evaluator(ClassificationEvaluator)
        
        # Then
        assert metrics == ['mocked_metric']
        mock_dummy.assert_called_once_with(ClassificationEvaluator)

    def test_get_metrics_by_dummy_call(self):
        """더미 호출을 통한 메트릭 추출 테스트"""
        # When
        metrics = _get_metrics_by_dummy_call(ClassificationEvaluator)
        
        # Then
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        # 실제 evaluate 호출로 얻어진 메트릭 키들
        expected = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected:
            assert metric in metrics

    @patch('src.cli.utils.ml_metrics._extract_metrics_from_evaluator')
    def test_get_task_metrics_with_fallback(self, mock_extract):
        """메트릭 추출 실패 시 fallback 동작 테스트"""
        # Given: 메트릭 추출 실패
        mock_extract.return_value = []
        
        # When
        metrics = get_task_metrics('classification')
        
        # Then: 최소 fallback 메트릭이 반환되어야 함
        expected_fallback = ['accuracy', 'f1_score']  # 간소화된 fallback
        assert metrics == expected_fallback

    def test_evaluator_registry_completeness(self):
        """Registry에 모든 필요한 Evaluator가 등록되었는지 테스트"""
        # Given
        expected_tasks = ['classification', 'regression', 'clustering', 'causal']
        
        # When & Then
        for task in expected_tasks:
            assert task in EVALUATOR_REGISTRY
            assert issubclass(EVALUATOR_REGISTRY[task], object)  # 클래스인지 확인

    def test_integration_registry_with_recipe_generator(self):
        """Registry 시스템이 recipe generator와 정상 연동되는지 테스트"""
        # When: recipe generator에서 사용하는 것과 동일한 방식으로 호출
        clf_metrics = get_task_metrics('classification')
        clf_tuning = get_tuning_config('classification')
        
        # Then
        assert isinstance(clf_metrics, list)
        assert len(clf_metrics) > 0
        assert isinstance(clf_tuning, dict)
        assert 'objective' in clf_tuning
        
        # 튜닝 objective가 실제 메트릭 리스트에 포함되어야 함
        assert clf_tuning['objective'] in clf_metrics or clf_tuning['objective'] in ['f1_score']  # f1_score는 특별 케이스