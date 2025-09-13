"""
Reproducibility utilities comprehensive testing
Follows tests/README.md philosophy with Context classes
Tests for src/utils/core/reproducibility.py

Author: Phase 2A Development
Date: 2025-09-13
"""

import pytest
import os
from unittest.mock import patch, Mock

from src.utils.core.reproducibility import set_global_seeds


class TestGlobalSeedsConfiguration:
    """전역 시드 설정 테스트 - Context 클래스 기반"""

    def test_set_global_seeds_default_value(self, component_test_context):
        """기본 시드 값(42) 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch.dict('os.environ', {}, clear=True):
                with patch('random.seed') as mock_random_seed:
                    with patch('numpy.random.seed') as mock_numpy_seed:
                        # Test default seed value
                        set_global_seeds()

                        # Verify default seed (42) was used
                        assert os.environ.get("PYTHONHASHSEED") == "42"
                        mock_random_seed.assert_called_once_with(42)
                        mock_numpy_seed.assert_called_once_with(42)

    def test_set_global_seeds_custom_value(self, component_test_context):
        """커스텀 시드 값 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            custom_seed = 12345

            with patch.dict('os.environ', {}, clear=True):
                with patch('random.seed') as mock_random_seed:
                    with patch('numpy.random.seed') as mock_numpy_seed:
                        set_global_seeds(custom_seed)

                        # Verify custom seed was used
                        assert os.environ.get("PYTHONHASHSEED") == str(custom_seed)
                        mock_random_seed.assert_called_once_with(custom_seed)
                        mock_numpy_seed.assert_called_once_with(custom_seed)

    def test_set_global_seeds_python_hashseed_environment(self, component_test_context):
        """PYTHONHASHSEED 환경 변수 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            seed_value = 99999

            # Clear environment first
            with patch.dict('os.environ', {}, clear=True):
                set_global_seeds(seed_value)

                # Verify PYTHONHASHSEED was set correctly
                assert "PYTHONHASHSEED" in os.environ
                assert os.environ["PYTHONHASHSEED"] == str(seed_value)

    def test_set_global_seeds_random_module(self, component_test_context):
        """Python random 모듈 시드 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('random.seed') as mock_random_seed:
                test_seed = 7777

                set_global_seeds(test_seed)

                mock_random_seed.assert_called_once_with(test_seed)

    def test_set_global_seeds_numpy_module(self, component_test_context):
        """NumPy random 시드 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('numpy.random.seed') as mock_numpy_seed:
                test_seed = 8888

                set_global_seeds(test_seed)

                mock_numpy_seed.assert_called_once_with(test_seed)


class TestPyTorchSeedsConfiguration:
    """PyTorch 시드 설정 테스트"""

    @patch('src.utils.core.reproducibility.torch', create=True)
    def test_set_global_seeds_pytorch_cpu(self, mock_torch, component_test_context):
        """PyTorch CPU 시드 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_torch.manual_seed = Mock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.cudnn = Mock()

            test_seed = 6666
            set_global_seeds(test_seed)

            # Verify PyTorch CPU seed was set
            mock_torch.manual_seed.assert_called_once_with(test_seed)

    @patch('src.utils.core.reproducibility.torch', create=True)
    def test_set_global_seeds_pytorch_cuda_available(self, mock_torch, component_test_context):
        """CUDA 사용 가능한 환경에서 PyTorch 시드 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_torch.manual_seed = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.manual_seed_all = Mock()
            mock_torch.backends.cudnn = Mock()

            test_seed = 5555
            set_global_seeds(test_seed)

            # Verify both CPU and CUDA seeds were set
            mock_torch.manual_seed.assert_called_once_with(test_seed)
            mock_torch.cuda.manual_seed_all.assert_called_once_with(test_seed)

    @patch('src.utils.core.reproducibility.torch', create=True)
    def test_set_global_seeds_pytorch_cudnn_deterministic(self, mock_torch, component_test_context):
        """PyTorch CUDNN 결정적 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_torch.manual_seed = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.manual_seed_all = Mock()
            mock_torch.backends.cudnn = Mock()

            set_global_seeds(42)

            # Verify CUDNN deterministic settings
            assert mock_torch.backends.cudnn.deterministic is True
            assert mock_torch.backends.cudnn.benchmark is False

    @patch('src.utils.core.reproducibility.torch', create=True)
    def test_set_global_seeds_pytorch_cuda_not_available(self, mock_torch, component_test_context):
        """CUDA 사용 불가능한 환경에서 PyTorch 시드 설정 테스트"""
        with component_test_context.classification_stack() as ctx:
            mock_torch.manual_seed = Mock()
            mock_torch.cuda.is_available.return_value = False
            mock_torch.cuda.manual_seed_all = Mock()
            mock_torch.backends.cudnn = Mock()

            set_global_seeds(42)

            # Verify only CPU seed was set (CUDA seed should not be called)
            mock_torch.manual_seed.assert_called_once_with(42)
            mock_torch.cuda.manual_seed_all.assert_not_called()


class TestExceptionHandling:
    """예외 처리 테스트"""

    def test_set_global_seeds_missing_random_module(self, component_test_context):
        """random 모듈 import 실패 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('random.seed', side_effect=ImportError("random module not available")):
                # Should not raise exception
                set_global_seeds(42)

    def test_set_global_seeds_missing_numpy_module(self, component_test_context):
        """numpy 모듈 import 실패 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('numpy.random.seed', side_effect=ImportError("numpy not available")):
                # Should not raise exception
                set_global_seeds(42)

    def test_set_global_seeds_missing_torch_module(self, component_test_context):
        """torch 모듈 import 실패 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            # Mock torch import failure by making it raise ImportError
            with patch('src.utils.core.reproducibility.torch', side_effect=ImportError("torch not available")):
                # Should not raise exception
                set_global_seeds(42)

    def test_set_global_seeds_os_environ_exception(self, component_test_context):
        """OS 환경 변수 설정 실패 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch.dict('os.environ', {}, clear=True):
                # Mock os.environ assignment to raise exception
                with patch('os.environ.__setitem__', side_effect=Exception("Environment error")):
                    # Should not raise exception
                    set_global_seeds(42)

    def test_set_global_seeds_logger_exception(self, component_test_context):
        """로거 import/사용 실패 시 안전한 처리 테스트"""
        with component_test_context.classification_stack() as ctx:
            with patch('src.utils.core.reproducibility.logger', side_effect=ImportError("logger not available")):
                # Should not raise exception
                set_global_seeds(42)


class TestSeedEffectiveness:
    """시드 설정 효과 검증 테스트"""

    def test_set_global_seeds_reproducible_random(self, component_test_context):
        """random 모듈의 재현성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            import random

            # Set seed and generate random numbers
            set_global_seeds(42)
            first_sequence = [random.random() for _ in range(5)]

            # Reset seed and generate again
            set_global_seeds(42)
            second_sequence = [random.random() for _ in range(5)]

            # Should be identical
            assert first_sequence == second_sequence

    def test_set_global_seeds_reproducible_numpy(self, component_test_context):
        """NumPy의 재현성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            import numpy as np

            # Set seed and generate random array
            set_global_seeds(123)
            first_array = np.random.random(5)

            # Reset seed and generate again
            set_global_seeds(123)
            second_array = np.random.random(5)

            # Should be identical
            np.testing.assert_array_equal(first_array, second_array)

    def test_set_global_seeds_different_seeds_different_results(self, component_test_context):
        """다른 시드 값으로 다른 결과 생성 검증 테스트"""
        with component_test_context.classification_stack() as ctx:
            import random
            import numpy as np

            # Generate with seed 42
            set_global_seeds(42)
            random_42 = [random.random() for _ in range(3)]
            numpy_42 = np.random.random(3)

            # Generate with seed 99
            set_global_seeds(99)
            random_99 = [random.random() for _ in range(3)]
            numpy_99 = np.random.random(3)

            # Results should be different
            assert random_42 != random_99
            assert not np.array_equal(numpy_42, numpy_99)


class TestReproducibilityIntegration:
    """재현성 통합 시나리오 테스트"""

    def test_complete_reproducibility_workflow(self, component_test_context):
        """완전한 재현성 워크플로우 테스트"""
        with component_test_context.classification_stack() as ctx:
            import random
            import numpy as np

            # Simulation of ML pipeline with reproducible random operations
            def simulate_ml_pipeline():
                # Data generation
                data = np.random.random((10, 3))

                # Random sampling
                indices = random.sample(range(10), 5)

                # Feature engineering with randomness
                features = data[indices] + np.random.normal(0, 0.1, (5, 3))

                return features.sum()

            # Run pipeline twice with same seed
            set_global_seeds(2023)
            result1 = simulate_ml_pipeline()

            set_global_seeds(2023)
            result2 = simulate_ml_pipeline()

            # Results should be identical
            assert abs(result1 - result2) < 1e-10

    def test_cross_library_seed_consistency(self, component_test_context):
        """크로스 라이브러리 시드 일관성 테스트"""
        with component_test_context.classification_stack() as ctx:
            import random
            import numpy as np

            # Set seed once and use multiple libraries
            set_global_seeds(777)

            # Generate using different libraries
            python_random = random.random()
            numpy_random = np.random.random()
            python_choice = random.choice([1, 2, 3, 4, 5])
            numpy_choice = np.random.choice([1, 2, 3, 4, 5])

            # Reset and generate again
            set_global_seeds(777)

            python_random_2 = random.random()
            numpy_random_2 = np.random.random()
            python_choice_2 = random.choice([1, 2, 3, 4, 5])
            numpy_choice_2 = np.random.choice([1, 2, 3, 4, 5])

            # All should be reproducible
            assert python_random == python_random_2
            assert numpy_random == numpy_random_2
            assert python_choice == python_choice_2
            assert numpy_choice == numpy_choice_2

    def test_environment_variable_persistence(self, component_test_context):
        """환경 변수 지속성 테스트"""
        with component_test_context.classification_stack() as ctx:
            test_seed = 54321

            # Set seed
            set_global_seeds(test_seed)

            # Verify environment variable persists
            assert os.environ.get("PYTHONHASHSEED") == str(test_seed)

            # Set different seed
            new_seed = 98765
            set_global_seeds(new_seed)

            # Environment should be updated
            assert os.environ.get("PYTHONHASHSEED") == str(new_seed)