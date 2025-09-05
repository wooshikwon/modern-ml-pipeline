"""
Unit tests for the reproducibility utility module.
Tests reproducibility and random seed management.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.utils.system.reproducibility import set_global_seeds


class TestSetGlobalSeeds:
    """Test set_seeds function."""
    
    def test_set_seeds_default(self):
        """Test setting seeds with default value."""
        try:
            set_global_seeds()
            assert True
        except Exception as e:
            pytest.fail(f"Default seed setting failed: {e}")
    
    def test_set_seeds_custom_value(self):
        """Test setting seeds with custom value."""
        try:
            set_global_seeds(12345)
            assert True
        except Exception as e:
            pytest.fail(f"Custom seed setting failed: {e}")
    
    def test_set_seeds_calls(self):
        """Test that set_global_seeds executes without error."""
        test_seed = 42
        try:
            set_global_seeds(test_seed)
            assert True
        except Exception as e:
            pytest.fail(f"Seed setting failed: {e}")
    
    def test_set_seeds_reproducibility(self):
        """Test that seeds actually provide reproducibility."""
        seed_value = 42
        
        # Set seed and generate random numbers
        set_global_seeds(seed_value)
        random1 = np.random.rand(5)
        
        # Reset seed and generate again
        set_global_seeds(seed_value)
        random2 = np.random.rand(5)
        
        # Should be identical
        np.testing.assert_array_equal(random1, random2)