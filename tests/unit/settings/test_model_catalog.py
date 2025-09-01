"""
Model Catalog Validation Tests
Phase 5: Pydantic 기반 모델 카탈로그 검증 테스트

CLAUDE.md 원칙 준수:
- TDD 기반 테스트
- 타입 힌트 필수
- Google Style Docstring
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.settings import (
    HyperparameterSpec,
    ModelSpec,
    ModelCatalog
)


class TestModelCatalogValidation:
    """Model Catalog 검증 시스템 테스트"""

    def test_hyperparameter_spec_validation(self):
        """HyperparameterSpec 검증 테스트"""
        # Given & When: 유효한 hyperparameter spec
        spec = HyperparameterSpec(
            type="int",
            range=[1, 100],
            default=50
        )
        
        # Then: 올바르게 생성되어야 함
        assert spec.type == "int"
        assert spec.range == [1, 100]
        assert spec.default == 50

    def test_hyperparameter_spec_invalid_type(self):
        """잘못된 타입의 HyperparameterSpec 테스트"""
        # Given & When & Then: 유효성 검증 오류가 발생해야 함
        with pytest.raises(Exception):  # Pydantic validation error
            HyperparameterSpec(
                type="invalid_type",
                range="not_a_list",
                default=None
            )

    def test_model_spec_validation(self):
        """ModelSpec 검증 테스트"""
        # Given & When: 유효한 model spec
        spec = ModelSpec(
            class_path="sklearn.ensemble.RandomForestClassifier",
            description="Random Forest Classifier",
            library="scikit-learn",
            hyperparameters={
                "fixed": {"random_state": 42},
                "tunable": {
                    "n_estimators": {
                        "type": "int",
                        "range": [10, 200],
                        "default": 100
                    }
                },
                "environment_defaults": {
                    "local": {"n_estimators": 10},
                    "prod": {"n_estimators": 100}
                }
            },
            supported_tasks=["binary_classification", "multiclass_classification"],
            feature_requirements={
                "numerical": True,
                "categorical": True,
                "text": False
            }
        )
        
        # Then: 올바르게 생성되어야 함
        assert spec.class_path == "sklearn.ensemble.RandomForestClassifier"
        assert spec.library == "scikit-learn"
        assert "fixed" in spec.hyperparameters
        assert "tunable" in spec.hyperparameters
        assert len(spec.supported_tasks) == 2

    def test_model_catalog_from_yaml(self):
        """YAML에서 ModelCatalog 로딩 테스트"""
        # Given: 임시 catalog YAML 파일
        catalog_content = {
            "Classification": [
                {
                    "class_path": "sklearn.ensemble.RandomForestClassifier",
                    "description": "Random Forest Classifier",
                    "library": "scikit-learn",
                    "hyperparameters": {
                        "fixed": {"random_state": 42},
                        "tunable": {
                            "n_estimators": {
                                "type": "int",
                                "range": [10, 200],
                                "default": 100
                            }
                        }
                    },
                    "supported_tasks": ["binary_classification"],
                    "feature_requirements": {"numerical": True, "categorical": False, "text": False}
                }
            ],
            "Regression": [
                {
                    "class_path": "sklearn.linear_model.LinearRegression",
                    "description": "Linear Regression",
                    "library": "scikit-learn",
                    "hyperparameters": {
                        "fixed": {"fit_intercept": True}
                    },
                    "supported_tasks": ["regression"],
                    "feature_requirements": {"numerical": True, "categorical": False, "text": False}
                }
            ]
        }
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(catalog_content, f)
            temp_path = f.name
        
        try:
            # When: YAML에서 ModelCatalog 로딩
            catalog = ModelCatalog.from_yaml(temp_path)
            
            # Then: 올바르게 로드되어야 함
            assert "Classification" in catalog.models
            assert "Regression" in catalog.models
            assert len(catalog.models["Classification"]) == 1
            assert len(catalog.models["Regression"]) == 1
            
            # Classification 모델 검증
            rf_model = catalog.models["Classification"][0]
            assert rf_model.class_path == "sklearn.ensemble.RandomForestClassifier"
            assert rf_model.library == "scikit-learn"
            
        finally:
            # 임시 파일 정리
            Path(temp_path).unlink()

    def test_model_catalog_find_model_spec(self):
        """모델 스펙 찾기 테스트"""
        # Given: ModelCatalog 인스턴스
        catalog_data = {
            "Classification": [
                ModelSpec(
                    class_path="sklearn.ensemble.RandomForestClassifier",
                    description="RF Classifier",
                    library="scikit-learn",
                    hyperparameters={},
                    supported_tasks=["classification"],
                    feature_requirements={}
                )
            ]
        }
        catalog = ModelCatalog(models=catalog_data)
        
        # When: 모델 스펙 찾기 (get_model_spec 사용)
        found_spec = catalog.get_model_spec("sklearn.ensemble.RandomForestClassifier")
        
        # Then: 올바른 스펙이 반환되어야 함
        assert found_spec is not None
        assert found_spec.class_path == "sklearn.ensemble.RandomForestClassifier"

    def test_model_catalog_find_model_spec_not_found(self):
        """존재하지 않는 모델 스펙 찾기 테스트"""
        # Given: ModelCatalog 인스턴스
        catalog_data = {
            "Classification": [
                ModelSpec(
                    class_path="sklearn.ensemble.RandomForestClassifier",
                    description="RF Classifier",
                    library="scikit-learn",
                    hyperparameters={},
                    supported_tasks=["classification"],
                    feature_requirements={}
                )
            ]
        }
        catalog = ModelCatalog(models=catalog_data)
        
        # When: 존재하지 않는 모델 스펙 찾기 (get_model_spec 사용)
        found_spec = catalog.get_model_spec("non.existent.Model")
        
        # Then: None이 반환되어야 함
        assert found_spec is None

    def test_validate_recipe_compatibility_success(self):
        """Recipe 호환성 검증 성공 테스트"""
        # Given: 호환되는 recipe와 catalog
        catalog_data = {
            "Classification": [
                ModelSpec(
                    class_path="sklearn.ensemble.RandomForestClassifier",
                    description="RF Classifier",
                    library="scikit-learn",
                    hyperparameters={
                        "tunable": {
                            "n_estimators": {
                                "type": "int",
                                "range": [10, 200],
                                "default": 100
                            }
                        }
                    },
                    supported_tasks=["classification"],
                    feature_requirements={}
                )
            ]
        }
        catalog = ModelCatalog(models=catalog_data)
        
        recipe = {
            "model": {
                "class_path": "sklearn.ensemble.RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": 50,  # 유효한 범위 내
                    "random_state": 42
                }
            }
        }
        
        # When: Recipe 호환성 검증
        is_compatible = catalog.validate_recipe_compatibility(recipe)
        
        # Then: 호환되어야 함
        assert is_compatible is True

    def test_validate_recipe_compatibility_model_not_found(self):
        """존재하지 않는 모델로 Recipe 호환성 검증 테스트"""
        # Given: 비어있는 catalog
        catalog = ModelCatalog(models={})
        
        recipe = {
            "model": {
                "class_path": "non.existent.Model",
                "hyperparameters": {}
            }
        }
        
        # When & Then: ValueError가 발생해야 함
        with pytest.raises(ValueError, match="Model.*not found in catalog"):
            catalog.validate_recipe_compatibility(recipe)

    def test_validate_hyperparameters_success(self):
        """하이퍼파라미터 검증 성공 테스트"""
        # Given: ModelCatalog와 유효한 recipe hyperparameters
        model_spec = ModelSpec(
            class_path="test.Model",
            description="Test Model",
            library="test",
            hyperparameters={
                "tunable": {
                    "param1": {
                        "type": "int",
                        "range": [1, 100],
                        "default": 50
                    },
                    "param2": {
                        "type": "float",
                        "range": [0.1, 1.0],
                        "default": 0.5
                    }
                }
            },
            supported_tasks=["test"],
            feature_requirements={}
        )
        
        catalog = ModelCatalog(models={"Test": [model_spec]})
        recipe_hyperparams = {
            "param1": 75,    # 유효한 범위 내
            "param2": 0.8,   # 유효한 범위 내
            "param3": "any"  # 카탈로그에 없는 파라미터 (허용)
        }
        
        # When: 하이퍼파라미터 검증
        is_valid = catalog._validate_hyperparameters(model_spec, recipe_hyperparams)
        
        # Then: 유효해야 함
        assert is_valid is True

    def test_validate_hyperparameters_out_of_range(self):
        """범위를 벗어난 하이퍼파라미터 검증 테스트"""
        # Given: ModelCatalog와 범위를 벗어난 recipe hyperparameters
        model_spec = ModelSpec(
            class_path="test.Model",
            description="Test Model", 
            library="test",
            hyperparameters={
                "tunable": {
                    "param1": {
                        "type": "int",
                        "range": [1, 100],
                        "default": 50
                    }
                }
            },
            supported_tasks=["test"],
            feature_requirements={}
        )
        
        catalog = ModelCatalog(models={"Test": [model_spec]})
        recipe_hyperparams = {
            "param1": 150  # 범위를 벗어남 (1-100 범위인데 150)
        }
        
        # When: 하이퍼파라미터 검증
        is_valid = catalog._validate_hyperparameters(model_spec, recipe_hyperparams)
        
        # Then: 유효하지 않아야 함
        assert is_valid is False