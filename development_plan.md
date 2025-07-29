# Source-Test Environment Separation Refactoring Plan

## 1. Goal and Principles

- **Primary Goal**: To achieve 100% purity of production code by completely removing all test-related files from production directories, including `src/` and `recipes/`.
- **Core Principle**: The `tests/` directory must be a self-contained, independent testing environment that holds all its necessary resources (configs, recipes, SQL) internally, without relying on external files.

## 2. Problems with the Current Structure (As-Is)

1.  **Production Code Pollution**: Test-specific files like `recipes/local_test/` and `recipes/local_classification_test.yaml` are mixed with actual operational recipes. This creates a risk of packaging and deploying unnecessary or insecure test files.
2.  **Unclear Boundaries**: It is difficult to immediately distinguish between production files and test files, leading to confusion and potential mistakes during maintenance.
3.  **Brittle Tests**: Test code relies on the structure of the project root (e.g., `recipes/`), making tests fragile and prone to breaking if the root directory structure changes.

## 3. Proposed New Structure (To-Be)

All resources required for testing will be moved into a standard `fixtures` directory within `tests/`. This `fixtures` directory will mimic the actual project structure for intuitive use.

```
modern-ml-pipeline/
├── src/
│   └── ... (Contains only pure production code)
├── recipes/
│   └── models/
│       ├── classification/
│       │   └── logistic_regression.yaml  <-- Only production recipes remain
│       └── ...
└── tests/
    ├── ...
    ├── fixtures/  <-- [NEW] All test resources are stored here
    │   ├── recipes/
    │   │   ├── local_classification_test.yaml
    │   │   ├── dev_classification_test.yaml
    │   │   └── local_test/
    │   │       └── ... (All test-specific recipes)
    │   └── sql/
    │       └── loaders/
    │           ├── e2e_mock_data.sql
    │           └── local_test_data.sql
    └── conftest.py  <-- Manages the test environment setup
```

## 4. Detailed Refactoring Plan

**Step 1: Create the `fixtures` directory for test assets**
- Create the `fixtures` directory and its sub-structure within `tests/` to store test recipes and SQL files.
- **Command**: `mkdir -p tests/fixtures/recipes tests/fixtures/sql/loaders`

**Step 2: Move all test-related files**
- Move all files and directories identified as test-specific from the `recipes/` directory to `tests/fixtures/recipes/`.
- **Targets**:
    - `recipes/dev_classification_test.yaml`
    - `recipes/e2e_classification_test.yaml`
    - `recipes/local_classification_test.yaml`
    - `recipes/local_test/` (the entire directory)
    - `recipes/sql/loaders/e2e_mock_data.sql`
    - `recipes/sql/loaders/local_test_data.sql`
- **Commands**: A series of `mv` commands, such as `mv recipes/local_test tests/fixtures/recipes/`.

**Step 3: Update file path references in the test code (Crucial Step)**
- Modify the test code to reference the new paths within `tests/fixtures/`.
- **Best Practice**: Utilize `pytest`'s `conftest.py` and fixtures to manage paths robustly. Create separate fixtures for different environments (e.g., `local_settings`, `dev_settings`) to test various merging scenarios.

- **Example code for `tests/conftest.py`**:
  ```python
  import pytest
  from pathlib import Path

  @pytest.fixture(scope="session")
  def tests_root() -> Path:
      """Fixture to return the root path of the tests directory."""
      return Path(__file__).parent

  @pytest.fixture
  def fixture_recipes_path(tests_root: Path) -> Path:
      """Fixture to return the path to the test recipes in fixtures."""
      return tests_root / "fixtures" / "recipes"
  ```

- **Example usage in a test file**:
  ```python
  # Example for tests/settings/test_settings.py
  def test_load_specific_recipe(fixture_recipes_path):
      recipe_path = fixture_recipes_path / "local_classification_test.yaml"
      settings = load_settings_by_file(str(recipe_path))
      assert settings.recipe.name == "local_classification_test"
  ```

## 5. Expected Benefits

-   **Production Code Purity**: `src/` and `recipes/` will contain only production code and data.
-   **Self-Contained Tests**: The `tests/` directory can be copied and run anywhere, producing identical results.
-   **Clear Structure**: Files are clearly separated by their purpose and role.
-   **Safe Deployment**: Packaging becomes safer by simply excluding the `tests/` directory, preventing any test-related files from being accidentally deployed.

## 6. Test Strategy by Module

This section will be iteratively updated to define the testing strategy for each core module, ensuring comprehensive coverage and robustness.

### 6.1. Settings Module (`tests/settings/test_settings.py`)

The test suite for the `settings` module must validate the full orchestration logic of `loaders.py`, not just a single happy path.

**✅ Current Status:**
- One test `test_load_settings_by_file` exists, covering a single successful case for the `local` environment.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Refactor `test_settings.py`**: Separate the single large test into smaller, focused unit tests based on the functionality they verify.
-   **Environment Merging Logic**:
    -   `test_dev_config_overrides_base`: Test that `dev.yaml` correctly overrides `base.yaml` when `APP_ENV=dev`.
    -   `test_prod_config_overrides_base`: Test that `prod.yaml` correctly overrides `base.yaml` when `APP_ENV=prod`.
-   **Jinja Template Rendering**:
    -   `test_jinja_rendering_with_context_params`: Test that a `.sql.j2` recipe is correctly rendered when `context_params` are provided.
    -   `test_jinja_rendering_without_context_params_raises_error`: Test that rendering a `.sql.j2` file without required `context_params` raises a `jinja2.UndefinedError`.
-   **Error Handling and Validation**:
    -   `test_loading_non_existent_recipe_raises_error`: Test that `load_settings_by_file` raises `FileNotFoundError` for a non-existent recipe.
    -   `test_recipe_with_missing_required_fields_raises_error`: Test that a recipe missing top-level fields (e.g., `model`) raises a `ValueError`.
    -   `test_recipe_with_invalid_type_raises_pydantic_error`: Test that a recipe with incorrect data types (e.g., `max_depth: "ten"`) raises a `pydantic.ValidationError`.
-   **Inference Settings Logic**:
    -   `test_create_settings_for_inference`: Write a dedicated unit test to verify that `create_settings_for_inference` correctly injects a dummy recipe into the config data.

### 6.2. Engine Module (`tests/engine/`)

The Factory is the heart of the system's assembly line. Tests must ensure that it correctly creates all components based on the provided settings and handles environment-specific logic properly.

**✅ Current Status:**
- A comprehensive test file `tests/components/test_factory.py` exists but is misplaced.
- It covers dynamic model creation and environment-specific augmenter creation.
- It contains outdated import paths (e.g., `from src.core...`).

**🎯 Improvement Plan & Missing Test Cases:**

-   **Structural Improvement**:
    -   **Move `tests/components/test_factory.py` to `tests/engine/test_factory.py`** to align the test structure with the source structure.
-   **Refactor Existing Tests**:
    -   Update all outdated import paths to use the new public APIs (e.g., `from src.engine import Factory`).
    -   Refactor tests to use fixtures from `conftest.py` (`local_settings`, `dev_settings`) instead of importing them directly.
-   **Component Creation Logic**:
    -   `test_create_evaluator_for_all_task_types`: Create a parameterized test that iterates through all `task_type`s (`classification`, `regression`, `causal`, `clustering`) and verifies that the correct `Evaluator` class is instantiated for each.
-   **Artifact Creation Logic (`PyfuncWrapper`)**:
    -   `test_create_pyfunc_wrapper_with_schema`: Verify that when `training_df` is provided to `create_pyfunc_wrapper`, the resulting artifact contains a non-null `signature` and `data_schema`.
    -   `test_create_pyfunc_wrapper_without_schema`: Verify that when `training_df` is *not* provided, the `signature` and `data_schema` attributes of the artifact are `None`.
-   **Internal Helper Logic**:
    -   `test_extract_hyperparameters`: Add a dedicated unit test for the `_extract_hyperparameters` method to ensure it correctly parses values from both `dict` and `HyperparametersSettings` objects.

### 6.3. Components Module (`tests/components/`)

Components are the core building blocks of the ML logic. Each component must be tested in isolation as a unit to ensure its correctness, including handling of various edge cases.

#### 6.3.1. Preprocessor (`tests/components/test_preprocessor.py`)

**✅ Current Status:**
- A very well-structured and detailed test suite already exists.
- It correctly tests the separation of `fit` and `transform` to prevent data leakage.
- It has high coverage of edge cases, including unseen categories, empty dataframes, and all-numerical/all-categorical data.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Refactor Tests to Use Data Fixtures**:
    -   Currently, test data is hardcoded inside test functions using `pd.DataFrame({...})`.
    -   **Action**: Create reusable `pytest` fixtures in `conftest.py` that provide standardized sample DataFrames (e.g., `sample_train_df`, `sample_test_df_with_unseen_categories`). This will reduce code duplication and improve test readability.
-   **I/O Functionality**:
    -   `test_preprocessor_save_and_load`: The `save()` and `load()` methods are currently untested. A test case should be added to verify that a fitted preprocessor can be saved to disk with `joblib` and then loaded back correctly, yielding the exact same transformation results.

#### 6.3.2. Trainer (`tests/components/test_trainer.py`)

**✅ Current Status:**
- A detailed test suite exists, effectively using `unittest.mock.patch` to isolate the `Trainer`'s internal logic.
- Core principles like conditional HPO (enabled/disabled) and data leakage prevention are well-tested.
- The execution flow (`split` -> `augment` -> `fit` -> `transform`) is thoroughly verified.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Refactor Existing Tests**:
    -   Update outdated import paths (e.g., `from src.core...`) to use the new public APIs.
    -   **Action**: Create a `pytest` fixture to encapsulate the complex `optuna.study` mock object setup, improving the readability of HPO-related tests.
-   **Task Type Coverage**:
    -   The current tests implicitly focus on the `classification` task type.
    -   `test_prepare_data_for_all_task_types`: Create a parameterized test that iterates through all task types (`classification`, `regression`, `causal`, `clustering`) to verify that the `_prepare_training_data` method correctly separates X, y, and additional data (like `treatment`) for each case.
    -   `test_fit_model_for_all_task_types`: Create a parameterized test to ensure the `_fit_model` method calls the model's `fit` function with the correct arguments based on the `task_type`.
-   **Error Handling**:
    -   Add a test case to verify that an appropriate error is raised if an unsupported `task_type` is provided in the recipe.

### 6.4. Pipelines Module (`tests/pipelines/`)

Pipeline tests are crucial for ensuring that all the unit-tested components work together correctly in an end-to-end flow. These are integration tests that should minimize mocking of the internal logic.

#### 6.4.1. Train Pipeline (`tests/pipelines/test_train_pipeline.py`)

**✅ Current Status:**
- A detailed end-to-end test suite exists, marked with `@pytest.mark.e2e`.
- It correctly implements test isolation by creating and cleaning up a temporary MLflow tracking URI.
- It thoroughly validates the completeness of the logged `PyfuncWrapper` artifact, including metadata for data leakage prevention, HPO results, and logic snapshots.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Decouple Tests from Source Code**:
    -   **Problem**: Currently, `train_pipeline.py` contains temporary mocking logic (`is_e2e_test_run`) to generate a mock DataFrame for tests. This pollutes the production code.
    -   **Action**: Remove the mocking logic from `train_pipeline.py`. In the test file, use `unittest.mock.patch` to mock the `DataAdapter.read` method directly, making it return a sample DataFrame. This achieves complete separation between source and test code.
-   **Refactor Existing Tests**:
    -   Update outdated import paths (e.g., `from src.core...`) to use the new public APIs.
    -   Create a `pytest` fixture in `conftest.py` to manage the temporary MLflow tracking URI, reducing code duplication across E2E tests.
-   **Expand E2E Scenarios**:
    -   Currently, the E2E test only covers the `local` environment scenario.
    -   `test_train_pipeline_e2e_in_dev_env`: Add a new E2E test that runs the training pipeline using `dev_settings`. This test will specifically verify that the real `Augmenter` (which uses the `FeatureStore`) is correctly integrated and executed, which is a critical difference from the `local` environment.

#### 6.4.2. Inference Pipeline (`tests/pipelines/test_inference_pipeline.py`)

**✅ Current Status:**
- An efficient E2E test suite exists that cleverly uses a `module-scoped fixture` to run training once and use the resulting artifact for all inference tests in the module.
- It thoroughly validates the inference output, MLflow logging, and the consistency of the used artifact.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Decouple Tests from Source Code**:
    -   **Problem**: Similar to the training pipeline, `inference_pipeline.py` contains temporary logic to mock data loading for E2E tests.
    -   **Action**: Remove this mocking logic. The test fixture that runs the initial training should use a recipe pointing to a small, real data file within `tests/fixtures/data`. The inference pipeline test can then `patch` the `DataAdapter.read` method if specific inference data is needed.
-   **Refactor Helper Functions**:
    -   The helper function `_is_jinja_template` is currently located inside `inference_pipeline.py`.
    -   **Action**: Move `_is_jinja_template` to `src/utils/system/templating_utils.py` to consolidate all templating-related logic in one place.
-   **Expand E2E Scenarios**:
    -   `test_inference_with_jinja_template`: Add an E2E test where the initial training uses a `.sql.j2` recipe. The inference test should then provide `context_params` to verify that dynamic SQL rendering works correctly during batch inference.
    -   `test_inference_raises_error_on_schema_drift`: Add a negative test case. It should run inference with input data that has a deliberately altered schema (e.g., missing a column, different data type) and verify that the `SchemaConsistencyValidator` catches the drift and raises a `ValueError`.

### 6.5. Serving Module (`tests/serving/`)

The serving module is the final delivery point of the model. Tests must ensure the API server starts correctly, handles requests robustly, and interacts with the online feature store as expected.

#### 6.5.1. API Server (`tests/serving/test_api.py`)

**✅ Current Status:**
- A very detailed test suite exists, covering E2E scenarios using a fixture-trained model.
- All self-describing metadata endpoints (`/model/metadata`, `/model/schema`, etc.) are thoroughly validated.
- Basic error handling for invalid inputs (`422`), paths (`404`), and methods (`405`) is covered.

**🎯 Improvement Plan & Missing Test Cases:**

-   **Restructure Test File**:
    -   **Problem**: The `test_api.py` file is monolithic, containing E2E tests, mocked unit tests, and compatibility tests together.
    -   **Action**: Split `test_api.py` into smaller, more focused files like `test_api_e2e.py` (for tests requiring a real trained model) and `test_api_endpoints.py` (for mocked unit tests of individual endpoints) to improve clarity and maintainability.
-   **Online Feature Store Integration**:
    -   `test_predict_endpoint_uses_online_features`: The current `/predict` test only checks for a successful response. It doesn't verify that the online feature store was actually used.
    -   **Action**: Add a test that `patch`es the `FeastAdapter.get_online_features` method and asserts that it was called with the correct primary key(s) from the request during a `/predict` call.
-   **Real-time Schema Validation**:
    -   `test_predict_endpoint_raises_error_on_schema_drift`: There is no test to verify that the API rejects requests that don't match the trained artifact's schema.
    -   **Action**: Add a negative test case that sends a JSON payload with a deliberately altered schema to the `/predict` endpoint and verifies that the API correctly returns a `400 Bad Request` status code.
