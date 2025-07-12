# Modern ML Pipeline

This project provides a modern, configuration-driven ML pipeline for training and serving uplift models. It is designed with MLOps principles in mind, emphasizing reproducibility, scalability, and a clear separation of concerns.

## Key Features

- **Configuration-Driven:** Easily experiment with different models, features, and hyperparameters by modifying YAML files without changing the core code.
- **Context-Aware Architecture:** The pipeline dynamically adapts its behavior for different execution contexts like training, batch inference, and real-time serving.
- **Extensible by Design:** Built on a foundation of abstract base classes, allowing for easy integration of new models, data sources, and preprocessing steps.
- **MLOps-Ready:** Standardized CLI, unified model packaging with MLflow, and fast, reproducible environments powered by `uv`.

## Development Environment Setup

This project uses `uv` for high-performance dependency management.

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install `uv`:**
    ```bash
    pip install uv
    ```

3.  **Sync dependencies:**
    Install all production and development dependencies using the lock file.
    ```bash
    uv pip sync requirements-dev.lock
    ```

## Usage

The project is controlled via a command-line interface.

### Training a Model

```bash
python main.py train --model-name "xgboost_x_learner"
```

### Running Batch Inference

```bash
python main.py batch-inference \
    --model-name "xgboost_x_learner" \
    --model-uri "models:/xgboost_x_learner/Production" \
    --input-sql-path "src/sql/inference_data.sql" \
    --output-table-id "uplift_predictions_q2_2024"
```

### Serving the API

```bash
python main.py serve-api --model-name "xgboost_x_learner"
```
