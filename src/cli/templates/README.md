# {{ project_name }}

ML Pipeline project powered by Modern ML Pipeline (MMP) framework.

## 🚀 5-Step ML Pipeline Workflow

### Step 1: Environment Setup
```bash
# Install dependencies with uv
uv sync

# Create environment configuration
uv run mmp get-config --env-name local

# Copy and configure environment variables
cp .env.local.template .env.local
# Edit .env.local with your credentials
```

### Step 2: Recipe Creation
```bash
# Create ML recipe interactively
uv run mmp get-recipe

# Or use a template
uv run mmp get-recipe --template classification
```

### Step 3: Model Training
```bash
# Train model with the recipe
uv run mmp train --recipe-file recipes/your_recipe.yaml --env-name local

# With hyperparameter tuning
uv run mmp train --recipe-file recipes/your_recipe.yaml --env-name local --tune
```

### Step 4: Model Inference
```bash
# Batch inference
uv run mmp batch-inference --run-id <mlflow_run_id> --env-name local

# Or serve as API
uv run mmp serve-api --run-id <mlflow_run_id> --env-name local --port 8000
```

### Step 5: Model Deployment
```bash
# Check system status
uv run mmp system-check --env-name prod

# Deploy to production
uv run mmp deploy --run-id <mlflow_run_id> --env-name prod
```

## 📁 Project Structure

```
{{ project_name }}/
├── README.md            # This file
├── pyproject.toml       # Project dependencies (uv)
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local development services
├── .env.local.template  # Environment variables template
├── configs/            # Environment configurations
│   ├── local.yaml      # Local development config
│   ├── dev.yaml        # Development server config
│   └── prod.yaml       # Production config
├── recipes/            # ML pipeline recipes
│   └── .gitkeep       
├── data/              # Data directory
│   ├── raw/           # Raw data
│   ├── processed/     # Processed data
│   └── features/      # Feature store data
├── sql/               # SQL scripts
│   ├── ddl/           # Table definitions
│   └── dml/           # Data queries
└── models/            # Saved models (auto-generated)
```

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- uv (for package management)
- Docker & Docker Compose (optional, for local services)

### Quick Start with Docker
```bash
# Start local services (MLflow, PostgreSQL, Redis)
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f
```

### Local Development
```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Lint code
uv run ruff check .

# Type check
uv run mypy .
```

## 📊 MLflow UI

After starting services with Docker Compose:
- MLflow UI: http://localhost:5002
- Default experiment: `{{ project_name }}-experiment`

## 🔧 Configuration

### Environment Variables
Each environment has its own `.env.{env_name}` file:
- `.env.local` - Local development
- `.env.dev` - Development server
- `.env.prod` - Production

### Config Files
Configuration hierarchy:
1. `configs/base.yaml` - Base configuration (inherited by all)
2. `configs/{env_name}.yaml` - Environment-specific overrides

### Recipe Structure
Recipes define ML pipeline specifications:
```yaml
name: my_recipe
model:
  class_path: sklearn.ensemble.RandomForestClassifier
  hyperparameters:
    n_estimators: 100
    random_state: 42
data:
  loader:
    name: sql_loader
    source_uri: "SELECT * FROM features"
evaluation:
  metrics: ["accuracy", "f1", "roc_auc"]
```

## 📝 Documentation

- [MMP Documentation](https://github.com/your-org/modern-ml-pipeline)
- [API Reference](./docs/api.md)
- [Recipe Guide](./docs/recipes.md)

## 🤝 Contributing

1. Create a feature branch
2. Make changes and test
3. Run quality checks: `uv run pre-commit run --all-files`
4. Submit pull request

## 📄 License

Copyright © {{ current_year }} {{ organization }}. All rights reserved.