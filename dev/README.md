# Development Environment

This directory contains development-specific files and tools for the modern-ml-pipeline project.

## Directory Structure

```
dev/
├── README.md              # This file
├── docker/               # Docker-related files
│   ├── docker-compose.yml # Development services (PostgreSQL, Redis)
│   └── Dockerfile         # Development container image
├── scripts/              # Development scripts
│   └── setup-dev-environment.sh  # Environment setup script
├── docs/                 # Development documentation
│   └── factoringlog.md   # Development notes and refactoring log
└── examples/             # Example scripts and usage demonstrations
    └── main.py           # Simple CLI entry point example
```

## Development Services

The `docker/docker-compose.yml` provides supporting services for development:

- **PostgreSQL** (port 5433): For testing database adapters
- **Redis** (port 6380): For testing caching and feature store integration

These services complement the main MLflow infrastructure provided by the `mmp-local-dev` project.

## Usage

### Setup Development Environment
```bash
cd dev/
./scripts/setup-dev-environment.sh
```

### Start Development Services
```bash
cd dev/docker/
docker-compose up -d
```

### Stop Development Services
```bash
cd dev/docker/
docker-compose down
```

## Integration with mmp-local-dev

This development environment is designed to work alongside the `mmp-local-dev` project:

- **mmp-local-dev**: Provides MLflow server, Feast, and core ML infrastructure
- **modern-ml-pipeline/dev/**: Provides supporting services and development tools

## Important Notes

- These files are **development-only** and are not included in PyPI releases
- The main modern-ml-pipeline package is a pure library with no infrastructure dependencies
- Use `mmp-local-dev` for full ML infrastructure setup
- Port numbers are chosen to avoid conflicts with mmp-local-dev services

## System Compatibility

- **Python 3.11+**: Required for the modern-ml-pipeline library
- **Docker & Docker Compose**: Required for development services
- **uv**: Recommended for package management and task running