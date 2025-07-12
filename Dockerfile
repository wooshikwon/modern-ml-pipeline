# Dockerfile for Modern ML Pipeline (v3.0 - uv)
# -----------------------------------------------------------------------------
# Multi-stage build optimized for speed and security using uv.
# -----------------------------------------------------------------------------

# --- Stage 1: `base` ---
# Purpose: A common base layer with Python and uv installed.
FROM python:3.10-slim as base

# Set environment variables
# ARG to allow overriding the environment at build time (e.g., --build-arg APP_ENV=dev)
ARG APP_ENV=prod
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    APP_ENV=$APP_ENV

# Install uv, the high-performance Python package installer
RUN pip install --no-cache-dir "uv==0.7.20"

# Set the working directory
WORKDIR /app

# --- Stage 2: `builder` ---
# Purpose: Install production and development dependencies into a virtual environment.
# This layer is cached effectively as long as the lock files don't change.
FROM base as builder

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the dependency definition files
COPY requirements.lock requirements-dev.lock ./

# Install dependencies using uv for maximum speed
# Install production dependencies first, then add development dependencies
RUN uv pip sync --no-cache --python /opt/venv/bin/python requirements.lock && \
    uv pip sync --no-cache --python /opt/venv/bin/python requirements-dev.lock

# --- Stage 3: `serve` (Final Serving Image) ---
# Purpose: A lightweight image with only the minimal code and production dependencies.
FROM base as serve

# Create a non-root user for security
RUN groupadd --system app && useradd --system --gid app app
USER app

# Copy the virtual environment with production dependencies from the 'builder' stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the necessary application files for serving
COPY --chown=app:app src/ src/
COPY --chown=app:app serving/ serving/
COPY --chown=app:app main.py .
COPY --chown=app:app config.yaml .
COPY --chown=app:app recipe/ recipe/

# Activate the virtual environment and define the entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "main.py", "serve-api"]

# Expose the port and set the default command
EXPOSE 8000
CMD ["--model-name", "xgboost_x_learner"]


# --- Stage 4: `train` (Final Training Image) ---
# Purpose: An image containing all code and all (dev) dependencies for training.
FROM base as train

# Create a non-root user
RUN groupadd --system app && useradd --system --gid app app
USER app

# Copy the virtual environment with all dependencies from the 'builder' stage
COPY --from=builder /opt/venv /opt/venv

# Copy the entire project source
COPY --chown=app:app . .

# Activate the virtual environment and define the entrypoint
ENTRYPOINT ["/opt/venv/bin/python", "main.py", "train"]

# Default command, can be overridden
CMD ["--model-name", "xgboost_x_learner"]
