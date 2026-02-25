# RDT Trading System Docker Image
# Sophisticated ML Trading Bot with Multi-Agent Architecture
FROM python:3.11-slim

# Build argument for selecting requirements file (default: full, web: minimal)
ARG REQUIREMENTS_FILE=requirements.txt
ARG INSTALL_ML=true

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    YF_SESSION=requests \
    TZ=America/New_York

# Set working directory
WORKDIR /app

# Install system dependencies (including build tools for ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files for better caching
COPY requirements.txt requirements-web.txt ./

# Install Python dependencies from selected requirements file
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Install additional ML dependencies only for full builds
RUN if [ "$INSTALL_ML" = "true" ]; then \
    pip install --no-cache-dir \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    hmmlearn>=0.3.0 \
    optuna>=3.0.0 \
    tensorflow-cpu>=2.15.0; \
    fi

# Fix curl_cffi impersonation issue (pin to compatible version)
RUN pip install --no-cache-dir 'curl_cffi>=0.7,<0.14'

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 trader

# Create directories for data persistence with correct ownership
RUN mkdir -p /app/data/logs /app/data/database /app/data/historical && \
    chown -R trader:trader /app

USER trader

# Default command (can be overridden)
ENTRYPOINT ["python", "main.py"]
CMD ["scanner"]
