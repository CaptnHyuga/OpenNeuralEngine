# Production Dockerfile for SNN Inference Server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY train.py ./
COPY src/ ./src/
COPY utils/ ./utils/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose inference port
EXPOSE 53801

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:53801/health || exit 1

# Set environment variables
ENV AIM_TRACKING_MODE=disabled
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Run inference server
CMD ["python", "scripts/launch_aim_inference.py"]
