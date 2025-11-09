# Backend Dockerfile for arXiv Paper Curator
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p data/raw data/processed data/vector_db

# Make entrypoint script executable
RUN chmod +x scripts/docker-entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["scripts/docker-entrypoint.sh"]

