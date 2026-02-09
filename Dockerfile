# ============================================================================
# LangGraph Chatbot - Dockerfile
# ============================================================================
# Multi-stage build for smaller image size

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY unified_backend.py .
COPY unified_frontend.py .
COPY .env.example .env

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/data

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

# Run the application
CMD ["streamlit", "run", "unified_frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
