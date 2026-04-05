# Email Triage OpenEnv — Dockerfile
# Builds a self-contained FastAPI server for the email triage environment.
# Compatible with Hugging Face Spaces (port 7860).

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all environment code
COPY . .

# Set PYTHONPATH so all imports resolve
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose HF Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]