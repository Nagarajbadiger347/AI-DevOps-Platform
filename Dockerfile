# ── Stage 1: build dependencies ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: production image ─────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Copy only installed packages from builder (no gcc, no build tools)
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/chroma_db && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check — hits /health every 30s, 3 failures = unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production: 4 workers, no reload, access log off (handled by Nginx)
CMD ["uvicorn", "app.orchestrator.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--no-access-log"]
