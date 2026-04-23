# Stage 1: deps
FROM python:3.13-slim AS deps
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: final
FROM python:3.13-slim AS final
WORKDIR /app

# non-root user
RUN useradd -r -u 1001 -g root nexusops

COPY --from=deps /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --chown=nexusops:root . .

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/logs /app/chroma_db && chown -R nexusops:root /app/logs /app/chroma_db

USER nexusops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "python manage.py migrate && uvicorn app.orchestrator.main:app --host 0.0.0.0 --proxy-headers --forwarded-allow-ips=*"]
