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

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UVICORN_WORKERS=2

RUN mkdir -p /app/logs /app/data /app/post_mortems && chown -R nexusops:root /app/logs /app/data /app/post_mortems

USER nexusops

EXPOSE 8000

# Healthcheck is configured in docker-compose.yml (longer start-period during
# migrations); avoid duplicate definitions.

# Multi-worker uvicorn — set UVICORN_WORKERS=1 in HA deployments where you run
# a separate monitor container, since the background loop in lifespan() fires
# once per worker. The monitor loop already de-duplicates triggers, so 2
# workers on a single host is fine.
CMD ["sh", "-c", "python manage.py migrate && exec uvicorn app.orchestrator.main:app --host 0.0.0.0 --workers ${UVICORN_WORKERS:-2} --proxy-headers --forwarded-allow-ips=*"]
