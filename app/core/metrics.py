"""
Prometheus metrics for SRE observability.

Includes SLO-ready metrics:
  - http_requests_total / http_request_duration_seconds  — for error-rate and latency SLOs
  - chat_pipeline_*                                      — chat action success/failure tracking
  - integration_call_duration_seconds                    — per-integration p95 latency
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Incident metrics
incidents_run_total = Counter(
    "incidents_run_total",
    "Total incidents initiated",
    ["tenant_id", "severity"],
)

incidents_failed_total = Counter(
    "incidents_failed_total",
    "Total incidents failed",
    ["tenant_id", "failure_reason"],
)

incident_duration_seconds = Histogram(
    "incident_pipeline_duration_seconds",
    "Incident pipeline latency",
    buckets=(5, 10, 30, 60, 120, 300),
)

# LLM metrics
llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "LLM call latency",
    ["provider", "model"],
)

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "status"],
)

# Database metrics
db_connections_in_use = Gauge(
    "db_connections_in_use",
    "Active PostgreSQL connections",
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query latency",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5),
)

# Approval metrics
approvals_pending = Gauge(
    "approvals_pending",
    "Pending approvals",
)

approvals_expired_total = Counter(
    "approvals_expired_total",
    "Expired approvals",
)

# Integration health
integration_failures_total = Counter(
    "integration_failures_total",
    "Integration API failures",
    ["service", "error_type"],
)

# ── SLO metrics ────────────────────────────────────────────────────────────

# HTTP SLO — error rate and latency (populated by middleware in orchestrator/main.py)
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

# Chat pipeline SLO — action success rate
chat_pipeline_actions_total = Counter(
    "chat_pipeline_actions_total",
    "Chat pipeline actions executed",
    ["action", "status"],   # status: success | error | dry_run | cancelled
)

chat_pipeline_errors_total = Counter(
    "chat_pipeline_errors_total",
    "Chat pipeline internal errors (unhandled exceptions)",
)

# Integration call latency — per service p95
integration_call_duration_seconds = Histogram(
    "integration_call_duration_seconds",
    "External integration call latency",
    ["service"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 15, 30),
)