"""
Prometheus metrics for SRE observability.
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