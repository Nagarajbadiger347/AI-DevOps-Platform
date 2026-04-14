# Compatibility shim — plugins/ has been consolidated.
# Import from their new canonical locations.
from app.agents.infra import k8s_checker, aws_checker  # noqa: F401
from app.integrations import grafana_checker, linux_checker  # noqa: F401
