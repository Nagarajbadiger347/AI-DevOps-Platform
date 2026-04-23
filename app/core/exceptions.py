"""Structured error handling for NexusOps.

All exceptions inherit from APIError and provide:
- code: machine-readable error code
- message: user-safe error message
- status_code: HTTP status code
- details: optional contextual info
"""

from typing import Any, Optional


class APIError(Exception):
    """Base class for all API errors."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Serialize to JSON response."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class TimeoutError(APIError):
    """Operation exceeded time limit."""

    def __init__(self, service: str = "service", timeout_sec: int = 30):
        super().__init__(
            code="TIMEOUT",
            message=f"Request to {service} took too long (>{timeout_sec}s)",
            status_code=504,
            details={"service": service, "timeout_seconds": timeout_sec},
        )


class CircuitBreakerOpen(APIError):
    """External service is temporarily unavailable (circuit breaker open)."""

    def __init__(self, service: str):
        super().__init__(
            code="SERVICE_UNAVAILABLE",
            message=f"{service} is temporarily unavailable (circuit breaker open)",
            status_code=503,
            details={"service": service},
        )


class InvalidRequest(APIError):
    """Request validation failed."""

    def __init__(self, message: str):
        super().__init__(
            code="INVALID_REQUEST",
            message=message,
            status_code=400,
        )


class AuthenticationFailed(APIError):
    """User authentication failed."""

    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(
            code="AUTHENTICATION_FAILED",
            message=reason,
            status_code=401,
        )


class PermissionDenied(APIError):
    """User lacks required permissions."""

    def __init__(self, resource: str = "resource"):
        super().__init__(
            code="PERMISSION_DENIED",
            message=f"You do not have permission to access {resource}",
            status_code=403,
        )


class NotFound(APIError):
    """Resource not found."""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource_type} '{resource_id}' not found",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class RateLimitExceeded(APIError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            code="RATE_LIMIT_EXCEEDED",
            message="Too many requests. Please slow down.",
            status_code=429,
            details={"retry_after_seconds": retry_after},
        )


class InternalServerError(APIError):
    """Internal server error."""

    def __init__(self, error_id: str):
        super().__init__(
            code="INTERNAL_ERROR",
            message="An unexpected error occurred. Please contact support.",
            status_code=500,
            details={"error_id": error_id},
        )


class DependencyUnavailable(APIError):
    """Required dependency is unavailable."""

    def __init__(self, dependency: str):
        super().__init__(
            code="DEPENDENCY_UNAVAILABLE",
            message=f"Required service '{dependency}' is unavailable",
            status_code=503,
            details={"dependency": dependency},
        )
