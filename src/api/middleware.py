"""Request/response logging and rate limiting middleware."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.api.rate_limiter import RateLimiter
from src.config import settings

logger = logging.getLogger(__name__)

# Shared rate limiter instance
_rate_limiter = RateLimiter(rpm=settings.RATE_LIMIT_RPM)

# Paths to rate limit
RATE_LIMITED_PATHS = {"/api/v1/tickets/process"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request and response details.

    Captures method, path, status code, and processing time for every request.
    Assigns a unique request ID for traceability.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process and log each request/response cycle.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response with added X-Request-ID header
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Request started | id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log completed request
        logger.info(
            "Request completed | id=%s method=%s path=%s status=%d duration=%.0fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = f"{duration_ms:.0f}"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces per-IP rate limiting on ticket processing.

    Only applies to specific endpoints (POST /api/v1/tickets/process).
    Returns 429 with retry information when the limit is exceeded.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check rate limit before processing the request.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response, or 429 if rate limited
        """
        # Only rate limit specific paths and methods
        if request.method == "POST" and request.url.path in RATE_LIMITED_PATHS:
            client_ip = request.client.host if request.client else "unknown"
            allowed, remaining, retry_after = _rate_limiter.is_allowed(client_ip)

            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please wait before submitting another ticket.",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response

        return await call_next(request)
