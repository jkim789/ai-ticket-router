"""
In-memory sliding window rate limiter.

Tracks request timestamps per client IP and enforces
a configurable requests-per-minute limit.
"""

import logging
import time
from collections import defaultdict
from typing import Tuple

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter using in-memory storage.

    Suitable for single-process deployments. Tracks request timestamps
    per key (typically client IP) and enforces a per-minute limit.
    """

    def __init__(self, rpm: int = 10, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            rpm: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.rpm = rpm
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> Tuple[bool, int, int]:
        """
        Check if a request from the given key is allowed.

        Args:
            key: Client identifier (typically IP address)

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > window_start
        ]

        if len(self._requests[key]) >= self.rpm:
            # Rate limited — calculate retry_after from oldest request in window
            oldest = self._requests[key][0]
            retry_after = int(oldest + self.window_seconds - now) + 1
            remaining = 0
            logger.warning("Rate limit exceeded for %s", key)
            return False, remaining, retry_after

        # Allow request
        self._requests[key].append(now)
        remaining = self.rpm - len(self._requests[key])
        return True, remaining, 0

    def cleanup(self) -> None:
        """Remove entries with no recent requests."""
        now = time.time()
        window_start = now - self.window_seconds
        expired_keys = [
            key for key, timestamps in self._requests.items()
            if not any(ts > window_start for ts in timestamps)
        ]
        for key in expired_keys:
            del self._requests[key]
