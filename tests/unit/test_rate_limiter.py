"""Unit tests for rate limiter."""

import time
from unittest.mock import patch

from src.api.rate_limiter import RateLimiter


def test_allows_requests_within_limit():
    """Test that requests within the limit are allowed."""
    limiter = RateLimiter(rpm=3)

    allowed, remaining, retry_after = limiter.is_allowed("192.168.1.1")
    assert allowed is True
    assert remaining == 2
    assert retry_after == 0


def test_tracks_remaining_count():
    """Test that remaining count decreases with each request."""
    limiter = RateLimiter(rpm=3)

    _, remaining, _ = limiter.is_allowed("192.168.1.1")
    assert remaining == 2

    _, remaining, _ = limiter.is_allowed("192.168.1.1")
    assert remaining == 1

    _, remaining, _ = limiter.is_allowed("192.168.1.1")
    assert remaining == 0


def test_blocks_requests_over_limit():
    """Test that requests over the limit are blocked."""
    limiter = RateLimiter(rpm=2)

    limiter.is_allowed("192.168.1.1")
    limiter.is_allowed("192.168.1.1")

    allowed, remaining, retry_after = limiter.is_allowed("192.168.1.1")
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


def test_retry_after_value():
    """Test that retry_after returns correct seconds to wait."""
    limiter = RateLimiter(rpm=1, window_seconds=60)

    limiter.is_allowed("192.168.1.1")

    _, _, retry_after = limiter.is_allowed("192.168.1.1")
    assert 1 <= retry_after <= 61


def test_separate_limits_per_ip():
    """Test that rate limits are tracked independently per IP."""
    limiter = RateLimiter(rpm=1)

    allowed1, _, _ = limiter.is_allowed("192.168.1.1")
    allowed2, _, _ = limiter.is_allowed("192.168.1.2")

    assert allowed1 is True
    assert allowed2 is True

    blocked1, _, _ = limiter.is_allowed("192.168.1.1")
    assert blocked1 is False

    # Different IP should still work
    allowed3, _, _ = limiter.is_allowed("192.168.1.3")
    assert allowed3 is True


def test_window_expiry_resets_limit():
    """Test that requests are allowed again after the window expires."""
    limiter = RateLimiter(rpm=1, window_seconds=60)

    limiter.is_allowed("192.168.1.1")
    allowed, _, _ = limiter.is_allowed("192.168.1.1")
    assert allowed is False

    # Simulate time passing beyond the window
    with patch("src.api.rate_limiter.time") as mock_time:
        mock_time.time.return_value = time.time() + 61
        allowed, remaining, _ = limiter.is_allowed("192.168.1.1")
        assert allowed is True
        assert remaining == 0  # 1 rpm - 1 request = 0 remaining


def test_cleanup_removes_expired_entries():
    """Test that cleanup removes IPs with no recent requests."""
    limiter = RateLimiter(rpm=5, window_seconds=60)

    limiter.is_allowed("192.168.1.1")
    limiter.is_allowed("192.168.1.2")

    assert len(limiter._requests) == 2

    # Simulate time passing
    with patch("src.api.rate_limiter.time") as mock_time:
        mock_time.time.return_value = time.time() + 61
        limiter.cleanup()

    assert len(limiter._requests) == 0
