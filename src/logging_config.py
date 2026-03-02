"""
Centralized structured logging configuration.

Configures Python logging to emit JSON logs with optional structured fields
like request_id, path, method, and status_code for correlation and analysis.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from src.config import settings


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter suitable for application and request logs."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Optional structured fields commonly used in this app
        for field in (
            "request_id",
            "path",
            "method",
            "status_code",
            "duration_ms",
            "ticket_id",
            "action",
        ):
            if hasattr(record, field):
                log[field] = getattr(record, field)

        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log, ensure_ascii=False)


def configure_logging() -> None:
    """
    Configure root logger with JSON formatter and app log level.

    This replaces any existing handlers installed by uvicorn or libraries so
    that all logs are consistent and structured.
    """
    root = logging.getLogger()

    # Remove any existing handlers to avoid duplicate logs
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))

