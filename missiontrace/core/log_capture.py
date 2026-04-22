"""
MissionTrace Log Capture — intercepts Python logging and attaches to current action.

Captures log records from the Python logging module and stores them in the
current MissionTrace action, making them visible in the MissionTrace dashboard.
"""

import logging
import traceback as tb_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from missiontrace.core.context import TraceContext


class MissionTraceLogHandler(logging.Handler):
    """
    Captures log records and attaches them to the current MissionTrace action.

    This handler intercepts Python logging calls and stores log events in the
    current action, which are later exported as OTLP span events when the action
    is sent to the backend.

    Usage:
        handler = MissionTraceLogHandler(ctx, level=logging.DEBUG)
        logging.root.addHandler(handler)
    """

    def __init__(self, ctx: "TraceContext", level: int = logging.DEBUG):
        super().__init__(level)
        self._ctx = ctx
        self.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record and attach it to the current action.

        Args:
            record: The LogRecord instance containing log information
        """
        try:
            # Get current action from MissionTrace context
            from missiontrace.core.context import get_current_action
            current_action = get_current_action()

            # Skip if no active action
            if not current_action:
                return

            # Build log event structure
            message = record.getMessage()

            # Truncate message to 1000 chars to avoid excessive payload size
            if len(message) > 1000:
                message = message[:997] + "..."

            log_event: dict[str, Any] = {
                "name": f"log.{record.levelname.lower()}",
                "timestamp": record.created,
                "attributes": {
                    "log.level": record.levelname,
                    "log.logger": record.name,
                    "log.message": message,
                }
            }

            # Add source location information
            if record.pathname:
                log_event["attributes"]["log.file"] = record.pathname
                log_event["attributes"]["log.line"] = record.lineno
                log_event["attributes"]["log.function"] = record.funcName

            # Add exception information if present
            if record.exc_info:
                exc_type, exc_value, exc_traceback = record.exc_info
                log_event["attributes"]["log.exception.type"] = exc_type.__name__
                log_event["attributes"]["log.exception.message"] = str(exc_value)

                # Format traceback (limit to last 10 frames to avoid huge payloads)
                tb_lines = tb_module.format_exception(exc_type, exc_value, exc_traceback)
                if len(tb_lines) > 10:
                    tb_lines = tb_lines[-10:]
                log_event["attributes"]["log.exception.traceback"] = "".join(tb_lines)

            # Add custom context from extra fields
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord.__dict__ and not key.startswith('_'):
                    log_event["attributes"][f"log.context.{key}"] = str(value)

            # Store in action's log_events list
            current_action.log_events.append(log_event)

        except Exception:
            # Never let log capture break the application
            # Use handleError to report the issue via logging's error handling
            self.handleError(record)
