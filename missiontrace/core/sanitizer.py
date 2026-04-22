"""
MissionTrace SDK — Input/Output Sanitizer

Redacts sensitive values (API keys, tokens, passwords) from action
inputs and outputs before they reach the transport buffer.
"""

from __future__ import annotations

import re
from typing import Any

# Default redaction patterns from the spec
_DEFAULT_PATTERNS: list[re.Pattern] = [
    # Common credential key-value patterns
    re.compile(
        r"(api[_\-]?key|token|secret|password|authorization)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    # OpenAI API keys
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    # GitHub personal access tokens
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    # Slack bot tokens
    re.compile(r"xoxb-[a-zA-Z0-9\-]+"),
    # Bearer auth headers
    re.compile(r"Bearer\s+[a-zA-Z0-9._\-]+"),
]

REDACTED = "[REDACTED]"


class Sanitizer:
    """Recursively redacts sensitive values from dicts / lists / strings."""

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self.patterns = list(_DEFAULT_PATTERNS)
        if extra_patterns:
            for p in extra_patterns:
                self.patterns.append(re.compile(p, re.IGNORECASE))

    def sanitize(self, data: Any) -> Any:
        """Return a deep-copy of *data* with sensitive values redacted."""
        if data is None:
            return None
        if isinstance(data, str):
            return self._sanitize_string(data)
        if isinstance(data, dict):
            return {k: self._sanitize_value(k, v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self.sanitize(item) for item in data)
        # Primitives (int, float, bool) pass through
        return data

    # -- internals ----------------------------------------------------------

    def _sanitize_string(self, value: str) -> str:
        result = value
        for pattern in self.patterns:
            result = pattern.sub(REDACTED, result)
        return result

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Redact entire value if the key itself looks sensitive."""
        sensitive_keys = {
            "api_key", "apikey", "api-key",
            "token", "secret", "password",
            "authorization", "auth_token",
            "access_token", "refresh_token",
            "private_key", "credentials",
        }
        if isinstance(key, str) and key.lower() in sensitive_keys:
            return REDACTED
        return self.sanitize(value)
