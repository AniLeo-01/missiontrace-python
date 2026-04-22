"""Tests for input/output sanitization."""

from missiontrace.core.sanitizer import REDACTED, Sanitizer


class TestSanitizer:
    def setup_method(self):
        self.san = Sanitizer()

    def test_openai_key(self):
        result = self.san.sanitize({"key": "sk-abcdefghijklmnopqrstuvwxyz1234567890"})
        assert REDACTED in result["key"]

    def test_github_token(self):
        result = self.san.sanitize("token: ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert "ghp_" not in result
        assert REDACTED in result

    def test_slack_token(self):
        result = self.san.sanitize("xoxb-123-456-abc")
        assert "xoxb" not in result

    def test_bearer_token(self):
        result = self.san.sanitize({"auth": "Bearer eyJhbGciOiJIUzI1NiJ9.test"})
        assert "eyJ" not in result["auth"]

    def test_sensitive_key_names(self):
        data = {
            "api_key": "super-secret-key",
            "password": "hunter2",
            "token": "abc123",
            "name": "safe-value",
        }
        result = self.san.sanitize(data)
        assert result["api_key"] == REDACTED
        assert result["password"] == REDACTED
        assert result["token"] == REDACTED
        assert result["name"] == "safe-value"

    def test_nested_dict(self):
        data = {"config": {"api_key": "secret", "name": "test"}}
        result = self.san.sanitize(data)
        assert result["config"]["api_key"] == REDACTED
        assert result["config"]["name"] == "test"

    def test_list_sanitization(self):
        data = ["sk-abcdefghijklmnopqrstuvwxyz1234567890", "safe"]
        result = self.san.sanitize(data)
        assert REDACTED in result[0]
        assert result[1] == "safe"

    def test_passthrough_primitives(self):
        assert self.san.sanitize(42) == 42
        assert self.san.sanitize(3.14) == 3.14
        assert self.san.sanitize(True) is True
        assert self.san.sanitize(None) is None

    def test_key_value_pattern(self):
        result = self.san.sanitize("api_key=my-secret-key-123")
        assert "my-secret-key-123" not in result

    def test_custom_pattern(self):
        san = Sanitizer(extra_patterns=[r"INTERNAL_\d+"])
        result = san.sanitize("code: INTERNAL_9999")
        assert "INTERNAL_9999" not in result
