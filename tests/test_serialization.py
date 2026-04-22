"""Tests for serialization utilities."""

from pydantic import BaseModel

from missiontrace.utils.serialization import capture_inputs, safe_serialize


class TestSafeSerialize:
    def test_primitives(self):
        assert safe_serialize(42) == 42
        assert safe_serialize(3.14) == 3.14
        assert safe_serialize(True) is True
        assert safe_serialize(None) is None

    def test_string_truncation(self):
        long_str = "a" * 5000
        result = safe_serialize(long_str, max_size=100)
        assert len(result) == 100

    def test_dict(self):
        data = {"key": "value", "nested": {"a": 1}}
        result = safe_serialize(data)
        assert result == data

    def test_list(self):
        data = [1, "hello", [2, 3]]
        result = safe_serialize(data)
        assert result == [1, "hello", [2, 3]]

    def test_pydantic_model(self):
        class MyModel(BaseModel):
            x: int = 1
            y: str = "hello"

        result = safe_serialize(MyModel())
        assert result == {"x": 1, "y": "hello"}

    def test_non_serializable_fallback(self):
        class Custom:
            def __repr__(self):
                return "Custom()"

        result = safe_serialize(Custom())
        assert result == "Custom()"


class TestCaptureInputs:
    def test_basic_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        result = capture_inputs(add, (3, 5), {})
        assert result == {"a": 3, "b": 5}

    def test_kwargs(self):
        def greet(name: str, greeting: str = "hello") -> str:
            return f"{greeting} {name}"

        result = capture_inputs(greet, (), {"name": "world"})
        assert result["name"] == "world"
        assert result["greeting"] == "hello"

    def test_fallback_on_failure(self):
        # Lambda with *args can't be easily bound
        result = capture_inputs(lambda *a: None, (1, 2, 3), {"x": 4})
        assert "arg_0" in result or "x" in result
