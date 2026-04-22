"""
MissionTrace SDK — Safe Serialization

Converts arbitrary Python objects to JSON-safe dicts with size caps.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

MAX_VALUE_SIZE = 2048  # 2KB default truncation


def safe_serialize(obj: Any, max_size: int = MAX_VALUE_SIZE) -> Any:
    """
    Convert *obj* to a JSON-safe value.  Falls back to repr() for
    non-serializable objects, truncated to *max_size* characters.
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:max_size] if len(obj) > max_size else obj
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v, max_size) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [safe_serialize(v, max_size) for v in obj]

    # Pydantic models
    if hasattr(obj, "model_dump"):
        return safe_serialize(obj.model_dump(), max_size)

    # Fallback — repr with truncation
    r = repr(obj)
    return r[:max_size] if len(r) > max_size else r


def capture_inputs(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """
    Build a dict of argument names → values from a function call.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return {k: safe_serialize(v) for k, v in bound.arguments.items()}
    except Exception:
        # If signature binding fails, just capture positional + keyword
        result: dict[str, Any] = {}
        for i, a in enumerate(args):
            result[f"arg_{i}"] = safe_serialize(a)
        for k, v in kwargs.items():
            result[k] = safe_serialize(v)
        return result
