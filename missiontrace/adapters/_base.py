"""
MissionTrace SDK — Base Adapter Contract

Every adapter (OpenAI, Anthropic, TinyFish, etc.) implements this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from missiontrace.core.context import TraceContext


class BaseAdapter(ABC):
    """
    Contract that all vendor/framework adapters must implement.

    - instrument(): monkey-patch the third-party SDK to emit Actions.
    - uninstrument(): restore originals (essential for test isolation).
    """

    def __init__(self, ctx: TraceContext) -> None:
        self._ctx = ctx

    @abstractmethod
    def instrument(self) -> None:
        """Patch the third-party SDK to emit Actions automatically."""
        ...

    @abstractmethod
    def uninstrument(self) -> None:
        """Restore original SDK behavior."""
        ...

    def _require_package(self, package_name: str, extras_name: str):
        """Lazy import with clear error message."""
        try:
            return __import__(package_name)
        except ImportError:
            raise ImportError(
                f"{package_name} required. Install: pip install missiontrace[{extras_name}]"
            ) from None
