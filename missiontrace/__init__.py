"""
MissionTrace SDK — Observability and Tracing for AI Agents

Usage:
    import missiontrace

    missiontrace.init(api_key="mt_xxx", project="my-project")

    # Level 1: Auto-instrumentation via adapters
    from missiontrace.adapters.openai import instrument
    instrument()

    # Level 2: Decorators
    @missiontrace.trace(action_type="tool_call", name="my_func")
    def my_func(x): ...

    # Level 3: Manual context managers
    with missiontrace.mission(name="research") as m:
        with missiontrace.action("inference", name="openai.chat", intent="generation") as act:
            result = call_llm()
            act.set_output({"response": result})
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Generator

from missiontrace.core.context import TraceContext, get_current_action, get_current_mission
from missiontrace.core.models import (
    Action,
    ActionType,
    ActionStatus,
    Mission,
    MissionStatus,
    TokenUsage,
    SamplingParams,
    ErrorInfo,
)
from missiontrace.core.sanitizer import Sanitizer
from missiontrace.core.transport import OTelTransport, PlaceholderExporter
from missiontrace.utils.serialization import capture_inputs, safe_serialize

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_ctx: TraceContext | None = None
_transport: OTelTransport | None = None
_sanitizer: Sanitizer | None = None
_initialized: bool = False
_capture_code_context: bool = False  # Enable code context capture


def _get_ctx() -> TraceContext:
    if _ctx is None:
        raise RuntimeError("MissionTrace not initialized. Call missiontrace.init() first.")
    return _ctx


def _get_transport() -> OTelTransport:
    if _transport is None:
        raise RuntimeError("MissionTrace not initialized. Call missiontrace.init() first.")
    return _transport


def _get_sanitizer() -> Sanitizer:
    if _sanitizer is None:
        raise RuntimeError("MissionTrace not initialized. Call missiontrace.init() first.")
    return _sanitizer


# ---------------------------------------------------------------------------
# init() — the entry point
# ---------------------------------------------------------------------------

def init(
    api_key: str = "",
    project: str = "",
    endpoint: str | None = None,
    flush_intervmt_s: float = 2.0,
    max_batch_size: int = 100,
    sanitizer_patterns: list[str] | None = None,
    service_name: str = "missiontrace-sdk",
    capture_logs: bool = True,
    log_level: str = "DEBUG",
    capture_code_context: bool = True,  # NEW: Enable code context capture for Action Context
    _exporter: Any = None,  # for testing — inject a custom exporter
) -> None:
    """
    Initialize the MissionTrace SDK.

    Args:
        api_key: MissionTrace API key (placeholder — backend not yet defined).
        project: Project identifier for grouping traces.
        endpoint: OTLP endpoint URL. If None, uses in-memory placeholder.
        flush_intervmt_s: Seconds between background flushes.
        max_batch_size: Max records per flush batch.
        sanitizer_patterns: Additional regex patterns for redaction.
        service_name: OpenTelemetry service name.
        capture_logs: Enable automatic capture of Python logging statements (default: True).
        log_level: Minimum log level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        capture_code_context: Enable automatic code context capture (docstrings, file paths, etc.) for Action Context generation (default: True).
    """
    global _ctx, _transport, _sanitizer, _initialized, _capture_code_context

    _ctx = TraceContext(project_id=project)
    _sanitizer = Sanitizer(extra_patterns=sanitizer_patterns)
    _capture_code_context = capture_code_context
    _transport = OTelTransport(
        service_name=service_name,
        endpoint=endpoint,
        api_key=api_key or None,
        flush_intervmt_s=flush_intervmt_s,
        max_batch_size=max_batch_size,
        exporter=_exporter,
    )
    _transport.attach(_ctx)

    # Install log capture handler if enabled
    if capture_logs:
        import logging
        from missiontrace.core.log_capture import MissionTraceLogHandler

        # Get the log level constant
        level = getattr(logging, log_level.upper(), logging.DEBUG)

        # Create and install handler on root logger, passing the context
        handler = MissionTraceLogHandler(_ctx, level=level)
        logging.root.addHandler(handler)

    _initialized = True


def is_initialized() -> bool:
    return _initialized


def shutdown() -> None:
    """Flush remaining data and shut down transport."""
    global _initialized
    if _transport:
        _transport.shutdown()
    _initialized = False


# ---------------------------------------------------------------------------
# @trace — decorator for custom functions (Level 2 API)
# ---------------------------------------------------------------------------

def trace(
    action_type: str = "tool_call",
    name: str | None = None,
    intent: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable:
    """
    Decorator that wraps a function as a traced Action.

    Usage:
        @missiontrace.trace(action_type="tool_call", name="apply_patch", intent="mutation")
        def apply_patch(file_path: str, diff: str) -> PatchResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        action_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _get_ctx()
            san = _get_sanitizer()

            raw_input = capture_inputs(func, args, kwargs)
            sanitized_input = san.sanitize(raw_input)

            at = ActionType(action_type)
            action = ctx.start_action(
                type=at, name=action_name, input=sanitized_input, intent=intent
            )
            if tags:
                action.tags.update(tags)

            try:
                result = func(*args, **kwargs)
                sanitized_output = san.sanitize(safe_serialize(result))
                ctx.end_action(action, output=sanitized_output)
                return result
            except Exception as exc:
                ctx.end_action(action, error=exc)
                raise  # Always re-raise — SDK is an observer

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _get_ctx()
            san = _get_sanitizer()

            raw_input = capture_inputs(func, args, kwargs)
            sanitized_input = san.sanitize(raw_input)

            at = ActionType(action_type)
            action = ctx.start_action(
                type=at, name=action_name, input=sanitized_input, intent=intent
            )
            if tags:
                action.tags.update(tags)

            try:
                result = await func(*args, **kwargs)
                sanitized_output = san.sanitize(safe_serialize(result))
                ctx.end_action(action, output=sanitized_output)
                return result
            except Exception as exc:
                ctx.end_action(action, error=exc)
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# mission() — context manager for Mission lifecycle (Level 3 API)
# ---------------------------------------------------------------------------

class MissionContext:
    """Wraps a Mission with convenience methods for the `with` block."""

    def __init__(self, mission: Mission) -> None:
        self._mission = mission

    @property
    def id(self) -> str:
        return self._mission.id

    @property
    def mission(self) -> Mission:
        return self._mission

    def set_metadata(self, key: str, value: Any) -> None:
        self._mission.metadata[key] = value


@contextmanager
def mission(
    name: str | None = None,
    trigger: str = "manual",
    metadata: dict[str, Any] | None = None,
) -> Generator[MissionContext, None, None]:
    """
    Context manager for a Mission lifecycle.

    Usage:
        with missiontrace.mission(name="research-acme") as m:
            ...
    """
    ctx = _get_ctx()
    m = ctx.start_mission(name=name, trigger=trigger, metadata=metadata)
    mc = MissionContext(m)
    try:
        yield mc
        ctx.end_mission(m, MissionStatus.COMPLETED)
    except Exception:
        ctx.end_mission(m, MissionStatus.FAILED)
        raise


# ---------------------------------------------------------------------------
# action() — context manager for Action lifecycle (Level 3 API)
# ---------------------------------------------------------------------------

class ActionContext:
    """Wraps an Action with convenience methods for the `with` block."""

    def __init__(self, action: Action, ctx: TraceContext, sanitizer: Sanitizer) -> None:
        self._action = action
        self._ctx = ctx
        self._sanitizer = sanitizer

    @property
    def id(self) -> str:
        return self._action.id

    @property
    def action(self) -> Action:
        return self._action

    def set_output(self, output: dict[str, Any]) -> None:
        self._action.output = self._sanitizer.sanitize(output)

    def set_intent(self, intent: str) -> None:
        """Set or override the action's intent classification."""
        self._action.intent = intent

    def set_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        totmt_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        self._action.token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            totmt_tokens=totmt_tokens,
            model=model,
        )


@contextmanager
def action(
    action_type: str,
    name: str = "",
    intent: str | None = None,
    input: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[ActionContext, None, None]:
    """
    Context manager for an Action lifecycle.

    Usage:
        with missiontrace.action("tool_call", name="openai.chat", intent="generation") as act:
            result = openai.chat(...)
            act.set_output({"response": result})
    """
    ctx = _get_ctx()
    san = _get_sanitizer()
    sanitized_input = san.sanitize(input) if input else {}
    at = ActionType(action_type)
    a = ctx.start_action(type=at, name=name, input=sanitized_input, intent=intent)
    if tags:
        a.tags.update(tags)

    # Capture code context if enabled
    if _capture_code_context:
        try:
            from missiontrace.core.code_inspector import capture_function_metadata, get_repository_info

            # Capture function metadata (docstring, file path, etc.)
            # Skip frames: 0=capture_function_metadata, 1=action generator, 2+=user code
            func_metadata = capture_function_metadata(skip_frames=3, capture_source=False)

            # Try to get repository info if in a git repo
            repo_info = get_repository_info(func_metadata.file_path) if func_metadata.file_path else None

            # Store in action metadata for later Action Context generation
            if not a.metadata:
                a.metadata = {}

            a.metadata["_code_metadata"] = {
                "function": func_metadata.to_dict(),
                "repository": repo_info,
            }
        except Exception as e:
            # Temporarily print the error for debugging
            import traceback
            print(f"DEBUG: Code inspection failed: {e}")
            traceback.print_exc()

    ac = ActionContext(a, ctx, san)
    try:
        yield ac
        if a.status == ActionStatus.RUNNING:
            ctx.end_action(a, output=a.output)
    except Exception as exc:
        ctx.end_action(a, error=exc)
        raise


async def attach_github_context(
    action: Action | ActionContext,
    repository: str,
    file_paths: list[str],
    branch: str = "main",
    mcp_client: Any = None,
) -> None:
    """Attach GitHub repository context to an action.

    Fetches code snippets via GitHub MCP and stores in action metadata.

    Args:
        action: The action to attach context to (from missiontrace.action() context manager)
        repository: GitHub repository in "owner/repo" format
        file_paths: List of file paths to fetch
        branch: Git branch (default: "main")
        mcp_client: Optional GitHub MCP client instance

    Example:
        with missiontrace.action("tool_call", name="apply_patch") as act:
            await missiontrace.attach_github_context(
                act,
                repository="anthropics/anthropic-sdk-python",
                file_paths=["src/anthropic/client.py"],
            )
            result = apply_patch()
            act.set_output(result)
    """
    ctx = _get_ctx()

    # Handle both Action and ActionContext
    if isinstance(action, ActionContext):
        actumt_action = action._action
    else:
        actumt_action = action

    await ctx.attach_github_context(actumt_action, repository, file_paths, branch, mcp_client)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "init",
    "shutdown",
    "is_initialized",
    "trace",
    "mission",
    "action",
    "attach_github_context",
    "MissionContext",
    "ActionContext",
    "Mission",
    "Action",
    "ActionType",
    "ActionStatus",
    "MissionStatus",
    "TokenUsage",
    "SamplingParams",
    "ErrorInfo",
    "PlaceholderExporter",
]
