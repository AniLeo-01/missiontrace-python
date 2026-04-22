"""
MissionTrace SDK — TinyFish Web Agent Adapter

Auto-instruments the TinyFish Python SDK to capture every run and its
individual steps as nested MissionTrace actions.

Instruments both sync and async clients for:
  1. agent.run()    — synchronous blocking run
  2. agent.queue()  — fire-and-forget (returns run_id)
  3. agent.stream() — SSE streaming with per-step PROGRESS events
  4. runs.get()     — single run retrieval (for polling async runs)
  5. runs.list()    — run listing / history

Captures per-run:
  - Goal, URL, browser_profile, proxy_config
  - Run ID, status, timing (started_at, finished_at)
  - Number of steps, result payload, error details
  - Each SSE PROGRESS event as a nested child action (stream mode)
  - Each step purpose description for debugging

Design principles:
  - Never throw: tracing failures are silently logged and dropped
  - Never alter: return values/exceptions always pass through unchanged
  - Minimal overhead: the original TinyFish SDK syntax is unchanged
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from missiontrace.adapters._base import BaseAdapter
from missiontrace.core.context import TraceContext
from missiontrace.core.models import ActionType
from missiontrace.core.sanitizer import Sanitizer
from missiontrace.utils.serialization import safe_serialize

logger = logging.getLogger("missiontrace.adapters.tinyfish")

_adapter_instance: TinyFishAdapter | None = None


# ---------------------------------------------------------------------------
# Helpers — build sanitized input/output dicts
# ---------------------------------------------------------------------------

def _build_agent_input(
    kwargs: dict,
    sanitizer: Sanitizer,
    method_name: str,
) -> dict[str, Any]:
    """Build sanitized input dict from an agent.run/queue/stream call."""
    data: dict[str, Any] = {
        "provider": "tinyfish",
        "method": method_name,
        "goal": kwargs.get("goal", ""),
        "url": kwargs.get("url", ""),
    }
    bp = kwargs.get("browser_profile")
    if bp is not None:
        data["browser_profile"] = str(bp.value) if hasattr(bp, "value") else str(bp)
    pc = kwargs.get("proxy_config")
    if pc is not None:
        data["proxy_config"] = safe_serialize(pc)
    return sanitizer.sanitize(data)


def _build_run_output(result: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized output from an AgentRunResponse (sync run)."""
    output: dict[str, Any] = {
        "run_id": getattr(result, "run_id", None),
        "status": str(getattr(result, "status", None)),
        "num_of_steps": getattr(result, "num_of_steps", None),
    }
    started = getattr(result, "started_at", None)
    if started is not None:
        output["started_at"] = str(started)
    finished = getattr(result, "finished_at", None)
    if finished is not None:
        output["finished_at"] = str(finished)

    res = getattr(result, "result", None)
    if res is not None:
        output["result"] = safe_serialize(res)

    err = getattr(result, "error", None)
    if err is not None:
        output["error"] = {
            "message": getattr(err, "message", str(err)),
            "category": str(getattr(err, "category", "UNKNOWN")),
            "retry_after": getattr(err, "retry_after", None),
            "help_url": getattr(err, "help_url", None),
            "help_message": getattr(err, "help_message", None),
        }
    return sanitizer.sanitize(output)


def _build_queue_output(result: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized output from an AgentRunAsyncResponse (queue)."""
    output: dict[str, Any] = {
        "run_id": getattr(result, "run_id", None),
    }
    err = getattr(result, "error", None)
    if err is not None:
        output["error"] = {
            "message": getattr(err, "message", str(err)),
            "category": str(getattr(err, "category", "UNKNOWN")),
        }
    return sanitizer.sanitize(output)


def _build_complete_event_output(event: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized output from a CompleteEvent (SSE stream end)."""
    output: dict[str, Any] = {
        "run_id": getattr(event, "run_id", None),
        "status": str(getattr(event, "status", None)),
        "timestamp": str(getattr(event, "timestamp", None)),
    }
    res = getattr(event, "result_json", None)
    if res is not None:
        output["result"] = safe_serialize(res)
    err = getattr(event, "error", None)
    if err is not None:
        output["error"] = {
            "message": getattr(err, "message", str(err)),
            "category": str(getattr(err, "category", "UNKNOWN")),
        }
    return sanitizer.sanitize(output)


def _build_progress_input(event: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized input from a ProgressEvent (one agent step)."""
    data: dict[str, Any] = {
        "run_id": getattr(event, "run_id", None),
        "purpose": getattr(event, "purpose", None),
        "timestamp": str(getattr(event, "timestamp", None)),
    }
    return sanitizer.sanitize(data)


def _build_runs_get_input(run_id: str, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized input for a runs.get() call."""
    return sanitizer.sanitize({"provider": "tinyfish", "method": "runs.get", "run_id": run_id})


def _build_runs_get_output(result: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized output from a Run (retrieve response)."""
    output: dict[str, Any] = {
        "run_id": getattr(result, "run_id", None),
        "status": str(getattr(result, "status", None)),
        "goal": getattr(result, "goal", None),
        "created_at": str(getattr(result, "created_at", None)),
        "started_at": str(getattr(result, "started_at", None)),
        "finished_at": str(getattr(result, "finished_at", None)),
        "streaming_url": getattr(result, "streaming_url", None),
    }
    res = getattr(result, "result", None)
    if res is not None:
        output["result"] = safe_serialize(res)
    err = getattr(result, "error", None)
    if err is not None:
        output["error"] = {
            "message": getattr(err, "message", str(err)),
            "category": str(getattr(err, "category", "UNKNOWN")),
        }
    bc = getattr(result, "browser_config", None)
    if bc is not None:
        output["browser_config"] = safe_serialize(bc)
    return sanitizer.sanitize(output)


def _build_runs_list_input(kwargs: dict, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized input for a runs.list() call."""
    data: dict[str, Any] = {"provider": "tinyfish", "method": "runs.list"}
    for key in ("cursor", "limit", "status", "goal", "created_after", "created_before", "sort_direction"):
        val = kwargs.get(key)
        if val is not None:
            data[key] = str(val) if not isinstance(val, (str, int)) else val
    return sanitizer.sanitize(data)


def _build_runs_list_output(result: Any, sanitizer: Sanitizer) -> dict[str, Any]:
    """Build sanitized output from a RunListResponse."""
    output: dict[str, Any] = {
        "num_runs": len(getattr(result, "data", [])),
    }
    pagination = getattr(result, "pagination", None)
    if pagination is not None:
        output["pagination"] = {
            "total": getattr(pagination, "total", None),
            "has_more": getattr(pagination, "has_more", None),
            "next_cursor": getattr(pagination, "next_cursor", None),
        }
    runs_data = getattr(result, "data", [])
    if runs_data:
        output["run_ids"] = [getattr(r, "run_id", None) for r in runs_data]
        output["statuses"] = [str(getattr(r, "status", None)) for r in runs_data]
    return sanitizer.sanitize(output)


# ---------------------------------------------------------------------------
# TinyFishAdapter — patches sync + async agent and runs resources
# ---------------------------------------------------------------------------

class TinyFishAdapter(BaseAdapter):
    """
    Adapter for the TinyFish Web Agent SDK.

    Instruments:
      - AgentResource.run / AsyncAgentResource.run     (sync + async)
      - AgentResource.queue / AsyncAgentResource.queue  (sync + async)
      - AgentResource.stream / AsyncAgentResource.stream (sync + async)
      - RunsResource.get / AsyncRunsResource.get        (sync + async)
      - RunsResource.list / AsyncRunsResource.list      (sync + async)

    Stream mode creates a parent action for the overall run, with nested
    child actions for each PROGRESS event (agent step) — giving full
    step-by-step debugging visibility.
    """

    def __init__(self, ctx: TraceContext, sanitizer: Sanitizer | None = None) -> None:
        super().__init__(ctx)
        self._sanitizer = sanitizer or Sanitizer()
        self._originals: dict[str, Any] = {}
        self._tinyfish_module: Any = None

    def instrument(self) -> None:
        tinyfish = self._require_package("tinyfish", "tinyfish")
        self._tinyfish_module = tinyfish

        # Agent resource patches
        self._patch_agent_run_sync(tinyfish)
        self._patch_agent_run_async(tinyfish)
        self._patch_agent_queue_sync(tinyfish)
        self._patch_agent_queue_async(tinyfish)
        self._patch_agent_stream_sync(tinyfish)
        self._patch_agent_stream_async(tinyfish)

        # Runs resource patches
        self._patch_runs_get_sync(tinyfish)
        self._patch_runs_get_async(tinyfish)
        self._patch_runs_list_sync(tinyfish)
        self._patch_runs_list_async(tinyfish)

        logger.info("TinyFish adapter instrumented (agent + runs, sync + async)")

    def uninstrument(self) -> None:
        if not self._tinyfish_module:
            return
        tf = self._tinyfish_module

        _restore_map = {
            "agent_run_sync": (lambda: tf.agent.AgentResource, "run"),
            "agent_run_async": (lambda: tf.agent.AsyncAgentResource, "run"),
            "agent_queue_sync": (lambda: tf.agent.AgentResource, "queue"),
            "agent_queue_async": (lambda: tf.agent.AsyncAgentResource, "queue"),
            "agent_stream_sync": (lambda: tf.agent.AgentResource, "stream"),
            "agent_stream_async": (lambda: tf.agent.AsyncAgentResource, "stream"),
            "runs_get_sync": (lambda: tf.runs.RunsResource, "get"),
            "runs_get_async": (lambda: tf.runs.AsyncRunsResource, "get"),
            "runs_list_sync": (lambda: tf.runs.RunsResource, "list"),
            "runs_list_async": (lambda: tf.runs.AsyncRunsResource, "list"),
        }

        for key, (cls_getter, method_name) in _restore_map.items():
            if key in self._originals:
                try:
                    setattr(cls_getter(), method_name, self._originals[key])
                except Exception:
                    pass

        self._originals.clear()
        logger.info("TinyFish adapter uninstrumented")

    # -- agent.run (sync) --------------------------------------------------

    def _patch_agent_run_sync(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AgentResource
        except AttributeError:
            logger.debug("AgentResource not found, skipping sync agent.run patch")
            return

        original = cls.run
        self._originals["agent_run_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.run")

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.run",
                input=input_data,
                intent="orchestration",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "agent.run"
            if kwargs.get("url"):
                action.tags["url"] = kwargs["url"]

            try:
                result = original(self_ref, *args, **kwargs)
                run_id = getattr(result, "run_id", None)
                if run_id:
                    action.tags["run_id"] = run_id
                output_data = _build_run_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.run = patched

    # -- agent.run (async) -------------------------------------------------

    def _patch_agent_run_async(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AsyncAgentResource
        except AttributeError:
            logger.debug("AsyncAgentResource not found, skipping async agent.run patch")
            return

        original = cls.run
        self._originals["agent_run_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.run")

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.run",
                input=input_data,
                intent="orchestration",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "agent.run"
            if kwargs.get("url"):
                action.tags["url"] = kwargs["url"]

            try:
                result = await original(self_ref, *args, **kwargs)
                run_id = getattr(result, "run_id", None)
                if run_id:
                    action.tags["run_id"] = run_id
                output_data = _build_run_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.run = patched

    # -- agent.queue (sync) ------------------------------------------------

    def _patch_agent_queue_sync(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AgentResource
        except AttributeError:
            return

        original = cls.queue
        self._originals["agent_queue_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.queue")

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.queue",
                input=input_data,
                intent="orchestration",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "agent.queue"
            if kwargs.get("url"):
                action.tags["url"] = kwargs["url"]

            try:
                result = original(self_ref, *args, **kwargs)
                run_id = getattr(result, "run_id", None)
                if run_id:
                    action.tags["run_id"] = run_id
                output_data = _build_queue_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.queue = patched

    # -- agent.queue (async) -----------------------------------------------

    def _patch_agent_queue_async(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AsyncAgentResource
        except AttributeError:
            return

        original = cls.queue
        self._originals["agent_queue_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.queue")

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.queue",
                input=input_data,
                intent="orchestration",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "agent.queue"
            if kwargs.get("url"):
                action.tags["url"] = kwargs["url"]

            try:
                result = await original(self_ref, *args, **kwargs)
                run_id = getattr(result, "run_id", None)
                if run_id:
                    action.tags["run_id"] = run_id
                output_data = _build_queue_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.queue = patched

    # -- agent.stream (sync) — captures every step as a child action -------

    def _patch_agent_stream_sync(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AgentResource
        except AttributeError:
            return

        original = cls.stream
        self._originals["agent_stream_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.stream")

            # Create parent action for the entire stream run
            parent_action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.stream",
                input=input_data,
                intent="orchestration",
            )
            parent_action.tags["provider"] = "tinyfish"
            parent_action.tags["method"] = "agent.stream"
            if kwargs.get("url"):
                parent_action.tags["url"] = kwargs["url"]

            # Wrap user callbacks to also capture as child actions
            original_on_started = kwargs.pop("on_started", None)
            original_on_streaming_url = kwargs.pop("on_streaming_url", None)
            original_on_progress = kwargs.pop("on_progress", None)
            original_on_heartbeat = kwargs.pop("on_heartbeat", None)
            original_on_complete = kwargs.pop("on_complete", None)

            step_counter = [0]

            def traced_on_started(event: Any) -> None:
                run_id = getattr(event, "run_id", None)
                if run_id:
                    parent_action.tags["run_id"] = run_id
                if original_on_started:
                    original_on_started(event)

            def traced_on_streaming_url(event: Any) -> None:
                streaming_url = getattr(event, "streaming_url", None)
                if streaming_url:
                    parent_action.tags["streaming_url"] = streaming_url
                if original_on_streaming_url:
                    original_on_streaming_url(event)

            def traced_on_progress(event: Any) -> None:
                step_counter[0] += 1
                purpose = getattr(event, "purpose", "unknown step")
                step_input = _build_progress_input(event, adapter._sanitizer)

                # Create a child action for this step
                step_action = adapter._ctx.start_action(
                    type=ActionType.TOOL_CALL,
                    name=f"tinyfish.agent.step[{step_counter[0]}]",
                    input=step_input,
                    intent="mutation",
                )
                step_action.tags["provider"] = "tinyfish"
                step_action.tags["step_number"] = str(step_counter[0])
                step_action.tags["purpose"] = purpose
                run_id = getattr(event, "run_id", None)
                if run_id:
                    step_action.tags["run_id"] = run_id

                adapter._ctx.end_action(
                    step_action,
                    output={"purpose": purpose, "step_number": step_counter[0]},
                )
                if original_on_progress:
                    original_on_progress(event)

            def traced_on_complete(event: Any) -> None:
                output_data = _build_complete_event_output(event, adapter._sanitizer)
                output_data["total_steps"] = step_counter[0]
                run_id = getattr(event, "run_id", None)
                if run_id:
                    parent_action.tags["run_id"] = run_id
                adapter._ctx.end_action(parent_action, output=output_data)
                if original_on_complete:
                    original_on_complete(event)

            try:
                stream = original(
                    self_ref,
                    *args,
                    on_started=traced_on_started,
                    on_streaming_url=traced_on_streaming_url,
                    on_progress=traced_on_progress,
                    on_heartbeat=original_on_heartbeat,
                    on_complete=traced_on_complete,
                    **kwargs,
                )
                return stream
            except Exception as exc:
                adapter._ctx.end_action(parent_action, error=exc)
                raise

        cls.stream = patched

    # -- agent.stream (async) — captures every step as a child action ------

    def _patch_agent_stream_async(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.agent.AsyncAgentResource
        except AttributeError:
            return

        original = cls.stream
        self._originals["agent_stream_async"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_agent_input(kwargs, adapter._sanitizer, "agent.stream")

            # Create parent action for the entire stream run
            parent_action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.agent.stream",
                input=input_data,
                intent="orchestration",
            )
            parent_action.tags["provider"] = "tinyfish"
            parent_action.tags["method"] = "agent.stream"
            if kwargs.get("url"):
                parent_action.tags["url"] = kwargs["url"]

            # Wrap user callbacks to also capture as child actions
            original_on_started = kwargs.pop("on_started", None)
            original_on_streaming_url = kwargs.pop("on_streaming_url", None)
            original_on_progress = kwargs.pop("on_progress", None)
            original_on_heartbeat = kwargs.pop("on_heartbeat", None)
            original_on_complete = kwargs.pop("on_complete", None)

            step_counter = [0]

            def traced_on_started(event: Any) -> None:
                run_id = getattr(event, "run_id", None)
                if run_id:
                    parent_action.tags["run_id"] = run_id
                if original_on_started:
                    original_on_started(event)

            def traced_on_streaming_url(event: Any) -> None:
                streaming_url = getattr(event, "streaming_url", None)
                if streaming_url:
                    parent_action.tags["streaming_url"] = streaming_url
                if original_on_streaming_url:
                    original_on_streaming_url(event)

            def traced_on_progress(event: Any) -> None:
                step_counter[0] += 1
                purpose = getattr(event, "purpose", "unknown step")
                step_input = _build_progress_input(event, adapter._sanitizer)

                step_action = adapter._ctx.start_action(
                    type=ActionType.TOOL_CALL,
                    name=f"tinyfish.agent.step[{step_counter[0]}]",
                    input=step_input,
                    intent="mutation",
                )
                step_action.tags["provider"] = "tinyfish"
                step_action.tags["step_number"] = str(step_counter[0])
                step_action.tags["purpose"] = purpose
                run_id = getattr(event, "run_id", None)
                if run_id:
                    step_action.tags["run_id"] = run_id

                adapter._ctx.end_action(
                    step_action,
                    output={"purpose": purpose, "step_number": step_counter[0]},
                )
                if original_on_progress:
                    original_on_progress(event)

            def traced_on_complete(event: Any) -> None:
                output_data = _build_complete_event_output(event, adapter._sanitizer)
                output_data["total_steps"] = step_counter[0]
                run_id = getattr(event, "run_id", None)
                if run_id:
                    parent_action.tags["run_id"] = run_id
                adapter._ctx.end_action(parent_action, output=output_data)
                if original_on_complete:
                    original_on_complete(event)

            try:
                stream = original(
                    self_ref,
                    *args,
                    on_started=traced_on_started,
                    on_streaming_url=traced_on_streaming_url,
                    on_progress=traced_on_progress,
                    on_heartbeat=original_on_heartbeat,
                    on_complete=traced_on_complete,
                    **kwargs,
                )
                return stream
            except Exception as exc:
                adapter._ctx.end_action(parent_action, error=exc)
                raise

        cls.stream = patched

    # -- runs.get (sync) ---------------------------------------------------

    def _patch_runs_get_sync(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.runs.RunsResource
        except AttributeError:
            return

        original = cls.get
        self._originals["runs_get_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_runs_get_input(run_id, adapter._sanitizer)

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.runs.get",
                input=input_data,
                intent="retrieval",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "runs.get"
            action.tags["run_id"] = run_id

            try:
                result = original(self_ref, run_id, *args, **kwargs)
                output_data = _build_runs_get_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.get = patched

    # -- runs.get (async) --------------------------------------------------

    def _patch_runs_get_async(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.runs.AsyncRunsResource
        except AttributeError:
            return

        original = cls.get
        self._originals["runs_get_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, run_id: str, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_runs_get_input(run_id, adapter._sanitizer)

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.runs.get",
                input=input_data,
                intent="retrieval",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "runs.get"
            action.tags["run_id"] = run_id

            try:
                result = await original(self_ref, run_id, *args, **kwargs)
                output_data = _build_runs_get_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.get = patched

    # -- runs.list (sync) --------------------------------------------------

    def _patch_runs_list_sync(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.runs.RunsResource
        except AttributeError:
            return

        original = cls.list
        self._originals["runs_list_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_runs_list_input(kwargs, adapter._sanitizer)

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.runs.list",
                input=input_data,
                intent="retrieval",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "runs.list"

            try:
                result = original(self_ref, *args, **kwargs)
                output_data = _build_runs_list_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.list = patched

    # -- runs.list (async) -------------------------------------------------

    def _patch_runs_list_async(self, tinyfish: Any) -> None:
        try:
            cls = tinyfish.runs.AsyncRunsResource
        except AttributeError:
            return

        original = cls.list
        self._originals["runs_list_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            input_data = _build_runs_list_input(kwargs, adapter._sanitizer)

            action = adapter._ctx.start_action(
                type=ActionType.TOOL_CALL,
                name="tinyfish.runs.list",
                input=input_data,
                intent="retrieval",
            )
            action.tags["provider"] = "tinyfish"
            action.tags["method"] = "runs.list"

            try:
                result = await original(self_ref, *args, **kwargs)
                output_data = _build_runs_list_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.list = patched


# ---------------------------------------------------------------------------
# Convenience module-level API
# ---------------------------------------------------------------------------

def instrument(ctx: TraceContext | None = None, sanitizer: Sanitizer | None = None) -> None:
    """
    Auto-instrument the TinyFish SDK (sync + async, agent + runs).

    Usage:
        import missiontrace
        missiontrace.init(api_key="mt_xxx", project="my-project")

        from missiontrace.adapters.tinyfish import instrument
        instrument()

        # Now all TinyFish calls are traced automatically
        from tinyfish import TinyFish
        client = TinyFish(api_key="tf_xxx")

        # Sync run — captured as a single action with run details
        response = client.agent.run(goal="Find price", url="https://example.com")

        # Stream — parent action + child action per step
        with client.agent.stream(goal="Extract data", url="https://example.com") as stream:
            for event in stream:
                print(event)

        # Queue + poll — each call captured independently
        queued = client.agent.queue(goal="Scrape page", url="https://example.com")
        run = client.runs.get(queued.run_id)
    """
    global _adapter_instance

    if ctx is None:
        import missiontrace
        ctx = missiontrace._ctx
        sanitizer = sanitizer or missiontrace._sanitizer

    if ctx is None:
        raise RuntimeError("MissionTrace not initialized. Call missiontrace.init() first.")

    _adapter_instance = TinyFishAdapter(ctx, sanitizer)
    _adapter_instance.instrument()


def uninstrument() -> None:
    """Remove TinyFish instrumentation and restore original SDK behavior."""
    global _adapter_instance
    if _adapter_instance:
        _adapter_instance.uninstrument()
        _adapter_instance = None
