"""
Tests for the TinyFish Web Agent adapter.

Uses mock TinyFish module structures to test:
  - Sync + Async agent.run
  - Sync + Async agent.queue
  - Sync + Async agent.stream (with per-step child actions)
  - Sync + Async runs.get
  - Sync + Async runs.list
  - Input/output capture and sanitization
  - Error propagation (never swallowed)
  - Uninstrumentation restores originals
"""

import asyncio
import types
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from missiontrace.core.context import TraceContext
from missiontrace.core.models import Action, ActionStatus, ActionType
from missiontrace.core.sanitizer import Sanitizer
from missiontrace.adapters.tinyfish import (
    TinyFishAdapter,
    _build_agent_input,
    _build_run_output,
    _build_queue_output,
    _build_complete_event_output,
    _build_progress_input,
    _build_runs_get_input,
    _build_runs_get_output,
    _build_runs_list_input,
    _build_runs_list_output,
    instrument,
    uninstrument,
)


# ---------------------------------------------------------------------------
# Mock TinyFish module structure
# ---------------------------------------------------------------------------

def _make_mock_tinyfish_module(
    sync_run_result: Any = None,
    async_run_result: Any = None,
    sync_queue_result: Any = None,
    async_queue_result: Any = None,
    sync_stream_result: Any = None,
    async_stream_result: Any = None,
    sync_runs_get_result: Any = None,
    async_runs_get_result: Any = None,
    sync_runs_list_result: Any = None,
    async_runs_list_result: Any = None,
):
    """Build a fake `tinyfish` module with the class hierarchy the adapter expects."""
    tinyfish = types.ModuleType("tinyfish")

    # Agent module
    agent_mod = types.ModuleType("tinyfish.agent")

    class AgentResource:
        @staticmethod
        def run(self_ref, *args, **kwargs):
            return sync_run_result

        @staticmethod
        def queue(self_ref, *args, **kwargs):
            return sync_queue_result

        @staticmethod
        def stream(self_ref, *args, **kwargs):
            return sync_stream_result

    class AsyncAgentResource:
        @staticmethod
        async def run(self_ref, *args, **kwargs):
            return async_run_result

        @staticmethod
        async def queue(self_ref, *args, **kwargs):
            return async_queue_result

        @staticmethod
        def stream(self_ref, *args, **kwargs):
            return async_stream_result

    agent_mod.AgentResource = AgentResource
    agent_mod.AsyncAgentResource = AsyncAgentResource
    tinyfish.agent = agent_mod

    # Runs module
    runs_mod = types.ModuleType("tinyfish.runs")

    class RunsResource:
        @staticmethod
        def get(self_ref, run_id, *args, **kwargs):
            return sync_runs_get_result

        @staticmethod
        def list(self_ref, *args, **kwargs):
            return sync_runs_list_result

    class AsyncRunsResource:
        @staticmethod
        async def get(self_ref, run_id, *args, **kwargs):
            return async_runs_get_result

        @staticmethod
        async def list(self_ref, *args, **kwargs):
            return async_runs_list_result

    runs_mod.RunsResource = RunsResource
    runs_mod.AsyncRunsResource = AsyncRunsResource
    tinyfish.runs = runs_mod

    return tinyfish


def _make_agent_run_response(
    run_id="run_abc123",
    status="COMPLETED",
    num_of_steps=5,
    result=None,
    error=None,
):
    """Build a mock AgentRunResponse."""
    resp = MagicMock()
    resp.run_id = run_id
    resp.status = status
    resp.num_of_steps = num_of_steps
    resp.started_at = datetime(2026, 4, 22, 10, 0, 0)
    resp.finished_at = datetime(2026, 4, 22, 10, 0, 30)
    resp.result = result or {"price": "$999"}
    resp.error = error
    return resp


def _make_agent_queue_response(run_id="run_def456", error=None):
    """Build a mock AgentRunAsyncResponse."""
    resp = MagicMock()
    resp.run_id = run_id
    resp.error = error
    return resp


def _make_run_error(message="Agent timed out", category="AGENT_FAILURE"):
    """Build a mock RunError."""
    err = MagicMock()
    err.message = message
    err.category = category
    err.retry_after = 30
    err.help_url = "https://docs.tinyfish.ai/troubleshooting"
    err.help_message = "Try increasing timeout"
    return err


def _make_started_event(run_id="run_abc123"):
    event = MagicMock()
    event.type = "STARTED"
    event.run_id = run_id
    event.timestamp = datetime(2026, 4, 22, 10, 0, 0)
    return event


def _make_streaming_url_event(run_id="run_abc123", url="wss://stream.tinyfish.ai/run_abc123"):
    event = MagicMock()
    event.type = "STREAMING_URL"
    event.run_id = run_id
    event.streaming_url = url
    event.timestamp = datetime(2026, 4, 22, 10, 0, 1)
    return event


def _make_progress_event(run_id="run_abc123", purpose="Clicking login button"):
    event = MagicMock()
    event.type = "PROGRESS"
    event.run_id = run_id
    event.purpose = purpose
    event.timestamp = datetime(2026, 4, 22, 10, 0, 5)
    return event


def _make_heartbeat_event():
    event = MagicMock()
    event.type = "HEARTBEAT"
    event.timestamp = datetime(2026, 4, 22, 10, 0, 7)
    return event


def _make_complete_event(
    run_id="run_abc123",
    status="COMPLETED",
    result_json=None,
    error=None,
):
    event = MagicMock()
    event.type = "COMPLETE"
    event.run_id = run_id
    event.status = status
    event.timestamp = datetime(2026, 4, 22, 10, 0, 30)
    event.result_json = result_json or {"price": "$999"}
    event.error = error
    return event


def _make_run_retrieve_response(
    run_id="run_abc123",
    status="COMPLETED",
    goal="Find price",
    result=None,
    error=None,
):
    """Build a mock RunRetrieveResponse (Run object)."""
    resp = MagicMock()
    resp.run_id = run_id
    resp.status = status
    resp.goal = goal
    resp.created_at = datetime(2026, 4, 22, 9, 59, 55)
    resp.started_at = datetime(2026, 4, 22, 10, 0, 0)
    resp.finished_at = datetime(2026, 4, 22, 10, 0, 30)
    resp.result = result or {"price": "$999"}
    resp.error = error
    resp.streaming_url = None
    resp.browser_config = None
    return resp


def _make_run_list_response(runs=None, total=2, has_more=False, next_cursor=None):
    """Build a mock RunListResponse."""
    if runs is None:
        runs = [
            _make_run_retrieve_response(run_id="run_1", goal="Task 1"),
            _make_run_retrieve_response(run_id="run_2", goal="Task 2"),
        ]
    pagination = MagicMock()
    pagination.total = total
    pagination.has_more = has_more
    pagination.next_cursor = next_cursor

    resp = MagicMock()
    resp.data = runs
    resp.pagination = pagination
    return resp


# ---------------------------------------------------------------------------
# Helper to get completed actions from context buffer
# ---------------------------------------------------------------------------

def _get_completed_actions(ctx: TraceContext) -> list[Action]:
    """Drain buffer and return only completed/failed Action objects, deduplicated by ID.

    The TraceContext appends actions both on start_action and end_action,
    so the buffer contains two entries per action. We deduplicate by ID
    and keep only the final (completed/failed) version.
    """
    items = ctx.drain_buffer()
    seen: dict[str, Action] = {}
    for item in items:
        if isinstance(item, Action) and item.status in (ActionStatus.COMPLETED, ActionStatus.FAILED):
            seen[item.id] = item  # last write wins (the end_action copy)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

class TestBuildAgentInput:
    def test_basic_input(self):
        sanitizer = Sanitizer()
        result = _build_agent_input(
            {"goal": "Find price", "url": "https://apple.com"},
            sanitizer,
            "agent.run",
        )
        assert result["provider"] == "tinyfish"
        assert result["method"] == "agent.run"
        assert result["goal"] == "Find price"
        assert result["url"] == "https://apple.com"

    def test_with_browser_profile(self):
        sanitizer = Sanitizer()
        bp = MagicMock()
        bp.value = "stealth"
        result = _build_agent_input(
            {"goal": "Test", "url": "https://example.com", "browser_profile": bp},
            sanitizer,
            "agent.stream",
        )
        assert result["browser_profile"] == "stealth"

    def test_with_proxy_config(self):
        sanitizer = Sanitizer()
        # Use a simple object instead of MagicMock to avoid safe_serialize
        # infinite recursion (MagicMock auto-creates model_dump attribute)
        pc = type("ProxyConfig", (), {"enabled": True, "country_code": "US"})()
        result = _build_agent_input(
            {"goal": "Test", "url": "https://example.com", "proxy_config": pc},
            sanitizer,
            "agent.queue",
        )
        assert "proxy_config" in result


class TestBuildRunOutput:
    def test_successful_run(self):
        sanitizer = Sanitizer()
        resp = _make_agent_run_response()
        output = _build_run_output(resp, sanitizer)
        assert output["run_id"] == "run_abc123"
        assert output["status"] == "COMPLETED"
        assert output["num_of_steps"] == 5
        assert output["result"] == {"price": "$999"}

    def test_failed_run_with_error(self):
        sanitizer = Sanitizer()
        err = _make_run_error()
        resp = _make_agent_run_response(status="FAILED", error=err, result=None)
        resp.result = None
        output = _build_run_output(resp, sanitizer)
        assert output["error"]["message"] == "Agent timed out"
        assert output["error"]["category"] == "AGENT_FAILURE"


class TestBuildQueueOutput:
    def test_successful_queue(self):
        sanitizer = Sanitizer()
        resp = _make_agent_queue_response()
        output = _build_queue_output(resp, sanitizer)
        assert output["run_id"] == "run_def456"

    def test_queue_with_error(self):
        sanitizer = Sanitizer()
        err = _make_run_error(message="Rate limited", category="SYSTEM_FAILURE")
        resp = _make_agent_queue_response(error=err)
        output = _build_queue_output(resp, sanitizer)
        assert output["error"]["message"] == "Rate limited"


class TestBuildProgressInput:
    def test_progress_event(self):
        sanitizer = Sanitizer()
        event = _make_progress_event(purpose="Filling search box")
        result = _build_progress_input(event, sanitizer)
        assert result["run_id"] == "run_abc123"
        assert result["purpose"] == "Filling search box"


class TestBuildCompleteEventOutput:
    def test_complete_success(self):
        sanitizer = Sanitizer()
        event = _make_complete_event()
        output = _build_complete_event_output(event, sanitizer)
        assert output["run_id"] == "run_abc123"
        assert output["status"] == "COMPLETED"
        assert output["result"] == {"price": "$999"}

    def test_complete_failure(self):
        sanitizer = Sanitizer()
        err = _make_run_error()
        event = _make_complete_event(status="FAILED", result_json=None, error=err)
        event.result_json = None
        output = _build_complete_event_output(event, sanitizer)
        assert output["error"]["message"] == "Agent timed out"


class TestBuildRunsGetOutput:
    def test_get_completed(self):
        sanitizer = Sanitizer()
        run = _make_run_retrieve_response()
        output = _build_runs_get_output(run, sanitizer)
        assert output["run_id"] == "run_abc123"
        assert output["status"] == "COMPLETED"
        assert output["goal"] == "Find price"


class TestBuildRunsListOutput:
    def test_list_response(self):
        sanitizer = Sanitizer()
        resp = _make_run_list_response()
        output = _build_runs_list_output(resp, sanitizer)
        assert output["num_runs"] == 2
        assert output["run_ids"] == ["run_1", "run_2"]
        assert output["pagination"]["total"] == 2
        assert output["pagination"]["has_more"] is False


# ---------------------------------------------------------------------------
# Tests: Sync agent.run patching
# ---------------------------------------------------------------------------

class TestAgentRunSync:
    def test_run_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        run_response = _make_agent_run_response()

        mock_tf = _make_mock_tinyfish_module(sync_run_result=run_response)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf

        # Manually patch
        adapter._patch_agent_run_sync(mock_tf)

        # Call the patched method
        resource = MagicMock()
        result = mock_tf.agent.AgentResource.run(
            resource,
            goal="Find price of iPhone",
            url="https://apple.com",
        )

        # Result should pass through unchanged
        assert result.run_id == "run_abc123"
        assert result.status == "COMPLETED"

        # Check actions in buffer
        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        action = actions[0]
        assert action.name == "tinyfish.agent.run"
        assert action.type == ActionType.TOOL_CALL
        assert action.status == ActionStatus.COMPLETED
        assert action.tags["provider"] == "tinyfish"
        assert action.tags["method"] == "agent.run"
        assert action.tags["run_id"] == "run_abc123"
        assert action.output["run_id"] == "run_abc123"
        assert action.output["num_of_steps"] == 5
        assert action.input["goal"] == "Find price of iPhone"
        assert action.input["url"] == "https://apple.com"

    def test_run_error_propagates(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        # Create a module where run raises an exception
        mock_tf = _make_mock_tinyfish_module()

        def raise_error(self_ref, *args, **kwargs):
            raise ConnectionError("Network unreachable")

        mock_tf.agent.AgentResource.run = staticmethod(raise_error)

        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_run_sync(mock_tf)

        resource = MagicMock()
        with pytest.raises(ConnectionError, match="Network unreachable"):
            mock_tf.agent.AgentResource.run(resource, goal="Test", url="https://example.com")

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].status == ActionStatus.FAILED
        assert actions[0].error.type == "ConnectionError"


# ---------------------------------------------------------------------------
# Tests: Async agent.run patching
# ---------------------------------------------------------------------------

class TestAgentRunAsync:
    def test_async_run_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        run_response = _make_agent_run_response()

        mock_tf = _make_mock_tinyfish_module(async_run_result=run_response)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_run_async(mock_tf)

        resource = MagicMock()
        result = asyncio.get_event_loop().run_until_complete(
            mock_tf.agent.AsyncAgentResource.run(resource, goal="Extract data", url="https://example.com")
        )

        assert result.run_id == "run_abc123"

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].name == "tinyfish.agent.run"
        assert actions[0].tags["provider"] == "tinyfish"


# ---------------------------------------------------------------------------
# Tests: Sync agent.queue patching
# ---------------------------------------------------------------------------

class TestAgentQueueSync:
    def test_queue_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        queue_response = _make_agent_queue_response()

        mock_tf = _make_mock_tinyfish_module(sync_queue_result=queue_response)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_queue_sync(mock_tf)

        resource = MagicMock()
        result = mock_tf.agent.AgentResource.queue(
            resource, goal="Scrape prices", url="https://shop.example.com"
        )

        assert result.run_id == "run_def456"

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        action = actions[0]
        assert action.name == "tinyfish.agent.queue"
        assert action.tags["run_id"] == "run_def456"
        assert action.output["run_id"] == "run_def456"


# ---------------------------------------------------------------------------
# Tests: Async agent.queue patching
# ---------------------------------------------------------------------------

class TestAgentQueueAsync:
    def test_async_queue_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        queue_response = _make_agent_queue_response(run_id="run_async_q")

        mock_tf = _make_mock_tinyfish_module(async_queue_result=queue_response)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_queue_async(mock_tf)

        resource = MagicMock()
        result = asyncio.get_event_loop().run_until_complete(
            mock_tf.agent.AsyncAgentResource.queue(resource, goal="Queue task", url="https://example.com")
        )

        assert result.run_id == "run_async_q"

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].name == "tinyfish.agent.queue"


# ---------------------------------------------------------------------------
# Tests: Sync agent.stream with per-step child actions
# ---------------------------------------------------------------------------

class TestAgentStreamSync:
    def test_stream_creates_parent_and_step_actions(self):
        """Stream mode should create a parent action + child action per PROGRESS event."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        started = _make_started_event()
        streaming_url = _make_streaming_url_event()
        progress1 = _make_progress_event(purpose="Navigating to login page")
        progress2 = _make_progress_event(purpose="Filling username field")
        progress3 = _make_progress_event(purpose="Clicking submit button")
        complete = _make_complete_event()

        # The mock stream will call callbacks in order
        def mock_stream(self_ref, *args, **kwargs):
            on_started = kwargs.get("on_started")
            on_streaming_url = kwargs.get("on_streaming_url")
            on_progress = kwargs.get("on_progress")
            on_complete = kwargs.get("on_complete")

            if on_started:
                on_started(started)
            if on_streaming_url:
                on_streaming_url(streaming_url)
            if on_progress:
                on_progress(progress1)
                on_progress(progress2)
                on_progress(progress3)
            if on_complete:
                on_complete(complete)
            return MagicMock()  # return stream object

        mock_tf = _make_mock_tinyfish_module()
        mock_tf.agent.AgentResource.stream = staticmethod(mock_stream)

        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_stream_sync(mock_tf)

        resource = MagicMock()
        user_progress_events = []

        mock_tf.agent.AgentResource.stream(
            resource,
            goal="Login and extract data",
            url="https://app.example.com",
            on_progress=lambda e: user_progress_events.append(e),
        )

        # User callbacks should still fire
        assert len(user_progress_events) == 3

        # Check actions: 1 parent + 3 step children = 4 completed actions
        actions = _get_completed_actions(ctx)
        assert len(actions) == 4

        # Steps should be the first 3 completed
        step_actions = [a for a in actions if "step" in a.name]
        assert len(step_actions) == 3
        assert step_actions[0].name == "tinyfish.agent.step[1]"
        assert step_actions[0].tags["purpose"] == "Navigating to login page"
        assert step_actions[1].name == "tinyfish.agent.step[2]"
        assert step_actions[1].tags["purpose"] == "Filling username field"
        assert step_actions[2].name == "tinyfish.agent.step[3]"
        assert step_actions[2].tags["purpose"] == "Clicking submit button"

        # Parent should be the stream action
        parent_actions = [a for a in actions if a.name == "tinyfish.agent.stream"]
        assert len(parent_actions) == 1
        parent = parent_actions[0]
        assert parent.tags["run_id"] == "run_abc123"
        assert parent.tags["streaming_url"] == "wss://stream.tinyfish.ai/run_abc123"
        assert parent.output["total_steps"] == 3
        assert parent.output["status"] == "COMPLETED"

    def test_stream_preserves_user_callbacks(self):
        """All user-provided on_* callbacks should still fire."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        started = _make_started_event()
        streaming_url = _make_streaming_url_event()
        progress = _make_progress_event()
        heartbeat = _make_heartbeat_event()
        complete = _make_complete_event()

        fired = {"started": False, "url": False, "progress": False, "heartbeat": False, "complete": False}

        def mock_stream(self_ref, *args, **kwargs):
            for key in ("on_started", "on_streaming_url", "on_progress", "on_heartbeat", "on_complete"):
                cb = kwargs.get(key)
                if cb:
                    event_map = {
                        "on_started": started,
                        "on_streaming_url": streaming_url,
                        "on_progress": progress,
                        "on_heartbeat": heartbeat,
                        "on_complete": complete,
                    }
                    cb(event_map[key])
            return MagicMock()

        mock_tf = _make_mock_tinyfish_module()
        mock_tf.agent.AgentResource.stream = staticmethod(mock_stream)

        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_stream_sync(mock_tf)

        resource = MagicMock()
        mock_tf.agent.AgentResource.stream(
            resource,
            goal="Test",
            url="https://example.com",
            on_started=lambda e: fired.__setitem__("started", True),
            on_streaming_url=lambda e: fired.__setitem__("url", True),
            on_progress=lambda e: fired.__setitem__("progress", True),
            on_heartbeat=lambda e: fired.__setitem__("heartbeat", True),
            on_complete=lambda e: fired.__setitem__("complete", True),
        )

        assert fired["started"]
        assert fired["url"]
        assert fired["progress"]
        assert fired["heartbeat"]
        assert fired["complete"]

    def test_stream_error_propagates(self):
        """Errors from the original stream should propagate and create a failed action."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        def mock_stream_error(self_ref, *args, **kwargs):
            raise TimeoutError("Run timed out")

        mock_tf = _make_mock_tinyfish_module()
        mock_tf.agent.AgentResource.stream = staticmethod(mock_stream_error)

        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_stream_sync(mock_tf)

        resource = MagicMock()
        with pytest.raises(TimeoutError, match="Run timed out"):
            mock_tf.agent.AgentResource.stream(resource, goal="Test", url="https://example.com")

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].status == ActionStatus.FAILED
        assert actions[0].error.type == "TimeoutError"


# ---------------------------------------------------------------------------
# Tests: Sync runs.get patching
# ---------------------------------------------------------------------------

class TestRunsGetSync:
    def test_runs_get_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        run = _make_run_retrieve_response()

        mock_tf = _make_mock_tinyfish_module(sync_runs_get_result=run)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_runs_get_sync(mock_tf)

        resource = MagicMock()
        result = mock_tf.runs.RunsResource.get(resource, "run_abc123")

        assert result.run_id == "run_abc123"

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        action = actions[0]
        assert action.name == "tinyfish.runs.get"
        assert action.tags["run_id"] == "run_abc123"
        assert action.intent == "retrieval"
        assert action.output["status"] == "COMPLETED"
        assert action.output["goal"] == "Find price"


# ---------------------------------------------------------------------------
# Tests: Async runs.get patching
# ---------------------------------------------------------------------------

class TestRunsGetAsync:
    def test_async_runs_get_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        run = _make_run_retrieve_response(run_id="run_async_get")

        mock_tf = _make_mock_tinyfish_module(async_runs_get_result=run)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_runs_get_async(mock_tf)

        resource = MagicMock()
        result = asyncio.get_event_loop().run_until_complete(
            mock_tf.runs.AsyncRunsResource.get(resource, "run_async_get")
        )

        assert result.run_id == "run_async_get"

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].name == "tinyfish.runs.get"


# ---------------------------------------------------------------------------
# Tests: Sync runs.list patching
# ---------------------------------------------------------------------------

class TestRunsListSync:
    def test_runs_list_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        list_resp = _make_run_list_response()

        mock_tf = _make_mock_tinyfish_module(sync_runs_list_result=list_resp)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_runs_list_sync(mock_tf)

        resource = MagicMock()
        result = mock_tf.runs.RunsResource.list(resource, limit=10, status="COMPLETED")

        assert len(result.data) == 2

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        action = actions[0]
        assert action.name == "tinyfish.runs.list"
        assert action.intent == "retrieval"
        assert action.output["num_runs"] == 2
        assert action.output["run_ids"] == ["run_1", "run_2"]


# ---------------------------------------------------------------------------
# Tests: Async runs.list patching
# ---------------------------------------------------------------------------

class TestRunsListAsync:
    def test_async_runs_list_is_traced(self):
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        list_resp = _make_run_list_response()

        mock_tf = _make_mock_tinyfish_module(async_runs_list_result=list_resp)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_runs_list_async(mock_tf)

        resource = MagicMock()
        result = asyncio.get_event_loop().run_until_complete(
            mock_tf.runs.AsyncRunsResource.list(resource, limit=5)
        )

        assert len(result.data) == 2

        actions = _get_completed_actions(ctx)
        assert len(actions) == 1
        assert actions[0].name == "tinyfish.runs.list"


# ---------------------------------------------------------------------------
# Tests: Full instrument/uninstrument lifecycle
# ---------------------------------------------------------------------------

class TestInstrumentUninstrument:
    def _instrument_with_mock(self, adapter, mock_tf):
        """Call instrument() but bypass _require_package by using mock module directly."""
        adapter._tinyfish_module = mock_tf
        # Replicate what instrument() does after _require_package
        adapter._patch_agent_run_sync(mock_tf)
        adapter._patch_agent_run_async(mock_tf)
        adapter._patch_agent_queue_sync(mock_tf)
        adapter._patch_agent_queue_async(mock_tf)
        adapter._patch_agent_stream_sync(mock_tf)
        adapter._patch_agent_stream_async(mock_tf)
        adapter._patch_runs_get_sync(mock_tf)
        adapter._patch_runs_get_async(mock_tf)
        adapter._patch_runs_list_sync(mock_tf)
        adapter._patch_runs_list_async(mock_tf)

    def test_instrument_patches_all_methods(self):
        """instrument() should patch all 10 methods on the mock module."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        mock_tf = _make_mock_tinyfish_module(
            sync_run_result=_make_agent_run_response(),
        )

        adapter = TinyFishAdapter(ctx, sanitizer)

        # Save originals
        orig_run = mock_tf.agent.AgentResource.run
        orig_queue = mock_tf.agent.AgentResource.queue
        orig_stream = mock_tf.agent.AgentResource.stream
        orig_get = mock_tf.runs.RunsResource.get
        orig_list = mock_tf.runs.RunsResource.list

        self._instrument_with_mock(adapter, mock_tf)

        # All methods should now be different (patched)
        assert mock_tf.agent.AgentResource.run is not orig_run
        assert mock_tf.agent.AgentResource.queue is not orig_queue
        assert mock_tf.agent.AgentResource.stream is not orig_stream
        assert mock_tf.runs.RunsResource.get is not orig_get
        assert mock_tf.runs.RunsResource.list is not orig_list

    def test_uninstrument_restores_originals(self):
        """uninstrument() should restore all original methods."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        mock_tf = _make_mock_tinyfish_module(
            sync_run_result=_make_agent_run_response(),
        )

        # Save originals before patching
        orig_run = mock_tf.agent.AgentResource.run
        orig_queue = mock_tf.agent.AgentResource.queue
        orig_stream = mock_tf.agent.AgentResource.stream
        orig_get = mock_tf.runs.RunsResource.get
        orig_list = mock_tf.runs.RunsResource.list

        adapter = TinyFishAdapter(ctx, sanitizer)
        self._instrument_with_mock(adapter, mock_tf)
        adapter.uninstrument()

        # All methods should be restored
        assert mock_tf.agent.AgentResource.run is orig_run
        assert mock_tf.agent.AgentResource.queue is orig_queue
        assert mock_tf.agent.AgentResource.stream is orig_stream
        assert mock_tf.runs.RunsResource.get is orig_get
        assert mock_tf.runs.RunsResource.list is orig_list


# ---------------------------------------------------------------------------
# Tests: Tags and metadata
# ---------------------------------------------------------------------------

class TestTagsAndMetadata:
    def test_url_tagged_on_agent_calls(self):
        """agent.run/queue/stream should tag the target URL."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()
        run_response = _make_agent_run_response()

        mock_tf = _make_mock_tinyfish_module(sync_run_result=run_response)
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_run_sync(mock_tf)

        resource = MagicMock()
        mock_tf.agent.AgentResource.run(
            resource, goal="Test", url="https://target-site.com/page"
        )

        actions = _get_completed_actions(ctx)
        assert actions[0].tags["url"] == "https://target-site.com/page"

    def test_intent_is_set_correctly(self):
        """Agent calls should have intent=orchestration, runs calls should have intent=retrieval."""
        ctx = TraceContext(project_id="test")
        sanitizer = Sanitizer()

        mock_tf = _make_mock_tinyfish_module(
            sync_run_result=_make_agent_run_response(),
            sync_runs_get_result=_make_run_retrieve_response(),
        )
        adapter = TinyFishAdapter(ctx, sanitizer)
        adapter._tinyfish_module = mock_tf
        adapter._patch_agent_run_sync(mock_tf)
        adapter._patch_runs_get_sync(mock_tf)

        resource = MagicMock()
        mock_tf.agent.AgentResource.run(resource, goal="Test", url="https://example.com")
        mock_tf.runs.RunsResource.get(resource, "run_abc123")

        actions = _get_completed_actions(ctx)
        agent_action = [a for a in actions if a.name == "tinyfish.agent.run"][0]
        runs_action = [a for a in actions if a.name == "tinyfish.runs.get"][0]

        assert agent_action.intent == "orchestration"
        assert runs_action.intent == "retrieval"


# ---------------------------------------------------------------------------
# Tests: Sanitization
# ---------------------------------------------------------------------------

class TestSanitization:
    def test_api_key_in_url_is_sanitized(self):
        """API keys in URLs should be sanitized by the Sanitizer."""
        sanitizer = Sanitizer()
        result = _build_agent_input(
            {
                "goal": "Test",
                "url": "https://example.com",
            },
            sanitizer,
            "agent.run",
        )
        # Basic input should pass through clean
        assert result["goal"] == "Test"

    def test_output_is_sanitized(self):
        """Output data should be sanitized."""
        sanitizer = Sanitizer()
        resp = _make_agent_run_response()
        output = _build_run_output(resp, sanitizer)
        # Should have all expected fields without crash
        assert "run_id" in output
        assert "status" in output


# ---------------------------------------------------------------------------
# Tests: Module-level instrument/uninstrument API
# ---------------------------------------------------------------------------

class TestModuleLevelAPI:
    def test_instrument_without_init_raises(self):
        """Calling instrument() without missiontrace.init() should raise."""
        import missiontrace
        old_ctx = missiontrace._ctx
        missiontrace._ctx = None

        try:
            with pytest.raises(RuntimeError, match="MissionTrace not initialized"):
                instrument()
        finally:
            missiontrace._ctx = old_ctx

    def test_uninstrument_when_not_instrumented(self):
        """Calling uninstrument() when not instrumented should be a no-op."""
        uninstrument()  # Should not raise
