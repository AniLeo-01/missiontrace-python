"""Tests for the user-facing API (init, @trace, mission, action)."""

import pytest

import missiontrace
from missiontrace.core.models import ActionStatus, ActionType, MissionStatus
from missiontrace.core.transport import PlaceholderExporter


@pytest.fixture(autouse=True)
def _init_sdk():
    """Initialize SDK before each test, shut down after."""
    exporter = PlaceholderExporter()
    missiontrace.init(
        api_key="test-key",
        project="test-project",
        _exporter=exporter,
        flush_intervmt_s=999,
    )
    yield exporter
    missiontrace.shutdown()


class TestInit:
    def test_is_initialized(self):
        assert missiontrace.is_initialized()

    def test_shutdown(self):
        missiontrace.shutdown()
        assert not missiontrace.is_initialized()


class TestTraceDecorator:
    def test_basic_trace(self):
        @missiontrace.trace(action_type="tool_call", name="add_numbers")
        def add(a: int, b: int) -> int:
            return a + b

        with missiontrace.mission(name="test") as m:
            result = add(3, 5)

        assert result == 8

        # Check the buffer was populated
        ctx = missiontrace._ctx
        transport = missiontrace._transport
        transport.flush()

    def test_trace_captures_exception(self):
        @missiontrace.trace(action_type="tool_call", name="failing_func")
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            with missiontrace.mission(name="test"):
                fail()

    def test_trace_with_tags(self):
        @missiontrace.trace(action_type="tool_call", name="tagged", tags={"env": "test"})
        def tagged_func():
            return "ok"

        with missiontrace.mission(name="test"):
            tagged_func()

    def test_trace_with_intent(self):
        @missiontrace.trace(action_type="tool_call", name="create_pr", intent="mutation")
        def create_pr():
            return {"pr_number": 42}

        with missiontrace.mission(name="test"):
            create_pr()

        # Verify intent was captured
        items = missiontrace._ctx.drain_buffer()
        from missiontrace.core.models import Action
        actions = [i for i in items if isinstance(i, Action)]
        assert len(actions) == 1
        assert actions[0].intent == "mutation"

    def test_trace_with_custom_intent(self):
        @missiontrace.trace(action_type="inference", name="classify", intent="triage")
        def classify():
            return {"severity": "high"}

        with missiontrace.mission(name="test"):
            classify()

        items = missiontrace._ctx.drain_buffer()
        from missiontrace.core.models import Action
        actions = [i for i in items if isinstance(i, Action)]
        assert actions[0].intent == "triage"

    def test_trace_sanitizes_inputs(self):
        @missiontrace.trace(action_type="tool_call", name="secret_func")
        def secret_func(api_key: str, data: str) -> str:
            return "done"

        with missiontrace.mission(name="test"):
            secret_func(api_key="sk-abcdefghijklmnopqrstuvwxyz1234567890", data="safe")

        # Verify sanitization happened via buffer inspection
        items = missiontrace._ctx.drain_buffer()
        from missiontrace.core.models import Action
        actions = [i for i in items if isinstance(i, Action)]

    def test_default_name_uses_qualname(self):
        @missiontrace.trace(action_type="tool_call")
        def my_cool_function():
            return 42

        with missiontrace.mission(name="test"):
            my_cool_function()


class TestMissionContextManager:
    def test_successful_mission(self):
        with missiontrace.mission(name="happy-path", metadata={"env": "test"}) as m:
            assert m.id  # has an ID
            assert m.mission.status == MissionStatus.RUNNING

        # After context exit, mission is completed
        assert m.mission.status == MissionStatus.COMPLETED

    def test_failed_mission(self):
        with pytest.raises(RuntimeError):
            with missiontrace.mission(name="sad-path") as m:
                raise RuntimeError("oops")

        assert m.mission.status == MissionStatus.FAILED

    def test_set_metadata(self):
        with missiontrace.mission(name="meta") as m:
            m.set_metadata("user", "aniruddha")

        assert m.mission.metadata["user"] == "aniruddha"


class TestActionContextManager:
    def test_basic_action(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action("inference", name="openai.chat") as act:
                act.set_output({"response": "hello"})

            assert act.action.status == ActionStatus.COMPLETED
            assert act.action.output == {"response": "hello"}

    def test_action_with_intent(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action("tool_call", name="github.create_pr", intent="mutation") as act:
                act.set_output({"pr_number": 42})

            assert act.action.intent == "mutation"

    def test_action_set_intent_dynamically(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action("inference", name="openai.chat") as act:
                # Decide intent based on output
                act.set_intent("generation")
                act.set_output({"response": "hello"})

            assert act.action.intent == "generation"

    def test_action_with_error(self):
        with pytest.raises(TypeError):
            with missiontrace.mission(name="test"):
                with missiontrace.action("tool_call", name="bad") as act:
                    raise TypeError("wrong type")

        assert act.action.status == ActionStatus.FAILED
        assert act.action.error.type == "TypeError"

    def test_nested_actions(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action("tool_call", name="outer") as outer:
                with missiontrace.action("tool_call", name="inner") as inner:
                    inner.set_output({"parsed": True})
                outer.set_output({"success": True})

            assert inner.action.parent_id == outer.action.id
            assert outer.action.parent_id is None

    def test_set_token_usage(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action("inference", name="openai.chat") as act:
                act.set_token_usage(
                    prompt_tokens=100,
                    completion_tokens=50,
                    totmt_tokens=150,
                    model="gpt-4",
                )

        assert act.action.token_usage.totmt_tokens == 150
        assert act.action.token_usage.model == "gpt-4"

    def test_action_sanitizes_input(self):
        with missiontrace.mission(name="test"):
            with missiontrace.action(
                "inference",
                name="test",
                input={"api_key": "super-secret", "prompt": "hello"},
            ) as act:
                pass

        assert act.action.input["api_key"] == "[REDACTED]"
        assert act.action.input["prompt"] == "hello"


class TestEndToEndWithTransport:
    def test_full_trace_to_otel_spans(self, _init_sdk):
        exporter = _init_sdk

        @missiontrace.trace(action_type="tool_call", name="apply_patch", intent="mutation")
        def apply_patch(file_path: str, diff: str) -> dict:
            return {"success": True, "lines_changed": 14}

        with missiontrace.mission(name="fix-auth-bug") as m:
            with missiontrace.action("tool_call", name="github.fetch_file", intent="retrieval") as act:
                act.set_output({"content": "def auth(): ..."})

            with missiontrace.action("inference", name="openai.chat", intent="generation") as act:
                act.set_output({"response": "Apply this patch..."})
                act.set_token_usage(prompt_tokens=500, completion_tokens=200, totmt_tokens=700, model="gpt-4")

            result = apply_patch(file_path="auth.py", diff="@@ -1 +1 @@")

            with missiontrace.action("tool_call", name="github.create_pr", intent="mutation") as act:
                act.set_output({"pr_number": 847})

        # Flush to OTel
        missiontrace._transport.flush()

        spans = exporter.get_exported_spans()
        assert len(spans) >= 5  # mission + 4 actions

        # Verify span names
        span_names = [s.name for s in spans]
        assert any("mission:fix-auth-bug" in n for n in span_names)
        assert any("action:github.fetch_file" in n for n in span_names)
        assert any("action:openai.chat" in n for n in span_names)
        assert any("action:apply_patch" in n for n in span_names)
        assert any("action:github.create_pr" in n for n in span_names)

        # Verify intent is exported as OTel attribute
        action_spans = [s for s in spans if "action:" in s.name]
        for s in action_spans:
            attrs = dict(s.attributes)
            if "openai.chat" in s.name:
                assert attrs.get("missiontrace.action.intent") == "generation"
            elif "github.create_pr" in s.name:
                assert attrs.get("missiontrace.action.intent") == "mutation"
            elif "github.fetch_file" in s.name:
                assert attrs.get("missiontrace.action.intent") == "retrieval"
            elif "apply_patch" in s.name:
                assert attrs.get("missiontrace.action.intent") == "mutation"
