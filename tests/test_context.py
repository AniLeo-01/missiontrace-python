"""Tests for context propagation (TraceContext + contextvars)."""

from missiontrace.core.context import (
    TraceContext,
    get_current_action,
    get_current_mission,
)
from missiontrace.core.models import (
    Action,
    ActionStatus,
    ActionType,
    Mission,
    MissionStatus,
)


class TestTraceContextMission:
    def test_start_and_end_mission(self):
        ctx = TraceContext(project_id="test-proj")
        m = ctx.start_mission(name="test-mission")

        assert m.status == MissionStatus.RUNNING
        assert m.project_id == "test-proj"
        assert get_current_mission() is m

        ctx.end_mission(m)
        assert m.status == MissionStatus.COMPLETED
        assert m.ended_at is not None

    def test_mission_failure(self):
        ctx = TraceContext()
        m = ctx.start_mission()
        ctx.end_mission(m, MissionStatus.FAILED)
        assert m.status == MissionStatus.FAILED


class TestTraceContextAction:
    def test_start_and_end_action(self):
        ctx = TraceContext()
        m = ctx.start_mission(name="m1")
        a = ctx.start_action(type=ActionType.INFERENCE, name="openai.chat")

        assert a.mission_id == m.id
        assert a.parent_id is None
        assert a.status == ActionStatus.RUNNING
        assert get_current_action() is a

        ctx.end_action(a, output={"response": "hello"})
        assert a.status == ActionStatus.COMPLETED
        assert a.output == {"response": "hello"}
        assert a.duration_ms is not None
        assert a.duration_ms >= 0

        ctx.end_mission(m)

    def test_action_with_intent(self):
        ctx = TraceContext()
        m = ctx.start_mission(name="m1")
        a = ctx.start_action(
            type=ActionType.TOOL_CALL, name="github.create_pr", intent="mutation"
        )
        assert a.intent == "mutation"
        ctx.end_action(a)
        ctx.end_mission(m)

    def test_action_with_custom_intent(self):
        ctx = TraceContext()
        m = ctx.start_mission()
        a = ctx.start_action(
            type=ActionType.INFERENCE, name="classify", intent="my_custom_step"
        )
        assert a.intent == "my_custom_step"
        ctx.end_action(a)
        ctx.end_mission(m)

    def test_action_intent_defaults_none(self):
        ctx = TraceContext()
        m = ctx.start_mission()
        a = ctx.start_action(type=ActionType.TOOL_CALL, name="test")
        assert a.intent is None
        ctx.end_action(a)
        ctx.end_mission(m)

    def test_nested_actions(self):
        """Parent-child relationships must propagate via contextvars."""
        ctx = TraceContext()
        m = ctx.start_mission(name="m1")

        outer = ctx.start_action(type=ActionType.TOOL_CALL, name="apply_patch")
        assert outer.parent_id is None

        inner = ctx.start_action(type=ActionType.TOOL_CALL, name="parse_diff")
        assert inner.parent_id == outer.id
        assert inner.mission_id == m.id

        # End inner first
        ctx.end_action(inner, output={"parsed": True})
        assert get_current_action() is outer  # restored

        ctx.end_action(outer, output={"success": True})
        ctx.end_mission(m)

    def test_action_with_error(self):
        ctx = TraceContext()
        m = ctx.start_mission()
        a = ctx.start_action(type=ActionType.INFERENCE, name="bad_call")

        err = ValueError("something went wrong")
        ctx.end_action(a, error=err)

        assert a.status == ActionStatus.FAILED
        assert a.error is not None
        assert a.error.type == "ValueError"
        assert "something went wrong" in a.error.message
        ctx.end_mission(m)

    def test_deeply_nested_actions(self):
        """3+ levels of nesting should work correctly."""
        ctx = TraceContext()
        m = ctx.start_mission()

        a1 = ctx.start_action(type=ActionType.TOOL_CALL, name="level1")
        a2 = ctx.start_action(type=ActionType.TOOL_CALL, name="level2")
        a3 = ctx.start_action(type=ActionType.TOOL_CALL, name="level3")

        assert a3.parent_id == a2.id
        assert a2.parent_id == a1.id
        assert a1.parent_id is None

        ctx.end_action(a3)
        assert get_current_action() is a2

        ctx.end_action(a2)
        assert get_current_action() is a1

        ctx.end_action(a1)
        ctx.end_mission(m)


class TestBufferDrain:
    def test_drain_returns_all_items(self):
        ctx = TraceContext()
        m = ctx.start_mission()
        a = ctx.start_action(type=ActionType.TOOL_CALL, name="test")
        ctx.end_action(a)
        ctx.end_mission(m)

        items = ctx.drain_buffer()
        assert len(items) == 2  # mission + action

        # Buffer is now empty
        assert len(ctx.drain_buffer()) == 0
