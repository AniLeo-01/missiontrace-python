"""Tests for OpenTelemetry transport layer."""

import time

from missiontrace.core.context import TraceContext
from missiontrace.core.models import (
    ActionType,
    MissionStatus,
)
from missiontrace.core.transport import OTelTransport, PlaceholderExporter


class TestPlaceholderExporter:
    def test_export_and_retrieve(self):
        exporter = PlaceholderExporter()
        # Simulate exporting (the real test is via OTelTransport)
        assert exporter.get_exported_spans() == []
        exporter.clear()

    def test_force_flush(self):
        exporter = PlaceholderExporter()
        assert exporter.force_flush() is True


class TestOTelTransport:
    def _make_transport(self) -> tuple[OTelTransport, PlaceholderExporter]:
        exporter = PlaceholderExporter()
        transport = OTelTransport(
            service_name="test-sdk",
            flush_intervmt_s=999,  # disable auto-flush for tests
            exporter=exporter,
        )
        return transport, exporter

    def test_export_mission(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext(project_id="proj-1")
        transport.attach(ctx)

        m = ctx.start_mission(name="test-mission")
        ctx.end_mission(m)

        count = transport.flush()
        assert count == 1

        spans = exporter.get_exported_spans()
        assert len(spans) == 1
        assert "mission:test-mission" in spans[0].name
        attrs = dict(spans[0].attributes)
        assert attrs["missiontrace.type"] == "mission"
        assert attrs["missiontrace.mission.id"] == m.id

    def test_export_action(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext(project_id="proj-1")
        transport.attach(ctx)

        m = ctx.start_mission(name="m1")
        a = ctx.start_action(type=ActionType.INFERENCE, name="openai.chat")
        ctx.end_action(a, output={"response": "hi"})
        ctx.end_mission(m)

        count = transport.flush()
        assert count == 2  # mission + action

        spans = exporter.get_exported_spans()
        action_spans = [s for s in spans if "action:" in s.name]
        assert len(action_spans) == 1
        attrs = dict(action_spans[0].attributes)
        assert attrs["missiontrace.action.type"] == "inference"
        assert attrs["missiontrace.action.name"] == "openai.chat"

    def test_export_action_with_intent(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext()
        transport.attach(ctx)

        m = ctx.start_mission(name="m1")
        a = ctx.start_action(
            type=ActionType.TOOL_CALL, name="github.create_pr", intent="mutation"
        )
        ctx.end_action(a)
        ctx.end_mission(m)

        transport.flush()
        spans = exporter.get_exported_spans()
        action_spans = [s for s in spans if "action:" in s.name]
        assert len(action_spans) == 1
        attrs = dict(action_spans[0].attributes)
        assert attrs["missiontrace.action.intent"] == "mutation"

    def test_export_action_without_intent_omits_attribute(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext()
        transport.attach(ctx)

        m = ctx.start_mission(name="m1")
        a = ctx.start_action(type=ActionType.TOOL_CALL, name="test")
        ctx.end_action(a)
        ctx.end_mission(m)

        transport.flush()
        spans = exporter.get_exported_spans()
        action_spans = [s for s in spans if "action:" in s.name]
        attrs = dict(action_spans[0].attributes)
        assert "missiontrace.action.intent" not in attrs

    def test_nested_actions_export(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext()
        transport.attach(ctx)

        m = ctx.start_mission(name="nested-test")
        outer = ctx.start_action(type=ActionType.TOOL_CALL, name="outer")
        inner = ctx.start_action(type=ActionType.TOOL_CALL, name="inner")
        ctx.end_action(inner)
        ctx.end_action(outer)
        ctx.end_mission(m)

        count = transport.flush()
        assert count == 3  # mission + 2 actions

        spans = exporter.get_exported_spans()
        action_spans = [s for s in spans if "action:" in s.name]
        assert len(action_spans) == 2

    def test_flush_empties_buffer(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext()
        transport.attach(ctx)

        m = ctx.start_mission()
        ctx.end_mission(m)

        transport.flush()
        assert len(exporter.get_exported_spans()) == 1

        # Second flush should yield nothing new
        count = transport.flush()
        assert count == 0

    def test_export_action_with_error(self):
        transport, exporter = self._make_transport()
        ctx = TraceContext()
        transport.attach(ctx)

        m = ctx.start_mission()
        a = ctx.start_action(type=ActionType.INFERENCE, name="bad_call")
        ctx.end_action(a, error=ValueError("test error"))
        ctx.end_mission(m)

        transport.flush()
        spans = exporter.get_exported_spans()
        action_spans = [s for s in spans if "action:" in s.name]
        assert len(action_spans) == 1

        attrs = dict(action_spans[0].attributes)
        assert attrs["missiontrace.error.type"] == "ValueError"
        assert attrs["missiontrace.error.message"] == "test error"
