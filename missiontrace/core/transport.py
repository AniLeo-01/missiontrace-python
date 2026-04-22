"""
MissionTrace SDK — OpenTelemetry Transport Layer

Maps MissionTrace primitives (Mission, Action) to OTel spans
and exports them via OTLP/HTTP to the MissionTrace backend.

Design:
  - Background daemon thread flushes every `flush_intervmt_s` seconds
  - Max batch size triggers early flush
  - Missions → root spans, Actions → child spans
  - atexit handler ensures final flush on process exit
  - Never throws — tracing failures are logged and silently dropped
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
    SimpleSpanProcessor,
)
from opentelemetry.trace import StatusCode, SpanKind

from missiontrace.core.models import (
    Action,
    ActionStatus,
    Mission,
    MissionStatus,
)

logger = logging.getLogger("missiontrace.transport")


# ---------------------------------------------------------------------------
# Placeholder exporter — stores spans in-memory for inspection / testing.
# Replace with OTLPSpanExporter when the backend is ready.
# ---------------------------------------------------------------------------

class PlaceholderExporter(SpanExporter):
    """
    In-memory span exporter for development and testing.
    When the MissionTrace backend is defined, swap this for:

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint="https://ingest.missiontrace.dev/v1/traces")
    """

    def __init__(self) -> None:
        self.exported_spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self.exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def get_exported_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self.exported_spans)

    def clear(self) -> None:
        with self._lock:
            self.exported_spans.clear()


class OTLPJsonExporter(SpanExporter):
    """
    Custom OTLP/HTTP JSON exporter that sends spans in JSON format
    instead of protobuf, which the MissionTrace backend expects.
    """

    def __init__(self, endpoint: str, headers: dict[str, str] | None = None) -> None:
        self.endpoint = endpoint
        self.headers = headers or {}
        self._lock = threading.Lock()

    def _span_to_json(self, span: ReadableSpan) -> dict:
        """Convert an OpenTelemetry ReadableSpan to OTLP JSON format."""
        attributes = []
        for key, value in (span.attributes or {}).items():
            if isinstance(value, bool):
                attributes.append({"key": key, "value": {"boolValue": value}})
            elif isinstance(value, int):
                attributes.append({"key": key, "value": {"intValue": str(value)}})
            elif isinstance(value, float):
                attributes.append({"key": key, "value": {"doubleValue": value}})
            else:
                attributes.append({"key": key, "value": {"stringValue": str(value)}})

        # Map status code
        status_code = 0  # UNSET
        status_message = ""
        if span.status:
            if span.status.status_code == StatusCode.OK:
                status_code = 1
            elif span.status.status_code == StatusCode.ERROR:
                status_code = 2
            status_message = span.status.description or ""

        # Convert events to OTLP format
        events = []
        if hasattr(span, 'events') and span.events:
            for event in span.events:
                event_attributes = []
                if hasattr(event, 'attributes') and event.attributes:
                    for key, value in event.attributes.items():
                        if isinstance(value, bool):
                            event_attributes.append({"key": key, "value": {"boolValue": value}})
                        elif isinstance(value, int):
                            event_attributes.append({"key": key, "value": {"intValue": str(value)}})
                        elif isinstance(value, float):
                            event_attributes.append({"key": key, "value": {"doubleValue": value}})
                        else:
                            event_attributes.append({"key": key, "value": {"stringValue": str(value)}})

                events.append({
                    "name": event.name,
                    "timeUnixNano": str(event.timestamp) if hasattr(event, 'timestamp') and event.timestamp else "0",
                    "attributes": event_attributes
                })

        return {
            "traceId": format(span.context.trace_id, "032x"),
            "spanId": format(span.context.span_id, "016x"),
            "parentSpanId": format(span.parent.span_id, "016x") if span.parent else "",
            "name": span.name,
            "kind": span.kind.value if span.kind else 1,
            "startTimeUnixNano": str(span.start_time) if span.start_time else "0",
            "endTimeUnixNano": str(span.end_time) if span.end_time else "0",
            "attributes": attributes,
            "events": events,  # NEW: Include span events
            "status": {"code": status_code, "message": status_message},
        }

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            import httpx

            # Build OTLP JSON payload
            span_data = [self._span_to_json(s) for s in spans]
            payload = {
                "resourceSpans": [
                    {
                        "resource": {"attributes": []},
                        "scopeSpans": [
                            {
                                "scope": {"name": "missiontrace", "version": "0.1.0"},
                                "spans": span_data,
                            }
                        ],
                    }
                ]
            }

            headers = {"Content-Type": "application/json", **self.headers}
            response = httpx.post(self.endpoint, json=payload, headers=headers, timeout=10.0)

            if response.status_code >= 400:
                logger.warning(f"OTLP export failed: {response.status_code} {response.text[:200]}")
                return SpanExportResult.FAILURE

            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.debug(f"OTLP export error: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------------------
# OTelTransport — bridges MissionTrace buffer → OTel spans
# ---------------------------------------------------------------------------

class OTelTransport:
    """
    Drains the TraceContext buffer and converts MissionTrace records to
    OpenTelemetry spans, then exports them via the configured exporter.
    """

    def __init__(
        self,
        service_name: str = "missiontrace-sdk",
        endpoint: str | None = None,
        api_key: str | None = None,
        flush_intervmt_s: float = 2.0,
        max_batch_size: int = 100,
        exporter: SpanExporter | None = None,
    ) -> None:
        self.flush_intervmt_s = flush_intervmt_s
        self.max_batch_size = max_batch_size
        self._shutdown = False

        # --- OTel provider setup ---
        resource = Resource.create(
            {
                "service.name": service_name,
                "missiontrace.sdk.version": "0.1.0",
            }
        )

        if exporter is not None:
            self._exporter = exporter
        elif endpoint:
            # Use custom OTLP JSON exporter (backend expects JSON, not protobuf)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._exporter = OTLPJsonExporter(
                endpoint=endpoint,
                headers=headers if headers else None,
            )
        else:
            self._exporter = PlaceholderExporter()

        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
        self._tracer = self._provider.get_tracer("missiontrace", "0.1.0")

        # Track OTel span contexts keyed by MissionTrace IDs
        # so child actions can be linked to parent spans.
        self._span_map: dict[str, trace.Span] = {}

        # Background flush thread
        self._trace_context_ref = None  # set via attach()
        self._flush_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        atexit.register(self.shutdown)

    def attach(self, trace_context: Any) -> None:
        """Bind to a TraceContext so the background thread can drain it."""
        self._trace_context_ref = trace_context
        self._start_flush_thread()

    # -- span conversion ----------------------------------------------------

    def _time_to_ns(self, wall_time: float) -> int:
        """Convert wall-clock time to epoch nanoseconds."""
        return int(wall_time * 1e9)

    def export_mission(self, mission: Mission) -> None:
        """Convert a Mission to an OTel root span."""
        try:
            # Convert wall-clock timestamps to epoch nanoseconds
            start_time_ns = self._time_to_ns(mission.started_at)
            end_time_ns = self._time_to_ns(mission.ended_at) if mission.ended_at else None

            span = self._tracer.start_span(
                name=f"mission:{mission.name or mission.id}",
                kind=SpanKind.INTERNAL,
                start_time=start_time_ns,
                attributes={
                    "missiontrace.type": "mission",
                    "missiontrace.mission.id": mission.id,
                    "missiontrace.project.id": mission.project_id,
                    "missiontrace.mission.trigger": mission.trigger,
                    "missiontrace.mission.status": mission.status.value,
                },
            )
            if mission.status == MissionStatus.FAILED:
                span.set_status(StatusCode.ERROR, "Mission failed")
            elif mission.status == MissionStatus.COMPLETED:
                span.set_status(StatusCode.OK)

            # Store for child action lookups
            self._span_map[mission.id] = span

            # End with explicit timestamp if mission is completed
            if end_time_ns:
                span.end(end_time=end_time_ns)
            else:
                span.end()
        except Exception:
            logger.debug("Failed to export mission span", exc_info=True)

    def export_action(self, action: Action) -> None:
        """Convert an Action to an OTel span, nested under parent if exists."""
        try:
            # Convert wall-clock timestamps to epoch nanoseconds
            start_time_ns = self._time_to_ns(action.started_at)
            end_time_ns = self._time_to_ns(action.ended_at) if action.ended_at else None

            # Determine parent context
            parent_ctx = None
            parent_span = self._span_map.get(action.parent_id or "")
            if parent_span is None:
                # Fall back to mission span
                parent_span = self._span_map.get(action.mission_id)
            if parent_span is not None:
                parent_ctx = trace.set_span_in_context(parent_span)

            base_attrs = {
                    "missiontrace.type": "action",
                    "missiontrace.action.id": action.id,
                    "missiontrace.action.type": action.type.value,
                    "missiontrace.action.name": action.name,
                    "missiontrace.mission.id": action.mission_id,
                    "missiontrace.action.parent_id": action.parent_id or "",
                    "missiontrace.action.status": action.status.value,
            }
            if action.intent:
                base_attrs["missiontrace.action.intent"] = action.intent

            span = self._tracer.start_span(
                name=f"action:{action.name}",
                kind=SpanKind.INTERNAL,
                context=parent_ctx,
                start_time=start_time_ns,
                attributes=base_attrs,
            )

            if action.duration_ms is not None:
                span.set_attribute("missiontrace.action.duration_ms", action.duration_ms)

            # Serialize input/output as JSON strings
            if action.input:
                import json
                try:
                    span.set_attribute("missiontrace.action.input", json.dumps(action.input))
                except (TypeError, ValueError):
                    span.set_attribute("missiontrace.action.input", str(action.input))

            if action.output:
                import json
                try:
                    span.set_attribute("missiontrace.action.output", json.dumps(action.output))
                except (TypeError, ValueError):
                    span.set_attribute("missiontrace.action.output", str(action.output))

            # Send metadata (including code context)
            if action.metadata:
                import json
                try:
                    span.set_attribute("missiontrace.action.metadata", json.dumps(action.metadata))
                except (TypeError, ValueError):
                    span.set_attribute("missiontrace.action.metadata", str(action.metadata))

            if action.token_usage:
                span.set_attribute(
                    "missiontrace.llm.prompt_tokens",
                    action.token_usage.prompt_tokens,
                )
                span.set_attribute(
                    "missiontrace.llm.completion_tokens",
                    action.token_usage.completion_tokens,
                )
                span.set_attribute(
                    "missiontrace.llm.totmt_tokens",
                    action.token_usage.totmt_tokens,
                )
                if action.token_usage.model:
                    span.set_attribute("missiontrace.llm.model", action.token_usage.model)

            if action.error:
                span.set_status(StatusCode.ERROR, action.error.message)
                span.set_attribute("missiontrace.error.type", action.error.type)
                span.set_attribute("missiontrace.error.message", action.error.message)
            elif action.status == ActionStatus.COMPLETED:
                span.set_status(StatusCode.OK)

            for k, v in action.tags.items():
                span.set_attribute(f"missiontrace.tag.{k}", v)

            # Add log events as span events
            for log_event in action.log_events:
                event_name = log_event.get("name", "log.info")
                event_timestamp = int(log_event.get("timestamp", time.time()) * 1e9)  # Convert to nanoseconds
                event_attrs = log_event.get("attributes", {})
                span.add_event(name=event_name, attributes=event_attrs, timestamp=event_timestamp)

            self._span_map[action.id] = span

            # End with explicit timestamp if action is completed
            if end_time_ns:
                span.end(end_time=end_time_ns)
            else:
                span.end()
        except Exception:
            logger.debug("Failed to export action span", exc_info=True)

    # -- buffer drain -------------------------------------------------------

    def flush(self) -> int:
        """
        Drain the TraceContext buffer and export all records.
        Returns the number of records exported.
        """
        if self._trace_context_ref is None:
            return 0

        items = self._trace_context_ref.drain_buffer()
        if not items:
            return 0

        count = 0
        for item in items:
            if isinstance(item, Mission):
                self.export_mission(item)
                count += 1
            elif isinstance(item, Action):
                self.export_action(item)
                count += 1

        return count

    # -- background thread --------------------------------------------------

    def _start_flush_thread(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="missiontrace-flush"
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        while not self._shutdown:
            try:
                self.flush()
            except Exception:
                logger.debug("Flush error (silently dropped)", exc_info=True)
            time.sleep(self.flush_intervmt_s)

    def shutdown(self) -> None:
        """Final flush + provider shutdown. Called by atexit."""
        self._shutdown = True
        try:
            self.flush()
            self._provider.shutdown()
        except Exception:
            logger.debug("Shutdown error", exc_info=True)

    @property
    def exporter(self) -> SpanExporter:
        return self._exporter
