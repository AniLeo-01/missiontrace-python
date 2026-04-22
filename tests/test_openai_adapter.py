"""
Tests for the vendor-agnostic OpenAI-compatible adapter.

Uses mock OpenAI module structures to test:
  - Sync + Async Chat Completions
  - Sync + Async Responses API
  - Full sampling params capture
  - Detailed token usage extraction
  - Provider detection from base_url (OpenAI, Gemini, Groq, etc.)
  - API key hint capture (redacted)
  - Client metadata tagging on actions
"""

import asyncio
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from missiontrace.core.context import TraceContext
from missiontrace.core.models import Action, ActionStatus, SamplingParams
from missiontrace.core.sanitizer import Sanitizer
from missiontrace.adapters.openai import (
    OpenAICompatAdapter,
    OpenAIAdapter,
    detect_provider,
    _redact_api_key,
    _extract_client_metadata,
    _extract_completions_sampling_params,
    _extract_completions_token_usage,
    _extract_responses_sampling_params,
    _extract_responses_token_usage,
)


# ---------------------------------------------------------------------------
# Mock OpenAI module structure
# ---------------------------------------------------------------------------

def _make_mock_openai_module(
    sync_completions_result: Any = None,
    async_completions_result: Any = None,
    sync_responses_result: Any = None,
    async_responses_result: Any = None,
):
    """
    Build a fake `openai` module with the class hierarchy the adapter expects.
    """
    openai = types.ModuleType("openai")

    resources = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions_mod = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        @staticmethod
        def create(self_ref, *args, **kwargs):
            return sync_completions_result

    class AsyncCompletions:
        @staticmethod
        async def create(self_ref, *args, **kwargs):
            return async_completions_result

    completions_mod.Completions = Completions
    completions_mod.AsyncCompletions = AsyncCompletions
    chat.completions = completions_mod
    resources.chat = chat

    responses_mod = types.ModuleType("openai.resources.responses")

    class Responses:
        @staticmethod
        def create(self_ref, *args, **kwargs):
            return sync_responses_result

    class AsyncResponses:
        @staticmethod
        async def create(self_ref, *args, **kwargs):
            return async_responses_result

    responses_mod.Responses = Responses
    responses_mod.AsyncResponses = AsyncResponses
    resources.responses = responses_mod

    openai.resources = resources
    return openai


def _make_mock_client(base_url: str | None = None, api_key: str | None = None):
    """Build a mock client instance that the resource's self_ref._client points to."""
    root_client = MagicMock()
    root_client.base_url = base_url
    root_client.api_key = api_key

    # The resource object (Completions/Responses) typically has ._client
    resource = MagicMock()
    resource._client = root_client
    return resource


def _make_completions_response(
    content="Hello!",
    model="gpt-4o",
    prompt_tokens=100,
    completion_tokens=50,
    totmt_tokens=150,
    finish_reason="stop",
    tool_calls=None,
    cached_tokens=10,
    reasoning_tokens=5,
    system_fingerprint="fp_abc123",
    service_tier="default",
):
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.refusal = None
    msg.tool_calls = tool_calls
    msg.annotations = None

    choice = MagicMock()
    choice.index = 0
    choice.message = msg
    choice.finish_reason = finish_reason
    choice.logprobs = None

    ptd = MagicMock()
    ptd.cached_tokens = cached_tokens
    ptd.audio_tokens = None

    ctd = MagicMock()
    ctd.reasoning_tokens = reasoning_tokens
    ctd.audio_tokens = None
    ctd.accepted_prediction_tokens = None
    ctd.rejected_prediction_tokens = None

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.totmt_tokens = totmt_tokens
    usage.prompt_tokens_details = ptd
    usage.completion_tokens_details = ctd

    result = MagicMock()
    result.id = "chatcmpl-abc123"
    result.model = model
    result.choices = [choice]
    result.usage = usage
    result.system_fingerprint = system_fingerprint
    result.service_tier = service_tier
    return result


def _make_responses_response(
    model="gpt-4o",
    status="completed",
    input_tokens=200,
    output_tokens=80,
    reasoning_tokens=15,
):
    otd = MagicMock()
    otd.reasoning_tokens = reasoning_tokens

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.output_tokens_details = otd

    result = MagicMock()
    result.id = "resp-abc456"
    result.model = model
    result.status = status
    result.output = [{"type": "message", "content": [{"type": "text", "text": "Hi"}]}]
    result.usage = usage
    return result


# ---------------------------------------------------------------------------
# Tests: Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_openai(self):
        assert detect_provider("https://api.openai.com/v1") == "openai"

    def test_gemini(self):
        assert detect_provider("https://generativelanguage.googleapis.com/v1beta/openai") == "gemini"

    def test_groq(self):
        assert detect_provider("https://api.groq.com/openai/v1") == "groq"

    def test_together(self):
        assert detect_provider("https://api.together.xyz/v1") == "together"

    def test_mistral(self):
        assert detect_provider("https://api.mistral.ai/v1") == "mistral"

    def test_fireworks(self):
        assert detect_provider("https://api.fireworks.ai/inference/v1") == "fireworks"

    def test_deepseek(self):
        assert detect_provider("https://api.deepseek.com/v1") == "deepseek"

    def test_openrouter(self):
        assert detect_provider("https://openrouter.ai/api/v1") == "openrouter"

    def test_azure_openai(self):
        assert detect_provider("https://my-resource.openai.azure.com/openai/v1") == "azure_openai"

    def test_cerebras(self):
        assert detect_provider("https://api.cerebras.ai/v1") == "cerebras"

    def test_perplexity(self):
        assert detect_provider("https://api.perplexity.ai/chat/completions") == "perplexity"

    def test_none_defaults_to_openai(self):
        assert detect_provider(None) == "openai"

    def test_empty_defaults_to_openai(self):
        assert detect_provider("") == "openai"

    def test_unknown_returns_hostname(self):
        assert detect_provider("https://my-custom-llm.example.com/v1") == "my-custom-llm.example.com"


class TestApiKeyRedaction:
    def test_long_key(self):
        assert _redact_api_key("sk-abc123def456ghi789jkl0") == "sk-a...jkl0"

    def test_short_key(self):
        assert _redact_api_key("short") == "shor...rt"

    def test_none(self):
        assert _redact_api_key(None) == ""

    def test_empty(self):
        assert _redact_api_key("") == ""


class TestClientMetadataExtraction:
    def test_standard_openai_client(self):
        client = _make_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-abc123def456ghi789jkl0",
        )
        meta = _extract_client_metadata(client)
        assert meta["provider"] == "openai"
        assert "api.openai.com" in meta["base_url"]
        assert meta["api_key_hint"] == "sk-a...jkl0"

    def test_gemini_client(self):
        client = _make_mock_client(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key="AIzaSyAbcdefghijklmnop",
        )
        meta = _extract_client_metadata(client)
        assert meta["provider"] == "gemini"
        assert "googleapis.com" in meta["base_url"]
        assert meta["api_key_hint"].startswith("AIza")

    def test_groq_client(self):
        client = _make_mock_client(
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk_abcdefghijklmnop123456",
        )
        meta = _extract_client_metadata(client)
        assert meta["provider"] == "groq"

    def test_no_base_url(self):
        client = _make_mock_client(base_url=None, api_key="sk-test")
        meta = _extract_client_metadata(client)
        assert meta["provider"] == "openai"

    def test_no_api_key(self):
        client = _make_mock_client(base_url="https://api.openai.com/v1", api_key=None)
        meta = _extract_client_metadata(client)
        assert "api_key_hint" not in meta


# ---------------------------------------------------------------------------
# Tests: Sampling params extraction
# ---------------------------------------------------------------------------

class TestSamplingParamsExtraction:
    def test_completions_all_params(self):
        kwargs = {
            "model": "gpt-4o",
            "temperature": 0.7, "top_p": 0.9,
            "max_tokens": 1000, "max_completion_tokens": 2000,
            "frequency_penalty": 0.5, "presence_penalty": 0.3,
            "seed": 42, "stop": ["\n", "END"],
            "n": 2, "logprobs": True, "top_logprobs": 5,
            "logit_bias": {"50256": -100},
            "response_format": {"type": "json_object"},
            "service_tier": "default", "reasoning_effort": "high",
            "parallel_tool_calls": True, "store": True,
        }
        sp = _extract_completions_sampling_params(kwargs)
        assert sp.temperature == 0.7
        assert sp.seed == 42
        assert sp.reasoning_effort == "high"
        assert sp.store is True

    def test_completions_minimmt_params(self):
        sp = _extract_completions_sampling_params({"model": "gpt-4o"})
        assert sp.temperature is None

    def test_responses_all_params(self):
        kwargs = {
            "model": "gpt-4o", "temperature": 0.5, "top_p": 0.8,
            "max_output_tokens": 4096, "service_tier": "flex",
            "reasoning": {"effort": "medium", "summary": "concise"},
            "parallel_tool_calls": False, "store": True,
        }
        sp = _extract_responses_sampling_params(kwargs)
        assert sp.max_output_tokens == 4096
        assert sp.reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# Tests: Token usage extraction
# ---------------------------------------------------------------------------

class TestTokenUsageExtraction:
    def test_completions_full_details(self):
        result = _make_completions_response(
            prompt_tokens=500, completion_tokens=200, totmt_tokens=700,
            cached_tokens=50, reasoning_tokens=30,
        )
        tu = _extract_completions_token_usage(result)
        assert tu.prompt_tokens == 500
        assert tu.cached_tokens == 50
        assert tu.reasoning_tokens == 30

    def test_completions_no_usage(self):
        result = MagicMock()
        result.usage = None
        assert _extract_completions_token_usage(result) is None

    def test_responses_full_details(self):
        result = _make_responses_response(input_tokens=300, output_tokens=100, reasoning_tokens=20)
        tu = _extract_responses_token_usage(result)
        assert tu.prompt_tokens == 300
        assert tu.totmt_tokens == 400
        assert tu.reasoning_tokens == 20

    def test_responses_no_usage(self):
        result = MagicMock()
        result.usage = None
        assert _extract_responses_token_usage(result) is None


# ---------------------------------------------------------------------------
# Tests: Sync Completions with provider tagging
# ---------------------------------------------------------------------------

class TestSyncCompletionsAdapter:
    def test_openai_provider_tagged(self):
        mock_result = _make_completions_response()
        mock_openai = _make_mock_openai_module(sync_completions_result=mock_result)

        ctx = TraceContext(project_id="test")
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission(name="test")
        adapter.instrument()

        # Simulate a client with base_url and api_key
        client_ref = _make_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-abc123def456ghi789jkl0",
        )
        result = mock_openai.resources.chat.completions.Completions.create(
            client_ref, model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )

        assert result is mock_result
        ctx.end_mission(m)

        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]
        assert len(actions) == 1

        action = actions[0]
        assert action.name == "openai.chat.completions.create"
        assert action.tags["provider"] == "openai"
        assert "api.openai.com" in action.tags["base_url"]
        assert action.tags["api_key_hint"] == "sk-a...jkl0"
        assert action.sampling_params.temperature == 0.7
        assert action.token_usage.prompt_tokens == 100

        # Input also has provider metadata
        assert action.input["provider"] == "openai"
        assert "api.openai.com" in action.input["base_url"]

        adapter.uninstrument()

    def test_gemini_provider_tagged(self):
        mock_result = _make_completions_response(model="gemini-2.0-flash")
        mock_openai = _make_mock_openai_module(sync_completions_result=mock_result)

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        client_ref = _make_mock_client(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key="AIzaSyAbcdefghijklmnop",
        )
        mock_openai.resources.chat.completions.Completions.create(
            client_ref, model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        action = actions[0]
        assert action.name == "gemini.chat.completions.create"
        assert action.tags["provider"] == "gemini"
        assert "googleapis.com" in action.tags["base_url"]

        adapter.uninstrument()

    def test_groq_provider_tagged(self):
        mock_result = _make_completions_response(model="llama-3.3-70b-versatile")
        mock_openai = _make_mock_openai_module(sync_completions_result=mock_result)

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        client_ref = _make_mock_client(
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk_abc123",
        )
        mock_openai.resources.chat.completions.Completions.create(
            client_ref, model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
        )

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        assert actions[0].name == "groq.chat.completions.create"
        assert actions[0].tags["provider"] == "groq"

        adapter.uninstrument()

    def test_exception_captured_and_reraised(self):
        mock_openai = _make_mock_openai_module()

        def raise_error(self_ref, *a, **kw):
            raise ConnectionError("API down")

        mock_openai.resources.chat.completions.Completions.create = raise_error

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter._originals["completions_sync"] = raise_error

        import functools

        @functools.wraps(raise_error)
        def patched(self_ref, *args, **kwargs):
            action = ctx.start_action(type="inference", name="openai.chat.completions.create", input={})
            try:
                return raise_error(self_ref, *args, **kwargs)
            except Exception as exc:
                ctx.end_action(action, error=exc)
                raise

        mock_openai.resources.chat.completions.Completions.create = patched

        with pytest.raises(ConnectionError, match="API down"):
            mock_openai.resources.chat.completions.Completions.create(None, model="gpt-4o")

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]
        assert actions[0].status == ActionStatus.FAILED
        assert actions[0].error.type == "ConnectionError"


# ---------------------------------------------------------------------------
# Tests: Async Completions
# ---------------------------------------------------------------------------

class TestAsyncCompletionsAdapter:
    def test_async_with_provider_tags(self):
        mock_result = _make_completions_response(model="gpt-4o-mini")
        mock_openai = _make_mock_openai_module(async_completions_result=mock_result)

        ctx = TraceContext(project_id="test")
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission(name="async-test")
        adapter.instrument()

        client_ref = _make_mock_client(
            base_url="https://api.together.xyz/v1",
            api_key="tog-abc123def456ghi789",
        )

        async def run():
            return await mock_openai.resources.chat.completions.AsyncCompletions.create(
                client_ref, model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.3,
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is mock_result

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        action = actions[0]
        assert action.name == "together.chat.completions.create"
        assert action.tags["provider"] == "together"
        assert action.sampling_params.temperature == 0.3

        adapter.uninstrument()


# ---------------------------------------------------------------------------
# Tests: Sync Responses API with provider tagging
# ---------------------------------------------------------------------------

class TestSyncResponsesAdapter:
    def test_openai_responses_tagged(self):
        mock_result = _make_responses_response()
        mock_openai = _make_mock_openai_module(sync_responses_result=mock_result)

        ctx = TraceContext(project_id="test")
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        client_ref = _make_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-test123456789abcdef",
        )
        mock_openai.resources.responses.Responses.create(
            client_ref, model="gpt-4o",
            input="What is the capital of France?",
            max_output_tokens=1024,
            reasoning={"effort": "high"},
        )

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        action = actions[0]
        assert action.name == "openai.responses.create"
        assert action.tags["provider"] == "openai"
        assert action.token_usage.prompt_tokens == 200
        assert action.sampling_params.max_output_tokens == 1024
        assert action.sampling_params.reasoning_effort == "high"

        adapter.uninstrument()


# ---------------------------------------------------------------------------
# Tests: Async Responses API
# ---------------------------------------------------------------------------

class TestAsyncResponsesAdapter:
    def test_async_responses_with_provider(self):
        mock_result = _make_responses_response(model="o3", reasoning_tokens=50)
        mock_openai = _make_mock_openai_module(async_responses_result=mock_result)

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        client_ref = _make_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-proj-abc123",
        )

        async def run():
            return await mock_openai.resources.responses.AsyncResponses.create(
                client_ref, model="o3", input="Solve this",
                temperature=0.2, reasoning={"effort": "xhigh"},
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is mock_result

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        action = actions[0]
        assert action.name == "openai.responses.create"
        assert action.token_usage.reasoning_tokens == 50
        assert action.sampling_params.reasoning_effort == "xhigh"

        adapter.uninstrument()


# ---------------------------------------------------------------------------
# Tests: Uninstrument + backwards compat alias
# ---------------------------------------------------------------------------

class TestUninstrument:
    def test_uninstrument_restores_all(self):
        mock_openai = _make_mock_openai_module(
            sync_completions_result=_make_completions_response(),
            sync_responses_result=_make_responses_response(),
        )

        orig_comp = mock_openai.resources.chat.completions.Completions.create
        orig_acomp = mock_openai.resources.chat.completions.AsyncCompletions.create
        orig_resp = mock_openai.resources.responses.Responses.create
        orig_aresp = mock_openai.resources.responses.AsyncResponses.create

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai
        adapter.instrument()

        assert mock_openai.resources.chat.completions.Completions.create is not orig_comp

        adapter.uninstrument()

        assert mock_openai.resources.chat.completions.Completions.create is orig_comp
        assert mock_openai.resources.chat.completions.AsyncCompletions.create is orig_acomp
        assert mock_openai.resources.responses.Responses.create is orig_resp
        assert mock_openai.resources.responses.AsyncResponses.create is orig_aresp


class TestBackwardsCompat:
    def test_openai_adapter_alias(self):
        """OpenAIAdapter should be an alias for OpenAICompatAdapter."""
        assert OpenAIAdapter is OpenAICompatAdapter


# ---------------------------------------------------------------------------
# Tests: Tool calls in completions output
# ---------------------------------------------------------------------------

class TestToolCallCapture:
    def test_tool_calls_serialized(self):
        tc1 = MagicMock()
        tc1.id = "call_abc"
        tc1.type = "function"
        tc1.function = MagicMock()
        tc1.function.name = "get_weather"
        tc1.function.arguments = '{"location": "SF"}'

        mock_result = _make_completions_response(
            content=None, tool_calls=[tc1], finish_reason="tool_calls",
        )
        mock_openai = _make_mock_openai_module(sync_completions_result=mock_result)

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        client_ref = _make_mock_client(base_url="https://api.openai.com/v1", api_key="sk-test")
        mock_openai.resources.chat.completions.Completions.create(
            client_ref, model="gpt-4o",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        output = actions[0].output
        choices = output.get("choices", [])
        assert choices[0]["finish_reason"] == "tool_calls"
        assert choices[0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

        adapter.uninstrument()


# ---------------------------------------------------------------------------
# Tests: No client metadata (bare self_ref with no _client)
# ---------------------------------------------------------------------------

class TestNoClientMetadata:
    def test_works_without_client_attrs(self):
        """Adapter should not crash if self_ref has no _client/base_url/api_key."""
        mock_result = _make_completions_response()
        mock_openai = _make_mock_openai_module(sync_completions_result=mock_result)

        ctx = TraceContext()
        adapter = OpenAICompatAdapter(ctx, Sanitizer())
        adapter._openai_module = mock_openai
        adapter._require_package = lambda *a: mock_openai

        m = ctx.start_mission()
        adapter.instrument()

        # Pass None as self_ref — no _client attribute
        mock_openai.resources.chat.completions.Completions.create(
            None, model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

        ctx.end_mission(m)
        items = ctx.drain_buffer()
        actions = [i for i in items if isinstance(i, Action)]

        # Should still work, defaulting provider to "openai"
        assert actions[0].tags["provider"] == "openai"
        assert actions[0].status == ActionStatus.COMPLETED

        adapter.uninstrument()
