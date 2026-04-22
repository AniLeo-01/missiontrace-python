"""
MissionTrace SDK — OpenAI-Compatible Adapter

Vendor-agnostic adapter for any provider that exposes an OpenAI-compatible
client interface (OpenAI, Gemini, Groq, Together, Mistral, Azure OpenAI, etc.).

Auto-instruments both sync and async clients for:
  1. Chat Completions API  — client.chat.completions.create
  2. Responses API         — client.responses.create

Captures per-call:
  - Provider identity (base_url, detected provider name, redacted api_key)
  - All sampling params (temperature, top_p, max_tokens, seed, etc.)
  - Detailed token usage breakdowns (cached, reasoning, audio, prediction)
  - Tool calls, finish reasons, model metadata
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any
from urllib.parse import urlparse

from missiontrace.adapters._base import BaseAdapter
from missiontrace.core.context import TraceContext
from missiontrace.core.models import ActionType, SamplingParams, TokenUsage
from missiontrace.core.sanitizer import Sanitizer
from missiontrace.utils.serialization import safe_serialize

logger = logging.getLogger("missiontrace.adapters.openai_compat")

_adapter_instance: OpenAICompatAdapter | None = None

# Keep old name as alias for backwards compat
OpenAIAdapter = None  # set at module bottom


# ---------------------------------------------------------------------------
# Provider detection from base_url
# ---------------------------------------------------------------------------

_PROVIDER_PATTERNS: list[tuple[str, str]] = [
    (r"api\.openai\.com", "openai"),
    (r"generativelanguage\.googleapis\.com", "gemini"),
    (r"api\.groq\.com", "groq"),
    (r"api\.together\.xyz", "together"),
    (r"api\.mistral\.ai", "mistral"),
    (r"api\.anthropic\.com", "anthropic"),
    (r"api\.fireworks\.ai", "fireworks"),
    (r"api\.perplexity\.ai", "perplexity"),
    (r"api\.deepseek\.com", "deepseek"),
    (r"openrouter\.ai", "openrouter"),
    (r"api\.cerebras\.ai", "cerebras"),
    (r"\.openai\.azure\.com", "azure_openai"),
]


def detect_provider(base_url: str | None) -> str:
    """Detect the LLM provider from the client's base_url."""
    if not base_url:
        return "openai"  # default when no base_url set
    url_str = str(base_url).lower()
    for pattern, provider in _PROVIDER_PATTERNS:
        if re.search(pattern, url_str):
            return provider
    # Unknown provider — return the hostname
    try:
        parsed = urlparse(url_str if "://" in url_str else f"https://{url_str}")
        return parsed.hostname or "unknown"
    except Exception:
        return "unknown"


def _redact_api_key(key: str | None) -> str:
    """Show first 4 and last 4 chars, redact the middle."""
    if not key:
        return ""
    key = str(key)
    if len(key) <= 12:
        return key[:4] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


def _extract_client_metadata(client_self: Any) -> dict[str, str]:
    """
    Extract base_url and api_key from an OpenAI-compatible client instance.

    The client instance is `self_ref` — the bound object whose .create() was called.
    We walk up to the root client to find base_url and api_key.
    """
    metadata: dict[str, str] = {}

    # The resource object (Completions, Responses) stores a ref to the root client
    # via ._client. base_url and api_key live on the root client.
    root_client = getattr(client_self, "_client", None)
    if root_client is None:
        # Sometimes the resource is nested deeper (chat.completions._client)
        root_client = client_self

    # Extract base_url
    base_url = getattr(root_client, "base_url", None)
    if base_url is not None:
        base_url_str = str(base_url).rstrip("/")
        metadata["base_url"] = base_url_str
        metadata["provider"] = detect_provider(base_url_str)
    else:
        metadata["provider"] = "openai"

    # Extract and redact api_key
    api_key = getattr(root_client, "api_key", None)
    if api_key:
        metadata["api_key_hint"] = _redact_api_key(api_key)

    return metadata


# ---------------------------------------------------------------------------
# Shared helpers — extract metrics from request kwargs and response objects
# ---------------------------------------------------------------------------

def _extract_completions_sampling_params(kwargs: dict) -> SamplingParams:
    """Pull every sampling / generation param from a Chat Completions request."""
    rf = kwargs.get("response_format")
    response_format_dict = None
    if rf is not None:
        response_format_dict = safe_serialize(rf) if not isinstance(rf, dict) else rf

    return SamplingParams(
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        max_tokens=kwargs.get("max_tokens"),
        max_completion_tokens=kwargs.get("max_completion_tokens"),
        frequency_penalty=kwargs.get("frequency_penalty"),
        presence_penalty=kwargs.get("presence_penalty"),
        seed=kwargs.get("seed"),
        stop=kwargs.get("stop"),
        n=kwargs.get("n"),
        logprobs=kwargs.get("logprobs"),
        top_logprobs=kwargs.get("top_logprobs"),
        logit_bias=kwargs.get("logit_bias"),
        response_format=response_format_dict,
        service_tier=kwargs.get("service_tier"),
        reasoning_effort=kwargs.get("reasoning_effort"),
        parallel_tool_calls=kwargs.get("parallel_tool_calls"),
        store=kwargs.get("store"),
    )


def _extract_responses_sampling_params(kwargs: dict) -> SamplingParams:
    """Pull every sampling / generation param from a Responses API request."""
    reasoning = kwargs.get("reasoning") or {}
    return SamplingParams(
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        service_tier=kwargs.get("service_tier"),
        reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None,
        parallel_tool_calls=kwargs.get("parallel_tool_calls"),
        store=kwargs.get("store"),
    )


def _extract_completions_token_usage(result: Any) -> TokenUsage | None:
    """Extract full token usage including detailed breakdowns from a Completions response."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return None

    tu = TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        totmt_tokens=getattr(usage, "totmt_tokens", 0) or 0,
        model=getattr(result, "model", None),
    )

    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        tu.cached_tokens = getattr(ptd, "cached_tokens", None)
        tu.audio_tokens_input = getattr(ptd, "audio_tokens", None)

    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        tu.reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
        tu.audio_tokens_output = getattr(ctd, "audio_tokens", None)
        tu.accepted_prediction_tokens = getattr(ctd, "accepted_prediction_tokens", None)
        tu.rejected_prediction_tokens = getattr(ctd, "rejected_prediction_tokens", None)

    return tu


def _extract_responses_token_usage(result: Any) -> TokenUsage | None:
    """Extract token usage from a Responses API response."""
    usage = getattr(result, "usage", None)
    if usage is None:
        return None

    tu = TokenUsage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        totmt_tokens=(getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0),
        model=getattr(result, "model", None),
    )

    otd = getattr(usage, "output_tokens_details", None)
    if otd is not None:
        tu.reasoning_tokens = getattr(otd, "reasoning_tokens", None)

    return tu


def _build_completions_input(
    kwargs: dict, args: tuple, sanitizer: Sanitizer, client_meta: dict[str, str],
) -> dict:
    """Build sanitized input dict from Chat Completions call args."""
    data = {
        "provider": client_meta.get("provider", "unknown"),
        "base_url": client_meta.get("base_url", ""),
        "api_key_hint": client_meta.get("api_key_hint", ""),
        "model": kwargs.get("model", args[0] if args else "unknown"),
        "messages": safe_serialize(kwargs.get("messages", args[1] if len(args) > 1 else [])),
        "tools": safe_serialize(kwargs.get("tools")),
        "tool_choice": safe_serialize(kwargs.get("tool_choice")),
        "functions": safe_serialize(kwargs.get("functions")),
        "function_call": safe_serialize(kwargs.get("function_call")),
    }
    return sanitizer.sanitize(data)


def _build_completions_output(result: Any, sanitizer: Sanitizer) -> dict:
    """Build sanitized output dict from a Chat Completions response."""
    output: dict[str, Any] = {
        "id": getattr(result, "id", None),
        "model": getattr(result, "model", None),
        "system_fingerprint": getattr(result, "system_fingerprint", None),
        "service_tier": getattr(result, "service_tier", None),
    }

    choices = getattr(result, "choices", None)
    if choices:
        serialized_choices = []
        for choice in choices:
            c: dict[str, Any] = {
                "index": getattr(choice, "index", None),
                "finish_reason": getattr(choice, "finish_reason", None),
            }
            msg = getattr(choice, "message", None)
            if msg:
                c["message"] = {
                    "role": getattr(msg, "role", None),
                    "content": safe_serialize(getattr(msg, "content", None)),
                    "refusal": getattr(msg, "refusal", None),
                }
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    c["message"]["tool_calls"] = safe_serialize([
                        {
                            "id": tc.id,
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ])
                annotations = getattr(msg, "annotations", None)
                if annotations:
                    c["message"]["annotations"] = safe_serialize(annotations)

            logprobs_data = getattr(choice, "logprobs", None)
            if logprobs_data is not None:
                c["logprobs"] = safe_serialize(logprobs_data)

            serialized_choices.append(c)
        output["choices"] = serialized_choices

    usage = getattr(result, "usage", None)
    if usage:
        output["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "totmt_tokens": getattr(usage, "totmt_tokens", 0),
        }

    return sanitizer.sanitize(output)


def _build_responses_input(
    kwargs: dict, args: tuple, sanitizer: Sanitizer, client_meta: dict[str, str],
) -> dict:
    """Build sanitized input dict from a Responses API call."""
    data = {
        "provider": client_meta.get("provider", "unknown"),
        "base_url": client_meta.get("base_url", ""),
        "api_key_hint": client_meta.get("api_key_hint", ""),
        "model": kwargs.get("model", args[0] if args else "unknown"),
        "input": safe_serialize(kwargs.get("input", args[1] if len(args) > 1 else None)),
        "instructions": safe_serialize(kwargs.get("instructions")),
        "tools": safe_serialize(kwargs.get("tools")),
        "previous_response_id": kwargs.get("previous_response_id"),
    }
    return sanitizer.sanitize(data)


def _build_responses_output(result: Any, sanitizer: Sanitizer) -> dict:
    """Build sanitized output dict from a Responses API response."""
    output: dict[str, Any] = {
        "id": getattr(result, "id", None),
        "model": getattr(result, "model", None),
        "status": getattr(result, "status", None),
    }

    raw_output = getattr(result, "output", None)
    if raw_output is not None:
        output["output"] = safe_serialize(raw_output)

    usage = getattr(result, "usage", None)
    if usage:
        output["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }

    return sanitizer.sanitize(output)


# ---------------------------------------------------------------------------
# OpenAICompatAdapter — patches sync + async, completions + responses
# ---------------------------------------------------------------------------

class OpenAICompatAdapter(BaseAdapter):
    """
    Vendor-agnostic adapter for any OpenAI-compatible client.

    Works with:
      - openai.OpenAI(api_key=...) → standard OpenAI
      - openai.OpenAI(api_key=..., base_url="https://generativelanguage.googleapis.com/v1beta/openai") → Gemini
      - openai.OpenAI(api_key=..., base_url="https://api.groq.com/openai/v1") → Groq
      - openai.OpenAI(api_key=..., base_url="https://api.together.xyz/v1") → Together
      - openai.AsyncOpenAI(...) → async variant of any of the above

    Instruments:
      - Completions.create / AsyncCompletions.create   (sync + async)
      - Responses.create / AsyncResponses.create        (sync + async)

    Each action is tagged with:
      - provider: detected from base_url (openai, gemini, groq, together, etc.)
      - base_url: the full base URL of the client
      - api_key_hint: first 4 + last 4 chars of the API key (safe for logging)
    """

    def __init__(self, ctx: TraceContext, sanitizer: Sanitizer | None = None) -> None:
        super().__init__(ctx)
        self._sanitizer = sanitizer or Sanitizer()
        self._originals: dict[str, Any] = {}
        self._openai_module: Any = None

    def instrument(self) -> None:
        openai = self._require_package("openai", "openai")
        self._openai_module = openai

        self._patch_completions_sync(openai)
        self._patch_completions_async(openai)
        self._patch_responses_sync(openai)
        self._patch_responses_async(openai)

        logger.info("OpenAI-compatible adapter instrumented (completions + responses, sync + async)")

    def uninstrument(self) -> None:
        if not self._openai_module:
            return
        openai = self._openai_module
        for key, original in self._originals.items():
            if key == "completions_sync":
                openai.resources.chat.completions.Completions.create = original
            elif key == "completions_async":
                openai.resources.chat.completions.AsyncCompletions.create = original
            elif key == "responses_sync":
                openai.resources.responses.Responses.create = original
            elif key == "responses_async":
                openai.resources.responses.AsyncResponses.create = original
        self._originals.clear()
        logger.info("OpenAI-compatible adapter uninstrumented")

    # -- Completions sync ---------------------------------------------------

    def _patch_completions_sync(self, openai: Any) -> None:
        try:
            cls = openai.resources.chat.completions.Completions
        except AttributeError:
            logger.debug("Completions class not found, skipping sync completions patch")
            return

        original = cls.create
        self._originals["completions_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            client_meta = _extract_client_metadata(self_ref)
            provider = client_meta.get("provider", "openai")
            input_data = _build_completions_input(kwargs, args, adapter._sanitizer, client_meta)
            sampling = _extract_completions_sampling_params(kwargs)

            action = adapter._ctx.start_action(
                type=ActionType.INFERENCE,
                name=f"{provider}.chat.completions.create",
                input=input_data,
            )
            action.sampling_params = sampling
            action.tags["provider"] = provider
            if "base_url" in client_meta:
                action.tags["base_url"] = client_meta["base_url"]
            if "api_key_hint" in client_meta:
                action.tags["api_key_hint"] = client_meta["api_key_hint"]

            try:
                result = original(self_ref, *args, **kwargs)
                action.token_usage = _extract_completions_token_usage(result)
                output_data = _build_completions_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.create = patched

    # -- Completions async --------------------------------------------------

    def _patch_completions_async(self, openai: Any) -> None:
        try:
            cls = openai.resources.chat.completions.AsyncCompletions
        except AttributeError:
            logger.debug("AsyncCompletions class not found, skipping async completions patch")
            return

        original = cls.create
        self._originals["completions_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            client_meta = _extract_client_metadata(self_ref)
            provider = client_meta.get("provider", "openai")
            input_data = _build_completions_input(kwargs, args, adapter._sanitizer, client_meta)
            sampling = _extract_completions_sampling_params(kwargs)

            action = adapter._ctx.start_action(
                type=ActionType.INFERENCE,
                name=f"{provider}.chat.completions.create",
                input=input_data,
            )
            action.sampling_params = sampling
            action.tags["provider"] = provider
            if "base_url" in client_meta:
                action.tags["base_url"] = client_meta["base_url"]
            if "api_key_hint" in client_meta:
                action.tags["api_key_hint"] = client_meta["api_key_hint"]

            try:
                result = await original(self_ref, *args, **kwargs)
                action.token_usage = _extract_completions_token_usage(result)
                output_data = _build_completions_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.create = patched

    # -- Responses sync -----------------------------------------------------

    def _patch_responses_sync(self, openai: Any) -> None:
        try:
            cls = openai.resources.responses.Responses
        except AttributeError:
            logger.debug("Responses class not found, skipping sync responses patch")
            return

        original = cls.create
        self._originals["responses_sync"] = original
        adapter = self

        @functools.wraps(original)
        def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            client_meta = _extract_client_metadata(self_ref)
            provider = client_meta.get("provider", "openai")
            input_data = _build_responses_input(kwargs, args, adapter._sanitizer, client_meta)
            sampling = _extract_responses_sampling_params(kwargs)

            action = adapter._ctx.start_action(
                type=ActionType.INFERENCE,
                name=f"{provider}.responses.create",
                input=input_data,
            )
            action.sampling_params = sampling
            action.tags["provider"] = provider
            if "base_url" in client_meta:
                action.tags["base_url"] = client_meta["base_url"]
            if "api_key_hint" in client_meta:
                action.tags["api_key_hint"] = client_meta["api_key_hint"]

            try:
                result = original(self_ref, *args, **kwargs)
                action.token_usage = _extract_responses_token_usage(result)
                output_data = _build_responses_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.create = patched

    # -- Responses async ----------------------------------------------------

    def _patch_responses_async(self, openai: Any) -> None:
        try:
            cls = openai.resources.responses.AsyncResponses
        except AttributeError:
            logger.debug("AsyncResponses class not found, skipping async responses patch")
            return

        original = cls.create
        self._originals["responses_async"] = original
        adapter = self

        @functools.wraps(original)
        async def patched(self_ref: Any, *args: Any, **kwargs: Any) -> Any:
            client_meta = _extract_client_metadata(self_ref)
            provider = client_meta.get("provider", "openai")
            input_data = _build_responses_input(kwargs, args, adapter._sanitizer, client_meta)
            sampling = _extract_responses_sampling_params(kwargs)

            action = adapter._ctx.start_action(
                type=ActionType.INFERENCE,
                name=f"{provider}.responses.create",
                input=input_data,
            )
            action.sampling_params = sampling
            action.tags["provider"] = provider
            if "base_url" in client_meta:
                action.tags["base_url"] = client_meta["base_url"]
            if "api_key_hint" in client_meta:
                action.tags["api_key_hint"] = client_meta["api_key_hint"]

            try:
                result = await original(self_ref, *args, **kwargs)
                action.token_usage = _extract_responses_token_usage(result)
                output_data = _build_responses_output(result, adapter._sanitizer)
                adapter._ctx.end_action(action, output=output_data)
                return result
            except Exception as exc:
                adapter._ctx.end_action(action, error=exc)
                raise

        cls.create = patched


# Backwards-compatible alias
OpenAIAdapter = OpenAICompatAdapter


# ---------------------------------------------------------------------------
# Convenience module-level API
# ---------------------------------------------------------------------------

def instrument(ctx: TraceContext | None = None, sanitizer: Sanitizer | None = None) -> None:
    """
    Auto-instrument OpenAI-compatible SDK (sync + async, completions + responses).

    Works with any provider using the OpenAI client interface:
        import openai
        # Standard OpenAI
        client = openai.OpenAI(api_key="sk-...")
        # Gemini via OpenAI compat
        client = openai.OpenAI(api_key="...", base_url="https://generativelanguage.googleapis.com/v1beta/openai")
        # Groq
        client = openai.OpenAI(api_key="...", base_url="https://api.groq.com/openai/v1")

    The adapter detects the provider from base_url and tags every action accordingly.
    """
    global _adapter_instance

    if ctx is None:
        import missiontrace
        ctx = missiontrace._ctx
        sanitizer = sanitizer or missiontrace._sanitizer

    if ctx is None:
        raise RuntimeError("MissionTrace not initialized. Call missiontrace.init() first.")

    _adapter_instance = OpenAICompatAdapter(ctx, sanitizer)
    _adapter_instance.instrument()


def uninstrument() -> None:
    global _adapter_instance
    if _adapter_instance:
        _adapter_instance.uninstrument()
        _adapter_instance = None
