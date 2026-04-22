# MissionTrace SDK

Observability and tracing SDK for AI agent systems. Captures structured trace data (missions, actions) from agent workflows and exports them via OpenTelemetry. External context (provider, external IDs, URLs) is stored as first-class fields on Action records by the backend, extracted from action tags.

## Quick Start

```bash
pip install -e .
```

```python
import missiontrace

missiontrace.init(api_key="mt_xxx", project="my-project")

with missiontrace.mission(name="research-acme-corp") as m:
    with missiontrace.action("inference", name="openai.chat", intent="generation") as act:
        result = call_llm("Summarize Acme Corp")
        act.set_output({"response": result})
        act.set_token_usage(prompt_tokens=500, completion_tokens=200, totmt_tokens=700, model="gpt-4")

    with missiontrace.action("tool_call", name="slack.post_message", intent="notification",
                           tags={"provider": "slack", "message_ts": "C041-123.456",
                                 "url": "https://acme.slack.com/archives/C041/p123"}) as act:
        msg = slack.post("Summary: ...")
        act.set_output({"ts": msg.ts, "channel": "#research"})
```

## Architecture

Three-layer design following the MissionTrace Architecture Specification:

```
missiontrace/
├── __init__.py          # Public API: init(), trace(), mission(), action()
├── core/
│   ├── models.py        # Mission, Action data models (Pydantic v2)
│   ├── context.py       # contextvars-based trace propagation
│   ├── sanitizer.py     # Input/output redaction (API keys, tokens, secrets)
│   └── transport.py     # OpenTelemetry span export (PlaceholderExporter → OTLP)
├── adapters/
│   ├── _base.py         # BaseAdapter ABC
│   └── openai.py        # Vendor-agnostic adapter for any OpenAI-compatible provider
└── utils/
    └── serialization.py # Safe serialization with truncation
```

## Three Levels of Instrumentation

### Level 1: Auto-Instrumentation (Adapters)

Zero code changes — patch any OpenAI-compatible SDK to emit traces automatically. A single `instrument()` call works for all client instances regardless of provider, base URL, or API key. The adapter auto-detects the provider at call time from the client's `base_url`.

#### OpenAI

```python
import missiontrace
from openai import OpenAI

missiontrace.init(api_key="mt_xxx", project="product-x")

from missiontrace.adapters.openai import instrument
instrument()

client = OpenAI()  # Uses OPENAI_API_KEY
response = client.chat.completions.create(model="gpt-4o", messages=[...])
# Traced as: openai.chat.completions.create
```

#### Google Gemini (via OpenAI-compatible endpoint)

```python
from openai import OpenAI

client = OpenAI(
    api_key="AIza...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
response = client.chat.completions.create(model="gemini-2.0-flash", messages=[...])
# Traced as: gemini.chat.completions.create
```

#### Groq

```python
from openai import OpenAI

client = OpenAI(
    api_key="gsk_...",
    base_url="https://api.groq.com/openai/v1",
)
response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[...])
# Traced as: groq.chat.completions.create
```

#### Together AI

```python
from openai import OpenAI

client = OpenAI(
    api_key="tog_...",
    base_url="https://api.together.xyz/v1",
)
response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-70B", messages=[...])
# Traced as: together.chat.completions.create
```

#### Async + Responses API

Works with both `OpenAI` and `AsyncOpenAI` clients, and instruments both the Chat Completions API and the Responses API:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()
response = await client.responses.create(model="gpt-4o", input="Summarize...")
# Traced as: openai.responses.create
```

#### Supported Providers (auto-detected)

| Provider | Base URL pattern | Trace prefix |
|----------|-----------------|-------------|
| OpenAI | `api.openai.com` | `openai.*` |
| Google Gemini | `generativelanguage.googleapis.com` | `gemini.*` |
| Groq | `api.groq.com` | `groq.*` |
| Together AI | `api.together.xyz` | `together.*` |
| Mistral | `api.mistral.ai` | `mistral.*` |
| Fireworks | `api.fireworks.ai` | `fireworks.*` |
| DeepSeek | `api.deepseek.com` | `deepseek.*` |
| OpenRouter | `openrouter.ai` | `openrouter.*` |
| Cerebras | `api.cerebras.ai` | `cerebras.*` |
| Perplexity | `api.perplexity.ai` | `perplexity.*` |
| Azure OpenAI | `*.openai.azure.com` | `azure_openai.*` |
| Any other | Custom base URL | `hostname.*` |

Every traced call captures: full sampling params, detailed token usage (cached, reasoning, audio, prediction tokens), tool calls, finish reasons, model metadata, provider name, base URL, and a redacted API key hint for debugging.

### Level 2: Decorator

Wrap custom functions to include them in the trace tree:

```python
@missiontrace.trace(action_type="tool_call", name="apply_patch", intent="mutation")
def apply_patch(file_path: str, diff: str) -> dict:
    parsed = parse_diff(diff)
    validate_syntax(parsed)
    return {"success": True, "lines_changed": 14}
```

Inputs are captured from the function signature, outputs from the return value. Exceptions are recorded and always re-raised.

### Level 3: Manual Context Managers

Full control for dynamic action names, conditional tracing, and rich external context:

```python
with missiontrace.mission(name="fix-auth-bug") as m:
    with missiontrace.action("tool_call", name="github.fetch_file", intent="retrieval",
                           tags={"provider": "github"}) as act:
        content = github.get_file("auth.py")
        act.set_output({"content": content})

    with missiontrace.action("tool_call", name="github.create_pr", intent="mutation",
                           tags={"provider": "github", "pr_number": "847",
                                 "url": "https://github.com/acme/repo/pull/847"}) as act:
        pr = github.create_pull(title="Fix auth bug")
        act.set_output({"pr_number": pr.number, "url": pr.html_url})
```

You can also set intent dynamically inside the block:

```python
with missiontrace.action("inference", name="openai.chat") as act:
    result = openai.chat(...)
    # Decide intent based on what happened
    act.set_intent("generation")
    act.set_output({"response": result})
```

## Data Model

Two primitives in a strict hierarchy:

- **Mission** — top-level trace container (one coherent unit of agent work)
- **Action** — a single step within a mission, forming a tree via parent-child relationships

### Action Type (mechanism)

What kind of operation this is — only two values:

| Type | Meaning | Examples |
|------|---------|----------|
| `inference` | AI model inference | `openai.chat.completions.create`, `openai.responses.create` |
| `tool_call` | Any tool/function invocation | `apply_patch()`, `github.create_pr`, `slack.post_message` |

### Intent (purpose)

Why the action is happening — a free-form string with well-known conventions from harness engineering:

| Intent | Meaning |
|--------|---------|
| `planning` | Deciding what to do next |
| `generation` | Producing content or code |
| `retrieval` | Fetching data from any source |
| `mutation` | Writing/changing state in an external system |
| `validation` | Checking correctness |
| `evaluation` | Assessing quality or classifying |
| `notification` | Alerting humans (Slack, email, etc.) |
| `orchestration` | Coordinating sub-agents or workflows |
| Any custom string | Your own domain-specific intents |

### External Context

When actions interact with external systems, use tags to provide context that the backend promotes to first-class fields for queryability:

```python
with missiontrace.action("tool_call", name="linear.create_issue", intent="mutation",
                       tags={"provider": "linear", "ticket_id": "ENG-501",
                             "url": "https://linear.app/acme/issue/ENG-501"}) as act:
    issue = linear.create_issue(title="Auth session leak")
    act.set_output({"issue_id": issue.id})
```

The backend extracts these tag keys into first-class `Action` fields:

| Tag Key | Backend Field |
|---------|--------------|
| `provider` | `Action.provider` |
| `pr_number`, `ticket_id`, `message_ts`, `commit_sha`, `id` | `Action.externmt_id` |
| `url`, `html_url`, `permalink`, `link` | `Action.externmt_url` |

## OpenTelemetry Transport

The SDK maps MissionTrace primitives to OTel spans:

- Missions become root spans
- Actions become child spans with proper parent-child linking
- Custom attributes use the `missiontrace.*` namespace
- `intent` is exported as `missiontrace.action.intent` (omitted when not set)

### Current: Placeholder Exporter

Spans are stored in-memory via `PlaceholderExporter` for development and testing.

### When Backend is Ready

Swap to OTLP/HTTP export by passing an endpoint:

```python
missiontrace.init(
    api_key="mt_xxx",
    project="product-x",
    endpoint="https://ingest.missiontrace.dev/v1/traces",
)
```

This automatically switches from `PlaceholderExporter` to `OTLPSpanExporter`.

## Input/Output Sanitization

The SDK redacts sensitive data before it reaches the transport buffer:

- OpenAI API keys (`sk-...`)
- GitHub tokens (`ghp_...`)
- Slack bot tokens (`xoxb-...`)
- Bearer auth headers
- Common credential key names (`api_key`, `password`, `token`, `secret`, etc.)

Add custom patterns:

```python
missiontrace.init(
    api_key="mt_xxx",
    project="my-project",
    sanitizer_patterns=[r"INTERNAL_\d+", r"my_custom_secret_\w+"],
)
```

## Metrics Captured

### Sampling Parameters (per action)

Every LLM call automatically captures the full set of sampling params from the request:

| Chat Completions API | Responses API |
|---------------------|---------------|
| `temperature`, `top_p` | `temperature`, `top_p` |
| `max_tokens`, `max_completion_tokens` | `max_output_tokens` |
| `frequency_penalty`, `presence_penalty` | `reasoning.effort` |
| `seed`, `stop`, `n` | `parallel_tool_calls` |
| `logprobs`, `top_logprobs`, `logit_bias` | `service_tier` |
| `response_format`, `reasoning_effort` | `store` |
| `service_tier`, `parallel_tool_calls`, `store` | |

### Token Usage (per action)

Detailed breakdowns beyond basic prompt/completion counts:

| Field | Source |
|-------|--------|
| `prompt_tokens`, `completion_tokens`, `totmt_tokens` | Both APIs |
| `cached_tokens` | Completions `prompt_tokens_details` |
| `reasoning_tokens` | Both APIs (completion/output details) |
| `audio_tokens_input`, `audio_tokens_output` | Completions audio |
| `accepted_prediction_tokens`, `rejected_prediction_tokens` | Completions prediction |

## Design Principles

- **Tracing never throws.** Backend down or buffer full? Records are silently dropped.
- **Tracing never alters behavior.** Exceptions are always re-raised. Return values are never modified.
- **< 5ms overhead per action (p99).** In-memory buffering with background daemon thread flush — no synchronous HTTP in the hot path.
- **Zero heavy dependencies in core.** Only `pydantic` and `opentelemetry-*` packages.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

115 tests covering models, context propagation, sanitization, OTel transport, the full public API (including intent on decorators, context managers, and dynamic set_intent), end-to-end trace capture with intent verification on OTel spans, and comprehensive vendor-agnostic adapter coverage (provider detection for 14 providers, API key redaction, client metadata extraction, sync/async completions + responses per provider, sampling params, token usage, tool calls, error handling, uninstrument, backwards compatibility).

## Optional Extras

```bash
pip install missiontrace[openai]      # OpenAI adapter
pip install missiontrace[anthropic]   # Anthropic adapter (coming soon)
pip install missiontrace[all-llm]     # All LLM vendor adapters
pip install missiontrace[all]         # Everything
```
