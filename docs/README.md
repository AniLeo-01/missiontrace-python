# MissionTrace Python SDK

> **Observability for AI Agents** - Track, debug, and understand your AI agent workflows with ease.

MissionTrace helps you see exactly what your AI agents are doing. Every LLM call, every tool execution, every step of your agent's workflow - captured and visualized in the MissionTrace dashboard.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Tracking Custom Code](#tracking-custom-code)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
pip install missiontrace
```

Or install with OpenAI support:

```bash
pip install missiontrace[openai]
```

### Your First Trace

Get up and running in 3 lines of code:

```python
import missiontrace

# 1. Initialize with your API key and project name
missiontrace.init(
    api_key="your-api-key",
    project="my-first-project",
    endpoint="http://localhost:8100/v1/traces"
)

# 2. Auto-instrument OpenAI
from missiontrace.adapters.openai import instrument
instrument()

# 3. Use OpenAI as normal - everything is automatically traced!
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

That's it! Open the MissionTrace dashboard to see your trace.

---

## Basic Usage

### Understanding Missions and Actions

MissionTrace organizes traces into two simple concepts:

- **Mission**: A complete task or workflow (e.g., "answer a customer question")
- **Action**: A single step within a mission (e.g., "call GPT-4", "search database")

```
Mission: "Help customer with order"
├── Action: Search order database
├── Action: Call GPT-4 to analyze order status
└── Action: Send response to customer
```

### Creating a Mission

Wrap your agent workflow in a mission to group related actions:

```python
import missiontrace

with missiontrace.mission(name="customer-support-request") as m:
    # All actions inside this block belong to this mission

    # Your agent code here...
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Where is my order?"}]
    )
```

### Adding Metadata

Add context to your missions for easier debugging:

```python
with missiontrace.mission(name="support-request") as m:
    # Add useful metadata
    m.set_metadata("customer_id", "cust_123")
    m.set_metadata("ticket_id", "TICKET-456")
    m.set_metadata("priority", "high")

    # Your agent code...
```

---

## Tracking Custom Code

### Using the Decorator (Recommended)

The easiest way to track your own functions:

```python
import missiontrace

@missiontrace.trace(action_type="tool_call", name="search_products")
def search_products(query: str) -> list:
    """Search the product database."""
    # Your search logic here
    return [{"id": 1, "name": "Widget", "price": 9.99}]

@missiontrace.trace(action_type="tool_call", name="send_email")
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to a customer."""
    # Your email logic here
    return True
```

Now every call to these functions is automatically tracked!

```python
with missiontrace.mission(name="process-order") as m:
    products = search_products("widgets")  # Automatically tracked
    send_email("customer@example.com", "Order Confirmed", "...")  # Automatically tracked
```

### Async Functions

Works seamlessly with async code:

```python
@missiontrace.trace(action_type="tool_call", name="fetch_weather")
async def fetch_weather(city: str) -> dict:
    """Fetch weather data from API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.json()
```

### Action Types

Use the right action type for clarity:

| Type | Use For |
|------|---------|
| `"inference"` | LLM calls, AI model invocations |
| `"tool_call"` | Database queries, API calls, file operations, any tool |

```python
@missiontrace.trace(action_type="inference", name="summarize")
def summarize_with_locmt_model(text: str) -> str:
    # Local LLM inference
    return model.generate(text)

@missiontrace.trace(action_type="tool_call", name="save_to_db")
def save_to_database(data: dict) -> bool:
    # Database operation
    return db.insert(data)
```

### Manual Action Tracking

For more control, use the context manager:

```python
with missiontrace.action("tool_call", name="complex_operation") as act:
    # Set input data (shown in dashboard)
    act.set_input({"query": "important search", "limit": 10})

    # Do your work
    result = perform_complex_operation()

    # Set output data (shown in dashboard)
    act.set_output({"results": result, "count": len(result)})
```

---

## Advanced Features

### Nested Actions

Actions automatically nest to show your workflow hierarchy:

```python
with missiontrace.mission(name="research-task") as m:
    with missiontrace.action("tool_call", name="gather_sources") as parent:
        # These are children of "gather_sources"
        with missiontrace.action("tool_call", name="search_web"):
            web_results = search_web("AI agents")

        with missiontrace.action("tool_call", name="search_papers"):
            paper_results = search_papers("AI agents")

    with missiontrace.action("inference", name="synthesize"):
        summary = synthesize(web_results, paper_results)
```

In the dashboard, you'll see:
```
Mission: research-task
├── gather_sources
│   ├── search_web
│   └── search_papers
└── synthesize
```

### Token Usage Tracking

Track LLM token usage for cost monitoring:

```python
with missiontrace.action("inference", name="generate_response") as act:
    response = my_llm_call()

    act.set_token_usage(
        prompt_tokens=150,
        completion_tokens=50,
        totmt_tokens=200,
        model="gpt-4"
    )
```

### Automatic Log Capture

Python logs are automatically captured and attached to the current action:

```python
import logging
logger = logging.getLogger(__name__)

@missiontrace.trace(action_type="tool_call", name="process_data")
def process_data(items: list):
    logger.info(f"Processing {len(items)} items")  # Captured!

    for item in items:
        logger.debug(f"Processing item: {item}")  # Captured!

    logger.info("Processing complete")  # Captured!
    return {"processed": len(items)}
```

All logs appear in the dashboard alongside the action.

### Intent Classification

Add intent to categorize what your actions do:

```python
@missiontrace.trace(action_type="tool_call", name="get_user", intent="retrieval")
def get_user(user_id: str):
    return db.users.find(user_id)

@missiontrace.trace(action_type="tool_call", name="update_user", intent="mutation")
def update_user(user_id: str, data: dict):
    return db.users.update(user_id, data)

@missiontrace.trace(action_type="inference", name="plan_next_step", intent="planning")
def plan_next_step(context: dict):
    return llm.plan(context)
```

Common intents:
- `planning` - Decision making, strategy
- `generation` - Creating content
- `retrieval` - Fetching data
- `mutation` - Changing data
- `validation` - Checking/verifying
- `notification` - Alerts, messages

### Tagging Actions

Add tags for filtering and searching:

```python
@missiontrace.trace(
    action_type="tool_call",
    name="api_call",
    tags={"service": "stripe", "operation": "charge"}
)
def charge_customer(customer_id: str, amount: float):
    return stripe.charges.create(customer=customer_id, amount=amount)
```

---

## Configuration

### Full Initialization Options

```python
missiontrace.init(
    # Required
    api_key="your-api-key",
    project="my-project",
    endpoint="http://localhost:8100/v1/traces",

    # Optional - Performance
    flush_intervmt_s=2.0,      # How often to send data (seconds)
    max_batch_size=100,        # Max traces per batch

    # Optional - Features
    capture_logs=True,         # Capture Python logs (default: True)
    log_level="DEBUG",         # Minimum log level to capture
    capture_code_context=True, # Capture function metadata (default: True)

    # Optional - Security
    sanitizer_patterns=[       # Regex patterns to redact sensitive data
        r"password=\w+",
        r"secret_key=\w+",
    ],
)
```

### Supported LLM Providers

The OpenAI adapter works with any OpenAI-compatible API:

```python
import openai
from missiontrace.adapters.openai import instrument

missiontrace.init(...)
instrument()

# OpenAI
client = openai.OpenAI(api_key="sk-...")

# Groq
client = openai.OpenAI(
    api_key="gsk-...",
    base_url="https://api.groq.com/openai/v1"
)

# Together AI
client = openai.OpenAI(
    api_key="...",
    base_url="https://api.together.xyz/v1"
)

# Google Gemini
client = openai.OpenAI(
    api_key="...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

# And more: Mistral, Fireworks, Perplexity, DeepSeek, OpenRouter, Cerebras, Azure OpenAI
```

The provider is automatically detected and shown in the dashboard.

### Cleanup

Always shut down cleanly to ensure all traces are sent:

```python
# At the end of your program
missiontrace.shutdown()
```

Or use a context manager for scripts:

```python
import missiontrace
import atexit

missiontrace.init(...)
atexit.register(missiontrace.shutdown)  # Ensures cleanup on exit
```

---

## Troubleshooting

### "MissionTrace not initialized" Error

Make sure you call `missiontrace.init()` before using any SDK features:

```python
import missiontrace

# This must come first!
missiontrace.init(api_key="...", project="...", endpoint="...")

# Now you can use the SDK
from missiontrace.adapters.openai import instrument
instrument()
```

### Traces Not Appearing in Dashboard

1. **Check your endpoint URL** - Make sure it's correct and the backend is running
2. **Check your API key** - Verify it's valid
3. **Call shutdown()** - Traces are batched; call `missiontrace.shutdown()` before your program exits
4. **Check network** - Ensure your app can reach the MissionTrace backend

### Missing Token Usage

Token usage is only captured for providers that return it in their responses. Most OpenAI-compatible APIs support this.

### Logs Not Being Captured

Check your log level configuration:

```python
missiontrace.init(
    capture_logs=True,    # Must be True (default)
    log_level="DEBUG",    # Set to capture DEBUG and above
    ...
)
```

---

## Quick Reference

### Decorators

```python
@missiontrace.trace(action_type="tool_call", name="my_function")
def my_function():
    pass

@missiontrace.trace(action_type="inference", name="llm_call", intent="generation")
async def llm_call():
    pass
```

### Context Managers

```python
# Mission (groups actions)
with missiontrace.mission(name="task-name") as m:
    m.set_metadata("key", "value")

# Action (individual operation)
with missiontrace.action("tool_call", name="action-name") as act:
    act.set_input({"key": "value"})
    act.set_output({"result": "value"})
    act.set_token_usage(prompt_tokens=100, completion_tokens=50, totmt_tokens=150)
```

### Lifecycle

```python
# Initialize
missiontrace.init(api_key="...", project="...", endpoint="...")

# Instrument
from missiontrace.adapters.openai import instrument
instrument()

# Use your code normally...

# Cleanup
missiontrace.shutdown()
```

---

## Need Help?

- Check out the [MissionTrace Dashboard](http://localhost:3000) to visualize your traces
- Report issues at the project repository
- Join our community for support and discussions

Happy tracing!
