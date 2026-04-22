# MissionTrace SDK

Observability and tracing SDK for AI agent systems.

## Installation

```bash
pip install missiontrace
```

**With adapters:**

```bash
pip install missiontrace[openai]    # OpenAI + compatible LLMs
pip install missiontrace[tinyfish]  # TinyFish web agents
pip install missiontrace[all]       # All adapters
```

## Quick Start

```python
import missiontrace

missiontrace.init(api_key="mt_xxx", project="my-project")

with missiontrace.mission(name="research-task") as m:
    with missiontrace.action("inference", name="openai.chat", intent="generation") as act:
        result = call_llm("Summarize Acme Corp")
        act.set_output({"response": result})
```

## Adapters

### OpenAI (+ Compatible Providers)

```bash
pip install missiontrace[openai]
```

```python
from missiontrace.adapters.openai import instrument
instrument()

client = OpenAI()
response = client.chat.completions.create(model="gpt-4o", messages=[...])
```

Auto-detects: OpenAI, Gemini, Groq, Together, Mistral, Fireworks, DeepSeek, OpenRouter, Cerebras, Perplexity, Azure OpenAI.

### TinyFish (Web Agents)

```bash
pip install missiontrace[tinyfish]
```

```python
from missiontrace.adapters.tinyfish import instrument
instrument()

client = TinyFish(api_key="tf_xxx")
response = client.agent.run(goal="Find the price", url="https://example.com")
```
