"""
MissionTrace SDK — Core Data Models

Two primitives in a strict hierarchy: Mission → Actions.
Artifacts are owned by the MissionTrace backend, not the SDK.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MissionStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ActionType(str, enum.Enum):
    INFERENCE = "inference"
    TOOL_CALL = "tool_call"


class ActionStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    totmt_tokens: int = 0
    model: str | None = None
    # Detailed breakdowns (from OpenAI token details)
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_tokens_input: int | None = None
    audio_tokens_output: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class SamplingParams(BaseModel):
    """All sampling / generation parameters captured from the LLM request."""
    # Chat Completions API params
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | str | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    logit_bias: dict[str, float] | None = None
    response_format: dict[str, Any] | None = None
    service_tier: str | None = None
    # Responses API params
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    # Shared
    parallel_tool_calls: bool | None = None
    store: bool | None = None


class ErrorInfo(BaseModel):
    type: str
    message: str
    traceback: str | None = None  # truncated to 10 frames


# ---------------------------------------------------------------------------
# Three primitives
# ---------------------------------------------------------------------------

class Mission(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    project_id: str = ""
    name: str | None = None
    trigger: str = "manual"  # manual | scheduled | webhook | api
    status: MissionStatus = MissionStatus.RUNNING
    started_at: float = Field(default_factory=time.time)  # Wall-clock time for accurate timestamps
    ended_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    feature_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Action(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    mission_id: str = ""
    parent_id: str | None = None
    type: ActionType = ActionType.TOOL_CALL
    intent: str | None = None  # planning|generation|retrieval|mutation|validation|evaluation|notification|orchestration|<custom>
    name: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    status: ActionStatus = ActionStatus.RUNNING
    started_at: float = Field(default_factory=time.time)  # Wall-clock time for accurate timestamps
    ended_at: float | None = None
    duration_ms: float | None = None
    token_usage: TokenUsage | None = None
    sampling_params: SamplingParams | None = None
    error: ErrorInfo | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Additional metadata (code context, etc.)
    log_events: list[dict[str, Any]] = Field(default_factory=list)  # Captured log entries

    model_config = {"arbitrary_types_allowed": True}


