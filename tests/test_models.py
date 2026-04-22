"""Tests for core data models."""

import uuid

from missiontrace.core.models import (
    Action,
    ActionStatus,
    ActionType,
    ErrorInfo,
    Mission,
    MissionStatus,
    SamplingParams,
    TokenUsage,
)


class TestMission:
    def test_defaults(self):
        m = Mission()
        assert len(m.id) == 32  # uuid4 hex
        assert m.status == MissionStatus.RUNNING
        assert m.trigger == "manual"
        assert m.metadata == {}
        assert m.ended_at is None

    def test_custom_fields(self):
        m = Mission(
            project_id="proj-1",
            name="research",
            trigger="webhook",
            metadata={"env": "prod"},
        )
        assert m.project_id == "proj-1"
        assert m.name == "research"
        assert m.trigger == "webhook"
        assert m.metadata["env"] == "prod"


class TestAction:
    def test_defaults(self):
        a = Action()
        assert a.status == ActionStatus.RUNNING
        assert a.type == ActionType.TOOL_CALL
        assert a.intent is None
        assert a.parent_id is None
        assert a.token_usage is None
        assert a.error is None

    def test_all_action_types(self):
        for at in ActionType:
            a = Action(type=at)
            assert a.type == at

    def test_action_type_values(self):
        """ActionType enum has exactly inference and tool_call."""
        assert set(ActionType) == {ActionType.INFERENCE, ActionType.TOOL_CALL}
        assert ActionType.INFERENCE.value == "inference"
        assert ActionType.TOOL_CALL.value == "tool_call"

    def test_with_intent(self):
        a = Action(type=ActionType.TOOL_CALL, intent="mutation")
        assert a.intent == "mutation"

    def test_with_custom_intent(self):
        a = Action(type=ActionType.INFERENCE, intent="my_custom_intent")
        assert a.intent == "my_custom_intent"

    def test_with_token_usage(self):
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, totmt_tokens=150, model="gpt-4")
        a = Action(token_usage=tu)
        assert a.token_usage.totmt_tokens == 150
        assert a.token_usage.model == "gpt-4"

    def test_with_error(self):
        err = ErrorInfo(type="ValueError", message="bad input", traceback="...")
        a = Action(error=err, status=ActionStatus.FAILED)
        assert a.error.type == "ValueError"
        assert a.status == ActionStatus.FAILED

    def test_with_sampling_params(self):
        sp = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=1000,
            frequency_penalty=0.5, seed=42,
        )
        a = Action(sampling_params=sp)
        assert a.sampling_params.temperature == 0.7
        assert a.sampling_params.seed == 42

    def test_sampling_params_defaults_none(self):
        a = Action()
        assert a.sampling_params is None


class TestTokenUsageDetailed:
    def test_detailed_breakdowns(self):
        tu = TokenUsage(
            prompt_tokens=500, completion_tokens=200, totmt_tokens=700,
            model="gpt-4o", cached_tokens=50, reasoning_tokens=30,
            audio_tokens_input=10, audio_tokens_output=5,
            accepted_prediction_tokens=15, rejected_prediction_tokens=3,
        )
        assert tu.cached_tokens == 50
        assert tu.reasoning_tokens == 30
        assert tu.audio_tokens_input == 10
        assert tu.audio_tokens_output == 5
        assert tu.accepted_prediction_tokens == 15
        assert tu.rejected_prediction_tokens == 3

    def test_detailed_fields_default_none(self):
        tu = TokenUsage(prompt_tokens=100, completion_tokens=50, totmt_tokens=150)
        assert tu.cached_tokens is None
        assert tu.reasoning_tokens is None


class TestSamplingParams:
    def test_completions_params(self):
        sp = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=1000,
            max_completion_tokens=2000, frequency_penalty=0.5,
            presence_penalty=0.3, seed=42, stop=["\n"],
            n=2, logprobs=True, top_logprobs=5,
            logit_bias={"50256": -100.0},
            response_format={"type": "json_object"},
            service_tier="default", reasoning_effort="high",
            parallel_tool_calls=True, store=True,
        )
        assert sp.temperature == 0.7
        assert sp.n == 2
        assert sp.logit_bias == {"50256": -100.0}
        assert sp.reasoning_effort == "high"

    def test_responses_params(self):
        sp = SamplingParams(
            max_output_tokens=4096, reasoning_effort="medium",
            temperature=0.5, top_p=0.8,
        )
        assert sp.max_output_tokens == 4096
        assert sp.reasoning_effort == "medium"

    def test_all_defaults_none(self):
        sp = SamplingParams()
        assert sp.temperature is None
        assert sp.max_tokens is None
        assert sp.max_output_tokens is None
        assert sp.reasoning_effort is None
        assert sp.logit_bias is None
