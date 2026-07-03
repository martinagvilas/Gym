# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeFailureMixin, judge_failure_metrics, run_judge
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.omniscience.app import (
    OmniscienceConfig,
    OmniscienceServer,
    OmniscienceVerifyRequest,
    _strip_thinking_traces,
    extract_text_from_response,
    parse_judge_grade,
)


class TestStripThinkingTraces:
    def test_strips_think_tags(self) -> None:
        assert _strip_thinking_traces("<think>some reasoning</think>The answer is 42") == "The answer is 42"

    def test_strips_thinking_tags(self) -> None:
        assert _strip_thinking_traces("<thinking>deep thought</thinking>Result: yes") == "Result: yes"

    def test_strips_unpaired_closing_think(self) -> None:
        assert _strip_thinking_traces("reasoning here</think>The actual answer") == "The actual answer"

    def test_strips_unpaired_closing_thinking(self) -> None:
        assert _strip_thinking_traces("reasoning here</thinking>The actual answer") == "The actual answer"

    def test_no_tags(self) -> None:
        assert _strip_thinking_traces("plain text") == "plain text"

    def test_multiline_think(self) -> None:
        assert _strip_thinking_traces("<think>\nline1\nline2\n</think>\nAnswer") == "Answer"


class TestParseJudgeGrade:
    def test_single_letter_a(self) -> None:
        assert parse_judge_grade("A") == "A"

    def test_single_letter_b(self) -> None:
        assert parse_judge_grade("B") == "B"

    def test_single_letter_c(self) -> None:
        assert parse_judge_grade("C") == "C"

    def test_single_letter_d(self) -> None:
        assert parse_judge_grade("D") == "D"

    def test_letter_on_last_line(self) -> None:
        assert parse_judge_grade("Some reasoning\nA") == "A"

    def test_letter_in_text(self) -> None:
        assert parse_judge_grade("The grade is A based on analysis") == "A"

    def test_fallback_to_b(self) -> None:
        assert parse_judge_grade("no clear grade here") == "B"

    def test_whitespace_handling(self) -> None:
        assert parse_judge_grade("  D  ") == "D"

    def test_d_in_multiline(self) -> None:
        assert parse_judge_grade("reasoning\nmore reasoning\nD") == "D"


class TestExtractTextFromResponse:
    def _make_response(self, text: str) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def test_extracts_text(self) -> None:
        response = self._make_response("Hello world")
        assert extract_text_from_response(response) == "Hello world"

    def test_strips_thinking(self) -> None:
        response = self._make_response("<think>reasoning</think>Answer")
        assert extract_text_from_response(response) == "Answer"

    def test_empty_output(self) -> None:
        response = NeMoGymResponse(
            id="test",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        assert extract_text_from_response(response) == ""


class TestOmniscienceServer:
    @fixture
    def config(self) -> OmniscienceConfig:
        return OmniscienceConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="judge_model",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _make_model_response(self, text: str) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="policy_resp",
            created_at=0.0,
            model="policy_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def _make_judge_response(self, grade: str) -> dict:
        return NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="judge_msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=grade, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    async def test_verify_correct(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("1989")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In what year did the Berlin Wall fall?",
            expected_answer="1989",
            domain="Humanities and Social Sciences",
            topic="History",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.is_correct == approx(1.0)
        assert result.is_incorrect == approx(0.0)
        assert result.is_partial == approx(0.0)
        assert result.is_not_attempted == approx(0.0)
        assert result.omniscience_index == approx(1.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_incorrect(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("B"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("1991")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In what year did the Berlin Wall fall?",
            expected_answer="1989",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "incorrect"
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(1.0)
        assert result.omniscience_index == approx(-1.0)
        assert result.is_hallucination == approx(1.0)

    async def test_verify_partial(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("C"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("About 28 million")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="In millions of dollars, how much profit did Company X make in 2024?",
            expected_answer="28",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "partial"
        assert result.is_partial == approx(1.0)
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.omniscience_index == approx(0.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_not_attempted(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("I don't have enough information to answer this question.")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some obscure question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.verdict == "not_attempted"
        assert result.is_not_attempted == approx(1.0)
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.omniscience_index == approx(0.0)
        assert result.is_hallucination == approx(0.0)

    async def test_verify_with_thinking_traces(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("<think>Let me recall...</think>Pancreas")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What organ in the human body produces insulin?",
            expected_answer="Pancreas",
        )

        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.verdict == "correct"
        assert result.extracted_answer == "Pancreas"

    async def test_verify_response_fields(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Test answer")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Test question?",
            expected_answer="Test answer",
        )

        result = await server.verify(request)
        dump = result.model_dump()
        assert "reward" in dump
        assert "extracted_answer" in dump
        assert "expected_answer" in dump
        assert "verdict" in dump
        assert "judge_output" in dump
        assert "is_correct" in dump
        assert "is_incorrect" in dump
        assert "is_partial" in dump
        assert "is_not_attempted" in dump
        assert "omniscience_index" in dump
        assert "is_hallucination" in dump
        assert "judge_failed" in dump
        assert "judge_failure_reason" in dump
        assert result.expected_answer == "Test answer"
        assert result.judge_failed is False
        assert result.judge_failure_reason is None

    async def test_verify_empty_response(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        server_mock.post = AsyncMock(return_value=response_mock)

        # Empty model output
        empty_response = NeMoGymResponse(
            id="empty",
            created_at=0.0,
            model="policy_model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=empty_response,
            question="Test?",
            expected_answer="Answer",
        )

        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.extracted_answer == ""

    async def test_judge_prompt_contains_question_and_answer(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("A"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("Paris")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="What is the capital of France?",
            expected_answer="Paris",
        )

        await server.verify(request)

        # Verify the judge was called with the right prompt content
        call_kwargs = server_mock.post.call_args
        json_payload = call_kwargs.kwargs["json"]
        judge_input = json_payload.input[0].content
        assert "What is the capital of France?" in judge_input
        assert "Paris" in judge_input

    async def test_verify_judge_error_after_exhausting_retries(self, config: OmniscienceConfig) -> None:
        config = config.model_copy(update={"judge_max_attempts": 2, "judge_retry_initial_backoff_seconds": 0.0})
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 503
        response_mock.text = AsyncMock(return_value="Authentication backend unavailable.")
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("some answer")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="q?",
            expected_answer="a",
        )

        result = await server.verify(request)
        assert result.verdict is None
        assert result.reward == approx(0.0)
        assert result.judge_failed is True
        assert result.is_correct == approx(0.0)
        assert result.is_incorrect == approx(0.0)
        assert result.is_partial == approx(0.0)
        assert result.is_not_attempted == approx(0.0)
        assert result.omniscience_index == approx(0.0)
        assert result.is_hallucination == approx(0.0)
        assert "503" in (result.judge_failure_reason or "")
        assert server_mock.post.await_count == config.judge_max_attempts

    async def test_verify_retry_recovers_on_second_attempt(self, config: OmniscienceConfig) -> None:
        config = config.model_copy(update={"judge_max_attempts": 3, "judge_retry_initial_backoff_seconds": 0.0})
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        fail_mock = AsyncMock()
        fail_mock.status = 503
        fail_mock.text = AsyncMock(return_value="transient")

        ok_mock = AsyncMock()
        ok_mock.status = 200
        ok_mock.json = AsyncMock(return_value=self._make_judge_response("A"))

        server_mock.post = AsyncMock(side_effect=[fail_mock, ok_mock])

        model_response = self._make_model_response("Paris")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="capital of France?",
            expected_answer="Paris",
        )

        result = await server.verify(request)
        assert result.verdict == "correct"
        assert result.is_correct == approx(1.0)
        assert result.judge_failed is False
        assert server_mock.post.await_count == 2

    async def test_verify_judge_error_on_non_dict_body(self, config: OmniscienceConfig) -> None:
        config = config.model_copy(update={"judge_max_attempts": 1, "judge_retry_initial_backoff_seconds": 0.0})
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value="Hit an exception in inner server")
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("some answer")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="q?",
            expected_answer="a",
        )

        result = await server.verify(request)
        assert result.verdict is None
        assert result.judge_failed is True
        assert "non_dict_body" in (result.judge_failure_reason or "")

    async def test_verify_no_think_tag_keeps_raw_generation(self, config: OmniscienceConfig) -> None:
        """No <think>/<thinking> tags at all: keep raw generation as the model's answer."""
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        server_mock.post = AsyncMock(return_value=response_mock)

        raw = "Let me think about this... I'm not sure about the answer"
        model_response = self._make_model_response(raw)
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.extracted_answer == raw
        assert result.verdict == "not_attempted"

    async def test_verify_unclosed_think_tag_blanks_generation(self, config: OmniscienceConfig) -> None:
        """<think> opened but never closed: response was truncated mid-reasoning, blank the generation."""
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value=self._make_judge_response("D"))
        server_mock.post = AsyncMock(return_value=response_mock)

        model_response = self._make_model_response("<think>reasoning that never finished")
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            question="Some question?",
            expected_answer="Some answer",
        )

        result = await server.verify(request)
        assert result.extracted_answer == ""
        assert result.verdict == "not_attempted"


class TestOmniscienceScoreFn:
    def test_correct(self) -> None:
        assert OmniscienceServer._omni_score_fn({"verdict": "correct"}) == {
            "judge_correct": 1.0,
            "judge_incorrect": 0.0,
            "judge_partially_correct": 0.0,
            "judge_abstained": 0.0,
        }

    def test_incorrect(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "incorrect"})
        assert scores["judge_correct"] == 0.0
        assert scores["judge_incorrect"] == -1.0

    def test_partial(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "partial"})
        assert scores["judge_partially_correct"] == 1.0
        assert scores["judge_correct"] == 0.0

    def test_not_attempted(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"verdict": "not_attempted"})
        assert scores["judge_abstained"] == 1.0
        assert scores["judge_correct"] == 0.0

    def test_judge_failed_is_all_zero(self) -> None:
        """judge_failed samples emit 0.0 for every bucket; mean/judge_failed comes from RewardProfiler."""
        scores = OmniscienceServer._omni_score_fn({"judge_failed": True})
        assert scores == {
            "judge_correct": 0.0,
            "judge_incorrect": 0.0,
            "judge_partially_correct": 0.0,
            "judge_abstained": 0.0,
        }

    def test_judge_failed_overrides_any_verdict(self) -> None:
        scores = OmniscienceServer._omni_score_fn({"judge_failed": True, "verdict": "correct"})
        assert scores["judge_correct"] == 0.0

    def test_unknown_verdict_is_all_zero(self) -> None:
        scores = OmniscienceServer._omni_score_fn({})
        assert scores == {
            "judge_correct": 0.0,
            "judge_incorrect": 0.0,
            "judge_partially_correct": 0.0,
            "judge_abstained": 0.0,
        }


class TestExtractTextNoStrip:
    def _make_response(self, text: str) -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def test_strip_false_preserves_reasoning(self) -> None:
        response = self._make_response("reasoning here</think>answer")
        assert extract_text_from_response(response, strip_thinking=False) == "reasoning here</think>answer"

    def test_strip_true_removes_reasoning(self) -> None:
        response = self._make_response("reasoning here</think>answer")
        assert extract_text_from_response(response, strip_thinking=True) == "answer"


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_success_returns_result_and_none(self) -> None:
        async def ok() -> str:
            return "hello"

        result, reason = await run_judge(ok())
        assert result == "hello"
        assert reason is None

    @pytest.mark.asyncio
    async def test_exception_becomes_reason(self) -> None:
        async def boom() -> str:
            raise RuntimeError("judge exploded")

        result, reason = await run_judge(boom())
        assert result is None
        assert reason is not None
        assert "RuntimeError" in reason
        assert "judge exploded" in reason

    @pytest.mark.asyncio
    async def test_pydantic_validation_error_becomes_reason(self) -> None:
        from pydantic import ValidationError

        async def bad_body() -> str:
            NeMoGymResponse.model_validate({"not": "valid"})
            return "unreachable"

        result, reason = await run_judge(bad_body())
        assert result is None
        assert reason is not None
        assert "ValidationError" in reason
        _ = ValidationError


class TestJudgeFailureMetrics:
    """Sanity-check the nemo_gym.judge helper as consumed by omniscience.

    The canonical unit tests live in tests/unit_tests/test_judge.py -- these
    duplicate the shape assertions so a Gym-side regression surfaces in the
    omniscience test run too.
    """

    def test_no_failures(self) -> None:
        tasks = [[{"reward": 1.0, "judge_failed": False}, {"reward": 0.0, "judge_failed": False}]]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 0
        assert m["reward[judge_ok_only]"] == approx(0.5)

    def test_all_failures(self) -> None:
        tasks = [[{"reward": 0.0, "judge_failed": True}, {"reward": 0.0, "judge_failed": True}]]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 2
        assert m["reward[judge_ok_only]"] is None

    def test_partial_failures_split_scores(self) -> None:
        tasks = [
            [
                {"reward": 1.0, "judge_failed": False},
                {"reward": 0.0, "judge_failed": True},
                {"reward": 1.0, "judge_failed": False},
            ]
        ]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 1
        assert m["reward[judge_ok_only]"] == approx(1.0)

    def test_empty_tasks(self) -> None:
        assert judge_failure_metrics([]) == {}

    def test_missing_judge_failed_treated_as_ok(self) -> None:
        tasks = [[{"reward": 1.0}, {"reward": 0.5}]]
        m = judge_failure_metrics(tasks)
        assert m["judge_failures"] == 0
        assert m["reward[judge_ok_only]"] == approx(0.75)


class TestJudgeFailureMixin:
    def test_defaults(self) -> None:
        instance = JudgeFailureMixin()
        assert instance.judge_failed is False
        assert instance.judge_failure_reason is None

    def test_setting_reason(self) -> None:
        instance = JudgeFailureMixin(judge_failed=True, judge_failure_reason="ClientError: rate limit")
        assert instance.judge_failed is True
        assert instance.judge_failure_reason == "ClientError: rate limit"


class TestComputeMetricsBothScores:
    """End-to-end: mixed batch of correct + incorrect + judge_failed samples
    yields both the "count failed as 0.0" score and the "skip failed" score
    from the same compute_metrics(tasks) call. Mirrors Ewa's stated core
    feature: two headline scores side by side.
    """

    @fixture
    def config(self) -> OmniscienceConfig:
        return OmniscienceConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _rollout(self, *, verdict: str | None, reward: float, judge_failed: bool = False) -> dict:
        return {
            "extracted_answer": "some answer" if not judge_failed else "",
            "verdict": verdict,
            "reward": reward,
            "judge_failed": judge_failed,
            "judge_failure_reason": "simulated" if judge_failed else None,
            "is_correct": 1.0 if verdict == "correct" else 0.0,
            "is_incorrect": 1.0 if verdict == "incorrect" else 0.0,
            "is_partial": 1.0 if verdict == "partial" else 0.0,
            "is_not_attempted": 1.0 if verdict == "not_attempted" else 0.0,
        }

    def test_mixed_batch_emits_both_scores(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        tasks = [
            [self._rollout(verdict="correct", reward=1.0)],
            [self._rollout(verdict="incorrect", reward=0.0)],
            [self._rollout(verdict="correct", reward=1.0)],
            [self._rollout(verdict=None, reward=0.0, judge_failed=True)],
        ]

        metrics = server.compute_metrics(tasks)

        assert metrics["judge_failures"] == 1
        assert metrics["reward[judge_ok_only]"] == approx(2.0 / 3.0)

        assert metrics["pass@1/judge_correct"] == approx(50.0)
        assert "pass@1/judge_omni_index" in metrics
        assert "pass@1/judge_omni_hallucination" in metrics

    def test_all_ok_batch(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        tasks = [
            [self._rollout(verdict="correct", reward=1.0)],
            [self._rollout(verdict="incorrect", reward=0.0)],
        ]
        metrics = server.compute_metrics(tasks)

        assert metrics["judge_failures"] == 0
        assert metrics["reward[judge_ok_only]"] == approx(0.5)

    def test_all_failed_batch_reward_ok_only_is_none(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        tasks = [[self._rollout(verdict=None, reward=0.0, judge_failed=True)] for _ in range(3)]
        metrics = server.compute_metrics(tasks)

        assert metrics["judge_failures"] == 3
        assert metrics["reward[judge_ok_only]"] is None

    def test_get_key_metrics_promotes_judge_failure_signals(self, config: OmniscienceConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        tasks = [
            [self._rollout(verdict="correct", reward=1.0)],
            [self._rollout(verdict=None, reward=0.0, judge_failed=True)],
        ]
        metrics = server.compute_metrics(tasks)
        key = server.get_key_metrics(metrics)

        assert "judge_failures" in key
        assert "reward[judge_ok_only]" in key


class TestVerifyExceptionPath:
    @fixture
    def config(self) -> OmniscienceConfig:
        return OmniscienceConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    async def test_client_error_becomes_judge_failed(self, config: OmniscienceConfig) -> None:
        config = config.model_copy(update={"judge_max_attempts": 2, "judge_retry_initial_backoff_seconds": 0.0})
        server_mock = MagicMock(spec=ServerClient)
        server = OmniscienceServer(config=config, server_client=server_mock)

        server_mock.post = AsyncMock(side_effect=ConnectionResetError("peer closed"))

        policy_response = NeMoGymResponse(
            id="policy_resp",
            created_at=0.0,
            model="policy_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg",
                    content=[NeMoGymResponseOutputText(annotations=[], text="some answer", type="output_text")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = OmniscienceVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=policy_response,
            question="q?",
            expected_answer="a",
        )

        result = await server.verify(request)
        assert result.verdict is None
        assert result.judge_failed is True
        assert result.reward == approx(0.0)
        assert result.judge_failure_reason is not None
        assert (
            "ConnectionResetError" in result.judge_failure_reason or "transport_error" in result.judge_failure_reason
        )
        assert server_mock.post.await_count == config.judge_max_attempts
