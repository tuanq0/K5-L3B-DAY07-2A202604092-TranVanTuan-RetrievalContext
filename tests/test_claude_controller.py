"""Offline tests for translating Claude tool calls into agent actions."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from starter.claude_controller import ClaudeController


def tool_response(action):
    block = SimpleNamespace(
        type="tool_use",
        name="agent_action",
        input=action,
    )
    return SimpleNamespace(content=[block], stop_reason="tool_use")


class ClaudeControllerTests(unittest.TestCase):
    @patch("starter.claude_controller.Anthropic")
    @patch("starter.claude_controller.load_dotenv")
    def test_decide_returns_search_action_and_sends_context(
        self, load_dotenv, anthropic
    ):
        client = Mock()
        client.messages.create.return_value = tool_response(
            {"action": "search", "query": "đất chợ dân sinh"}
        )
        anthropic.return_value = client
        controller = ClaudeController(api_key="test-key")

        result = controller.decide(
            "Đất chợ dân sinh là gì?",
            [
                {
                    "queries": ["đất chợ"],
                    "context": "[chunk-1] ...",
                    "context_ids": ["chunk-1"],
                }
            ],
        )

        self.assertEqual(
            result, {"action": "search", "query": "đất chợ dân sinh"}
        )
        request = client.messages.create.call_args.kwargs
        self.assertEqual(request["model"], "claude-haiku-5-5")
        self.assertIn("[chunk-1] ...", request["messages"][0]["content"])
        self.assertEqual(request["tool_choice"], {"type": "any"})

    @patch("starter.claude_controller.Anthropic")
    @patch("starter.claude_controller.load_dotenv")
    def test_decide_returns_final_answer_and_citations(
        self, load_dotenv, anthropic
    ):
        client = Mock()
        client.messages.create.return_value = tool_response(
            {
                "action": "answer",
                "answer": "Thuộc đất công cộng [chunk-1].",
                "citations": ["chunk-1"],
            }
        )
        anthropic.return_value = client
        controller = ClaudeController(api_key="test-key")

        result = controller.decide("Đất chợ dân sinh là gì?", [])

        self.assertEqual(
            result,
            {
                "action": "answer",
                "answer": "Thuộc đất công cộng [chunk-1].",
                "citations": ["chunk-1"],
            },
        )

    @patch("starter.claude_controller.load_dotenv")
    def test_missing_api_key_fails_before_client_creation(self, load_dotenv):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                ClaudeController()

    @patch("starter.claude_controller.Anthropic")
    @patch("starter.claude_controller.load_dotenv")
    def test_response_without_tool_call_is_an_error(self, load_dotenv, anthropic):
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[], stop_reason="end_turn"
        )
        anthropic.return_value = client
        controller = ClaudeController(api_key="test-key")

        with self.assertRaisesRegex(RuntimeError, "did not return an agent action"):
            controller.decide("question", [])


if __name__ == "__main__":
    unittest.main()
