"""Unit tests for chromegpt utility functions."""

from unittest.mock import MagicMock, patch

import pytest

from chromegpt.main import run_chromegpt
from chromegpt.tools.utils import (
    is_complete_sentence,
    prettify_text,
    truncate_string_from_last_occurrence,
)


class TestIsCompleteSentence:
    """Tests for is_complete_sentence."""

    def test_period_ending(self) -> None:
        assert is_complete_sentence("This is a sentence.") is True

    def test_exclamation_ending(self) -> None:
        assert is_complete_sentence("Hello world!") is True

    def test_question_ending(self) -> None:
        assert is_complete_sentence("Is this a sentence?") is True

    def test_trailing_whitespace(self) -> None:
        assert is_complete_sentence("This is a sentence.  ") is True

    def test_incomplete_sentence(self) -> None:
        assert is_complete_sentence("This is not a sentence") is False

    def test_empty_string(self) -> None:
        assert is_complete_sentence("") is False

    def test_only_punctuation(self) -> None:
        assert is_complete_sentence("!") is True


class TestPrettifyText:
    """Tests for prettify_text."""

    def test_basic(self) -> None:
        assert prettify_text("Hello World") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert prettify_text("Hello   World") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        assert prettify_text("  Hello World  ") == "hello world"

    def test_with_limit(self) -> None:
        assert prettify_text("Hello World", limit=5) == "hello"

    def test_limit_none(self) -> None:
        assert prettify_text("Hello World", limit=None) == "hello world"

    def test_newlines_and_tabs(self) -> None:
        assert prettify_text("Hello\n\tWorld") == "hello world"


class TestTruncateStringFromLastOccurrence:
    """Tests for truncate_string_from_last_occurrence."""

    def test_basic(self) -> None:
        result = truncate_string_from_last_occurrence("Hello. World. Foo", ".")
        assert result == "Hello. World."

    def test_no_occurrence(self) -> None:
        result = truncate_string_from_last_occurrence("Hello World", ".")
        assert result == "Hello World"

    def test_single_occurrence(self) -> None:
        result = truncate_string_from_last_occurrence("Hello.World", ".")
        assert result == "Hello."

    def test_character_at_end(self) -> None:
        result = truncate_string_from_last_occurrence("Hello.", ".")
        assert result == "Hello."

    def test_empty_string(self) -> None:
        result = truncate_string_from_last_occurrence("", ".")
        assert result == ""


class TestRunChromegpt:
    """Tests for run_chromegpt agent selection."""

    def test_invalid_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="Agent invalid-agent not found"):
            run_chromegpt(task="test", agent="invalid-agent")

    @patch("chromegpt.main.ZeroShotAgent")
    def test_zero_shot_agent_selected(self, mock_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_cls.return_value = mock_agent

        result = run_chromegpt(task="test task", agent="zero-shot")

        mock_cls.assert_called_once_with(model="gpt-3.5-turbo", verbose=False)
        mock_agent.run.assert_called_once_with(["test task"])
        assert result == "done"

    @patch("chromegpt.main.AutoGPTAgent")
    def test_auto_gpt_agent_selected(self, mock_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_cls.return_value = mock_agent

        result = run_chromegpt(task="test task", agent="auto-gpt", model="gpt-4")

        mock_cls.assert_called_once_with(model="gpt-4", verbose=False)
        mock_agent.run.assert_called_once_with(["test task"])
        assert result == "done"

    @patch("chromegpt.main.BabyAGIAgent")
    def test_baby_agi_agent_selected(self, mock_cls: MagicMock) -> None:
        mock_agent = MagicMock()
        mock_agent.run.return_value = "done"
        mock_cls.return_value = mock_agent

        result = run_chromegpt(task="test task", agent="baby-agi")

        mock_cls.assert_called_once_with(model="gpt-3.5-turbo", verbose=False)
        mock_agent.run.assert_called_once_with(["test task"])
        assert result == "done"
