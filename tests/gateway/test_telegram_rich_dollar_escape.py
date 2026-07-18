"""Tests for bare-``$`` escaping in Telegram Rich Messages (issue #66746).

Telegram Bot API 10.1 Rich Messages treat ``$...$`` as inline LaTeX math
delimiters (https://core.telegram.org/bots/api#rich-markdown-style). Hermes
forwards the LLM's raw markdown to ``sendRichMessage``/``editMessageText``
unescaped, so two or more bare ``$`` currency figures in the same message
(e.g. ``$395k`` and ``$483k``) get paired by Telegram's own client-side
parser and rendered as a single garbled inline-math span, with any
``**bold**`` markers straddling the pair breaking too.

``_rich_message_payload`` must escape lone ``$`` to ``\\$`` before sending,
while leaving intentional ``$$...$$`` block math, and code spans/blocks,
completely untouched.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these
tests construct a real ``TelegramAdapter``.
"""

import pytest

from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _escape_rich_bare_dollars,
)


@pytest.fixture()
def adapter():
    """Bare adapter instance — _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


class TestBareDollarEscaping:
    """Verify _rich_message_payload escapes bare $ so Telegram doesn't parse it as math."""

    def test_two_bare_dollar_amounts_are_escaped(self, adapter):
        """The exact reported repro shape: two currency figures + bold gap marker."""
        content = (
            "- **Red collateral**: API shows $395k vs UI $483k **88k gap**\n"
            "- **Red debt**: API $233k vs UI $168k **65k gap**"
        )
        md = adapter._rich_message_payload(content)["markdown"]
        assert "\\$395k" in md
        assert "\\$483k" in md
        assert "\\$233k" in md
        assert "\\$168k" in md
        # No UNESCAPED bare $ may survive (would be parsed as math delimiters).
        # Every $ in the output must be immediately preceded by a backslash.
        import re as _re
        unescaped = _re.findall(r'(?<!\\)\$', md)
        assert not unescaped, f"found unescaped bare $ in: {md!r}"

    def test_single_bare_dollar_is_escaped(self, adapter):
        content = "Price is $50 today."
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "Price is \\$50 today."

    def test_block_math_is_preserved(self, adapter):
        """$$...$$ is intentional block math (see _needs_rich_rendering) and
        must survive untouched — this is the existing, wanted behavior."""
        content = "Outside details: $$x^2 + y^2$$"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$$x^2 + y^2$$" in md
        assert "\\$\\$" not in md

    def test_fenced_code_block_dollars_untouched(self, adapter):
        content = "Run this:\n```bash\necho $(date)\n```"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "echo $(date)" in md

    def test_inline_code_dollars_untouched(self, adapter):
        content = "Set `echo $HOME` in your shell."
        md = adapter._rich_message_payload(content)["markdown"]
        assert "`echo $HOME`" in md

    def test_mixed_prose_code_and_block_math(self, adapter):
        content = "Price is $50. Formula: $$E=mc^2$$. Code: `cost=$100`"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "\\$50" in md
        assert "$$E=mc^2$$" in md
        assert "`cost=$100`" in md

    def test_no_dollar_signs_unaffected(self, adapter):
        content = "No currency figures here at all."
        assert adapter._rich_message_payload(content)["markdown"] == content

    def test_skip_entity_detection_flag_preserved_with_dollar_escaping(self, adapter):
        payload = adapter._rich_message_payload(
            "Cost: $50", skip_entity_detection=True
        )
        assert payload.get("skip_entity_detection") is True
        assert payload["markdown"] == "Cost: \\$50"


class TestEscapeRichBareDollarsHelper:
    """Direct unit tests of the module-level _escape_rich_bare_dollars helper."""

    def test_empty_string(self):
        assert _escape_rich_bare_dollars("") == ""

    def test_no_dollars_returns_unchanged(self):
        assert _escape_rich_bare_dollars("plain text") == "plain text"

    def test_multiple_bare_dollars_all_escaped(self):
        result = _escape_rich_bare_dollars("$1 $2 $3")
        assert result == "\\$1 \\$2 \\$3"

    def test_block_math_spanning_multiple_lines_preserved(self):
        content = "Before\n$$\nE = mc^2\n$$\nAfter $5 more"
        result = _escape_rich_bare_dollars(content)
        assert "$$\nE = mc^2\n$$" in result
        assert "\\$5" in result
