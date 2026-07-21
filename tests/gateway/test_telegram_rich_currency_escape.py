"""Tests for currency-dollar escaping in Telegram Rich Messages (issue #66746).

Telegram Bot API 10.1 Rich Messages treat ``$...$`` as inline LaTeX math
delimiters (https://core.telegram.org/bots/api#rich-markdown-style). Two or
more bare ``$`` currency figures in the same message (e.g. ``$395k``,
``$483k``) get paired by Telegram's own client-side parser into a single
garbled inline-math span, with any ``**bold**`` markers straddling the pair
breaking too.

``_rich_message_payload`` must escape ``$`` immediately followed by a digit
(the currency signal) to ``\\$``, while leaving intentional inline math
(``$E=mc^2$``, ``$\\alpha$``), ``$$...$$`` block math, and code spans/blocks
completely untouched.

This is the key differentiator from #66779 and #66784, both of which escaped
EVERY bare ``$`` and were flagged by the maintainer (teknium1 via
hermes-sweeper) for destroying the ``$...$`` inline math syntax that the
system prompt (agent/prompt_builder.py:869-872) actively instructs the agent
to use.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these
tests construct a real ``TelegramAdapter``.
"""

import re

import pytest

from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _escape_rich_currency_dollars,
)


@pytest.fixture()
def adapter():
    """Bare adapter instance — _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


class TestCurrencyDollarEscaping:
    """Verify _rich_message_payload escapes currency $ but preserves math."""

    def test_two_currency_amounts_are_escaped(self, adapter):
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
        # No UNESCAPED currency $ may survive.
        unescaped = re.findall(r'(?<!\\)\$(?=\d)', md)
        assert not unescaped, f"found unescaped currency $ in: {md!r}"

    def test_single_currency_amount_escaped(self, adapter):
        content = "Price is $50 today."
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "Price is \\$50 today."

    def test_inline_math_preserved(self, adapter):
        """$E=mc^2$ is intentional inline math — the system prompt instructs
        the agent to use $...$ math. Must NOT be escaped."""
        content = "Energy: $E=mc^2$ and momentum $p=mv$."
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$E=mc^2$" in md
        assert "$p=mv$" in md
        assert "\\$" not in md

    def test_inline_math_with_greek_letters_preserved(self, adapter):
        content = "The value $\\alpha + \\beta$ is constant."
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$\\alpha + \\beta$" in md

    def test_inline_math_with_variables_preserved(self, adapter):
        content = "Solve $x^2 + y^2 = r^2$ for $r$."
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$x^2 + y^2 = r^2$" in md
        assert "$r$" in md

    def test_block_math_preserved(self, adapter):
        """$$...$$ block math must survive untouched."""
        content = "Outside details: $$x^2 + y^2$$"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$$x^2 + y^2$$" in md
        assert "\\$\\$" not in md

    def test_block_math_multiline_preserved(self, adapter):
        """$$ delimiters and math content survive. Newlines inside the block
        get hard-break treatment from _rich_normalize_linebreaks — that's
        pre-existing behavior (the protected-region regex only covers fenced
        code and tables, not $$ blocks). The key assertion: the $$ delimiters
        and math content are NOT corrupted by dollar escaping."""
        content = "Before\n$$\nE = mc^2\n$$\nAfter"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "$$" in md
        assert "E = mc^2" in md
        assert "\\$" not in md  # no dollar escaping touched the block math

    def test_fenced_code_block_dollars_untouched(self, adapter):
        content = "Run this:\n```bash\necho $(date)\n```"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "echo $(date)" in md

    def test_inline_code_dollars_untouched(self, adapter):
        content = "Set `echo $HOME` in your shell."
        md = adapter._rich_message_payload(content)["markdown"]
        assert "`echo $HOME`" in md

    def test_mixed_currency_math_and_code(self, adapter):
        """The hardest realistic case: currency, inline math, block math,
        and code all in one message."""
        content = (
            "Price is $50. Formula: $E=mc^2$. Block: $$p=mv$$. "
            "Code: `cost=$100`."
        )
        md = adapter._rich_message_payload(content)["markdown"]
        assert "\\$50" in md          # currency escaped
        assert "$E=mc^2$" in md       # inline math preserved
        assert "$$p=mv$$" in md       # block math preserved
        assert "`cost=$100`" in md    # code untouched

    def test_no_dollar_signs_unaffected(self, adapter):
        content = "No currency figures here at all."
        assert adapter._rich_message_payload(content)["markdown"] == content

    def test_skip_entity_detection_flag_preserved(self, adapter):
        payload = adapter._rich_message_payload(
            "Cost: $50", skip_entity_detection=True
        )
        assert payload.get("skip_entity_detection") is True
        assert payload["markdown"] == "Cost: \\$50"

    def test_dollar_five_as_math_degrades_gracefully(self, adapter):
        """$5$ as intentional inline math is vanishingly rare. If it occurs,
        the $ before the digit gets escaped — degrading to literal "$5$"
        instead of math-styled "5". This is acceptable: no data corruption."""
        content = "The answer is $5$ dollars."
        md = adapter._rich_message_payload(content)["markdown"]
        # The first $ (before digit) is escaped; the closing $ (before space)
        # is NOT followed by a digit, so it's preserved.
        assert "\\$5$" in md


class TestEscapeRichCurrencyDollarsHelper:
    """Direct unit tests of the module-level _escape_rich_currency_dollars."""

    def test_empty_string(self):
        assert _escape_rich_currency_dollars("") == ""

    def test_no_dollars_returns_unchanged(self):
        assert _escape_rich_currency_dollars("plain text") == "plain text"

    def test_multiple_currency_dollars_all_escaped(self):
        result = _escape_rich_currency_dollars("$1 $2 $3")
        assert result == "\\$1 \\$2 \\$3"

    def test_dollar_followed_by_letter_not_escaped(self):
        """$E, $x, $r — these are math signals, not currency."""
        result = _escape_rich_currency_dollars("$E=mc^2$ $x$ $r$")
        assert result == "$E=mc^2$ $x$ $r$"

    def test_dollar_followed_by_backslash_not_escaped(self):
        """$\\alpha — LaTeX command, not currency."""
        result = _escape_rich_currency_dollars("$\\alpha$")
        assert result == "$\\alpha$"

    def test_dollar_at_end_of_string_not_escaped(self):
        """Trailing $ with nothing after it — not currency."""
        result = _escape_rich_currency_dollars("cost is $")
        assert result == "cost is $"

    def test_dollar_followed_by_space_not_escaped(self):
        result = _escape_rich_currency_dollars("$ something")
        assert result == "$ something"

    def test_currency_with_comma_separators(self):
        result = _escape_rich_currency_dollars("$1,234,567")
        assert result == "\\$1,234,567"

    def test_currency_with_decimal(self):
        result = _escape_rich_currency_dollars("$99.99")
        assert result == "\\$99.99"

    def test_block_math_spanning_multiple_lines_preserved(self):
        content = "Before\n$$\nE = mc^2\n$$\nAfter $5 more"
        result = _escape_rich_currency_dollars(content)
        assert "$$\nE = mc^2\n$$" in result
        assert "\\$5" in result
