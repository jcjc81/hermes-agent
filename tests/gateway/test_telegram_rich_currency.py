"""LOCAL PATCH (#66746): currency ``$`` escaping in Telegram rich messages.

Bot API 10.1 ``sendRichMessage`` treats ``$...$`` as inline math, so two bare currency
figures ("$395k and $483k") pair into one garbled math span. ``_rich_message_payload``
escapes only a ``$`` directly followed by a digit and leaves intentional inline math,
``$$`` block math and code untouched.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these tests
construct a real ``TelegramAdapter``.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture()
def adapter():
    """Bare adapter instance: _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


def _md(adapter, content: str) -> str:
    return adapter._rich_message_payload(content)["markdown"]


def test_two_currency_figures_are_escaped(adapter):
    assert _md(adapter, "Asking $395k, sold at $483k") == r"Asking \$395k, sold at \$483k"


def test_inline_math_is_untouched(adapter):
    assert _md(adapter, r"Energy $E=mc^2$ and $\alpha$") == r"Energy $E=mc^2$ and $\alpha$"


def test_block_math_is_untouched(adapter):
    assert _md(adapter, "$$5 + 5 = 10$$") == "$$5 + 5 = 10$$"


def test_code_is_untouched(adapter):
    content = "Run `echo $5` then:\n```\nprice=$10\n```"
    md = _md(adapter, content)
    assert "`echo $5`" in md
    assert "price=$10" in md
    assert r"\$" not in md


def test_escape_composes_with_hard_breaks(adapter):
    """Escaping runs alongside upstream's newline normalization, not instead of it."""
    assert _md(adapter, "Cost $5\nTotal $10") == "Cost \\$5  \nTotal \\$10"
