"""LOCAL PATCH tests: ``/model <alias>`` reports the env var its key came from.

Ported from live commit 500f5bed363 (tests/hermes_cli/test_model_alias_credentials_83612.py, renamed
upstream to test_model_alias_credentials.py) onto upstream PR #148256's implementation. The switch result
names the env var the alias key came from, so callers can bind the session to it (per-request refresh and
restart re-read that var, never the provider default).
"""

from tests.hermes_cli.test_model_alias_credentials import ALIAS_HOST, _switch_to_alias


def test_key_env_alias_reports_its_var(monkeypatch):
    monkeypatch.setenv("THETA_API_KEY", "sk-theta-key")
    result, _ = _switch_to_alias(monkeypatch, {
        "model": "theta-1", "provider": "custom", "base_url": ALIAS_HOST, "key_env": "THETA_API_KEY",
    })
    assert result.success, result.error_message
    assert result.api_key == "sk-theta-key"
    assert result.key_env == "THETA_API_KEY"


def test_env_template_alias_reports_its_var(monkeypatch):
    monkeypatch.setenv("THETA_TEMPLATE_KEY", "sk-theta-template")
    result, _ = _switch_to_alias(monkeypatch, {
        "model": "theta-1", "provider": "custom", "base_url": ALIAS_HOST,
        "api_key": "${THETA_TEMPLATE_KEY}",
    })
    assert result.success, result.error_message
    assert result.key_env == "THETA_TEMPLATE_KEY"


def test_literal_key_alias_has_no_var(monkeypatch):
    result, _ = _switch_to_alias(monkeypatch, {
        "model": "theta-1", "provider": "custom", "base_url": ALIAS_HOST, "api_key": "sk-literal",
    })
    assert result.success, result.error_message
    assert result.key_env == ""
