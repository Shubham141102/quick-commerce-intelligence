"""Optional LLM layer (Phase 6G) with a fake client — no network: grounding, guardrails and silent fallback."""

import pandas as pd

from src.rag.composer import Answer
from src.rag.llm import (
    ClaudeClient,
    OpenAICompatibleClient,
    client_from_config,
    client_from_key,
    grounded_rewrite,
    numbers_in,
    verify,
)

EVIDENCE = Answer(
    question="Which SKUs are at high risk and what does the policy say?", route="hybrid", tool="at_risk_skus",
    markdown=("**SKUs at risk of running out** — all stores · High tier, as of 30 Sep 2025\n\n- **High risk** — 10\n"
              "- **Units to order** — 47\n\n**Policy guidance**\n\n> High-risk SKUs must be reviewed the same day.\n>\n"
              "> — *POL-INV §2 Stock risk tiers*"),
    table=pd.DataFrame({"product": ["Baby Wipes"], "closing_stock": [0], "suggested_qty": [4]}),
    citations=["POL-INV §2 Stock risk tiers"])


class Fake:
    name = "fake-model"

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


GOOD = ("10 SKUs are at high risk across all stores, as of 30 Sep 2025.\n- 47 units should be ordered.\n"
        "- High-risk SKUs must be reviewed the same day (POL-INV §2 Stock risk tiers).")


def test_numbers_are_normalised():
    assert numbers_in("₹1,57,101 and 20.5% on 30 Sep") == {"157101", "20.5", "30"}


def test_verified_reply_replaces_text_and_keeps_the_evidence():
    fake = Fake(GOOD)
    out = grounded_rewrite(EVIDENCE, fake)
    assert out.markdown == GOOD and out.evidence_markdown == EVIDENCE.markdown
    assert out.mode == "llm (fake-model, verified)" and out.table is EVIDENCE.table
    system, user = fake.calls[0]
    assert "ONLY the evidence" in system and "POL-INV §2 Stock risk tiers" in user and "Baby Wipes" in user


def test_invented_number_is_rejected_and_template_kept():
    out = grounded_rewrite(EVIDENCE, Fake(GOOD.replace("47 units", "52 units")))
    assert out.markdown == EVIDENCE.markdown and out.mode == "template (LLM reply rejected: unverified number 52)"


def test_missing_citation_is_rejected():
    out = grounded_rewrite(EVIDENCE, Fake(GOOD.replace(" (POL-INV §2 Stock risk tiers)", "")))
    assert out.markdown == EVIDENCE.markdown and "missing citation" in out.mode


def test_errors_fall_back_silently():
    out = grounded_rewrite(EVIDENCE, Fake(error=TimeoutError("slow")))
    assert out.markdown == EVIDENCE.markdown and out.mode == "template (LLM unavailable: TimeoutError)"
    assert verify("", EVIDENCE) == "empty reply" and verify("x" * 3000, EVIDENCE) == "reply too long"


def test_refusals_and_help_are_never_rephrased():
    fake = Fake(GOOD)
    for route in ("not_allowed", "out_of_scope", "help", "clarify"):
        a = Answer("q", route, "fixed text")
        assert grounded_rewrite(a, fake).markdown == "fixed text"
    assert fake.calls == [] and grounded_rewrite(EVIDENCE, None) is EVIDENCE


def test_client_from_config_providers(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert client_from_config({}) is None                                    # nothing configured: template mode
    assert client_from_config({"provider": "groq"}) is None                  # Groq needs a key
    g = client_from_config({"provider": "groq", "api_key": "gsk-test"})
    assert isinstance(g, OpenAICompatibleClient) and g.model == "openai/gpt-oss-120b"
    assert g.base_url == "https://api.groq.com/openai/v1" and g.name == "gpt-oss-120b via Groq"
    o = client_from_config({"provider": "ollama"})                           # local: no key needed
    assert o.base_url.startswith("http://localhost") and o.api_key == ""
    custom = client_from_config({"provider": "groq", "api_key": "k", "model": "llama-3.1-8b-instant"})
    assert custom.model == "llama-3.1-8b-instant" and custom.name == "llama-3.1-8b-instant via Groq"
    assert client_from_config({"provider": "nonsense", "api_key": "k"}) is None
    assert isinstance(client_from_config({"provider": "anthropic", "api_key": "sk"}), ClaudeClient)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_API_KEY", "gsk-env")
    assert client_from_config({}).api_key == "gsk-env"                       # environment variables also work


def test_client_from_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client_from_key(None) is None and client_from_key("  ") is None
    c = client_from_key("sk-test")
    assert isinstance(c, ClaudeClient) and c.name == "claude-haiku-4-5" and c.model == "claude-haiku-4-5-20251001"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert client_from_key(None).api_key == "sk-env"


def test_demo_secrets_script_keeps_the_anthropic_section():
    from scripts.create_demo_secrets import _other_sections
    text = ('[auth.users.inventory]\nname = "x"\npassword_hash = "h"\n\n[anthropic]\napi_key = "sk-keep"\n\n'
            '[auth.users.admin]\nname = "y"\n')
    kept = _other_sections(text)
    assert "[anthropic]" in kept and 'api_key = "sk-keep"' in kept and not any("auth" in ln for ln in kept)


def test_typographic_hyphens_are_normalised_before_verifying():
    nb = GOOD.replace("POL-INV", "POL‑INV").replace("High-risk", "High‑risk")   # what gpt-oss emits
    out = grounded_rewrite(EVIDENCE, Fake(nb))
    assert out.mode.startswith("llm") and "POL-INV §2 Stock risk tiers" in out.markdown and "‑" not in out.markdown


def test_truncated_reply_falls_back():
    from src.rag.llm import TruncatedReply
    out = grounded_rewrite(EVIDENCE, Fake(error=TruncatedReply("reply stopped at the token limit")))
    assert out.markdown == EVIDENCE.markdown and out.mode == "template (LLM unavailable: TruncatedReply)"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    import io
    import json
    import urllib.error
    import urllib.request

    from src.rag import llm
    calls, waits = [], []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) <= 2:                       # throttled twice, then OK
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"retry-after": "3"}, None)
        return io.BytesIO(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request("https://example.test", data=b"{}", method="POST")
    assert llm._post_with_retry(req, 5, sleep=waits.append) == {"ok": True}
    assert len(calls) == 3 and waits == [3.0, 3.0]


def test_numbers_written_as_words_are_checked_too():
    worded = GOOD.replace("10 SKUs", "Ten SKUs")                    # correct, but a word escapes the digit check
    out = grounded_rewrite(EVIDENCE, Fake(worded))
    assert out.markdown == EVIDENCE.markdown and out.mode == "template (LLM reply rejected: number written as a word (ten))"
    assert verify(GOOD.replace("10 SKUs", "Four SKUs"), EVIDENCE) == "number written as a word (four)"
