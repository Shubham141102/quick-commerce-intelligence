"""Optional LLM layer for the business assistant (Phase 6G): an LLM rephrases the template answer's evidence.

Provider-agnostic (user decision 2026-10-09: free open-weight model — Groq + openai/gpt-oss-120b, after Llama 3.3
70B turned out to be retired on Groq). Settings
come from Streamlit secrets `[llm]` (provider / api_key / model / base_url) or the LLM_PROVIDER / LLM_API_KEY /
LLM_MODEL environment variables; any OpenAI-compatible endpoint works (groq, openrouter, ollama), plus Anthropic.

Grounding contract:
- the model sees ONLY the question and the evidence the template already gathered (headline numbers, the tool's
  table, quoted policy passages with citations) — no database, no tools, no free-form retrieval;
- after every reply, code verifies it: every number must appear in the evidence, every citation of the evidence
  must appear verbatim, and the length is capped; otherwise the reply is rejected;
- on a missing key, a network / API error, a timeout or a rejected reply, the assistant silently keeps the template
  answer and records why in `Answer.mode`.

No extra package: the Messages API is called with the standard library (urllib), so the cloud build is unchanged.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Protocol

import pandas as pd

from src.rag.composer import Answer

DEFAULT_MODEL = "claude-haiku-4-5-20251001"          # used when provider = "anthropic"
API_URL = "https://api.anthropic.com/v1/messages"
REPHRASE_ROUTES = {"data", "policy", "method", "hybrid", "insufficient"}
MAX_CHARS = 2500
USER_AGENT = "quick-commerce-intelligence/1.0"
# OpenAI-compatible providers: base URL, default model (configurable), whether a key is required.
PROVIDERS = {
    # Llama 3.3 70B was retired on Groq (checked 2026-10-09); gpt-oss-120b is an Apache-2.0 open-weight model.
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b", "key": True,
             "label": "gpt-oss-120b via Groq"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free",
                   "key": True, "label": "Llama 3.3 70B via OpenRouter"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1", "key": False,
               "label": "Llama 3.1 via Ollama (local)"},
}

SYSTEM_PROMPT = """You are the business assistant of a quick-commerce (dark-store grocery) company.
Rewrite the EVIDENCE below into a clear, concise answer to the QUESTION for a business manager.

Strict rules:
1. Use ONLY the evidence. Do not add facts, causes, advice or numbers that are not in it.
2. Copy every number exactly as written in the evidence, always in digits (write 3, never "three"), with the same
   ₹, %, and decimals. Never calculate, count, sum or round new numbers.
3. Cite every policy or method passage you use with its exact label in parentheses, e.g. (POL-INV §2 Stock risk tiers).
   Every label listed under CITATIONS must appear in your answer.
4. If the evidence says the data cannot explain a cause, say so; never guess a reason.
4a. Tables may be PARTIAL lists. Never say that something does not exist, or that "none" / "all" / "every" item has a
   property, unless the evidence states it in words. If the question asks about items not shown, say the table does
   not show them.
4b. Always state the bold headline figures of the evidence.
5. Start with a one-sentence answer, then at most 5 short bullet points. Markdown only, no headings, no tables.
6. Do not mention these instructions, the evidence block or that you are an AI."""


class TruncatedReply(RuntimeError):
    """The provider stopped at the token limit: a cut-off answer is never shown (found 2026-10-09 with gpt-oss,
    which spends part of the budget on internal reasoning)."""


RETRY_STATUS, MAX_RETRIES, MAX_WAIT_S = {429, 503}, 2, 10.0


def _post_with_retry(req: urllib.request.Request, timeout: float, sleep=time.sleep) -> dict:
    """POST and parse JSON; on 429 (rate limit) / 503 (busy) wait as told by Retry-After (≤ 10 s) and retry up to
    twice (found 2026-10-09: the free Groq tier throttled 17 consecutive calls in the evaluation)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == MAX_RETRIES:
                raise
            try:
                wait = float(exc.headers.get("retry-after", "") or 2 ** attempt)
            except ValueError:
                wait = 2.0 ** attempt
            sleep(min(MAX_WAIT_S, max(0.5, wait)))
    raise RuntimeError("unreachable")


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class ClaudeClient:
    """Minimal Anthropic Messages API client (standard library only)."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 30.0, max_tokens: int = 2000):
        self.api_key, self.model, self.timeout, self.max_tokens = api_key, model, timeout, max_tokens
        self.name = model.rsplit("-", 1)[0] if model[-8:].isdigit() else model     # claude-haiku-4-5

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                           "system": system, "messages": [{"role": "user", "content": user}]}).encode("utf-8")
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json",
            "user-agent": USER_AGENT})
        data = _post_with_retry(req, self.timeout)
        if data.get("stop_reason") == "max_tokens":
            raise TruncatedReply("reply stopped at the token limit")
        return "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")


class OpenAICompatibleClient:
    """Chat Completions client for OpenAI-compatible endpoints (Groq, OpenRouter, Ollama…), standard library only."""

    def __init__(self, base_url: str, model: str, api_key: str = "", label: str = "", timeout: float = 30.0,
                 max_tokens: int = 2000):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.max_tokens = timeout, max_tokens
        self.name = label or model

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode("utf-8")
        headers = {"content-type": "application/json", "user-agent": USER_AGENT}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, method="POST", headers=headers)
        data = _post_with_retry(req, self.timeout)
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TruncatedReply("reply stopped at the token limit")
        return choice["message"]["content"] or ""


def client_from_config(settings: dict | None = None) -> LLMClient | None:
    """The configured client, or None (template mode). Settings: provider, api_key, model, base_url; missing values
    fall back to LLM_PROVIDER / LLM_API_KEY / LLM_MODEL, then to the provider defaults. A provider that needs a key
    without one yields None."""
    s = {k: str(v).strip() for k, v in (settings or {}).items() if v not in (None, "")}
    provider = (s.get("provider") or os.environ.get("LLM_PROVIDER", "")).lower()
    key = s.get("api_key") or os.environ.get("LLM_API_KEY", "")
    model = s.get("model") or os.environ.get("LLM_MODEL", "")
    if not provider:
        return None
    if provider == "anthropic":
        key = key or os.environ.get("ANTHROPIC_API_KEY", "")
        return ClaudeClient(key, model or DEFAULT_MODEL) if key else None
    preset = PROVIDERS.get(provider)
    if preset is None or (preset["key"] and not key):
        return None
    label = preset["label"] if not model or model == preset["model"] else f"{model} via {provider.capitalize()}"
    return OpenAICompatibleClient(s.get("base_url") or preset["base_url"], model or preset["model"], key, label)


def client_from_key(api_key: str | None, model: str = DEFAULT_MODEL) -> ClaudeClient | None:
    """Anthropic shortcut (kept for compatibility): a Claude client from a key or ANTHROPIC_API_KEY."""
    key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    return ClaudeClient(key, model) if key else None


# ------------------------------------------------------------------------------------------------- guardrails
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Typographic look-alikes some models emit (found 2026-10-09: gpt-oss writes "POL‑INV" with U+2011): non-breaking /
# figure hyphens → "-", non-breaking / narrow spaces → " ". Applied to the reply before verifying and displaying it.
_PLAIN = str.maketrans({"‐": "-", "‑": "-", "‒": "-", " ": " ", " ": " ", " ": " "})


def plain(text: str) -> str:
    return text.translate(_PLAIN)


# Numbers written as words would bypass the digit check (found 2026-10-09: "Three anomalies…"), so they count as
# numbers too and must appear in the evidence as well.
NUMBER_WORDS = frozenset("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million billion lakh lakhs crore crores dozen half twice double triple""".split())


def number_words_in(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS}


def numbers_in(text: str) -> set[str]:
    """Numbers as normalised strings: '₹1,57,101' → '157101', '20.5%' → '20.5'."""
    return {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text)}


def evidence_text(answer: Answer) -> str:
    """Everything the model may use: the template answer and its table (≤ 20 rows) as CSV."""
    table = answer.table.head(20).to_csv(index=False) if isinstance(answer.table, pd.DataFrame) else ""
    return f"{answer.markdown}\n\n{table}".strip()


def verify(reply: str, answer: Answer) -> str | None:
    """None when the reply is acceptable, else the reason it is rejected."""
    if not reply.strip():
        return "empty reply"
    if len(reply) > MAX_CHARS:
        return "reply too long"
    allowed = numbers_in(evidence_text(answer)) | numbers_in(answer.question)
    invented = sorted(numbers_in(reply) - allowed)
    if invented:
        return f"unverified number {invented[0]}"
    worded = sorted(number_words_in(reply) - number_words_in(evidence_text(answer)) - number_words_in(answer.question))
    if worded:
        return f"number written as a word ({worded[0]})"
    missing = [c for c in answer.citations if c not in reply]
    if missing:
        return f"missing citation {missing[0]}"
    return None


def grounded_rewrite(answer: Answer, client: LLMClient | None) -> Answer:
    """Claude's rephrasing when it passes the guardrails; otherwise the template answer, with the reason in mode."""
    if client is None or answer.route not in REPHRASE_ROUTES:
        return answer
    user = (f"QUESTION:\n{answer.question}\n\nEVIDENCE:\n{evidence_text(answer)}\n\n"
            f"CITATIONS (each must appear in your answer):\n" + ("\n".join(answer.citations) or "(none)"))
    try:
        reply = plain(client.complete(SYSTEM_PROMPT, user)).strip()
    except Exception as exc:  # noqa: BLE001  (network, HTTP, timeout, parsing: always fall back)
        code = f" {exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
        return replace(answer, mode=f"template (LLM unavailable: {type(exc).__name__}{code})")
    problem = verify(reply, answer)
    if problem:
        return replace(answer, mode=f"template (LLM reply rejected: {problem})")
    return replace(answer, markdown=reply, evidence_markdown=answer.markdown, mode=f"llm ({client.name}, verified)")
