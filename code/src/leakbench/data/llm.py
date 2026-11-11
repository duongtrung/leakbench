"""Minimal OpenAI-compatible LLM client --- the credential-gated path that turns the
*programmatic* controller into a **neural** one (RQ8) and lets a framework adapter
drive real tools with a real model.

This is the **only** module in ``src/`` that talks to the internet, and it does so
only when you have explicitly configured a provider and supplied a key. Nothing here
runs at import time, in the test suite, or in any default experiment: the generator,
the measured harness, and the loopback live capture remain fully offline.

Providers (both speak the OpenAI ``/chat/completions`` schema, so one client covers
them):

- ``openrouter`` -> ``https://openrouter.ai/api/v1``   key: ``OPENROUTER_API_KEY``
- ``groq``       -> ``https://api.groq.com/openai/v1``  key: ``GROQ_API_KEY``
- ``openai``     -> ``https://api.openai.com/v1``       key: ``OPENAI_API_KEY``
- ``custom``     -> whatever you pass as ``base_url``   key: ``LLM_API_KEY``

Keys are read from the environment, or from a ``.env`` file next to the repository
root (which ``.gitignore`` excludes). Key lookup is case-insensitive, so
``OpenRouter_API_KEY`` and ``OPENROUTER_API_KEY`` both work.

Only the standard library is used, so no dependency is added for a path most runs do
not take.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from leakbench._version import __version__
from leakbench.utils.logging import get_logger

logger = get_logger(__name__)

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    # AWS Bedrock exposes an OpenAI-compatible surface at /openai/v1 and accepts a
    # Bedrock API key as a bearer token, so it needs no SigV4 signing here. It is in
    # the sweep as a *serving-infrastructure control*, not as another model: running a
    # model through a second, independent provider separates "this is what the model
    # does" from "this is what one provider's queueing does" -- which matters for a
    # paper measuring a timing channel. Region is part of the host name; override
    # base_url to move it.
    "bedrock": {
        "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        "key_env": "BEDROCK_API_KEY",
        "default_model": "openai.gpt-oss-20b-1:0",
    },
    "custom": {
        "base_url": "",
        "key_env": "LLM_API_KEY",
        "default_model": "",
    },
}

_RETRY_STATUS = (408, 429, 500, 502, 503, 504)

# Sent on every request. See the comment at the header block: some providers sit
# behind a WAF that rejects urllib's default User-Agent.
USER_AGENT = f"LeakBench/{__version__} (+https://github.com/leakbench)"


class LLMError(RuntimeError):
    """The endpoint could not be reached, or answered with something unusable."""


# --------------------------------------------------------------------- keys
_EXPORT_WRAPPER = re.compile(
    r"^export\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?P<value>.*)$", re.DOTALL)


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Parse a ``.env`` file into a dict (no dependency, no side effects).

    Values are **not** logged or echoed anywhere. Missing file -> empty dict.
    """
    if path is None:
        # repository root == three levels up from src/leakbench/data/llm.py
        path = Path(__file__).resolve().parents[3] / ".env"
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        # People paste the shell line a console handed them, quotes and all:
        #     MY_KEY='export SOME_TOKEN=abc123'
        # Taking that literally sends "export SOME_TOKEN=abc123" as the credential and
        # the provider answers 401, which reads like a bad key rather than a bad paste.
        # Strip one leading `export NAME=` wrapper if the value carries it.
        m = _EXPORT_WRAPPER.match(value)
        if m:
            value = m.group("value").strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


def resolve_api_key(env_name: str, dotenv_path: str | Path | None = None) -> str | None:
    """Find a key by name in the environment or a ``.env``, case-insensitively."""
    target = env_name.lower()
    for key, value in os.environ.items():
        if key.lower() == target and value:
            return value
    for key, value in load_dotenv(dotenv_path).items():
        if key.lower() == target and value:
            return value
    return None


# ------------------------------------------------------------------- client
@dataclass
class LLMClient:
    """A tiny chat-completions client with retries and usage accounting.

    Parameters
    ----------
    provider:
        one of :data:`PROVIDERS`.
    model:
        model id; falls back to the provider's default.
    base_url, api_key:
        override the provider preset (``provider="custom"`` requires ``base_url``).
    temperature, max_tokens:
        sampling controls. A planner wants low temperature and short replies.
    timeout_s, max_retries:
        transport controls. 429/5xx are retried with exponential backoff, which is
        what a free-tier key needs most.
    """

    provider: str = "openrouter"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 128
    timeout_s: float = 60.0
    max_retries: int = 4
    dotenv_path: str | Path | None = None
    calls: int = field(default=0, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}; expected one of {sorted(PROVIDERS)}."
            )
        preset = PROVIDERS[self.provider]
        self.base_url = (self.base_url or preset["base_url"]).rstrip("/")
        self.model = self.model or preset["default_model"] or None
        if not self.base_url:
            raise ValueError("base_url is required for provider='custom'.")
        if not self.model:
            raise ValueError("model is required (no default for this provider).")
        if self.api_key is None:
            self.api_key = resolve_api_key(preset["key_env"], self.dotenv_path)
        if not self.api_key:
            raise LLMError(
                f"No API key found for provider {self.provider!r}. Set "
                f"{preset['key_env']} in the environment or in a .env file at the "
                f"repository root (it is gitignored). Keys are never written to any "
                f"result file."
            )

    # ---------------------------------------------------------------- calling
    def complete(self, prompt: str) -> str:
        """Send one prompt, return the assistant's text (the ``Planner`` contract)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Identify the client. Required in practice, not just politeness:
                # urllib's default "Python-urllib/x.y" is blocked outright by the
                # WAF in front of some providers (Groq answers HTTP 403 "error code
                # 1010" to it), which looks like a bad key until you look closely.
                "User-Agent": USER_AGENT,
                # OpenRouter asks callers to identify themselves; harmless elsewhere.
                "HTTP-Referer": "https://github.com/leakbench",
                "X-Title": "LeakBench",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode())
                # A gateway may deliver a provider-side failure as HTTP 200 with an
                # error object in the body -- OpenRouter answers a rate limit that way.
                # Retrying only on HTTP status therefore misses exactly the class of
                # error that most deserves a retry, and it surfaced as "unexpected
                # response shape" after 27 minutes of a capture had already run.
                err = data.get("error") if isinstance(data, dict) else None
                if err:
                    code = err.get("code") if isinstance(err, dict) else None
                    message = (err.get("message") if isinstance(err, dict) else str(err))
                    if code in _RETRY_STATUS and attempt < self.max_retries:
                        delay = min(30.0, 2.0**attempt)
                        logger.warning(
                            "%s returned error %s in a 200 body (%s); retrying in %.0fs "
                            "(attempt %d/%d)", self.provider, code, message, delay,
                            attempt + 1, self.max_retries)
                        time.sleep(delay)
                        continue
                    raise LLMError(
                        f"{self.provider} returned error {code}: {message}")
                return self._extract(data)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRY_STATUS or attempt == self.max_retries:
                    detail = exc.read().decode(errors="replace")[:400]
                    raise LLMError(
                        f"{self.provider} returned HTTP {exc.code}: {detail}"
                    ) from exc
                delay = min(30.0, 2.0**attempt)
                logger.warning("%s HTTP %d; retrying in %.0fs (attempt %d/%d)",
                               self.provider, exc.code, delay, attempt + 1,
                               self.max_retries)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise LLMError(f"could not reach {self.base_url}: {exc}") from exc
                time.sleep(min(30.0, 2.0**attempt))
        raise LLMError(f"exhausted retries against {self.base_url}: {last_error}")

    def _extract(self, data: dict[str, Any]) -> str:
        """Pull the assistant text out of an OpenAI-shaped response.

        Reasoning models need care. They spend their budget on a hidden chain of
        thought and return ``content: null`` with the text under ``reasoning`` (or
        ``reasoning_content``, depending on the provider); if the budget runs out
        mid-thought, ``finish_reason`` is ``length`` and there is no answer at all.
        Falling back to the reasoning field lets a planner still find the JSON action
        when the model wrote it there, and the explicit budget error tells the caller
        what to do instead of surfacing a confusing "no usable tool call" downstream.
        """
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from exc
        text = message.get("content")
        if not text:
            for alt in ("reasoning_content", "reasoning"):
                candidate = message.get(alt)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
        if not text:
            finish = (data["choices"][0] or {}).get("finish_reason")
            if finish == "length":
                raise LLMError(
                    f"{self.model} exhausted its {self.max_tokens}-token budget before "
                    "emitting an answer (reasoning models spend the budget thinking); "
                    "raise max_tokens"
                )
            raise LLMError(f"empty completion from {self.model}: {str(data)[:300]}")
        usage = data.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        return str(text)

    # ------------------------------------------------------------- reporting
    def usage(self) -> dict[str, Any]:
        """Call/token counts for the run record. Never includes the key."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def make_completion_fn(provider: str = "openrouter", model: str | None = None,
                       **kwargs: Any) -> tuple[Callable[[str], str], LLMClient]:
    """Build a ``complete(prompt) -> str`` callable plus the client behind it.

    The callable is exactly what :class:`leakbench.data.live_capture.LLMPlanner` and
    :class:`leakbench.data.adapters.ReActAdapter` expect, so wiring a real model into
    either is a one-liner. The client is returned as well so a runner can record its
    usage in the result file.
    """
    client = LLMClient(provider=provider, model=model, **kwargs)
    return client.complete, client
