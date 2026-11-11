"""LLM client + LLM-driven adapter tests.

**No test here contacts the network.** The client's transport is stubbed and the
adapters are driven by a local fake `complete()`, so the whole file runs offline —
which is also the guarantee the repository makes about `src/`: the only module that
can leave the machine is `data/llm.py`, and only when a key is configured.
"""

import io
import json
import urllib.error

import numpy as np
import pytest

from leakbench.data.adapters import (
    LangChainAdapter,
    MCPRouterAdapter,
    ReActAdapter,
    _parse_step,
)
from leakbench.data.llm import (
    PROVIDERS,
    LLMClient,
    LLMError,
    load_dotenv,
    make_completion_fn,
    resolve_api_key,
)
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import TOOL_CATEGORIES, reference_taxonomy


# ------------------------------------------------------------------ key handling
def test_dotenv_is_parsed_without_dependencies(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# comment\nFOO_API_KEY="abc123"\nBAR=  plain \n\nBROKEN\n')
    env = load_dotenv(p)
    assert env["FOO_API_KEY"] == "abc123"
    assert env["BAR"] == "plain"
    assert "BROKEN" not in env


def test_key_lookup_is_case_insensitive(tmp_path, monkeypatch):
    """A `.env` written as OpenRouter_API_KEY must still be found."""
    p = tmp_path / ".env"
    p.write_text("OpenRouter_API_KEY=sk-test\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_api_key("OPENROUTER_API_KEY", p) == "sk-test"


def test_environment_wins_over_dotenv(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert resolve_api_key("GROQ_API_KEY", p) == "from-env"


def test_missing_key_raises_with_actionable_message(tmp_path, monkeypatch):
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider["key_env"], raising=False)
    with pytest.raises(LLMError, match="No API key found"):
        LLMClient(provider="groq", dotenv_path=tmp_path / "absent.env")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="not-a-provider", api_key="x")


def test_custom_provider_requires_a_base_url():
    with pytest.raises(ValueError, match="base_url"):
        LLMClient(provider="custom", api_key="x", model="m")


# ---------------------------------------------------------------- transport
class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_payload(content="{}"):
    return json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }).encode()


def test_client_parses_and_accounts_usage(monkeypatch):
    secret = "sk-do-not-leak-this-value"
    client = LLMClient(provider="groq", api_key=secret, model="m")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse(_ok_payload("hello")))
    assert client.complete("hi") == "hello"
    usage = client.usage()
    assert usage["calls"] == 1 and usage["prompt_tokens"] == 11
    # The usage block goes into result files, so it must never carry the key.
    assert "api_key" not in usage
    assert secret not in json.dumps(usage)


def test_client_retries_on_rate_limit(monkeypatch):
    client = LLMClient(provider="groq", api_key="k", model="m", max_retries=2)
    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b""))
        return _FakeResponse(_ok_payload("ok"))

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert client.complete("hi") == "ok"
    assert calls["n"] == 2


def test_client_does_not_retry_a_bad_key(monkeypatch):
    client = LLMClient(provider="groq", api_key="k", model="m", max_retries=3)
    calls = {"n": 0}

    def unauthorized(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", unauthorized)
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete("hi")
    assert calls["n"] == 1


def test_client_rejects_an_unexpected_response_shape(monkeypatch):
    client = LLMClient(provider="groq", api_key="k", model="m")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse(b'{"nope": 1}'))
    with pytest.raises(LLMError, match="unexpected response shape"):
        client.complete("hi")


def test_make_completion_fn_returns_a_planner_compatible_callable(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _FakeResponse(_ok_payload("x")))
    fn, client = make_completion_fn(provider="openrouter", model="m", api_key="k")
    assert fn("prompt") == "x"
    assert client.usage()["calls"] == 1


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize("reply,expected", [
    ('{"tool": "search", "scale": 1.2}', ("search", 1.2)),
    ('```json\n{"tool": "calc"}\n```', ("calc", 1.0)),
    ('Sure! {"tool": "db_query", "scale": 0.5} hope that helps', ("db_query", 0.5)),
    ('{"tool": "search", "scale": 99}', ("search", 1.5)),      # clamped
    ('{"done": true}', None),
    ('{"tool": "telepathy"}', None),                            # not a real tool
    ('no json here', None),
])
def test_step_parsing_is_tolerant_but_strict(reply, expected):
    assert _parse_step(reply) == expected


# ------------------------------------------------------------------ adapters
def _fake_complete(sequence):
    it = iter(sequence)

    def complete(prompt: str) -> str:
        assert "Available tools" in prompt
        try:
            return next(it)
        except StopIteration:
            return '{"done": true}'

    return complete


def test_react_adapter_without_an_endpoint_refuses():
    with pytest.raises(NotImplementedError, match="complete"):
        list(ReActAdapter().run_task(0))


def test_react_adapter_drives_real_tools():
    replies = ['{"tool": "search", "scale": 1.0}',
               '{"tool": "code_exec", "scale": 1.2}',
               '{"tool": "db_query", "scale": 0.8}',
               '{"done": true}']
    adapter = ReActAdapter(complete=_fake_complete(replies), max_steps=6)
    events = list(adapter.run_task(0))
    assert [e.category for e in events] == ["search", "code_exec", "db_query"]
    # Sizes come from executing the genuine backends, so they are non-trivial.
    assert all(e.request_size > 0 and e.response_size > 0 for e in events)
    assert events[0].gap == 0.0 and events[1].gap > 0.0


def test_react_adapter_feeds_real_signals_back_into_the_prompt():
    seen: list[str] = []

    def complete(prompt: str) -> str:
        seen.append(prompt)
        return '{"tool": "search"}' if len(seen) < 3 else '{"done": true}'

    list(ReActAdapter(complete=complete, max_steps=4).run_task(0))
    assert "No tools called yet" in seen[0]
    # The second prompt must carry the first call's genuine output signals.
    assert "n_hits" in seen[1] and "search ->" in seen[1]


def test_react_adapter_produces_a_usable_dataset():
    replies = ['{"tool": "search"}', '{"tool": "file_io"}', '{"done": true}']
    tax = reference_taxonomy()
    adapter = ReActAdapter(complete=lambda p: _fake_complete(replies)(p), max_steps=3)
    # Cycle a fresh reply stream per call so every task gets the same short plan.
    adapter.complete = lambda p, _r=replies: (
        '{"tool": "search"}' if "No tools called yet" in p else '{"done": true}'
    )
    ds = adapter.sample_dataset(tax, 1)
    assert len(ds) == tax.num_classes
    assert ds.source == "adapter:react"
    X, y = featurize_dataset(ds, "platform")
    assert X.shape[0] == tax.num_classes


def test_react_adapter_reports_a_model_that_never_calls_a_tool():
    adapter = ReActAdapter(complete=lambda p: "I refuse", max_steps=3)
    with pytest.raises(ValueError, match="no usable tool call"):
        list(adapter.run_task(0))


def test_mcp_router_adapter_without_an_endpoint_refuses():
    with pytest.raises(NotImplementedError, match="complete"):
        MCPRouterAdapter().sample_dataset(reference_taxonomy(), 1)


def test_langchain_adapter_is_still_an_honest_stub():
    with pytest.raises(NotImplementedError, match="LangChain"):
        list(LangChainAdapter().run_task(0))


def test_tool_categories_are_offered_to_the_model():
    adapter = ReActAdapter(complete=lambda p: '{"done": true}')
    prompt = adapter._prompt("literature_review", [])
    for tool in TOOL_CATEGORIES:
        assert tool in prompt
    assert "literature review" in prompt


def test_adapter_gaps_are_real_inter_arrival_times():
    replies = ['{"tool": "file_io"}'] * 4
    adapter = ReActAdapter(complete=_fake_complete(replies), max_steps=4)
    events = list(adapter.run_task(0))
    gaps = np.array([e.gap for e in events[1:]])
    assert gaps.size >= 2
    assert np.all(gaps > 0)


# ------------------------------------------------- reasoning models (content: null)
def _resp(message, finish_reason="stop"):
    return {"choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_extract_falls_back_to_the_reasoning_field():
    """Reasoning models return content=null and put the text under `reasoning`.

    Regression guard: without the fallback the planner sees 'None' and reports
    "the model produced no usable tool call", which reads like a prompt problem
    rather than a response-shape one.
    """
    from leakbench.data.llm import LLMClient

    c = LLMClient(provider="openrouter", api_key="test-key-not-real")
    assert c._extract(_resp({"content": '{"tool": "search"}'})) == '{"tool": "search"}'
    assert c._extract(_resp({"content": None, "reasoning": '{"tool": "code"}'})) \
        == '{"tool": "code"}'
    assert c._extract(_resp({"content": "", "reasoning_content": "x"})) == "x"


def test_extract_names_the_token_budget_when_thinking_runs_out():
    from leakbench.data.llm import LLMClient, LLMError

    c = LLMClient(provider="openrouter", api_key="test-key-not-real", max_tokens=200)
    with pytest.raises(LLMError, match="token budget"):
        c._extract(_resp({"content": None, "reasoning": ""}, finish_reason="length"))
    with pytest.raises(LLMError, match="empty completion"):
        c._extract(_resp({"content": None}, finish_reason="stop"))


def test_planner_retries_an_unparseable_reply_before_giving_up():
    """A prose reply must not kill the session; a real ReAct loop asks again.

    Regression guard: reasoning models routinely burn a whole budget thinking and
    answer on the retry. Without this, one bad completion ends a 150-session capture.
    """
    from leakbench.data.live_capture import LLMPlanner

    replies = iter(["I need to think about which tool to use...",
                    '{"tool": "search", "scale": 1.0}'])
    planner = LLMPlanner(complete=lambda _p: next(replies), max_steps=2)
    step = next(planner.program("literature_review"))
    assert step[0] == "search"          # the planner yields a (tool, scale) pair
    assert planner.parse_failures == 1


def test_planner_gives_up_after_the_retry_budget_and_says_why():
    from leakbench.data.live_capture import LLMPlanner

    planner = LLMPlanner(complete=lambda _p: "thinking, thinking", max_steps=2,
                         parse_retries=1)
    with pytest.raises(ValueError, match="2 attempts"):
        next(planner.program("literature_review"))
    assert planner.parse_failures == 2


def test_planner_does_not_retry_an_explicit_done_signal():
    """A model saying it finished is a success, not a parse failure.

    Regression guard: retrying it would burn two extra completions at the end of
    every session and record a failure where none occurred.
    """
    from leakbench.data.live_capture import LLMPlanner, StepObservation

    calls = []

    def complete(prompt):
        calls.append(prompt)
        return ('{"tool": "search", "scale": 1.0}' if len(calls) == 1
                else '{"done": true}')

    planner = LLMPlanner(complete=complete, max_steps=4, parse_retries=2)
    program = planner.program("literature_review")
    step = next(program)
    try:
        program.send(StepObservation(tool=step[0], scale=step[1],
                                     signal={"n_hits": 3.0}, response_bytes=10))
    except StopIteration:
        pass
    assert len(calls) == 2          # not 4: the done signal was not retried
    assert planner.parse_failures == 0


def test_dotenv_strips_a_pasted_export_wrapper(tmp_path):
    """`MY_KEY='export SOME_TOKEN=abc'` must yield `abc`, not the whole shell line.

    Consoles hand out credentials as an export statement; pasting it verbatim is the
    obvious thing to do. Taking it literally sends the wrapper as the credential and
    the provider answers 401, which looks like a bad key rather than a bad paste.
    """
    from leakbench.data.llm import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "PLAIN=abc123\n"
        "WRAPPED='export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-xyz'\n"
        'DQUOTED="export SOME_TOKEN=tok-1"\n'
        "# comment\n"
        "EMPTY=\n"
    )
    got = load_dotenv(env)
    assert got["PLAIN"] == "abc123"
    assert got["WRAPPED"] == "bedrock-api-key-xyz"
    assert got["DQUOTED"] == "tok-1"
    assert got["EMPTY"] == ""


def test_done_detection_ignores_the_word_in_a_chain_of_thought():
    """Deliberating about stopping is not stopping.

    The planner prompt documents `{"done": true}` as the stop format, so a reasoning
    model reasons about it out loud. A substring test on the word reads that
    deliberation as a decision and ends the session with zero tool calls, which is
    exactly how gpt-oss-20b failed before this became a parse.
    """
    from leakbench.data.live_capture import LLMPlanner

    thinking = ('<reasoning>The user wants JSON. I could reply {"done": true} if the '
                'task were finished, but no tool has been called yet, so I should '
                'search first.</reasoning>\n{"tool": "search", "scale": 1.0}')
    assert LLMPlanner._parse(thinking) == ("search", 1.0)
    assert LLMPlanner._is_done(thinking) is False

    assert LLMPlanner._is_done('{"done": true}') is True
    assert LLMPlanner._is_done('```json\n{"done": true}\n```') is True
    assert LLMPlanner._is_done("I think we are done here.") is False
    assert LLMPlanner._is_done('{"done": false}') is False


def test_parse_step_prefers_the_answer_over_json_quoted_while_reasoning():
    """A narrated JSON example must not swallow the real answer.

    Taking text[first_brace:last_brace] spans from the brace inside the narration to
    the brace closing the answer and parses as nothing. This is the root cause of the
    "planner produced no steps" failures on every reasoning model we tried.
    """
    from leakbench.data.adapters import _parse_step, json_objects

    thinking = ('<reasoning>I could reply {"done": true} if finished, but no tool has '
                'run yet.</reasoning>\n{"tool": "search", "scale": 1.0}')
    assert len(json_objects(thinking)) == 2
    assert _parse_step(thinking) == ("search", 1.0)


def test_json_objects_is_not_confused_by_braces_inside_strings():
    from leakbench.data.adapters import json_objects

    spans = json_objects('{"note": "a } brace and a { one", "tool": "search"}')
    assert len(spans) == 1
    assert json.loads(spans[0])["tool"] == "search"


def test_json_objects_handles_nesting_and_escapes():
    from leakbench.data.adapters import json_objects

    assert json_objects('{"a": {"b": 1}} tail {"c": 2}') == ['{"a": {"b": 1}}', '{"c": 2}']
    assert json_objects(r'{"q": "he said \"hi\" }"}') == [r'{"q": "he said \"hi\" }"}']
    assert json_objects("no json here") == []


def test_error_in_a_200_body_is_retried_not_mistaken_for_a_bad_shape(monkeypatch):
    """A gateway can deliver a rate limit as HTTP 200 with an error object.

    Retrying only on HTTP status misses exactly the class of failure that most
    deserves a retry. Before this, a 429 in the body surfaced as "unexpected response
    shape" and cost 27 minutes of an in-flight capture.
    """
    import io
    import urllib.request

    from leakbench.data.llm import LLMClient

    bodies = [
        json.dumps({"error": {"code": 429, "message": "Provider returned error"}}),
        json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
    ]

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(_req, timeout=None):
        return _Resp(bodies.pop(0).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("leakbench.data.llm.time.sleep", lambda _s: None)
    c = LLMClient(provider="openrouter", api_key="test-key-not-real")
    assert c.complete("hi") == "ok"
    assert not bodies                      # both responses were consumed: it retried


def test_non_retryable_error_body_names_the_provider_code(monkeypatch):
    import io
    import urllib.request

    from leakbench.data.llm import LLMClient, LLMError

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda _r, timeout=None: _Resp(
        json.dumps({"error": {"code": 402, "message": "insufficient credits"}}).encode()))
    c = LLMClient(provider="openrouter", api_key="test-key-not-real")
    with pytest.raises(LLMError, match="402.*insufficient credits"):
        c.complete("hi")
