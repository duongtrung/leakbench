"""Live-capture smoke test (RQ8). Opens real loopback sockets, so it is kept tiny.

Skips gracefully if loopback sockets are unavailable in the sandbox.
"""

import json
import socket

import numpy as np
import pytest

from leakbench.observables.maps import featurize_dataset


def _loopback_available() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _loopback_available(),
                                reason="loopback sockets unavailable")

VALID_MODES = {"tls_captured_proxy", "tcp_captured_proxy"}


@pytest.fixture(scope="module")
def live_dataset():
    from leakbench.data.live_capture import LIVE_CLASSES, LiveCaptureHarness

    return LiveCaptureHarness(seed=3).capture_dataset(3, classes=LIVE_CLASSES)


def test_live_capture_tiny(live_dataset):
    from leakbench.data.live_capture import LIVE_CLASSES

    ds = live_dataset
    assert len(ds) == len(LIVE_CLASSES) * 3
    assert ds.source == "live"
    # Captured (not modeled) wire sizes, taken off the capture proxy.
    assert ds.metadata["capture_mode"] in VALID_MODES
    assert ds.metadata["capture"] == "captured_at_tcp_proxy"
    assert all(t.metadata["capture_mode"] in VALID_MODES for t in ds.traces)
    # The observable pipeline runs on captured traces unchanged.
    X, y = featurize_dataset(ds, "network")
    assert X.shape[0] == len(ds)
    assert np.all(np.isfinite(X))


def test_capture_mode_matches_what_actually_ran(live_dataset):
    """The recorded mode must never claim TLS that did not happen."""
    ds = live_dataset
    if ds.metadata["capture_mode"] == "tls_captured_proxy":
        assert ds.metadata["tls"] == "real"
        assert ds.metadata["tls_generator"] in ("openssl", "cryptography")
        assert ds.metadata["mean_handshake_bytes"] > 0
        assert "TLS framing modeled over captured TCP" not in ds.metadata["gaps"]
    else:
        assert ds.metadata["tls"] == "modeled_over_captured_tcp"
        assert "TLS framing modeled over captured TCP" in ds.metadata["gaps"]
    # The loopback gap is never silently dropped.
    assert "loopback not WAN" in ds.metadata["gaps"]
    assert ds.metadata["link"] == "loopback"


def test_planner_branches_on_real_tool_output(live_dataset):
    """Plan length must vary: the controller reads genuine tool results."""
    lengths = {t.n for t in live_dataset.traces}
    assert len(lengths) > 1
    assert live_dataset.metadata["planner_kind"] == "programmatic_stochastic"


def test_inter_arrival_gaps_are_real_and_positive(live_dataset):
    gaps = np.concatenate([t.gaps()[1:] for t in live_dataset.traces if t.n > 1])
    assert gaps.size > 0
    # Inter-arrival gaps include the previous call's real round trip, so they are
    # strictly positive (an end-to-start definition would give ~0 by construction).
    assert float(np.median(gaps)) > 0.0


def test_link_kind_is_resolved_not_assumed():
    """Passing a server host must not make the run *claim* a WAN link."""
    from leakbench.data.live_capture import _link_kind

    assert _link_kind(None) == "loopback"
    assert _link_kind("127.0.0.1") == "loopback"
    assert _link_kind("localhost") == "loopback"
    assert _link_kind("10.0.0.5") == "remote"


def test_two_host_code_path_runs_against_a_separate_server():
    """The WAN runbook drives a server the harness did not start.

    Exercised here against a server on this host (so no second machine is needed);
    what it verifies is the code path and the recorded metadata, not a WAN link ---
    and the recorded link correctly stays ``loopback``.
    """
    from leakbench.data.live_capture import (
        LIVE_CLASSES,
        LiveCaptureHarness,
        _ToolServer,
    )
    from leakbench.data.tlsutil import generate_self_signed

    material = generate_self_signed()
    if material is None:
        pytest.skip("no TLS material available on this host")
    server = _ToolServer(seed=5, tls=material, bind_host="127.0.0.1")
    server.start()
    try:
        harness = LiveCaptureHarness(
            seed=5, server_host="127.0.0.1", server_port=server.port,
            server_ca_file=str(material.cert_path),
        )
        ds = harness.capture_dataset(2, classes=LIVE_CLASSES[:3])
    finally:
        server.close()
        material.cleanup()

    assert len(ds) == 6
    assert ds.metadata["capture_mode"] == "tls_captured_proxy"
    assert ds.metadata["server"] == f"127.0.0.1:{server.port}"
    assert ds.metadata["link"] == "loopback"


def test_remote_tls_without_a_ca_file_is_refused():
    from leakbench.data.live_capture import LiveCaptureHarness

    harness = LiveCaptureHarness(server_host="10.0.0.5", server_port=8443)
    with pytest.raises(ValueError, match="server_ca_file"):
        harness.capture_dataset(1)


def test_remote_without_a_port_is_refused():
    from leakbench.data.live_capture import LiveCaptureHarness

    harness = LiveCaptureHarness(server_host="10.0.0.5")
    with pytest.raises(ValueError, match="server_port"):
        harness.capture_dataset(1)


def test_llm_planner_refuses_to_fake_a_neural_controller():
    from leakbench.data.live_capture import LLMPlanner

    with pytest.raises(NotImplementedError, match="LLM endpoint"):
        next(LLMPlanner().program("literature_review"))


def test_llm_planner_drives_the_step_interface():
    """With a completion callable supplied, the neural path yields real steps."""
    from leakbench.data.live_capture import LLMPlanner

    replies = iter(['{"tool": "search", "scale": 1.0}',
                    '```json\n{"tool": "fetch", "scale": 0.5}\n```',
                    '{"done": true}'])
    planner = LLMPlanner(complete=lambda prompt: next(replies))
    program = planner.program("literature_review")
    from leakbench.data.live_capture import StepObservation

    steps = []
    obs = None
    while True:
        try:
            step = program.send(obs) if obs is not None else next(program)
        except StopIteration:
            break
        steps.append(step)
        obs = StepObservation(tool=step[0], scale=step[1], signal={"n_hits": 3.0},
                              response_bytes=10)
    assert steps == [("search", 1.0), ("fetch", 0.5)]


def test_llm_planner_prompt_carries_real_signals():
    from leakbench.data.live_capture import LLMPlanner, StepObservation

    prompt = LLMPlanner._prompt(
        "market_scan", [StepObservation("search", 1.0, {"n_hits": 7.0}, 128)]
    )
    assert "market scan" in prompt
    assert json.dumps({"n_hits": 7.0}) in prompt
    assert "Available tools" in prompt


def test_llm_planner_tolerates_prose_and_fences():
    """Models wrap JSON in prose; a strict json.loads made every session fail."""
    from leakbench.data.live_capture import LLMPlanner

    for reply in ('{"tool": "search"}',
                  '```json\n{"tool": "search", "scale": 1.0}\n```',
                  'Sure, next I would call {"tool": "search"} to look things up.'):
        assert LLMPlanner._parse(reply) == ("search", 1.0)


def test_llm_planner_reports_a_model_that_never_emits_a_call():
    """The failure must name the model's actual reply, not surface as 'no steps'."""
    from leakbench.data.live_capture import LLMPlanner

    planner = LLMPlanner(complete=lambda p: "I am not going to answer in JSON.")
    with pytest.raises(ValueError, match="no usable tool call"):
        next(planner.program("trip_planning"))


def test_capture_proxy_survives_a_slow_controller():
    """A quiet period longer than the proxy's connect timeout must not kill the session.

    Regression guard: the capture proxy used to keep its 5-second *connect* timeout on
    the relay socket, so it became a read timeout. A programmatic planner pauses for
    milliseconds and never hit it; a neural controller pauses for seconds and every
    RQ8 run died with a bare `ConnectionResetError`/`BrokenPipeError`. Takes ~6 s by
    design -- the pause is the thing being tested.
    """
    import time as _time

    from leakbench.data.live_capture import LIVE_CLASSES, LiveCaptureHarness

    class _SlowPlanner:
        def program(self, task_name):
            yield ("search", 1.0)
            _time.sleep(6.0)          # longer than the old 5 s relay timeout
            yield ("fetch", 1.0)

    ds = LiveCaptureHarness(seed=9, planner=_SlowPlanner()).capture_dataset(
        1, classes=LIVE_CLASSES[:1]
    )
    assert len(ds) == 1
    trace = ds.traces[0]
    assert [c.category for c in trace.calls] == ["search", "fetch"]
    # The long pause is recorded as a real inter-arrival gap, not lost.
    assert trace.gaps()[1] > 5.0


def test_capture_redraws_a_session_the_planner_declines():
    """One declined draw must not cost the whole capture.

    A stochastic planner can answer `{"done": true}` before calling anything. That
    session has no observable, so it is redrawn -- and counted, because a high count
    means the collected traces are a filtered sample.
    """
    from leakbench.data.live_capture import LiveCaptureHarness

    calls = {"n": 0}

    def flaky_complete(_prompt: str) -> str:
        calls["n"] += 1
        # Decline the very first draw of the run, then behave.
        if calls["n"] == 1:
            return '{"done": true}'
        return ('{"done": true}' if calls["n"] % 4 == 0
                else '{"tool": "search", "scale": 1.0}')

    from leakbench.data.live_capture import LLMPlanner

    harness = LiveCaptureHarness(seed=5,
                                 planner=LLMPlanner(complete=flaky_complete, max_steps=3))
    ds = harness.capture_dataset(1, classes=("literature_review", "trip_planning"))
    assert len(ds) == 2                       # both classes collected despite the decline
    assert ds.metadata["empty_sessions_redrawn"] >= 1
    assert all(t.n >= 1 for t in ds.traces)


def test_first_turn_prompt_does_not_offer_the_stop_option():
    """Offering `done` before any tool has run invites an empty trace."""
    from leakbench.data.live_capture import LLMPlanner, StepObservation

    first = LLMPlanner._prompt("literature_review", [])
    assert '"done"' not in first
    assert "at least one tool call" in first

    later = LLMPlanner._prompt(
        "literature_review", [StepObservation("search", 1.0, {"n_hits": 3.0}, 64)])
    assert '"done"' in later          # stopping is a real decision once tools have run
