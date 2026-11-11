#!/usr/bin/env python3
"""RQ8 over a **wide-area** link --- the paper's last remaining fidelity gap.

Layout (`docs/ec2_experiment_guide.md` §5a, tunnelled variant):

    host B (cloud):   python scripts/run_tool_server.py --bind 127.0.0.1 --port 8443
    host A (client):  ssh -N -L 18443:127.0.0.1:8443 user@B      # the WAN hop
                      python scripts/run_wan_capture.py --server-port 18443 \
                          --server-ca leakbench_server_cert.pem

Everything except the link is identical to the loopback run, so the two are directly
comparable: same tools, same six overlapping classes, same stochastic planner, same
capture proxy, same estimator, same seed.

**What is real and what is not.** Every tool call genuinely crosses the public
internet and back --- the measured application RTT is ~280 ms, against sub-millisecond
on loopback --- so propagation delay, jitter and congestion are in the timing channel
for the first time. TLS is end-to-end between this client and the remote server, so
the record framing the proxy captures is real. What is *not* raw is TCP segmentation:
because the security group permits only port 22, the TLS stream rides inside an SSH
tunnel, so the packet boundaries on the wire are SSH's rather than the TLS stream's
own. The proxy therefore sees real TLS records with real WAN timing, and a separate
on-wire capture (summarized by `scripts/analyze_wan_pcap.py`) is corroboration that the
path is a real internet path, not a second observable. This is stated in the output JSON
so the limitation travels with the number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from leakbench.data.live_capture import LIVE_CLASSES, LiveCaptureHarness
from leakbench.estimators.leakage import estimate_leakage
from leakbench.observables.maps import featurize_dataset
from leakbench.utils.logging import get_logger
from leakbench.utils.provenance import capture_provenance

logger = get_logger("leakbench.wan")


class _TimestampedHarness(LiveCaptureHarness):
    """Stamp each session with wall-clock start/end so a packet capture taken on the
    far end can be related to the run without trusting either host's clock."""

    def run_session(self, *args, **kwargs):
        t0 = time.time()
        trace = super().run_session(*args, **kwargs)
        trace.metadata["wall_start"] = t0
        trace.metadata["wall_end"] = time.time()
        return trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--server-host", default="127.0.0.1",
                    help="local end of the tunnel (or the server's address directly)")
    ap.add_argument("--server-port", type=int, required=True)
    ap.add_argument("--server-ca", required=True)
    ap.add_argument("--link", default="wan_ssh_tunnel",
                    help="recorded verbatim in the output as the link description")
    ap.add_argument("--output", default="outputs/wan/live_wan.json")
    args = ap.parse_args()

    harness = _TimestampedHarness(
        seed=args.seed,
        server_host=args.server_host,
        server_port=args.server_port,
        server_ca_file=args.server_ca,
    )

    t_start = time.time()
    ds = harness.capture_dataset(args.traces_per_class)
    elapsed = time.time() - t_start

    labels = sorted({t.label for t in ds.traces})
    prior = np.full(len(labels), 1.0 / len(labels))

    vantages = {}
    for v in ("network", "platform", "providers"):
        X, y = featurize_dataset(ds, v)
        leak = estimate_leakage(X, y, vantage=v, prior=prior, bootstrap=200,
                                seed=args.seed)
        vantages[v] = {"advantage": leak.advantage,
                       "advantage_ci": list(leak.advantage_ci),
                       "mi_bits": leak.mi_lower_bound_bits,
                       "best_estimator": leak.best_estimator}
        logger.info("%s: advantage=%.3f  MI=%.3f", v, leak.advantage,
                    leak.mi_lower_bound_bits)

    # The two published projections, on WAN-captured bytes.
    from leakbench.observables.maps import network_feature_slices
    sl = network_feature_slices()
    X, y = featurize_dataset(ds, "network")
    projections = {}
    for name, blocks in (("size_only", ("volume", "size_hist", "size_stats")),
                         ("timing_only", ("timing_hist", "timing_stats"))):
        cols = np.zeros(X.shape[1], dtype=bool)
        for b in blocks:
            cols[sl[b]] = True
        projections[name] = estimate_leakage(
            X[:, cols], y, prior=prior, bootstrap=100, seed=args.seed).advantage

    gaps = np.concatenate([t.gaps()[1:] for t in ds.traces if t.n > 1])
    result = {
        "rq": "RQ8_live_wan",
        "link": args.link,
        "link_note": "client and capture proxy on the client host; tool server in "
                     "AWS us-east-1; every call crosses the public internet",
        "capture_mode": ds.traces[0].metadata.get("capture_mode"),
        "classes": list(LIVE_CLASSES),
        "n_traces": len(ds.traces),
        "traces_per_class": args.traces_per_class,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "mean_trace_length": float(np.mean([t.n for t in ds.traces])),
        "inter_call_gap_s": {
            "median": float(np.median(gaps)),
            "p10": float(np.percentile(gaps, 10)),
            "p90": float(np.percentile(gaps, 90)),
        },
        "vantages": vantages,
        "projections": projections,
        "gaps_remaining": [
            "TCP segmentation on the wire is the SSH tunnel's, not the TLS stream's "
            "own (only port 22 is reachable); TLS records and timing are real."
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov = capture_provenance(
        seed=args.seed, taxonomy_hash=ds.taxonomy_hash, taxonomy_name=ds.taxonomy_name,
        config={"traces_per_class": args.traces_per_class, "link": args.link,
                "server_host": args.server_host, "server_port": args.server_port},
        note="client and capture proxy on this host; tool server remote").to_dict()
    prov["elapsed_s"] = elapsed
    out.write_text(json.dumps(
        {"status": "live_captured_traffic_over_wan", "result": result,
         "provenance": prov,
         "per_session_wall_clock": [
             {"label": int(t.label), "start": t.metadata["wall_start"],
              "end": t.metadata["wall_end"]} for t in ds.traces]},
        indent=2))
    logger.info("wrote %s (%d sessions in %.0f s)", out, len(ds.traces), elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
