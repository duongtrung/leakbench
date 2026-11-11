"""Wire models --- turn *true* byte sizes into *on-the-wire* sizes an on-path
observer actually sees (paper §Impl "Measured tool-execution harness", §Eval RQ6).

Two models, applied in sequence for the network vantage on measured traces:

1. **TLS-1.3/GCM record framing** (paper: "16 KB records, per-record overhead"): a
   payload is split into ceil(size / 16384) records, each carrying a fixed per-record
   overhead (header + AEAD tag). This is what any TLS endpoint does to the byte
   stream before it hits the wire.

2. **HTTP/2 wire model** (network vantage only): a real on-path observer sees
   multiplexed bursts, not clean per-call boundaries. Adjacent tool messages coalesce
   into shared records with probability ``p``; cross-traffic adds multiplicative size
   noise ``sigma``; sizes are MTU-quantized.

The RQ6 robustness sweep varies ``p ∈ {0.2,0.45,0.7}`` and ``sigma ∈ {0.05,0.15,0.30}``
(nine configs); the paper reports the network advantage stays in [0.44, 0.55] and
below the platform value across all nine.

These models are *approximations of a real stack*, disclosed as such; the live
experiment (RQ8, ``live_capture.py``) instead **captures** real socket bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leakbench.data.schemas import ToolCall, Trace, TraceDataset

# TLS-1.3 constants (approximate, documented as a model).
TLS_RECORD_SIZE = 16384          # max application-data record payload
TLS_RECORD_OVERHEAD = 22         # 5-byte header + 16-byte GCM tag + type byte
MTU = 1500                       # for MTU quantization on the wire


def tls_frame_size(payload_bytes: float) -> float:
    """On-the-wire size of ``payload_bytes`` after TLS-1.3 record framing."""
    payload_bytes = max(1.0, float(payload_bytes))
    n_records = int(np.ceil(payload_bytes / TLS_RECORD_SIZE))
    return payload_bytes + n_records * TLS_RECORD_OVERHEAD


def mtu_quantize(size: float, mtu: int = MTU) -> float:
    """Round a byte count up to a whole number of MTU-sized segments."""
    return float(np.ceil(max(1.0, size) / mtu) * mtu)


@dataclass
class HTTP2WireModel:
    """HTTP/2 multiplexing model for the network vantage (paper §Impl RQ6).

    Parameters
    ----------
    coalesce_prob:
        probability ``p`` that an adjacent message coalesces into the previous
        record (record coalescing).
    cross_traffic_sigma:
        std of multiplicative log-normal cross-traffic size noise.
    seed:
        RNG seed for the (stochastic) coalescing and noise.

    Notes
    -----
    The model keeps **one** RNG and advances it across calls to :meth:`apply`, so
    every trace gets an independent draw. Deriving the per-trace seed from the trace
    itself (e.g. from its label and length, as an earlier version did) is a subtle but
    real methodological bug: it makes the coalescing pattern a deterministic function
    of the label, so same-class traces receive identical wire distortion, which
    couples the observable to the secret through the *model* rather than through the
    traffic and inflates the network advantage. Cross-traffic in a real network is
    independent of which task the client is running.

    Determinism therefore holds at the level of a dataset pass (same seed + same trace
    order ⇒ same result), which is what :func:`apply_wire_model_dataset` provides. Call
    :meth:`reset` to rewind the stream.
    """

    coalesce_prob: float = 0.45
    cross_traffic_sigma: float = 0.15
    seed: int = 20270214

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Rewind the noise stream (so a repeated dataset pass is reproducible)."""
        self._rng = np.random.default_rng(self.seed)

    def apply(self, trace: Trace) -> Trace:
        """Return a new trace whose sizes are on-the-wire sizes for the network
        observer. Categories are *kept on the record* (the network map ignores them;
        the platform/provider maps use the pre-wire trace)."""
        rng = self._rng
        new_calls: list[ToolCall] = []
        carry = 0.0  # bytes carried forward when a message coalesces
        for i, call in enumerate(trace.calls):
            # TLS-frame the true up+down payload.
            framed = tls_frame_size(call.request_size) + tls_frame_size(call.response_size)
            framed += carry
            carry = 0.0
            # Cross-traffic multiplicative noise.
            noise = float(rng.lognormal(0.0, self.cross_traffic_sigma))
            wire = framed * noise
            # Record coalescing: with prob p, defer this message onto the next one.
            if i < len(trace.calls) - 1 and rng.random() < self.coalesce_prob:
                carry = wire
                continue
            wire = mtu_quantize(wire)
            # Encode the wire size into an opaque single "record" call. We keep the
            # original category label only for bookkeeping; the network map never
            # reads it. Split arbitrarily into up/down halves for schema validity.
            new_calls.append(
                ToolCall(
                    category=call.category,
                    request_size=max(1.0, wire * 0.4),
                    response_size=max(1.0, wire * 0.6),
                    gap=call.gap,
                )
            )
        if carry > 0 and new_calls:
            # Flush any trailing carried bytes into the last record.
            last = new_calls[-1]
            new_calls[-1] = ToolCall(
                category=last.category,
                request_size=last.request_size,
                response_size=last.response_size + carry,
                gap=last.gap,
            )
        if not new_calls:  # degenerate: keep at least one record
            new_calls = list(trace.calls)
        return Trace(
            calls=new_calls,
            label=trace.label,
            family=trace.family,
            metadata={**trace.metadata, "wire": "http2"},
        )


def apply_tls_framing(trace: Trace) -> Trace:
    """Apply only TLS record framing (no HTTP/2 multiplexing) --- used for the
    provider vantage on measured traces, where per-call sizes are visible but still
    framed."""
    calls = [
        ToolCall(
            category=c.category,
            request_size=tls_frame_size(c.request_size),
            response_size=tls_frame_size(c.response_size),
            gap=c.gap,
        )
        for c in trace.calls
    ]
    return Trace(calls=calls, label=trace.label, family=trace.family,
                 metadata={**trace.metadata, "wire": "tls"})


def apply_wire_model_dataset(
    dataset: TraceDataset, model: HTTP2WireModel | None = None
) -> TraceDataset:
    """Apply the HTTP/2 wire model to every trace in a dataset (network vantage).

    The model's noise stream is rewound first, so repeating this call on the same
    dataset with the same model reproduces the same wire exactly, while individual
    traces still receive independent draws.
    """
    model = model or HTTP2WireModel()
    model.reset()
    return TraceDataset(
        traces=[model.apply(t) for t in dataset.traces],
        taxonomy_name=dataset.taxonomy_name,
        taxonomy_hash=dataset.taxonomy_hash,
        source=dataset.source,
        metadata={**dataset.metadata, "wire_model": "http2"},
    )
