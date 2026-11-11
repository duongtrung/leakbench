#!/usr/bin/env python3
"""Summarize the on-wire capture taken during the WAN run (`scripts/run_wan_capture.py`).

This exists to answer one question a reviewer should ask: *did the traffic actually
cross a wide-area network, or is "WAN" a label on another loopback run?* It reads the
raw pcap with no third-party dependency and reports the properties that only a real
internet path produces --- an MTU well below the host's 9001-byte interface MTU (so
path-MTU discovery clamped it), a round-trip time in the hundreds of milliseconds, and
whatever retransmissions and out-of-order delivery the path happened to inflict.

It is deliberately **not** a second observable. The measured traffic rides inside an
SSH tunnel (the security group permits only port 22), so on-wire packet boundaries are
the tunnel's framing, not the TLS stream's own. The leakage numbers come from the
capture proxy, which sees the real TLS records with real WAN timing.

Usage:  python scripts/analyze_wan_pcap.py /tmp/wan.pcap --client-ip $CLIENT_IP
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

PCAP_MAGICS = {
    0xA1B2C3D4: ("<", 1e-6),   # little-endian, microsecond
    0xD4C3B2A1: (">", 1e-6),
    0xA1B23C4D: ("<", 1e-9),   # nanosecond
    0x4D3CB2A1: (">", 1e-9),
}


def read_pcap(path: Path):
    """Yield (timestamp, ethernet frame bytes). Classic pcap only; that is what
    `tcpdump -w` writes by default."""
    data = path.read_bytes()
    magic = struct.unpack("<I", data[:4])[0]
    if magic not in PCAP_MAGICS:
        magic = struct.unpack(">I", data[:4])[0]
    if magic not in PCAP_MAGICS:
        raise SystemExit(f"{path} is not a classic pcap (magic {magic:#x}); "
                         "re-capture without --pcapng, or convert it.")
    endian, tick = PCAP_MAGICS[magic]
    snaplen = struct.unpack(endian + "I", data[16:20])[0]
    off = 24
    while off + 16 <= len(data):
        ts_s, ts_frac, caplen, _origlen = struct.unpack(endian + "IIII", data[off:off + 16])
        off += 16
        yield ts_s + ts_frac * tick, data[off:off + caplen]
        off += caplen
    _ = snaplen


def parse(frame: bytes):
    """Return (src_ip, dst_ip, src_port, dst_port, ip_total_len, seq, payload_len, flags)
    for IPv4/TCP frames, else None. Works on truncated captures: we only need headers."""
    if len(frame) < 34 or frame[12:14] != b"\x08\x00":
        return None
    ip = frame[14:]
    ihl = (ip[0] & 0x0F) * 4
    if ip[9] != 6 or len(ip) < ihl + 20:
        return None
    total_len = struct.unpack(">H", ip[2:4])[0]
    src = ".".join(str(b) for b in ip[12:16])
    dst = ".".join(str(b) for b in ip[16:20])
    tcp = ip[ihl:]
    sport, dport = struct.unpack(">HH", tcp[0:4])
    seq = struct.unpack(">I", tcp[4:8])[0]
    doff = (tcp[12] >> 4) * 4
    flags = tcp[13]
    payload_len = max(0, total_len - ihl - doff)
    return src, dst, sport, dport, total_len, seq, payload_len, flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--client-ip", required=True,
                    help="public address the client appeared as, to split directions")
    ap.add_argument("--host-mtu", type=int, default=9001,
                    help="the capturing interface's own MTU, for the PMTU comparison")
    ap.add_argument("--max-turnaround-s", type=float, default=2.0,
                    help="intervals longer than this are the session being idle (SSH "
                         "keepalives, gaps between runs), not a request waiting on the "
                         "path; they are excluded from the round-trip statistic and "
                         "counted separately")
    ap.add_argument("--output", default="outputs/wan/wire_summary.json")
    args = ap.parse_args()

    up, down, seen = [], [], {}
    retrans = 0
    t_first = t_last = None
    for ts, frame in read_pcap(Path(args.pcap)):
        p = parse(frame)
        if p is None:
            continue
        src, _dst, sport, _dport, total_len, seq, plen, _flags = p
        t_first = ts if t_first is None else t_first
        t_last = ts
        if plen == 0:                      # pure ACKs carry no application bytes
            continue
        key = (src, sport, seq, plen)
        if key in seen:
            retrans += 1
        seen[key] = ts
        (up if src == args.client_ip else down).append((ts, total_len, plen))

    if not up and not down:
        raise SystemExit("no data-carrying TCP packets matched; check --client-ip "
                         "and the tcpdump filter")

    def stats(rows, label):
        sizes = np.array([r[1] for r in rows], dtype=float)
        return {
            "packets": len(rows),
            "bytes_on_wire": float(sizes.sum()),
            "size_median": float(np.median(sizes)),
            "size_p90": float(np.percentile(sizes, 90)),
            "size_max": float(sizes.max()),
            "_label": label,
        }

    u, d = stats(up, "client->server"), stats(down, "server->client")
    all_ts = sorted([r[0] for r in up + down])
    gaps = np.diff(np.array(all_ts))
    # Request/response turnaround. Packets arrive in bursts (one request, then the
    # response spread over many segments), so measuring every upstream packet against
    # the next downstream one just samples *within* a burst and reports sub-millisecond
    # nonsense. Collapse each upstream burst first, then time its last packet to the
    # first downstream packet that follows: that is one request -> first response byte.
    #
    # Note *where* this capture is taken: at the server's NIC. So up-burst -> first
    # down measures the server's own service time and says nothing about the path.
    # The round trip shows up in the other direction --- the server's last response
    # packet to the client's next request --- because that interval contains the full
    # propagation delay in both directions plus the client's think time.
    BURST_GAP_S = 0.05

    def burst_ends(ts_list):
        return [t for t, nxt in zip(ts_list, ts_list[1:] + [float("inf")])
                if nxt - t > BURST_GAP_S]

    def follow(ends, others):
        out, i = [], 0
        for t in ends:
            while i < len(others) and others[i] <= t:
                i += 1
            if i < len(others):
                out.append(others[i] - t)
        return np.array(out) if out else np.array([0.0])

    up_ts = sorted(r[0] for r in up)
    down_ts = sorted(r[0] for r in down)
    service_a = follow(burst_ends(up_ts), down_ts)    # server-side processing
    turns_raw = follow(burst_ends(down_ts), up_ts)     # round trip + client think time
    # Drop idle stretches: with the capture left running between and around the
    # sessions, the keepalive interval would otherwise dominate the upper quantiles.
    turns_a = turns_raw[turns_raw <= args.max_turnaround_s]
    n_idle = int(turns_raw.size - turns_a.size)
    if turns_a.size == 0:
        turns_a = np.array([0.0])

    # PMTU evidence must come from the *egress* direction. This capture is taken on
    # the server, so client->server packets arrive through the NIC's receive path and
    # generic receive offload may already have merged segments before tcpdump sees
    # them -- which is why the upstream maximum can exceed the path MTU. The
    # server->client direction is what the host actually put on the wire.
    largest = d["size_max"]
    summary = {
        "pcap": args.pcap,
        "capture_seconds": float(t_last - t_first),
        "client_ip": args.client_ip,
        "upstream": u,
        "downstream": d,
        "retransmitted_or_duplicate_segments": retrans,
        "largest_egress_ip_packet": largest,
        "largest_ingress_ip_packet": u["size_max"],
        "ingress_note": "may exceed the path MTU: receive offload can merge segments "
                        "before the capture point, so only the egress figure is "
                        "evidence about the path",
        "host_interface_mtu": args.host_mtu,
        "path_mtu_clamped": bool(largest < args.host_mtu),
        "server_service_time_s": {
            "median": float(np.median(service_a)),
            "p90": float(np.percentile(service_a, 90)),
            "_what": "server NIC: request in -> first response byte out",
        },
        "round_trip_plus_client_s": {
            "median": float(np.median(turns_a)),
            "p10": float(np.percentile(turns_a, 10)),
            "p90": float(np.percentile(turns_a, 90)),
            "_what": "server NIC: last response byte out -> next request in; "
                     "contains the full propagation delay both ways",
            "n_intervals": int(turns_a.size),
            "n_idle_intervals_excluded": n_idle,
            "idle_threshold_s": args.max_turnaround_s,
        },
        "inter_packet_gap_s": {
            "median": float(np.median(gaps)) if gaps.size else 0.0,
            "p99": float(np.percentile(gaps, 99)) if gaps.size else 0.0,
        },
        "note": "corroboration that the link is a real internet path, not a second "
                "observable: the measured traffic rides inside an SSH tunnel, so these "
                "packet boundaries are the tunnel's framing",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "wire_capture_summary",
                               "result": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
