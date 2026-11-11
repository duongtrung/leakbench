#!/usr/bin/env python3
"""Run the RQ8 tool server standalone, for the **WAN** variant of the live experiment.

Two-host layout (``docs/ec2_experiment_guide.md`` §5a):

    host B (server):  python scripts/run_tool_server.py --bind 0.0.0.0 --port 8443
                      # copy the printed certificate path to host A
    host A (client):  python scripts/run_live_capture.py --traces-per-class 70 \
                          --server-host <B private IP> --server-port 8443 \
                          --server-ca cert.pem

The capture proxy runs on the **client** host, so it sits on the client's uplink and
records the bytes as they cross the real link — which is exactly the network vantage
the paper models. The server here is the same real tool backend used by the loopback
experiment; nothing about the tools, the protocol, or the estimator changes.

Security note: this server executes only its own fixed local tools (in-process Python
arithmetic, in-memory SQLite, temp-file I/O, corpus search, JSON payloads); it does
not evaluate anything a client sends. Still, expose the port **only** to the client
host's security group, never to the internet.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from leakbench.data.live_capture import _ToolServer
from leakbench.data.tlsutil import generate_self_signed
from leakbench.utils.logging import get_logger

logger = get_logger("leakbench.tool_server")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0", help="bind address")
    ap.add_argument("--port", type=int, default=8443, help="bind port (0 = pick one)")
    ap.add_argument("--seed", type=int, default=20270214)
    ap.add_argument("--no-tls", action="store_true", help="serve plaintext TCP")
    ap.add_argument("--cert-out", default="leakbench_server_cert.pem",
                    help="where to copy the certificate the client must trust")
    args = ap.parse_args()

    material = None if args.no_tls else generate_self_signed()
    if not args.no_tls and material is None:
        raise SystemExit(
            "No TLS material could be provisioned (install `cryptography` or the "
            "`openssl` binary), or pass --no-tls to serve plaintext."
        )

    server = _ToolServer(seed=args.seed, tls=material, bind_host=args.bind,
                         bind_port=args.port)
    server.start()
    cert_path = None
    if material is not None:
        cert_path = Path(args.cert_out).resolve()
        shutil.copyfile(material.cert_path, cert_path)

    print(f"LeakBench tool server listening on {args.bind}:{server.port}")
    print(f"  TLS: {'off (plaintext TCP)' if material is None else 'on'}")
    if cert_path is not None:
        print(f"  certificate written to {cert_path}")
        print(f"  copy it to the client host and pass --server-ca {cert_path.name}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        server.close()
        if material is not None:
            material.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
