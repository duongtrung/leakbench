# WAN RTT sweep — what to create, and why each choice

Goal: turn the single wide-area data point into a **dose–response curve**. The paper
claims real propagation delay erases the timing channel while leaving the size channel
intact. One WAN point is consistent with that; four points at different RTTs test it.

This replaces the SSH-tunnel workaround, so it also closes the last stated limitation
(on-wire packet boundaries being the tunnel's framing rather than the TLS stream's own).

## Why two EC2 instances rather than laptop → EC2

| | laptop → EC2 | EC2 → EC2 |
|---|---|---|
| reproducible by a reviewer | no | **yes** |
| survives your IP changing | no (it changed twice in one day) | **yes** |
| security-group source | a public IP allowlist | **the client's SG** — AWS-idiomatic |
| lets us vary RTT | no, one fixed path | **yes, by choosing regions** |

The last row is the one that matters scientifically.

## What to create

**1. Keep the existing instance as the SERVER** (`us-east-1`). Nothing to change on it
except one inbound rule, below. Tell me its current public IP — it changes on every
stop/start, which is why we could not reach it.

**2. Create ONE client instance.** `t3.small`, Ubuntu 24.04, 20 GB gp3. Region: see the
sweep table. About **US$0.021/hour**; the whole sweep is under an hour of it.

**3. One inbound rule on the SERVER's security group:**

| field | value |
|---|---|
| Type | Custom TCP |
| Port range | **8443** |
| Source | **the client instance's security group id** (type `sg-` and pick it) |

Using the client's SG as the source rather than an IP is the point: no allowlist to
maintain, nothing breaks when an address changes, and the port is never open to the
internet. Leave port 22 as it is.

**4. Give the client instance outbound internet** (default SG allows this) so it can
`pip install`. No inbound rule is needed on the *client*.

## The sweep

Same server in `us-east-1` throughout; only the client's region moves. Expected RTTs
are typical AWS inter-region figures — the run measures the real ones and records them.

| point | client region | expected RTT | what it isolates |
|---|---|---|---|
| 1 | (already have) loopback | ~0.3 ms | no network at all |
| 2 | `us-east-2` (Ohio) | ~15 ms | a network, but a short one |
| 3 | `ap-southeast-1` (Singapore) | ~230 ms | long-haul |
| 4 | `sa-east-1` (São Paulo) | ~340 ms | longest reachable |

If you would rather create only one client, **`ap-southeast-1`** is the single most
valuable point: it is the realistic distance for your users and it is where the timing
channel should be most degraded. Points 2 and 4 turn the result into a curve; changing
a client's region means launching it in that region, so a three-point sweep is three
short-lived instances, not one.

## What I do once it exists

Give me the client's public IP and key (or add my key to it) and I will:

1. `rsync` the repo to the client, install, run `pytest` as a sanity check.
2. Start the tool server on the server instance bound to `0.0.0.0:8443`.
3. Run `scripts/run_wan_capture.py --server-host <server private or public IP>
   --server-port 8443 --server-ca <cert>` from the client, two seeds per region.
4. Capture on the server's interface and summarize with `scripts/analyze_wan_pcap.py`,
   which now reports egress PMTU, round trip and retransmissions.
5. Produce the RTT-sweep figure and fold it into RQ8.

Use the **private** IP if you put both instances in the same VPC (only possible for the
same region); across regions use the public IP, which is what the SG rule above covers.

## Cost, honestly

Three `t3.small` clients for an hour each plus cross-region data transfer for a few
hundred MB: well under **US$0.50** in total. The dominant cost is wall-clock — each
capture is ~12 minutes at 70 traces/class, so two seeds per region is ~25 minutes of
running per point.
