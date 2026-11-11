# E1: single-provider vantage (10/11/2026)

Reviewer B #6: the providers vantage assumes that every tool provider colludes. What does one provider learn on its own?

Code: `e1/single_provider.py` computes the results and `e1/summarize.py` builds the table. Raw results are in `e1/single_provider.json` and the table in `e1/table_single_provider.csv`. The whole run takes about 30 s on one workstation.

## Method

The provider of tool category *c* sees only the calls addressed to it. For each session it gets:
- the request and response size of each of its calls;
- how many calls it received;
- the time between its own consecutive calls.

It does not see the other tools, their sizes, or when the session started. We assume the provider can link the calls of one session to each other, for example through the same API key or connection.

Its view is featurized with the network map O_net, which covers sizes, count and timing. The estimator, classifier bank, bootstrap and seeds are unchanged.

A session that never called *c* is an all-zero vector. So the provider is credited with knowing that the session existed. That makes these numbers an upper bound for metadata leakage to one provider.

**Sanity check:** the joint numbers recomputed by this script match the earlier runs exactly for all five trace sets:
- synthetic: 0.6409 / 0.8746 / 0.9928;
- WAN: 0.3786 / 0.8214 / 0.8143;
- loopback: 0.4214 / 0.8071 / 0.7929;
- local seeds: 0.80 / 1.00 / 1.00 and 0.88 / 0.96 / 0.96.

## Result

| Level | Seed | Network | Platform | Providers, colluding | One provider, median | One provider, max |
|---|---|---|---|---|---|---|
| Synthetic (12 cl.) | 20270214 | 0.64 [0.62, 0.66] | 0.87 [0.86, 0.89] | 0.99 [0.99, 1.00] | 0.31 | 0.34 [0.32, 0.36] (search, in 54% of sessions) |
| L-prog loopback (6 cl.) | 20270215 | 0.42 [0.33, 0.51] | 0.81 [0.74, 0.88] | 0.79 [0.73, 0.86] | 0.21 | 0.44 [0.34, 0.51] (search, in 64% of sessions) |
| WAN, scripted (6 cl.) | 20270215 | 0.38 [0.29, 0.47] | 0.82 [0.75, 0.89] | 0.81 [0.74, 0.87] | 0.23 | 0.40 [0.32, 0.49] (search, in 64% of sessions) |
| L-LLM local Q4 (6 cl.) | 20270214 | 0.80 [0.66, 0.90] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.40 | 0.80 [0.68, 0.90] (search, in 83% of sessions) |
| L-LLM local Q4 (6 cl.) | 20270215 | 0.88 [0.78, 0.96] | 0.96 [0.90, 1.00] | 0.96 [0.90, 1.00] | 0.38 | 0.66 [0.54, 0.80] (search, in 83% of sessions) |

## Findings

1. **Collusion is what puts the providers vantage on top.** On its own, a provider learns far less than all providers together:
   - The median single provider sits at 0.21–0.40, against 0.79–1.00 for colluding providers.
   - The median single provider is also below the on-path network in every row.
2. **The most-used provider is about as strong as the network observer.** The best single provider is always `search`, the tool called in the most sessions:
   - Live runs: it reaches 0.40–0.80, against 0.38–0.88 for the network. The confidence intervals overlap in every live row.
   - Synthetic generator: it is well below the network (0.34 vs 0.64).
3. **The ordering for the paper:** one provider ≲ network < platform, and colluding providers are the upper bound.

## What to say in the paper

> A single tool provider, which sees only the calls addressed to it, reaches a median advantage of 0.21–0.40. The most widely used provider (search) reaches 0.34–0.80, comparable to the on-path network. The colluding-providers vantage is therefore an upper bound that requires collusion; no single provider comes close to it.

This fits as one row or column in Table 1 ("One provider: median / max"), or as one sentence.

## Caveats (state them where the number is used)

- **Metadata only.** A real tool provider also reads the plaintext of the requests it serves, for example the search query. These numbers measure what the provider learns from traffic metadata, consistent with the other vantages. They are not everything a provider knows. Say so in one clause, otherwise a reviewer will.
- **Linking and existence are assumed.** The provider is assumed to link the calls of one session and to know about sessions that did not call it. Both choices favour the provider, so the result is an upper bound for one provider's metadata view.
- **A conditional variant was computed but should not be reported.** It looks only at sessions that called the provider and is stored in the JSON under `called`. On the live traces it rests on 25–150 sessions, often spanning only 2–3 classes, so it swings between 0.0 and 1.0 with very wide intervals. On the synthetic data it gives 0.44–0.62.
- **The hosted L-LLM run is not covered.** Its traces were not kept in the original run. The local L-LLM rows inherit the local-planner caveats from E2: every session ran to the 12-step cap, and the timing channel is open. Part of the search provider's 0.66–0.80 may come from that timing channel.
