"""Emit the body rows of tables/t_crossmodel.tex from the sweep summary.

The cross-model table grows every time another planner finishes, and hand-editing it
each time is how a paper ends up with a row that no run produced. This prints the rows
and the survival counts straight from the stored summary, so updating the table is a
paste rather than a transcription.

Usage:  python scripts/emit_crossmodel_rows.py
"""
import json
import pathlib

SHORT = {
    "meta-llama/llama-3.3-70b-instruct":        ("llama-3.3-70b", "Meta"),
    "google/gemma-3-27b-it":                    ("gemma-3-27b", "Google"),
    "mistralai/mistral-nemo":                   ("mistral-nemo", "Mistral"),
    "mistralai/mistral-small-3.2-24b-instruct": ("mistral-small-24b", "Mistral"),
    "qwen/qwen3-30b-a3b-instruct-2507":         ("qwen3-30b-a3b", "Alibaba"),
    "openai/gpt-oss-20b":                       ("gpt-oss-20b", "OpenAI, via OpenRouter"),
    "openai.gpt-oss-20b-1:0":                   ("gpt-oss-20b", "OpenAI, via Bedrock"),
}
r = json.loads(pathlib.Path("outputs/llm_sweep/summary.json").read_text())["result"]
rows = []
for x in r["rows"]:
    short, lab = SHORT.get(x["model"], (x["model"].split("/")[-1], x.get("provider", "")))
    a = x["advantage"]
    rows.append(f"{short} \\textsf{{({lab})}}".ljust(46) +
                f"& ${a['network']:.2f}$ & ${a['platform']:.2f}$ & ${a['providers']:.2f}$ "
                f"& ${x['size_only']:.2f}$ & ${x['timing_only']:.2f}$ \\\\")
n = r["n_models"]
so, to = r["size_only_range"], r["timing_only_range"]
print("\n".join(rows))
print(f"\n% net<plat {r['net_below_platform']}  size>time {r['size_beats_timing']}  "
      f"prov>=plat {r['providers_ge_platform_pointwise']}  "
      f"significant violations: {r['significant_refinement_violations'] or 'none'}")
print(f"% size-only {so[0]:.2f}-{so[1]:.2f}   timing-only {to[0]:.2f}-{to[1]:.2f}   n={n}")
