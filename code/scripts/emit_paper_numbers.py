#!/usr/bin/env python3
"""Emit every paper table row and figure coordinate **from measured results**.

    python scripts/emit_paper_numbers.py --input outputs/scale_study \
        --output outputs/scale_study/paper_numbers.tex

The paper's `tables/t_*.tex` and `plots/p_*.tex` currently carry hard-coded values.
This script reads the result JSON produced by `run_scale_study.py` and prints, for
each of them, the exact LaTeX fragment to paste in --- so every number in the
submission traces back to a run with recorded provenance instead of a literal.

Nothing is invented: a paper object whose stage was not run is reported as MISSING
rather than filled with a plausible value. Numbers are formatted to the precision the
paper uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VANTAGE_ORDER = ("network", "platform", "providers")


def _load(indir: Path, stage: str) -> dict[str, Any] | None:
    p = indir / f"{stage}.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))["result"]


def _missing(name: str, stage: str) -> str:
    return f"% {name}: MISSING -- stage '{stage}' was not run; do not fill this in by hand.\n"


def t_vantage(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_vantage.tex", "vantage")
    out = ["% tables/t_vantage.tex -- body rows (RQ1)"]
    chips = {"network": r"\netchip{} network", "platform": r"\platchip{} platform",
             "providers": r"\provchip{} providers"}
    for v in VANTAGE_ORDER:
        d = r["vantages"][v]
        lo, hi = d["advantage_ci"]
        out.append(f"{chips[v]:<26} & ${d['accuracy']:.2f}$ & "
                   f"${d['advantage']:.2f}\\;[{lo:.2f},{hi:.2f}]$ & "
                   f"${d['bayes_risk']:.2f}$ & ${d['mi_lower_bound_bits']:.2f}$ \\\\")
    return "\n".join(out) + "\n"


def p_vantage(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_vantage.tex", "vantage")
    pts = []
    for i, v in enumerate(VANTAGE_ORDER, start=1):
        d = r["vantages"][v]
        lo, hi = d["advantage_ci"]
        half = max(d["advantage"] - lo, hi - d["advantage"])
        pts.append(f"({i},{d['advantage']:.3f}) +- (0,{half:.3f})")
    mi = ", ".join(f"{r['vantages'][v]['mi_lower_bound_bits']:.2f}" for v in VANTAGE_ORDER)
    return ("% plots/p_vantage.tex -- coordinates\n  coordinates {"
            + " ".join(pts) + "};\n"
            + f"% caption: recovered information: {mi} bits.\n")


def t_coarse(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_coarse.tex", "coarse")
    out = ["% tables/t_coarse.tex -- body rows (4-way family vs 12-way class)"]
    chips = {"network": r"\netchip{} network", "platform": r"\platchip{} platform",
             "providers": r"\provchip{} providers"}
    for v in VANTAGE_ORDER:
        d = r["rows"][v]
        out.append(f"{chips[v]:<26} & ${d['family_advantage_reestimated']:.2f}$ & "
                   f"${d['family_mi_bits']:.2f}$ & ${d['class_advantage']:.2f}$ & "
                   f"${d['class_mi_bits']:.2f}$ \\\\")
    return "\n".join(out) + "\n"


def p_ablation(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_ablation.tex", "ablation")
    a = r["ablations"]
    order = ["length_only", "timing_only", "size_only", "size_plus_timing_all"]
    pts = " ".join(f"({i},{a[k]['advantage']:.3f})" for i, k in enumerate(order, start=1))
    return "% plots/p_ablation.tex -- coordinates\n  coordinates {" + pts + "};\n"


def p_subgroup(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_subgroup.tex", "subgroup")
    band, fam = r["by_length_band"], r["within_family_discrimination"]
    pts = [f"(short,{band.get('short', float('nan')):.3f})",
           f"(long,{band.get('long', float('nan')):.3f})"]
    for f in ("research", "coding", "personal", "ops"):
        d = fam.get(f, {})
        if "advantage" in d:
            pts.append(f"({f},{d['advantage']:.3f})")
    return ("% plots/p_subgroup.tex -- coordinates (length bands = full 12-way; "
            "family = within-family k-way)\n  coordinates {" + " ".join(pts) + "};\n")


def t_subgroup(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_subgroup.tex", "subgroup")
    band, band_mi = r["by_length_band"], r["by_length_band_mi_bits"]
    fam = r["within_family_discrimination"]
    out = ["% tables/t_subgroup.tex -- body rows"]
    out.append(f"                              & short ($n\\le$ med) & ${band.get('short', 0):.2f}$ "
               f"& ${band_mi.get('short', 0):.2f}$ \\\\")
    out.append(f"                              & long ($n>$ med)    & ${band.get('long', 0):.2f}$ "
               f"& ${band_mi.get('long', 0):.2f}$ \\\\")
    for f in ("research", "coding", "personal", "ops"):
        d = fam.get(f, {})
        if "advantage" in d:
            out.append(f"                                & {f} & ${d['advantage']:.2f}$ "
                       f"& ${d['mi_bits']:.2f}$ \\\\")
    return "\n".join(out) + "\n"


def t_defense(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_defense.tex", "defenses")
    out = ["% tables/t_defense.tex -- body rows"]
    for row in r["rows"]:
        out.append(f"{row['defense']:<14} & --- & ${row['advantage']:.2f}$ & "
                   f"${row['mi_bits']:.2f}$ & ${row['bandwidth_overhead'] * 100:+.0f}\\%$ & "
                   f"${row['latency_overhead'] * 100:+.0f}\\%$ \\\\")
    return "\n".join(out) + "\n"


def p_tradeoff(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_tradeoff.tex", "defenses")
    pts = " ".join(f"({row['bandwidth_overhead'] * 100:.0f},{row['advantage']:.3f})"
                   for row in r["rows"])
    return ("% plots/p_tradeoff.tex -- frontier coordinates (x = bandwidth overhead %)\n"
            "  coordinates {" + pts + "};\n")


def p_validation(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_validation.tex", "validation")
    ana = " ".join(f"({row['p']:g},{row['analytic_mi_bits']:.2f})" for row in r["rows"])
    est = " ".join(f"({row['p']:g},{row['estimated_mi_bits']:.2f})" for row in r["rows"])
    return ("% plots/p_validation.tex -- analytic then estimated\n"
            "  coordinates {" + ana + "};\n"
            "  coordinates {" + est + "};\n"
            f"% max |MI - analytic| = {r['max_mi_abs_error']:.3f} bits\n")


def p_separability(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_separability.tex", "separability")
    knob = r.get("separability_knob", "dirichlet_alpha")
    adv = " ".join(f"({row[knob]:g},{row['advantage']:.3f})" for row in r["rows"])
    mi = " ".join(f"({row[knob]:g},{row['mi_bits']:.2f})" for row in r["rows"])
    warn = ""
    if not r.get("advantage_monotone", True):
        warn = ("% WARNING: the advantage is NOT monotone along the separability knob.\n"
                "% Do not caption this figure as monotone -- see "
                "audits/paper_vs_measured.md.\n")
    out = [warn + f"% plots/p_separability.tex -- x axis is {knob}",
           "% (swept from confusable to separable; note the axis is REVERSED vs alpha's",
           "%  numeric order, so label it 'class separability', not 'alpha')",
           "  coordinates {" + adv + "};",
           "  coordinates {" + mi + "};"]
    fam = r.get("family_structure_knob")
    if fam:
        frows = " ".join(f"({row['signature_upweight']:g},{row['advantage']:.3f})"
                         for row in fam["rows"])
        out += ["% companion curve: the FAMILY-STRUCTURE knob (signature up-weight).",
                "% It decreases advantage, because the signature is shared by a family,",
                "% so up-weighting it makes same-family classes more alike.",
                "  coordinates {" + frows + "};"]
    return "\n".join(out) + "\n"


def p_scalability(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("plots/p_scalability.tex", "scalability")
    pts = " ".join(f"({row['n_traces']},{row['estimate_seconds']:.2f})" for row in r["rows"])
    return ("% plots/p_scalability.tex -- coordinates (x = traces, y = seconds)\n"
            "  coordinates {" + pts + "};\n"
            f"% log-log slope = {r['loglog_slope']:.2f} (1.0 would be linear)\n")


def t_measured_and_plot(r: dict[str, Any] | None, gen: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_measured.tex + plots/p_measured.tex", "measured_multiseed")
    out = ["% tables/t_measured.tex -- body rows",
           "% NOTE: the +- is the spread across seeds; the interval is the bootstrap CI",
           "% at the reference seed, as the paper's own caption specifies. The two answer",
           "% different questions, so the CI need not contain the cross-seed mean --- say so",
           "% in the caption rather than leaving a reader to trip over it."]
    chips = {"network": r"\netchip{} network", "platform": r"\platchip{} platform",
             "providers": r"\provchip{} providers"}
    for v in VANTAGE_ORDER:
        s = r["summary"][v]
        lo, hi = s["advantage_ci_at_reference_seed"]
        out.append(f"{chips[v]:<24} & ${s['mean']:.2f} \\pm {s['std']:.2f}$ & "
                   f"$[{lo:.2f}, {hi:.2f}]$ & ${s['mi_bits_at_reference_seed']:.2f}$ \\\\")
    out.append("")
    out.append("% plots/p_measured.tex -- generator series then measured series")
    if gen:
        g = " ".join(f"({i},{gen['vantages'][v]['advantage']:.2f}) +- (0,0.0)"
                     for i, v in enumerate(VANTAGE_ORDER, start=1))
        out.append("  coordinates {" + g + "};")
    m = " ".join(f"({i},{r['summary'][v]['mean']:.2f}) +- (0,{r['summary'][v]['std']:.2f})"
                 for i, v in enumerate(VANTAGE_ORDER, start=1))
    out.append("  coordinates {" + m + "};")
    for step, s in r["ordering_steps"].items():
        out.append(f"% {step}: holds in {s['holds_in_seeds']}/{s['n_seeds']} seeds, "
                   f"t={s['t_statistic']:.1f}, p={s['p_value_one_sided']:.1g}")
    return "\n".join(out) + "\n"


def t_robustness(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_robustness.tex", "robustness")
    b = r["by_attacker"]
    order = ("random_forest", "hist_gbdt", "logistic", "knn")
    vals = " & ".join(f"${b[k]['advantage']:.2f}$" for k in order if k in b)
    return ("% tables/t_robustness.tex -- body row\n"
            f"\\textbf{{Advantage}} & {vals} \\\\\n")


def t_openworld(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_openworld.tex", "openworld")
    out = ["% tables/t_openworld.tex -- body rows"]
    for row in r["rows"]:
        flag = "  % DEGENERATE: " + row["degenerate"] if "degenerate" in row else ""
        out.append(f"${row['monitored_base_rate'] * 100:.0f}\\%$ & ${row['precision']:.2f}$ & "
                   f"${row['recall']:.2f}$ & ${row['f1']:.2f}$ \\\\{flag}")
    out.append(f"% within-taxonomy 'other': F1 = {r['within_taxonomy']['f1']:.2f}")
    return "\n".join(out) + "\n"


def t_taxablation(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("tables/t_taxablation.tex", "taxonomy_sweep")
    out = ["% tables/t_taxablation.tex -- body rows"]
    for row in r["rows"]:
        out.append(f"{row['num_classes']:<3} & ${row['entropy_bits']:.2f}$ & "
                   f"${row['advantage']:.2f}$ & ${row['mi_bits']:.2f}$ \\\\")
    return "\n".join(out) + "\n"


def t_live(prog: dict[str, Any] | None, neural: dict[str, Any] | None,
           wan: dict[str, Any] | None) -> str:
    """Table 'Live' now carries three configurations side by side, so it is assembled
    from three separate runs: the programmatic loopback stage, the neural-planner run,
    and the wide-area run. Each is optional and reported as MISSING if absent, because
    a silently dropped column would read as 'we did not need it'."""
    if not any((prog, neural, wan)):
        return _missing("tables/t_live.tex", "live / live_neural / wan")

    def first(r):
        return r["per_seed"][0] if r and r.get("per_seed") else None

    cols = [("L-prog", first(prog)), ("L-LLM", first(neural)), ("WAN", wan)]
    chips = {"network": r"\netchip{} network (captured wire)",
             "platform": r"\platchip{} platform", "providers": r"\provchip{} providers"}
    out = ["% tables/t_live.tex -- body rows, three configurations (first seed each)",
           "% columns: " + " | ".join(n for n, _ in cols)]

    def cell(e, v):
        if not e:
            return "MISSING"
        d = e["vantages"][v]
        return f"${d['advantage']:.2f}$ (${d['mi_bits']:.2f}$)"

    for v in VANTAGE_ORDER:
        out.append(f"{chips[v]:<34} & " + " & ".join(cell(e, v) for _, e in cols) + r" \\")
    for proj, label in (("size_only", r"\quad size-only (Whisper-Leak-like)"),
                        ("timing_only", r"\quad timing-only                  ")):
        vals = [f"${e['projections'][proj]:.2f}$" if e else "MISSING" for _, e in cols]
        out.append(f"{label} & " + " & ".join(vals) + r" \\")

    if wan:
        d = wan["vantages"]
        out.append("% WAN 95% CIs: " + ", ".join(
            f"[{d[v]['advantage_ci'][0]:.2f},{d[v]['advantage_ci'][1]:.2f}]"
            for v in VANTAGE_ORDER))
        g = wan.get("inter_call_gap_s", {})
        if g:
            out.append(f"% WAN median inter-call gap: {g['median']*1000:.0f} ms")
        out.append(f"% WAN link: {wan.get('link')}; {wan.get('n_traces')} sessions in "
                   f"{wan.get('elapsed_s', 0):.0f} s")
    for tag, r in (("L-prog", prog), ("L-LLM", neural)):
        if r:
            out.append(f"% {tag}: capture={r['capture_mode']}, link={r.get('link')}, "
                       f"planner={r.get('planner_kind')}, seeds={r['seeds']}")
            if len(r["per_seed"]) > 1:
                sd = r["per_seed"][1]
                out.append(f"%   {tag} second seed: " + ", ".join(
                    f"{v}={sd['vantages'][v]['advantage']:.2f}" for v in VANTAGE_ORDER))
    return "\n".join(out) + "\n"


def t_crossmodel(r: dict[str, Any] | None) -> str:
    """Table 'Cross-model': one live capture per model, plus the provider control.

    The point of the table is not the individual numbers but which claims survive
    every model. Emitted with the survival counts so the caption cannot overstate
    them by accident.
    """
    if not r:
        return _missing("tables/t_crossmodel.tex", "llm_sweep/summary")
    out = ["% tables/t_crossmodel.tex -- body rows (one live capture per model)"]
    for row in r["rows"]:
        a = row["advantage"]
        short = row["model"].split("/")[-1].replace("-instruct", "")
        provider = row.get("provider", "")
        label = f"{short} \\,\\textsf{{({provider})}}" if provider else short
        out.append(f"{label:<46} & ${a['network']:.2f}$ & ${a['platform']:.2f}$ & "
                   f"${a['providers']:.2f}$ & ${row['size_only']:.2f}$ & "
                   f"${row['timing_only']:.2f}$ \\\\")
    out.append(f"% net < platform: {r['net_below_platform']}")
    out.append(f"% providers >= platform (pointwise): "
               f"{r['providers_ge_platform_pointwise']}")
    out.append(f"% significant refinement violations: "
               f"{r['significant_refinement_violations'] or 'none'}")
    out.append(f"% size beats timing: {r['size_beats_timing']}")
    out.append(f"% size-only range: {r['size_only_range'][0]:.2f}"
               f"--{r['size_only_range'][1]:.2f}; timing-only range: "
               f"{r['timing_only_range'][0]:.2f}--{r['timing_only_range'][1]:.2f}")
    return "\n".join(out) + "\n"


def t_wire(r: dict[str, Any] | None) -> str:
    """The packet-capture evidence that the WAN link is a real internet path."""
    if not r:
        return _missing("WAN wire capture (quoted in tables/t_live.tex caption)",
                        "wan/wire_summary")
    rt, sv = r["round_trip_plus_client_s"], r["server_service_time_s"]
    return "\n".join([
        "% WAN wire capture (quoted in the t_live caption and sec:live)",
        f"%   round trip {rt['median']*1000:.0f} ms median over {rt['n_intervals']} "
        f"intervals (p10 {rt['p10']*1000:.0f}, p90 {rt['p90']*1000:.0f})",
        f"%   server service time {sv['median']*1000:.1f} ms median",
        f"%   largest server->client packet {r['largest_egress_ip_packet']:.0f} B vs "
        f"host MTU {r['host_interface_mtu']} (PMTU clamped: {r['path_mtu_clamped']})",
        f"%   (ingress max {r['largest_ingress_ip_packet']:.0f} B -- receive offload, "
        f"not path evidence)",
        f"%   retransmitted/duplicate segments: "
        f"{r['retransmitted_or_duplicate_segments']}",
    ]) + "\n"


def t_projection(r: dict[str, Any] | None) -> str:
    if not r:
        return _missing("case study (tables/t_projection.tex numbers)", "projections")
    out = ["% Section 'Case study: reproducing a deployed attack' -- measured values"]
    for name, d in r["rows"].items():
        out.append(f"%   {name:<22} advantage {d['advantage']:.3f}  ({d['vantage']})")
    return "\n".join(out) + "\n"


def prose(indir: Path) -> str:
    """Numbers the paper states in running text rather than in a float."""
    out = ["% ---- values quoted in prose ----"]
    fi = _load(indir, "feature_importance")
    if fi:
        parts = ", ".join(f"{k} {v:.0%}" for k, v in
                          sorted(fi["importance_by_block"].items(), key=lambda kv: -kv[1]))
        out.append(f"% RQ2 feature importance: {parts}")
        out.append(f"%   size-related {fi['size_related_share']:.0%}, "
                   f"timing-related {fi['timing_related_share']:.0%}")
    gv = _load(indir, "generator_variance")
    if gv:
        s = gv["summary"]
        out.append("% Variance across generator draws: " + ", ".join(
            f"{v} {s[v]['mean']:.2f}+-{s[v]['std']:.2f}" for v in VANTAGE_ORDER))
    ps = _load(indir, "prior_sensitivity")
    if ps:
        u = ps["rows"]["uniform"]["vantages"]["platform"]["advantage"]
        z = ps["rows"]["zipf"]["vantages"]["platform"]["advantage"]
        out.append(f"% Non-uniform prior: platform advantage {z:.2f} (Zipf) vs {u:.2f} (uniform)")
    ws = _load(indir, "wire_sweep")
    if ws:
        lo, hi = ws["network_advantage_range"]
        out.append(f"% Wire-model sweep: network advantage in [{lo:.2f}, {hi:.2f}], "
                   f"below platform ({ws['platform_advantage']:.2f}) in "
                   f"{ws['configs_below_platform']}/{ws['n_configs']} configs")
    ov = _load(indir, "overlap_sweep")
    if ov:
        out.append("% Overlap sensitivity (unstated in the paper):")
        for row in ov["rows"]:
            out.append(f"%   overlap {row['overlap']:.2f}: net {row['network']:.2f} "
                       f"(pre-wire {row['network_pre_wire']:.2f}), plat {row['platform']:.2f}, "
                       f"prov {row['providers']:.2f}")
    ea = _load(indir, "estimator_agreement")
    if ea:
        out.append(f"% Estimator agreement: vantage ranking {ea['vantage_ranking']}, "
                   f"agrees={ea['agreement']['vantage_ranking_agrees']}")
    lc = _load(indir, "learning_curve")
    if lc:
        out.append(f"% Learning curve last-step delta: {lc['last_step_delta']}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="outputs/scale_study")
    ap.add_argument("--output", default=None, help="write here instead of stdout")
    args = ap.parse_args()
    indir = Path(args.input)
    if not indir.is_dir():
        raise SystemExit(f"{indir} does not exist -- run scripts/run_scale_study.py first.")

    summary_path = indir / "summary.json"
    header = ["% LeakBench measured numbers for the paper.",
              "% Generated by scripts/emit_paper_numbers.py -- do not edit by hand."]
    if summary_path.exists():
        s = json.load(open(summary_path, encoding="utf-8"))
        prov, meta = s.get("provenance", {}), s.get("meta", {})
        header += [
            f"% run: {meta.get('started_utc')} on {meta.get('platform')}",
            f"% seed {prov.get('seed')}, taxonomy {prov.get('taxonomy_name')} "
            f"(hash {prov.get('taxonomy_hash')}), LeakBench {prov.get('leakbench_version')}",
            f"% status: {s.get('status')}",
        ]

    blocks = [
        t_vantage(_load(indir, "vantage")),
        p_vantage(_load(indir, "vantage")),
        t_coarse(_load(indir, "coarse")),
        p_ablation(_load(indir, "ablation")),
        t_subgroup(_load(indir, "subgroup")),
        p_subgroup(_load(indir, "subgroup")),
        t_defense(_load(indir, "defenses")),
        p_tradeoff(_load(indir, "defenses")),
        p_validation(_load(indir, "validation")),
        p_separability(_load(indir, "separability")),
        p_scalability(_load(indir, "scalability")),
        t_measured_and_plot(_load(indir, "measured_multiseed"), _load(indir, "vantage")),
        t_robustness(_load(indir, "robustness")),
        t_openworld(_load(indir, "openworld")),
        t_taxablation(_load(indir, "taxonomy_sweep")),
        t_live(_load(indir, "live"),
               _load(indir.parent / "live_neural", "live"),
               _load(indir.parent / "wan", "live_wan")),
        t_wire(_load(indir.parent / "wan", "wire_summary")),
        t_crossmodel(_load(indir.parent / "llm_sweep", "summary")),
        t_projection(_load(indir, "projections")),
        prose(indir),
    ]
    text = "\n".join(header) + "\n\n" + "\n\n".join(b.rstrip() for b in blocks) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        n_missing = text.count("MISSING")
        print(f"Wrote {args.output}" + (f"  ({n_missing} MISSING blocks)" if n_missing else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
