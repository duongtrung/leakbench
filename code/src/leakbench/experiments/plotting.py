"""Plotting --- generate figures FROM result files (never from hard-coded numbers).

Every plot produced from synthetic data is stamped with the banner
"synthetic smoke-test output, not a paper result" so a viewer can never mistake a
demo figure for a measurement of a deployed system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from leakbench.utils.io import ensure_dir  # noqa: E402

SYNTHETIC_BANNER = "synthetic smoke-test output, not a paper result"


def _stamp(ax, synthetic: bool) -> None:
    if synthetic:
        ax.figure.text(0.5, 0.01, SYNTHETIC_BANNER, ha="center", va="bottom",
                       fontsize=8, color="crimson", style="italic")


def plot_vantage(result: dict[str, Any], out_path: str | Path, synthetic: bool = True) -> Path:
    adv = result["advantage_by_vantage"]
    vantages = list(adv.keys())
    values = [adv[v] for v in vantages]
    cis = []
    for v in vantages:
        ci = result["vantages"][v]["advantage_ci"]
        cis.append(ci)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    lowers = [values[i] - cis[i][0] for i in range(len(values))]
    uppers = [cis[i][1] - values[i] for i in range(len(values))]
    ax.bar(vantages, values, color=["#E76F51", "#2A9D8F", "#6A4C93"],
           yerr=[lowers, uppers], capsize=4)
    ax.set_ylabel("attacker advantage")
    ax.set_ylim(0, 1)
    ax.set_title("Leakage by vantage (RQ1)")
    _stamp(ax, synthetic)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_ablation(result: dict[str, Any], out_path: str | Path, synthetic: bool = True) -> Path:
    abls = result["ablations"]
    names = list(abls.keys())
    values = [abls[n]["advantage"] for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(names, values, color="#1F4E79")
    ax.set_ylabel("attacker advantage")
    ax.set_ylim(0, 1)
    ax.set_title("Feature ablation, network vantage (RQ2)")
    ax.tick_params(axis="x", rotation=20)
    _stamp(ax, synthetic)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_defenses(result: dict[str, Any], out_path: str | Path, synthetic: bool = True) -> Path:
    rows = result["rows"]
    bw = [r["bandwidth_overhead"] * 100 for r in rows]
    adv = [r["advantage"] for r in rows]
    labels = [r["defense"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(bw, adv, color="#BC4749", zorder=3)
    for x, y, lab in zip(bw, adv, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("bandwidth overhead (%)")
    ax.set_ylabel("residual attacker advantage")
    ax.set_ylim(0, 1)
    ax.set_title("Privacy-utility frontier (RQ4)")
    _stamp(ax, synthetic)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_validation(result: dict[str, Any], out_path: str | Path, synthetic: bool = True) -> Path:
    rows = result["rows"]
    ps = [r["p"] for r in rows]
    ana = [r["analytic_mi_bits"] for r in rows]
    est = [r["estimated_mi_bits"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ps, ana, "o-", label="analytic MI", color="#1F4E79")
    ax.plot(ps, est, "s--", label="estimated MI (lower bound)", color="#E76F51")
    ax.set_xlabel("crossover p")
    ax.set_ylabel("mutual information (bits)")
    ax.set_title("Estimator validation (RQ5)")
    ax.legend(fontsize=8)
    _stamp(ax, synthetic)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = ensure_dir(Path(out_path).parent) / Path(out_path).name
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
