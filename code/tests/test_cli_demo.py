"""CLI smoke test + end-to-end demo test (required tests: CLI runs on synthetic
data; no fake result files shipped as real)."""

import json
from pathlib import Path

import pytest

from leakbench.cli import main
from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.demo import run_synthetic_demo


def test_cli_info_runs(capsys):
    rc = main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "LeakBench" in out
    assert "H(C)" in out


def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_synthetic_demo_end_to_end(tmp_path):
    cfg = ExperimentConfig(
        name="test_demo", num_classes=12, traces_per_class=12, bootstrap=10,
        output_dir=str(tmp_path / "demo"),
    )
    out = run_synthetic_demo(cfg)
    # All expected result files exist.
    for stage in ("vantage", "ablation", "subgroup", "defenses", "validation", "summary"):
        assert (out / f"{stage}.json").exists()
    # Vantage ordering: providers must refine both (advantage >= network, platform).
    vant = json.loads((out / "vantage.json").read_text())["result"]["advantage_by_vantage"]
    assert vant["providers"] + 1e-9 >= vant["network"]
    assert vant["providers"] + 1e-9 >= vant["platform"]


def test_demo_outputs_labeled_synthetic(tmp_path):
    """No demo result may masquerade as a paper result: every file must carry the
    synthetic label."""
    cfg = ExperimentConfig(
        name="t", num_classes=12, traces_per_class=12, bootstrap=10,
        output_dir=str(tmp_path / "demo"),
    )
    out = run_synthetic_demo(cfg)
    for jf in Path(out).glob("*.json"):
        payload = json.loads(jf.read_text())
        blob = json.dumps(payload).lower()
        assert "synthetic" in blob, f"{jf.name} is not labeled synthetic"
