"""Real-trace ingestion tests (docs/data_format.md layout).

No data is downloaded: every fixture is written into a tmp_path by the test itself,
which is exactly how a user supplies real traces.
"""

import json

import pytest

from leakbench.data.generator import CalibratedGenerator
from leakbench.data.loader import (
    TraceFormatError,
    dump_trace_dataset,
    load_real_dataset,
    load_taxonomy_json,
)
from leakbench.experiments.config import ExperimentConfig
from leakbench.experiments.runners import build_dataset
from leakbench.metrics.openworld import OTHER_LABEL
from leakbench.observables.maps import featurize_dataset
from leakbench.taxonomy import reference_taxonomy


def _write_dataset(root, traces_per_class=3, split="train"):
    ds = CalibratedGenerator(taxonomy=reference_taxonomy(), seed=5).sample_dataset(
        traces_per_class
    )
    (root / split).mkdir(parents=True, exist_ok=True)
    dump_trace_dataset(ds, root / split / "class_all.jsonl")
    return ds


def test_roundtrip_dump_and_load(tmp_path):
    original = _write_dataset(tmp_path)
    bundle = load_real_dataset(tmp_path)
    assert len(bundle.train) == len(original)
    assert bundle.train.source == "real"
    assert not bundle.has_explicit_test
    # The observable pipeline runs on ingested traces unchanged.
    X, y = featurize_dataset(bundle.train, "providers")
    assert X.shape[0] == len(original)
    assert set(y) == set(original.labels())


def test_explicit_test_split_is_honored(tmp_path):
    _write_dataset(tmp_path, 3, "train")
    _write_dataset(tmp_path, 2, "test")
    bundle = load_real_dataset(tmp_path)
    assert bundle.has_explicit_test
    assert len(bundle.test) == 12 * 2
    assert len(bundle.combined()) == 12 * 5


def test_taxonomy_json_is_used(tmp_path):
    _write_dataset(tmp_path, 2)
    tax_doc = {
        "name": "custom/12c-4f",
        "classes": [
            {"index": i, "name": f"task_{i}", "family": fam}
            for i, fam in enumerate(["research", "coding", "personal", "ops"] * 3)
        ],
    }
    (tmp_path / "taxonomy.json").write_text(json.dumps(tax_doc))
    bundle = load_real_dataset(tmp_path)
    assert bundle.taxonomy.name == "custom/12c-4f"
    assert bundle.taxonomy.num_classes == 12


def test_splits_json_folds_classes_into_other(tmp_path):
    _write_dataset(tmp_path, 2)
    (tmp_path / "splits.json").write_text(json.dumps({"open_world_other": [10, 11]}))
    bundle = load_real_dataset(tmp_path)
    labels = list(bundle.train.labels())
    assert OTHER_LABEL in labels
    assert 10 not in labels and 11 not in labels


def test_missing_root_raises_and_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        load_real_dataset(tmp_path / "nope")


def test_bad_category_names_file_and_line(tmp_path):
    (tmp_path / "train").mkdir(parents=True)
    rec = {"label": 0, "family": "research",
           "calls": [{"category": "not_a_tool", "request_size": 1.0,
                      "response_size": 2.0, "gap": 0.0}]}
    (tmp_path / "train" / "a.jsonl").write_text(json.dumps({"label": 0, "calls": [
        {"category": "search", "request_size": 1.0, "response_size": 2.0, "gap": 0.0}]})
        + "\n" + json.dumps(rec) + "\n")
    with pytest.raises(TraceFormatError, match="a.jsonl:2"):
        load_real_dataset(tmp_path)


def test_negative_size_is_rejected(tmp_path):
    (tmp_path / "train").mkdir(parents=True)
    rec = {"label": 0, "calls": [{"category": "search", "request_size": -1.0,
                                  "response_size": 2.0, "gap": 0.0}]}
    (tmp_path / "train" / "a.jsonl").write_text(json.dumps(rec) + "\n")
    with pytest.raises(TraceFormatError):
        load_real_dataset(tmp_path)


def test_label_out_of_range_is_rejected(tmp_path):
    (tmp_path / "train").mkdir(parents=True)
    rec = {"label": 99, "calls": [{"category": "search", "request_size": 1.0,
                                   "response_size": 2.0, "gap": 0.0}]}
    (tmp_path / "train" / "a.jsonl").write_text(json.dumps(rec) + "\n")
    with pytest.raises(TraceFormatError, match="outside"):
        load_real_dataset(tmp_path)


def test_taxonomy_json_rejects_unknown_family(tmp_path):
    doc = {"name": "x", "classes": [{"index": 0, "name": "a", "family": "astrology"}]}
    p = tmp_path / "taxonomy.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(TraceFormatError, match="family"):
        load_taxonomy_json(p)


def test_config_source_real_is_wired(tmp_path):
    _write_dataset(tmp_path, 4)
    cfg = ExperimentConfig(source="real", data_root=str(tmp_path), num_classes=12)
    ds, tax = build_dataset(cfg)
    assert len(ds) == 48
    assert tax.num_classes == 12


def test_source_real_without_root_raises():
    cfg = ExperimentConfig(source="real", data_root=None)
    with pytest.raises(ValueError, match="data.root"):
        build_dataset(cfg)
