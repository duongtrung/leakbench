"""Real agent-trace ingestion --- the ``data.root`` path of ``docs/data_format.md``.

LeakBench **never downloads data**. This module reads trace files that a user has
placed on disk themselves (see ``docs/ec2_experiment_guide.md`` §3), validates them
against the canonical record schema (Def. "Agent trace"), and returns the same
:class:`~leakbench.data.schemas.TraceDataset` object the generator / measured
harness produce --- so observation maps, defenses, and the estimator run unchanged.

Expected layout (``docs/data_format.md``)::

    <data.root>/
    ├── taxonomy.json        # optional: {name, classes:[{index,name,family}], prior:[...]}
    ├── train/class_00.jsonl # one JSON trace record per line
    ├── test/                # optional; else LeakBench does its own stratified split
    └── splits.json          # optional: {"open_world_other": [class indices]}

Validation is strict and *loud*: a malformed record raises with the offending file
and line number rather than being silently dropped, because a silently-skipped trace
would change a leakage number without any record of why.

STATUS: implemented (ingestion). Providing the data is the user's responsibility;
nothing here contacts the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from leakbench.data.schemas import Trace, TraceDataset
from leakbench.metrics.openworld import OTHER_LABEL
from leakbench.taxonomy import (
    FAMILY_SIGNATURES,
    TOOL_CATEGORIES,
    TaskClass,
    Taxonomy,
    reference_taxonomy,
)

TRACE_SUFFIXES = (".jsonl", ".ndjson")


class TraceFormatError(ValueError):
    """A real-data file violates the ``docs/data_format.md`` schema."""


# ------------------------------------------------------------------ taxonomy
def load_taxonomy_json(path: str | Path) -> Taxonomy:
    """Load a ``taxonomy.json`` describing the classes/prior of a real dataset.

    Schema::

        {"name": "my-tax/6c-2f",
         "classes": [{"index": 0, "name": "...", "family": "research"}, ...],
         "prior": [...]}            # optional; uniform if absent

    Families must be one of the four taxonomy families, because the signature-tool
    structure (and therefore the coarse 4-way secret) is defined per family.
    """
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        raw = json.load(fh)
    if "classes" not in raw:
        raise TraceFormatError(f"{p}: taxonomy.json must contain a 'classes' array.")

    classes: list[TaskClass] = []
    for i, c in enumerate(raw["classes"]):
        idx = int(c.get("index", i))
        fam = str(c["family"])
        if fam not in FAMILY_SIGNATURES:
            raise TraceFormatError(
                f"{p}: class {idx} has family {fam!r}; expected one of "
                f"{sorted(FAMILY_SIGNATURES)}."
            )
        classes.append(TaskClass(index=idx, name=str(c["name"]), family=fam))
    classes.sort(key=lambda c: c.index)
    if [c.index for c in classes] != list(range(len(classes))):
        raise TraceFormatError(f"{p}: class indices must be 0..{len(classes) - 1} with no gaps.")

    if "prior" in raw and raw["prior"] is not None:
        prior = np.asarray(raw["prior"], dtype=float)
        if prior.shape != (len(classes),):
            raise TraceFormatError(
                f"{p}: prior has length {prior.shape[0]}, expected {len(classes)}."
            )
        if not np.isclose(prior.sum(), 1.0):
            raise TraceFormatError(f"{p}: prior must sum to 1 (got {prior.sum():.6f}).")
    else:
        prior = np.full(len(classes), 1.0 / len(classes))

    return Taxonomy(classes=tuple(classes), prior=prior,
                    name=str(raw.get("name", f"real/{len(classes)}c")))


# --------------------------------------------------------------------- traces
def _validate_record(rec: Any, path: Path, line_no: int, num_classes: int | None) -> Trace:
    if not isinstance(rec, dict):
        raise TraceFormatError(f"{path}:{line_no}: record is not a JSON object.")
    if "calls" not in rec or "label" not in rec:
        raise TraceFormatError(
            f"{path}:{line_no}: record must have 'label' and 'calls' "
            f"(see docs/data_format.md)."
        )
    calls = rec["calls"]
    if not isinstance(calls, list) or not calls:
        raise TraceFormatError(f"{path}:{line_no}: 'calls' must be a non-empty array.")
    for j, c in enumerate(calls):
        missing = {"category", "request_size", "response_size", "gap"} - set(c)
        if missing:
            raise TraceFormatError(
                f"{path}:{line_no}: call {j} is missing {sorted(missing)}."
            )
        if c["category"] not in TOOL_CATEGORIES:
            raise TraceFormatError(
                f"{path}:{line_no}: call {j} has category {c['category']!r}; expected one "
                f"of {list(TOOL_CATEGORIES)}."
            )
    label = int(rec["label"])
    if num_classes is not None and label != OTHER_LABEL and not (0 <= label < num_classes):
        raise TraceFormatError(
            f"{path}:{line_no}: label {label} outside 0..{num_classes - 1} "
            f"(or {OTHER_LABEL} for open-world 'other')."
        )
    try:
        trace = Trace.from_record(rec)
    except ValueError as exc:  # schema constraints from ToolCall.__post_init__
        raise TraceFormatError(f"{path}:{line_no}: {exc}") from exc
    trace.metadata.setdefault("source", "real")
    trace.metadata["source_file"] = path.name
    return trace


def iter_trace_file(path: str | Path, num_classes: int | None = None) -> Iterator[Trace]:
    """Yield validated traces from one newline-delimited-JSON trace file."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"{p}:{line_no}: invalid JSON ({exc.msg}).") from exc
            yield _validate_record(rec, p, line_no, num_classes)


def load_split(
    root: str | Path,
    split: str,
    taxonomy: Taxonomy,
    *,
    other_classes: frozenset[int] = frozenset(),
) -> list[Trace]:
    """Load every ``*.jsonl`` under ``<root>/<split>/`` (recursively)."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"{split_dir} does not exist. LeakBench never downloads data; place trace "
            f"files there yourself in the docs/data_format.md layout."
        )
    files = sorted(
        f for f in split_dir.rglob("*") if f.suffix in TRACE_SUFFIXES and f.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"No {'/'.join(TRACE_SUFFIXES)} trace files under {split_dir}."
        )
    traces: list[Trace] = []
    for f in files:
        for t in iter_trace_file(f, num_classes=taxonomy.num_classes):
            if t.label in other_classes:
                t.metadata["original_label"] = int(t.label)
                t.label = OTHER_LABEL
            elif t.label != OTHER_LABEL and t.family is None:
                t.family = taxonomy.family_of(t.label)
            traces.append(t)
    return traces


@dataclass
class RealTraceBundle:
    """What ``data.root`` yields: a taxonomy plus one or two labeled splits."""

    taxonomy: Taxonomy
    train: TraceDataset
    test: TraceDataset | None = None
    open_world_other: tuple[int, ...] = ()
    root: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_explicit_test(self) -> bool:
        return self.test is not None and len(self.test) > 0

    def combined(self) -> TraceDataset:
        """Train + test as one dataset (LeakBench then does its own stratified split).

        Use this only when the provider did *not* ship a canonical split; when
        ``has_explicit_test`` is true, honoring their split is the comparable choice.
        """
        if not self.has_explicit_test:
            return self.train
        assert self.test is not None
        return TraceDataset(
            traces=list(self.train.traces) + list(self.test.traces),
            taxonomy_name=self.train.taxonomy_name,
            taxonomy_hash=self.train.taxonomy_hash,
            source=self.train.source,
            metadata={**self.train.metadata, "merged_splits": ["train", "test"]},
        )


def load_real_dataset(root: str | Path) -> RealTraceBundle:
    """Load a real trace dataset from ``data.root`` (``docs/data_format.md``).

    Raises
    ------
    FileNotFoundError
        if ``root`` (or its ``train/`` directory) is absent --- LeakBench does not
        fetch it for you.
    TraceFormatError
        on any schema violation, naming the file and line.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"data.root {root_path} does not exist. LeakBench never downloads data; "
            f"see docs/data_format.md for the expected layout."
        )

    tax_file = root_path / "taxonomy.json"
    taxonomy = load_taxonomy_json(tax_file) if tax_file.exists() else reference_taxonomy()

    other_classes: tuple[int, ...] = ()
    splits_file = root_path / "splits.json"
    if splits_file.exists():
        with open(splits_file, encoding="utf-8") as fh:
            splits = json.load(fh)
        other_classes = tuple(int(i) for i in splits.get("open_world_other", []))

    other_set = frozenset(other_classes)
    train_traces = load_split(root_path, "train", taxonomy, other_classes=other_set)
    test_traces: list[Trace] = []
    if (root_path / "test").is_dir():
        test_traces = load_split(root_path, "test", taxonomy, other_classes=other_set)

    def _wrap(traces: list[Trace], split: str) -> TraceDataset:
        return TraceDataset(
            traces=traces,
            taxonomy_name=taxonomy.name,
            taxonomy_hash=taxonomy.content_hash(),
            source="real",
            metadata={"split": split, "root": str(root_path), "n_traces": len(traces)},
        )

    return RealTraceBundle(
        taxonomy=taxonomy,
        train=_wrap(train_traces, "train"),
        test=_wrap(test_traces, "test") if test_traces else None,
        open_world_other=other_classes,
        root=str(root_path),
        metadata={
            "taxonomy_source": "taxonomy.json" if tax_file.exists() else "reference",
            "n_train": len(train_traces),
            "n_test": len(test_traces),
        },
    )


def dump_trace_dataset(dataset: TraceDataset, path: str | Path) -> Path:
    """Write a dataset in the ``docs/data_format.md`` JSONL form.

    Useful to persist adapter output (or a generator draw) so a later run ingests it
    through the same ``data.root`` path a real dataset takes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for t in dataset.traces:
            fh.write(json.dumps(t.to_record()) + "\n")
    return p
