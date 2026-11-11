# Reproducibility

## Seeds

All randomness derives from a single logged seed (paper §Impl). The reference seed
is `20270214`. `utils/seeding.py` provides `seed_everything(seed)` (global Python +
NumPy) and `make_rng(seed)` (an explicit, independent `np.random.Generator`, which
library code prefers so concurrent components stay deterministic).

Determinism is unit-tested: `tests/test_estimator.py::test_estimator_reproducible_under_seed`
and `tests/test_generator.py::test_generator_deterministic_under_seed`.

## Provenance

Every result file embeds a `provenance` block (`utils/provenance.py`):

```json
{
  "leakbench_version": "0.1.0",
  "python_version": "3.11.x",
  "platform": "...",
  "seed": 20270214,
  "taxonomy_hash": "8ba36962dec5",
  "taxonomy_name": "tax-v1/12c-4f",
  "config": { ... full ExperimentConfig ... }
}
```

Results are **keyed by taxonomy hash** so a later, larger taxonomy does not silently
invalidate comparison (paper §Design: reports keyed by taxonomy hash, framework
version, defense params, seed).

## Configs

A run is fully determined by its YAML config (`configs/`), which is captured verbatim
into provenance. To reproduce a run, re-run with the same config and seed.

## Outputs

Result files live under `output_dir` (default `outputs/...`), gitignored except
`.gitkeep`. Each carries a `status` field:

- `synthetic smoke-test output, not a paper result` — from the generator demo;
- `real_computation_on_synthetic_or_local_data` — generator/measured experiment runs;
- `real_computation_at_paper_scale` — a full `scripts/run_scale_study.py` run;
- `quick_check_not_paper_scale` — `run_scale_study.py --quick` (pipeline check only);
- `live_captured_traffic (<capture_mode>)` — the RQ8 live script, where
  `capture_mode` is the transport actually captured (`tls_captured_proxy` or
  `tcp_captured_proxy`) and the dataset metadata separately records the `link`
  (`loopback` or `remote`), resolved from the server address rather than assumed.

No file is ever stamped as a measurement of a deployed product.

## What is *not* reproducible bit-for-bit

The live experiment (RQ8) records wall-clock inter-arrival gaps off a real socket, so
its timing features differ run to run even at a fixed seed; the seed fixes the
planner's stochastic choices and the server's tool-parameter draws, not the host's
scheduling. Sizes are reproducible; timing is a measurement. The measured harness has
the same property for its gap features. Both record this in their dataset metadata.

## Environment

Python 3.11+. Dependencies pinned by lower bound in `requirements.txt` /
`pyproject.toml` (numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml). No GPU, no
internet (beyond loopback sockets for the optional live experiment). Estimator
hyperparameters are fixed across all conditions (App. "Equation Audit"): RF 200 trees,
kNN k=7, stratified 60/40 split, bootstrap B=200 (main) / 50 (sweeps), fixed size and
inter-arrival histogram buckets.

## Determinism caveats

- The **measured harness** records true wall-clock timing, which varies run to run.
  The tool *category sequence and byte sizes* are seeded and reproducible; the
  absolute `gap` values are not (they reflect real machine timing, by design). This
  is intrinsic to a "measured" source and disclosed as such.
- The **live experiment** captures real socket bytes; kernel coalescing introduces
  run-to-run variation in the captured chunk sizes. The planner's branching is seeded.
