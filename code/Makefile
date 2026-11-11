.PHONY: help install install-dev test lint typecheck verify demo experiment scale \
        scale-quick supporting paper-numbers plots clean

help:
	@echo "LeakBench make targets:"
	@echo "  install      pip install the package (editable)"
	@echo "  install-dev  install with dev extras (pytest, ruff, mypy)"
	@echo "  test         run the pytest suite (synthetic data only)"
	@echo "  lint         ruff check"
	@echo "  typecheck    mypy src"
	@echo "  demo         run the synthetic smoke-test demo"
	@echo "  experiment   run the full generator experiment (configs/default.yaml)"
	@echo "  scale-quick  end-to-end check of every stage at tiny sizes (~1 min)"
	@echo "  scale        the full paper-scale study -- RUN THIS ON A SERVER"
	@echo "  supporting   the supporting experiments the paper cites (no credentials)"
	@echo "  paper-numbers regenerate every table/figure value from stored results"
	@echo "  verify       everything that must pass before believing a number"
	@echo "  plots        regenerate plots from outputs/"
	@echo "  clean        remove caches and outputs"
	@echo ""
	@echo "Needs something this repo cannot bundle:"
	@echo "  live LLM / cross-model sweep  -> an API key (scripts/run_model_sweep.py)"
	@echo "  wide-area RTT sweep           -> a second host (docs/wan_rtt_sweep_setup.md)"

install:
	pip install -e . --break-system-packages

install-dev:
	pip install -e ".[dev]" --break-system-packages

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy src

demo:
	python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml

experiment:
	python scripts/run_experiment.py --config configs/default.yaml --stage all

scale-quick:
	python scripts/run_scale_study.py --quick --output outputs/scale_quick

scale:
	python scripts/run_scale_study.py --output outputs/scale_study

# The experiments the paper cites beyond the main study. All are CPU-only and need no
# credentials, so an artifact evaluator can run them unattended. `scale` must have run
# first: several of these read its stored outputs rather than recomputing them.
supporting:
	python scripts/val_scaling.py
	python scripts/defenses_platform.py
	python scripts/measured_repeatability.py
	python scripts/bigram_ablation.py
	python scripts/probe_subsets.py
	python scripts/vantage_ordering_vs_n.py

paper-numbers:
	python scripts/emit_paper_numbers.py --output outputs/scale_study/paper_numbers.tex
	@grep -c MISSING outputs/scale_study/paper_numbers.tex | grep -qx 0 \
	  || (echo "FAIL: some table rows have no measured value behind them"; exit 1)
	@echo "OK: every paper value is generated from a stored result"

# The gate before trusting any number: the suite, the linters, the cross-stage
# identities (the same quantity computed by different stages must agree), and the
# emitter reporting no MISSING rows.
verify: test lint typecheck
	python scripts/check_cross_stage.py
	$(MAKE) paper-numbers

plots:
	python scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/plots

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ src/**/__pycache__
	find outputs -type f ! -name '.gitkeep' -delete 2>/dev/null || true
