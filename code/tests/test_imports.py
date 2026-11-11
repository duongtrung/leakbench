"""Package-import smoke tests (required test #1)."""


def test_top_level_import():
    import leakbench

    assert leakbench.__version__


def test_submodule_imports():
    # Every public sub-package must import cleanly with no side effects.
    import leakbench.cli  # noqa: F401
    import leakbench.data  # noqa: F401
    import leakbench.data.adapters  # noqa: F401
    import leakbench.data.generator  # noqa: F401
    import leakbench.data.measured  # noqa: F401
    import leakbench.data.wire  # noqa: F401
    import leakbench.defenses  # noqa: F401
    import leakbench.estimators  # noqa: F401
    import leakbench.experiments  # noqa: F401
    import leakbench.metrics  # noqa: F401
    import leakbench.observables  # noqa: F401
    import leakbench.taxonomy  # noqa: F401
    import leakbench.validation  # noqa: F401


def test_cli_parser_builds():
    from leakbench.cli import build_parser

    parser = build_parser()
    assert parser.prog == "leakbench"
