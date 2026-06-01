# Contributing

Thanks for contributing to CasNet! This project uses [uv](https://docs.astral.sh/uv/) for environment
management and [ruff](https://docs.astral.sh/ruff/) + [mypy](https://mypy-lang.org/) for quality checks.

## Development setup

```bash
uv sync               # core + dev dependencies into .venv
```

Add optional stacks when working on the training or quantum-chemistry code:

```bash
uv sync --extra ml --extra qc
```

## Running the checks

```bash
uv run pytest                       # test suite
uv run pytest --cov=casnet          # with coverage
uv run ruff check .                 # lint
uv run ruff format --check .        # formatting (drop --check to auto-format)
uv run mypy                         # type check (currently scoped to the core)
```

All four must pass before opening a PR. CI runs the same checks on Python 3.10 and 3.11.

## Conventions

- **Behaviour-preserving by default.** The quantum-chemistry numerics are pinned to the original
  dependency era; avoid changes that could alter model outputs or CASSCF results. When a refactor risks
  changing behaviour, add a characterization test first.
- **Keep the core lightweight.** Modules under `casnet.data.utils`, `casnet.config`,
  `casnet.data.splits`, and `casnet.data.geometries` must not import the heavy ML/QC stack, so they stay
  importable and unit-testable without `torch`/`pyscf`. PySCF/Torch-dependent code lives in dedicated
  modules.
- **Tests** mirror the package under `tests/`. Tests needing optional extras should guard their imports
  with `pytest.importorskip(...)` so the default run stays fast.
- **Linting scope.** The modernized core is linted and type-checked strictly. The legacy
  quantum-chemistry / evaluation / figure modules carry scoped per-file ignores (see `pyproject.toml`)
  pending deeper, separately-verifiable refactors; new code there should still aim to be clean.

## Type checking

`mypy` currently covers the pure-Python core (it runs without the heavy stack, keeping CI fast). As the
`ml`/`qc` modules gain type coverage, add them to the `files` list under `[tool.mypy]`.
