# CasNet

Neural-network orbital guesses for accelerating CASSCF convergence in quantum-chemistry calculations.

CasNet trains [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) (PaiNN) models to
predict CASSCF molecular-orbital coefficients / Fock matrices directly from a molecular geometry. The
predicted orbitals are then used as the initial guess for a CASSCF calculation (via PySCF or OpenMolcas),
reducing the number of macro/micro/inner iterations needed to converge. The repository contains the full
research pipeline used for the CasNet paper, benchmarked on fulvene.

> **Status:** this repository was recently modernized (uv + `src/` layout, ruff/mypy, tests, CI). The
> pure-Python data utilities install and run anywhere; the training / quantum-chemistry paths require
> additional opt-in dependencies and external software (see [Optional extras](#optional-extras)).

## Installation

CasNet uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone git@github.com:rhjvanworkum/CasNet.git
cd CasNet
uv sync            # creates .venv with the lightweight core + dev tools
```

This installs the lightweight core (NumPy/SciPy/ASE/h5py/tqdm) — enough to run the data-preparation
utilities, the examples, and the test suite.

### Optional extras

The heavy machine-learning and quantum-chemistry stacks are opt-in so that the default install and CI stay
fast. Install what you need:

```bash
uv sync --extra ml      # PyTorch 1.13 + SchNetPack 2.0 + Lightning (training / inference)
uv sync --extra qc      # PySCF (quantum-chemistry backend)
uv sync --extra viz     # matplotlib for figure plotting
uv sync --extra wandb   # Weights & Biases experiment tracking
```

> **Note:** the `ml` extra pins the original PyTorch 1.13 era and therefore requires **Python 3.10**
> (the version in `.python-version`). The lightweight core works on 3.10–3.12.

**External prerequisites** (not installed by uv):

- **Custom PySCF fork** — the iteration-counting evaluation scripts expect a PySCF fork whose CASSCF
  `kernel()` additionally returns macro/micro/inner iteration counts. Install it over the stock package
  in your environment (e.g. `uv pip install -e /path/to/pyscf-fork`).
- **OpenMolcas** — the OpenMolcas backend calls the `pymolcas` executable; set `MOLCAS_PATH` to point at it.
- **3D orbital rendering** — `evaluation/paper_figures/plot_orbitals.py` needs `mayavi`, `vtk`, and `PyQt5`
  (`uv pip install mayavi vtk PyQt5`).

### Configuration via environment variables

Paths are resolved at runtime (with sensible fallbacks) instead of being hard-coded:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CASNET_BASE_DIR` (or legacy `base_dir`) | Experiment data root | current directory |
| `CASNET_DATA_STORAGE` | ASE databases + split files | `./data_storage` |
| `CASNET_CHECKPOINTS` | Trained-model checkpoints | `./checkpoints` |
| `MOLCAS_PATH` | OpenMolcas `pymolcas` executable | `pymolcas` |
| `WANDB_PROJECT` | W&B project name (if `--use_wandb`) | `casnet` |

## Quickstart

Run the bundled examples (core only, no external software needed):

```bash
uv run examples/01_generate_split.py        # reproducible train/val/test split
uv run examples/02_interpolate_geometries.py # build a geometry scan + XYZ I/O
uv run examples/03_phase_correct_orbitals.py # align orbital phases across geometries
```

Run the tests and quality checks:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Usage

The full pipeline (requires the `ml` + `qc` extras and the external software above):

```bash
# 1. Generate a train/val/test split
uv run casnet-generate-split --preset geom_scan --name geom_scan_250 --seed 42

# 2. Run CASSCF calculations to build training data (PySCF fork)
uv run python -m casnet.data.casscf.pyscf.run_casscf_calculations \
    --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/ --basis sto_6g

# 3. Save results into an ASE database
uv run python -m casnet.data.db.save_casscf_calculations_to_db \
    --geometry_folder geometries/geom_scan_200/ --output_folder pyscf/geom_scan_200_sto_6g/

# 4. Train an ML-MO or ML-F model
uv run casnet-train --db_name geom_scan_200_sto_6g.db --split_name geom_scan_250.npz \
    --property mo_coeffs_adjusted --model_name gs200_sto_6g_MO
uv run casnet-train --db_name geom_scan_200_sto_6g.db --split_name geom_scan_250.npz \
    --property F --model_name gs200_sto_6g_F

# 5. Evaluate convergence / energy errors
uv run python -m casnet.evaluation.pyscf.compute_SCF_iterations \
    --geometry_folder geometries/geom_scan_200/ --split_name geom_scan_250.npz \
    --mode ML-MO --model gs200_sto_6g_MO --basis sto_6g
```

See [`examples/`](examples/) for runnable demonstrations of the core building blocks.

## Project structure

```
src/casnet/
├── config.py              # centralized constants + env-var path helpers
├── cli.py                 # `casnet-train` console-script entry point
├── data/
│   ├── utils.py           # pure-Python: Atom, XYZ I/O, CasscfResult, phase correction
│   ├── molden.py          # Molden output (PySCF)
│   ├── casscf/            # CASSCF runners (PySCF fork + OpenMolcas)
│   ├── db/                # ASE-database serialization
│   ├── geometries/        # geometry generation
│   └── splits/            # seeded train/val/test split generation
├── model/                 # PaiNN model, training (Lightning), inference, losses
└── evaluation/            # model-loss / SCF-iteration / energy-error scripts + paper figures
tests/                     # pytest suite for the pure-Python core
examples/                  # runnable, self-contained examples + bundled sample data
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and linting.

## License

[MIT](LICENSE).
