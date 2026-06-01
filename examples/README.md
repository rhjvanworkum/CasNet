# Examples

Self-contained, runnable examples that demonstrate the lightweight core of
CasNet. They use only the default dependencies (NumPy) and the small bundled
geometries in [`data/`](data/) — no GPU, PySCF, or OpenMolcas required.

Run any of them with `uv run`:

```bash
uv run examples/01_generate_split.py
uv run examples/02_interpolate_geometries.py
uv run examples/03_phase_correct_orbitals.py
```

| Example | Shows |
| --- | --- |
| `01_generate_split.py` | Reproducible, seeded train/val/test split generation |
| `02_interpolate_geometries.py` | Building a geometry scan from two endpoints + XYZ I/O |
| `03_phase_correct_orbitals.py` | Aligning orbital phases across geometries |

The full training and quantum-chemistry workflows (which require the `ml` and
`qc` extras plus the custom PySCF fork — see the top-level
[README](../README.md)) are documented there rather than as runnable examples,
since they depend on the external quantum-chemistry stack and trained
checkpoints.
