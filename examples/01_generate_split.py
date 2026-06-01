"""Generate a reproducible train/validation/test split.

Run with:

    uv run examples/01_generate_split.py

Demonstrates :func:`casnet.data.splits.generate.generate_split`, which replaces
the old unseeded split scripts. Uses only the lightweight core (NumPy).
"""

import tempfile
from pathlib import Path

from casnet.data.splits.generate import PRESETS, generate_split, save_split


def main() -> None:
    # The "geom_scan" preset: 250 geometries, 80/10/10 split.
    train, val, test = generate_split(seed=42, **PRESETS["geom_scan"])
    print(f"geom_scan split -> train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"first 5 train indices: {train[:5].tolist()}")

    # Splits are reproducible: same seed -> identical indices.
    train2, _, _ = generate_split(seed=42, **PRESETS["geom_scan"])
    print(f"reproducible with same seed: {bool((train == train2).all())}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "geom_scan_250.npz"
        save_split(str(out), train, val, test)
        print(f"saved split to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
