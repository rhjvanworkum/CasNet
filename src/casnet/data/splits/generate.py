"""Generate reproducible train/validation/test splits.

Replaces the three previous standalone scripts (``generate_geom_scan_split``,
``generate_normal_dist_split``, ``generate_wigner_dist_split``), which shared the
same logic but used an unseeded global ``np.random.shuffle`` and therefore were
not reproducible. The split logic now lives in :func:`generate_split` and accepts
an explicit ``seed``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

#: Named presets reproducing the original three split scripts.
PRESETS: dict[str, dict[str, Any]] = {
    "geom_scan": dict(n_total=250, train_frac=0.8, val_frac=0.1),
    "normal": dict(n_total=5000, train_frac=0.9, val_frac=0.1),
    "wigner": dict(
        n_total=250, train_frac=0.8, val_frac=0.1, pool_size=225, fixed_test=range(225, 250)
    ),
}


def generate_split(
    n_total: int,
    train_frac: float,
    val_frac: float,
    *,
    seed: int | None = None,
    pool_size: int | None = None,
    fixed_test: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (train, val, test) index split.

    Args:
        n_total: Dataset size used to derive the train/val counts.
        train_frac: Fraction of ``n_total`` used for training.
        val_frac: Fraction of ``n_total`` used for validation.
        seed: Seed for the RNG; pass an integer for reproducible splits.
        pool_size: Number of indices to shuffle into train/val. Defaults to
            ``n_total``. Use a smaller pool when a fixed test set is held out.
        fixed_test: Explicit test indices. When given, all remaining shuffled
            pool indices beyond the train count become the validation set.

    Returns:
        ``(train_idx, val_idx, test_idx)`` arrays.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(pool_size if pool_size is not None else n_total)
    rng.shuffle(pool)

    n_train = int(train_frac * n_total)
    n_val = int((train_frac + val_frac) * n_total) - n_train

    train_idx = pool[:n_train]
    if fixed_test is not None:
        val_idx = pool[n_train:]
        test_idx = np.asarray(list(fixed_test))
    else:
        val_idx = pool[n_train : n_train + n_val]
        test_idx = pool[n_train + n_val :]

    return train_idx, val_idx, test_idx


def save_split(
    save_path: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """Save a split to a ``.npz`` archive with ``train_idx``/``val_idx``/``test_idx`` keys."""
    np.savez(save_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--name", required=True, help="Output split name (without extension)")
    parser.add_argument("--out-dir", default="./data_storage", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    train_idx, val_idx, test_idx = generate_split(seed=args.seed, **PRESETS[args.preset])
    os.makedirs(args.out_dir, exist_ok=True)
    save_split(os.path.join(args.out_dir, f"{args.name}.npz"), train_idx, val_idx, test_idx)
    print(
        f"Wrote split '{args.name}' "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )


if __name__ == "__main__":
    main()
