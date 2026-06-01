"""Tests for reproducible train/val/test split generation."""

import numpy as np

from casnet.data.splits.generate import PRESETS, generate_split, save_split


def test_geom_scan_preset_sizes():
    train, val, test = generate_split(seed=0, **PRESETS["geom_scan"])
    assert (len(train), len(val), len(test)) == (200, 25, 25)
    # disjoint and complete cover of 0..249
    assert set(train) | set(val) | set(test) == set(range(250))
    assert len(set(train) & set(val)) == 0


def test_normal_preset_sizes():
    train, val, test = generate_split(seed=0, **PRESETS["normal"])
    assert (len(train), len(val), len(test)) == (4500, 500, 0)


def test_wigner_preset_holds_out_fixed_test():
    train, val, test = generate_split(seed=0, **PRESETS["wigner"])
    assert (len(train), len(val), len(test)) == (200, 25, 25)
    assert list(test) == list(range(225, 250))
    # train/val are drawn only from the 0..224 pool
    assert set(train) | set(val) <= set(range(225))


def test_seed_is_reproducible():
    a = generate_split(seed=123, **PRESETS["geom_scan"])
    b = generate_split(seed=123, **PRESETS["geom_scan"])
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_different_seeds_differ():
    a = generate_split(seed=1, **PRESETS["geom_scan"])
    b = generate_split(seed=2, **PRESETS["geom_scan"])
    assert not np.array_equal(a[0], b[0])


def test_save_split_roundtrip(tmp_path):
    train, val, test = generate_split(seed=0, **PRESETS["geom_scan"])
    path = tmp_path / "split.npz"
    save_split(str(path), train, val, test)
    loaded = np.load(path)
    assert np.array_equal(loaded["train_idx"], train)
    assert np.array_equal(loaded["val_idx"], val)
    assert np.array_equal(loaded["test_idx"], test)
