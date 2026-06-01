"""Tests for orbital phase correction."""

import numpy as np

from casnet.data.utils import phase_correct_orbitals


def test_flips_negatively_overlapping_columns():
    ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[-1.0, 0.0], [0.0, 1.0]])  # first column points opposite to ref

    corrected = phase_correct_orbitals(ref, target)

    # first column flipped to align with ref, second left alone
    assert np.allclose(corrected[:, 0], [1.0, 0.0])
    assert np.allclose(corrected[:, 1], [0.0, 1.0])


def test_identity_when_already_aligned():
    ref = np.eye(3)
    target = np.eye(3)
    assert np.allclose(phase_correct_orbitals(ref, target), np.eye(3))


def test_does_not_mutate_input():
    ref = np.eye(2)
    target = np.array([[-1.0, 0.0], [0.0, -1.0]])
    target_copy = target.copy()
    phase_correct_orbitals(ref, target)
    assert np.allclose(target, target_copy)


def test_double_correction_is_stable():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((4, 4))
    target = rng.standard_normal((4, 4))
    once = phase_correct_orbitals(ref, target)
    twice = phase_correct_orbitals(ref, once)
    assert np.allclose(once, twice)
